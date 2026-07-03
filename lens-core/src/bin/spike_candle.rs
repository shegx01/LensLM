//! candle + Metal validation & benchmark harness (issue #91).
//!
//! RETAINED dev tool (not shipped — gated behind `native-ml-metal`, and the
//! `[[bin]]` builds only with that feature). Reproduces the evidence behind the
//! candle+Metal decision (see `.omc/plans/issue-91-candle-metal-spike-results.md`)
//! so parity/throughput/offload can be re-measured on demand — e.g. after a candle
//! upgrade or on new Apple-Silicon hardware. The production assertion guardrail
//! lives in `tests/candle_metal_parity.rs`; this bin is for measurement.
//! Answers the four questions that greenlit the path:
//!   Q4 — cross-engine cosine parity (candle-CPU/Metal vs production fastembed/ORT).
//!        (recall@5 docs=candle/query=fastembed lives in the `recall` subcommand.)
//!   Q1 — candle-CPU vs ORT/fastembed-CPU throughput (replace vs supplement).
//!   Q2 — does candle-Metal actually FREE the CPU on a sustained bulk re-embed?
//!
//! Run (NEVER execute the ./target binary directly — provenance lock):
//!   cargo run -p lens-core --features native-ml-metal --bin spike_candle -- [all|parity|throughput|offload]
//!
//! Gated behind `native-ml-metal` (aarch64-apple-darwin only).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use lens_core::chunk::chunk_blocks_deterministic;
use lens_core::embedder::{CandleNomicEmbedder, Compute, Embedder, FastembedEmbedder};
use lens_core::parse::{SourceKind, parse_blocks};

/// Main eval corpus stems (matches `eval.rs` MAIN_DOCS) — raw recall@5 = 1.00 with
/// fastembed today; the floor is MAIN_RECALL_FLOOR = 0.75.
const MAIN_DOCS: &[&str] = &["espresso", "photosynthesis", "rust_ownership", "tides"];
const RECALL_FLOOR: f32 = 0.75;
const K: usize = 5;

fn spike_cache_dir() -> PathBuf {
    let base = std::env::var("SPIKE_CACHE_DIR").unwrap_or_else(|_| {
        format!(
            "{}/lenslm-spike-candle",
            std::env::temp_dir().to_string_lossy()
        )
    });
    PathBuf::from(base)
}

/// A small varied doc corpus (representative of real chunk text — prose,
/// technical, list-y) for parity + throughput. Short-to-medium so tokenization
/// stays in the common < 512-token regime.
const CORPUS: &[&str] = &[
    "The Voyager Golden Record carries sounds and images of Earth into interstellar space.",
    "Rust's ownership model eliminates data races at compile time without a garbage collector.",
    "Mean pooling averages token embeddings weighted by the attention mask before normalization.",
    "The Antikythera mechanism is an ancient Greek analog computer for predicting eclipses.",
    "LanceDB stores vectors in a columnar Arrow format with disk-based approximate search.",
    "Photosynthesis converts carbon dioxide and water into glucose using light energy.",
    "A transformer's self-attention lets every token attend to every other token in the sequence.",
    "The Metal API exposes Apple GPU compute kernels for on-device machine learning.",
    "Coreference resolution links pronouns and definite descriptions to their named antecedents.",
    "Nomic embed uses rotary position embeddings instead of absolute position tables.",
    "Retrieval-augmented generation grounds a language model's answers in fetched documents.",
    "The disc was chosen to represent humanity to any extraterrestrial civilization that finds it.",
    "SwiGLU gates one linear projection with the SiLU activation of a parallel projection.",
    "Cosine similarity measures the angle between two vectors, ignoring their magnitude.",
    "Apple Silicon unifies CPU, GPU, and Neural Engine memory on a single die.",
    "Tokenizers split text into subword units drawn from a fixed WordPiece vocabulary.",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    match candle_core::Device::new_metal(0) {
        Ok(_) => println!("candle Metal device 0: OK"),
        Err(e) => println!("candle Metal device 0: UNAVAILABLE ({e})"),
    }
    let cache = spike_cache_dir();
    println!("spike cache dir: {}\nmode: {mode}\n", cache.display());

    match mode.as_str() {
        "parity" => parity(&cache)?,
        "throughput" => throughput(&cache)?,
        "offload" => offload(&cache)?,
        "recall" => recall(&cache)?,
        "all" => {
            parity(&cache)?;
            recall(&cache)?;
            throughput(&cache)?;
            offload(&cache)?;
        }
        other => {
            eprintln!("unknown mode {other:?}; use all|parity|recall|throughput|offload");
            std::process::exit(2);
        }
    }
    Ok(())
}

// ── Q4 (recall half): cross-engine recall@5 ─────────────────────────────────
// The production config the device policy creates: notebook DOCS embedded on
// candle, retrieval QUERY embedded on fastembed. Reproduces eval.rs's exact
// chunking (same nomic tokenizer, same deterministic ids) so the authored gold
// ids line up, then brute-force cosine top-5 (no LanceDB needed for the invariant
// check). Prints the fastembed/fastembed control alongside — they must match, and
// both must clear RECALL_FLOOR (0.75).
fn recall(cache: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("######## Q4 — CROSS-ENGINE RECALL@{K} (docs=candle, query=fastembed) ########");

    // Reuse the SAME tokenizer eval uses so chunk boundaries — and thus the
    // content-derived gold ids in queries.json — match exactly.
    let rt = tokio::runtime::Runtime::new()?;
    let tokenizer = rt.block_on(lens_core::resolve_nomic_tokenizer(cache))?;

    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("eval");

    // Chunk the main corpus deterministically.
    let mut chunk_ids: Vec<String> = Vec::new();
    let mut chunk_texts: Vec<String> = Vec::new();
    for stem in MAIN_DOCS {
        let text = std::fs::read_to_string(fixtures.join(format!("{stem}.md")))?;
        let blocks = parse_blocks(&text, SourceKind::Markdown);
        for c in chunk_blocks_deterministic(&text, &blocks, &tokenizer)? {
            chunk_ids.push(c.id);
            chunk_texts.push(c.text);
        }
    }
    println!(
        "corpus: {} chunks over {} docs",
        chunk_ids.len(),
        MAIN_DOCS.len()
    );

    #[derive(serde::Deserialize)]
    struct Q {
        query: String,
        gold_chunk_ids: Vec<String>,
    }
    let queries: Vec<Q> =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join("queries.json"))?)?;

    let fe = FastembedEmbedder::new(cache)?;
    let cand = CandleNomicEmbedder::new(cache, Compute::Cpu)?;

    let refs: Vec<&str> = chunk_texts.iter().map(String::as_str).collect();
    let fe_docs = fe.embed_documents(&refs)?;
    let cand_docs = cand.embed_documents(&refs)?;

    // recall@5 for a given doc-vector set, always querying via fastembed.
    let measure = |doc_vecs: &[Vec<f32>]| -> Result<f32, Box<dyn std::error::Error>> {
        let mut hits = 0usize;
        for q in &queries {
            let qv = fe.embed_query(&q.query)?;
            let mut scored: Vec<(usize, f32)> = doc_vecs
                .iter()
                .enumerate()
                .map(|(i, dv)| (i, cosine(&qv, dv)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top: HashSet<&str> = scored
                .iter()
                .take(K)
                .map(|(i, _)| chunk_ids[*i].as_str())
                .collect();
            if q.gold_chunk_ids.iter().any(|g| top.contains(g.as_str())) {
                hits += 1;
            }
        }
        Ok(hits as f32 / queries.len() as f32)
    };

    let control = measure(&fe_docs)?;
    let cross = measure(&cand_docs)?;
    println!(
        "\nfastembed→fastembed (control) recall@{K}: {control:.4}\n\
         candle→fastembed   (production) recall@{K}: {cross:.4}\nfloor: {RECALL_FLOOR:.4}"
    );
    let verdict = if cross + 1e-6 >= RECALL_FLOOR && (cross - control).abs() < 1e-6 {
        "RECALL OK — cross-engine matches control AND clears the floor."
    } else if cross + 1e-6 >= RECALL_FLOOR {
        "RECALL OK — clears the floor (differs slightly from control)."
    } else {
        "RECALL FAIL — cross-engine drops below the floor."
    };
    println!("VERDICT: {verdict}\n");
    Ok(())
}

// ── Q4: cross-engine parity ────────────────────────────────────────────────
// The load-bearing invariant: docs embedded on candle must match the SAME text
// embedded on fastembed/ORT within tolerance, or a candle-doc / fastembed-query
// notebook mis-retrieves. Report cosine(candle_cpu, fastembed) and
// cosine(candle_metal, fastembed) per doc, plus the mins.
fn parity(cache: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("######## Q4 — CROSS-ENGINE PARITY (candle vs fastembed) ########");
    let fe = FastembedEmbedder::new(cache)?;
    let cpu = CandleNomicEmbedder::new(cache, Compute::Cpu)?;
    let metal = CandleNomicEmbedder::new(cache, Compute::Metal).ok();

    let fe_v = fe.embed_documents(CORPUS)?;
    let cpu_v = cpu.embed_documents(CORPUS)?;
    let metal_v = match &metal {
        Some(m) => Some(m.embed_documents(CORPUS)?),
        None => None,
    };

    println!(
        "\n{:<58} {:>12} {:>12}",
        "doc (truncated)", "cpu↔fe", "metal↔fe"
    );
    let mut min_cpu = f32::INFINITY;
    let mut min_metal = f32::INFINITY;
    let mut sum_cpu = 0.0f32;
    for (i, doc) in CORPUS.iter().enumerate() {
        let c_cpu = cosine(&cpu_v[i], &fe_v[i]);
        min_cpu = min_cpu.min(c_cpu);
        sum_cpu += c_cpu;
        let c_metal = metal_v
            .as_ref()
            .map(|mv| cosine(&mv[i], &fe_v[i]))
            .unwrap_or(f32::NAN);
        min_metal = min_metal.min(c_metal);
        let short: String = doc.chars().take(54).collect();
        println!("{short:<58} {c_cpu:>12.6} {c_metal:>12.6}");
    }
    println!(
        "\nmean cpu↔fe   : {:.6}\nmin  cpu↔fe   : {:.6}\nmin  metal↔fe : {:.6}",
        sum_cpu / CORPUS.len() as f32,
        min_cpu,
        min_metal
    );
    // Also candle-cpu vs candle-metal self-parity (should be ~1.0).
    if let Some(mv) = &metal_v {
        let mut min_self = f32::INFINITY;
        for i in 0..CORPUS.len() {
            min_self = min_self.min(cosine(&cpu_v[i], &mv[i]));
        }
        println!("min  cpu↔metal (self): {min_self:.6}");
    }
    let verdict = if min_cpu >= 0.9995 {
        "PARITY OK (>= 0.9995)"
    } else if min_cpu >= 0.99 {
        "CLOSE (0.99–0.9995) — check recall, may still retrieve fine"
    } else {
        "PARITY RISK (< 0.99) — cross-engine mixing unsafe for this model"
    };
    println!("\nVERDICT: {verdict}\n");
    Ok(())
}

// ── Q1: throughput (candle-cpu vs fastembed-cpu vs candle-metal) ────────────
fn throughput(cache: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("######## Q1 — THROUGHPUT (docs/sec, batch=16) ########");
    let batch: Vec<&str> = CORPUS.to_vec();
    let rounds = 20usize;
    let total_docs = batch.len() * rounds;

    let fe = FastembedEmbedder::new(cache)?;
    let cpu = CandleNomicEmbedder::new(cache, Compute::Cpu)?;
    let metal = CandleNomicEmbedder::new(cache, Compute::Metal).ok();

    // Warmup each (first batch pays lazy init / kernel compile).
    fe.embed_documents(&batch)?;
    cpu.embed_documents(&batch)?;
    if let Some(m) = &metal {
        m.embed_documents(&batch)?;
    }

    let bench = |label: &str, f: &dyn Fn() -> Result<(), Box<dyn std::error::Error>>| {
        let t = Instant::now();
        for _ in 0..rounds {
            f().unwrap();
        }
        let secs = t.elapsed().as_secs_f64();
        let dps = total_docs as f64 / secs;
        println!("{label:<16} {secs:>8.3}s  {dps:>10.1} docs/s");
    };

    bench("fastembed-cpu", &|| {
        fe.embed_documents(&batch)?;
        Ok(())
    });
    bench("candle-cpu", &|| {
        cpu.embed_documents(&batch)?;
        Ok(())
    });
    if let Some(m) = &metal {
        bench("candle-metal", &|| {
            m.embed_documents(&batch)?;
            Ok(())
        });
    }
    println!();
    Ok(())
}

// ── Q2: CPU offload on sustained bulk re-embed ──────────────────────────────
// Measures AVG CPU CORES BUSY = (process CPU-seconds) / (wall-seconds) via
// getrusage(RUSAGE_SELF) — the sudo-free analog of the CoreML spike's
// powermetrics. A CPU-bound engine pegs multiple cores (avg cores >> 1); a
// genuinely GPU-offloaded engine leaves the CPU mostly idle (avg cores ≈ the
// tokenize/dispatch overhead, ~1 or less). Runs the SAME sustained workload on
// candle-cpu, fastembed-cpu, and candle-metal so the offload delta is direct.
fn offload(cache: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("######## Q2 — METAL CPU OFFLOAD (avg CPU cores on sustained bulk) ########");
    let big_rounds = 60usize;
    let batch: Vec<&str> = CORPUS.to_vec();
    let total = batch.len() * big_rounds;

    let fe = FastembedEmbedder::new(cache)?;
    let cpu = CandleNomicEmbedder::new(cache, Compute::Cpu)?;
    let metal = CandleNomicEmbedder::new(cache, Compute::Metal).ok();

    // warmups
    fe.embed_documents(&batch)?;
    cpu.embed_documents(&batch)?;
    if let Some(m) = &metal {
        m.embed_documents(&batch)?;
    }

    println!(
        "\n{:<16} {:>10} {:>10} {:>12} {:>12}",
        "engine", "wall(s)", "cpu(s)", "avg cores", "docs/s"
    );
    let run = |label: &str, e: &dyn Embedder| {
        let cpu0 = process_cpu_seconds();
        let t = Instant::now();
        for _ in 0..big_rounds {
            e.embed_documents(&batch).unwrap();
        }
        let wall = t.elapsed().as_secs_f64();
        let cpu_s = process_cpu_seconds() - cpu0;
        let cores = cpu_s / wall;
        let dps = total as f64 / wall;
        println!("{label:<16} {wall:>10.3} {cpu_s:>10.3} {cores:>12.2} {dps:>12.1}");
        cores
    };

    let fe_cores = run("fastembed-cpu", &fe);
    let cpu_cores = run("candle-cpu", &cpu);
    if let Some(m) = &metal {
        let metal_cores = run("candle-metal", m);
        let baseline = fe_cores.max(cpu_cores);
        let offload_pct = (1.0 - metal_cores / baseline) * 100.0;
        println!(
            "\ncandle-metal uses {metal_cores:.2} avg cores vs {baseline:.2} for the busiest \
             CPU engine → ~{offload_pct:.0}% CPU freed."
        );
        let verdict = if metal_cores < baseline * 0.5 {
            "OFFLOAD CONFIRMED — Metal frees the CPU on sustained bulk."
        } else {
            "WEAK OFFLOAD — Metal still burns comparable CPU (dispatch-bound?)."
        };
        println!("VERDICT: {verdict}");
    }
    Ok(())
}

/// Total CPU seconds (user + system) consumed by THIS process so far, via
/// getrusage(RUSAGE_SELF). Used to derive avg-cores-busy for the offload gate.
fn process_cpu_seconds() -> f64 {
    // SAFETY: getrusage with a valid RUSAGE_SELF target and a zeroed rusage out
    // param; the struct is POD and fully written by the call on success.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) != 0 {
            return 0.0;
        }
        let u = ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 / 1e6;
        let s = ru.ru_stime.tv_sec as f64 + ru.ru_stime.tv_usec as f64 / 1e6;
        u + s
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}
