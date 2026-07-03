//! Cross-engine parity guardrail for the candle Metal embedding backend (issue #91).
//!
//! The device policy ([`lens_core::embedder::select_compute`]) deliberately lets a
//! notebook's document vectors come from the candle engine while its query vectors
//! come from fastembed/ONNX. That is only correct if the two engines agree — the
//! LOAD-BEARING invariant of the whole design. This test asserts it directly:
//! candle-CPU vs fastembed cosine ≥ 0.999 per input.
//!
//! Feature-gated (`native-ml-metal`, aarch64-apple-darwin) and `#[ignore]` by
//! default: it downloads ~547 MB (candle nomic F32 safetensors) + ~130 MB
//! (fastembed ONNX). Run explicitly, e.g. on a macOS-aarch64 lane:
//!
//!   cargo test -p lens-core --features native-ml-metal --test candle_metal_parity \
//!     -- --ignored --nocapture
//!
//! The full parity + throughput + offload + recall evidence is recorded in
//! `.omc/plans/issue-91-candle-metal-spike-results.md`.

#![cfg(feature = "native-ml-metal")]

use lens_core::embedder::{CandleNomicEmbedder, Compute, Embedder, FastembedEmbedder};

/// Minimum acceptable cross-engine cosine. Measured at 1.000000; 0.999 is a
/// generous guardrail that still catches a real divergence (e.g. an accidental
/// fp16 regression, which drops parity to ~0.998).
const PARITY_FLOOR: f32 = 0.999;

const DOCS: &[&str] = &[
    "The Voyager Golden Record carries sounds and images of Earth.",
    "Rust's ownership model eliminates data races at compile time.",
    "Mean pooling averages token embeddings weighted by the attention mask.",
];

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
#[ignore = "downloads ~677 MB of model weights; run on a macOS-aarch64 lane"]
fn candle_cpu_matches_fastembed_within_tolerance() {
    let dir = tempfile::tempdir().expect("tempdir");
    let fe = FastembedEmbedder::new(dir.path()).expect("fastembed init");
    let candle = CandleNomicEmbedder::new(dir.path(), Compute::Cpu).expect("candle-cpu init");

    let fe_v = fe.embed_documents(DOCS).expect("fastembed embed");
    let cand_v = candle.embed_documents(DOCS).expect("candle embed");

    assert_eq!(fe_v.len(), cand_v.len());
    for (i, (a, b)) in fe_v.iter().zip(&cand_v).enumerate() {
        assert_eq!(a.len(), 768, "fastembed dim");
        assert_eq!(b.len(), 768, "candle dim");
        let c = cosine(a, b);
        assert!(
            c >= PARITY_FLOOR,
            "doc {i}: cross-engine cosine {c:.6} < floor {PARITY_FLOOR} — \
             candle/fastembed parity broke (cross-engine mixing would mis-retrieve)"
        );
    }
}

/// candle-CPU and candle-Metal must produce identical vectors (F32 on Metal is
/// bit-faithful to CPU); if this ever diverges, an fp16 or kernel regression crept
/// in. Skipped automatically if no Metal device is constructible.
#[test]
#[ignore = "downloads ~547 MB of model weights; run on a macOS-aarch64 lane"]
fn candle_metal_matches_candle_cpu() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cpu = CandleNomicEmbedder::new(dir.path(), Compute::Cpu).expect("candle-cpu init");
    let metal = match CandleNomicEmbedder::new(dir.path(), Compute::Metal) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("no Metal device ({e}); skipping metal-vs-cpu parity");
            return;
        }
    };
    let cpu_v = cpu.embed_documents(DOCS).expect("cpu embed");
    let metal_v = metal.embed_documents(DOCS).expect("metal embed");
    for (i, (a, b)) in cpu_v.iter().zip(&metal_v).enumerate() {
        let c = cosine(a, b);
        assert!(
            c >= 0.9999,
            "doc {i}: candle cpu↔metal cosine {c:.6} < 0.9999 (F32 should be exact)"
        );
    }
}
