//! candle NomicBERT embedding backend (CPU + Apple Metal GPU) — issue #91.
//!
//! A second [`Embedder`] implementation alongside [`crate::embedder::FastembedEmbedder`],
//! backed by `candle-transformers`' native `nomic_bert` model. Unlike the
//! `fastembed`/ONNX path it runs the WHOLE forward pass on ONE device — so on the
//! Apple Metal GPU it delivers a clean CPU offload (measured ~99% of CPU cores
//! freed) and ~2.6× the bulk throughput, while producing vectors that are
//! numerically identical to fastembed (cosine 1.000000, recall@5 identical — see
//! `.omc/plans/issue-91-candle-metal-spike-results.md`). That parity is what lets
//! the device be a per-job runtime choice ([`crate::embedder::device`]) rather than
//! a persisted notebook property.
//!
//! Weights load as [`DType::F32`] on BOTH devices: fp16 on Apple Silicon is only
//! ~1.1× faster but drops cross-engine cosine parity to ~0.998; F32 holds ~0.9999.
//!
//! Feature-gated (`native-ml-metal`, aarch64-apple-darwin only). Currently wires
//! the default `nomic-embed-text-v1.5`; other GPU-eligible models
//! (`accelerate_hint = true`) fall back to CPU until wired here.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{Config, NomicBertModel, l2_normalize, mean_pooling};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::LensError;
use crate::embedder::Embedder;
use crate::embedder::device::Compute;
use crate::embedder::registry::{DEFAULT_EMBED_MODEL_ID, EmbeddingModelSpec};

/// HuggingFace repo the candle backend loads nomic weights from. NOTE this is the
/// ORIGINAL nomic repo (F32 safetensors + `tokenizer.json`), distinct from
/// fastembed's `Qdrant/…-onnx` mirror — the two engines fetch different artifacts.
const NOMIC_HF_REPO: &str = "nomic-ai/nomic-embed-text-v1.5";

/// PINNED repo revision (a specific commit, NOT `main`) so the download is
/// deterministic and the [`NOMIC_SAFETENSORS_SHA256`] check below stays valid — an
/// upstream `main` update can't silently change the weights under us.
const NOMIC_HF_REVISION: &str = "e9b6763023c676ca8431644204f50c2b100d9aab";

/// Expected SHA-256 of `model.safetensors` at [`NOMIC_HF_REVISION`] (its HF LFS
/// `oid`). Verified after fetch — the SAME supply-chain integrity pattern the
/// Kokoro model download uses (`KOKORO_MODEL_SHA256`). hf-hub already fetches over
/// HTTPS into content-addressed blobs, but the ~547 MB file is mmapped straight
/// into the process, so we verify it end-to-end before trusting it.
const NOMIC_SAFETENSORS_SHA256: &str =
    "9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c";

/// Max tokens per input. nomic-v1.5 trained at 2048; our chunks are ~512, so this
/// only guards a pathological input. Truncation MUST be set explicitly — the
/// serialized `tokenizer.json` bakes in neither truncation nor padding, and an
/// over-length input would otherwise crash the forward pass.
const MAX_TOKENS: usize = 2048;

/// Maps a [`Compute`] to the concrete candle [`Device`]. `Metal` on a machine
/// without a constructible Metal device is a hard error here — the caller
/// ([`crate::LensEngine::embedder_for`]) is responsible for falling back to the
/// fastembed CPU path, so this stays strict and surfaces the failure.
fn candle_device(compute: Compute) -> Result<Device, LensError> {
    match compute {
        Compute::Cpu => Ok(Device::Cpu),
        Compute::Metal => Device::new_metal(0)
            .map_err(|e| LensError::Model(format!("candle Metal device init failed: {e}"))),
    }
}

/// candle-backed NomicBERT embedder (768-dim), device-selectable.
pub struct CandleNomicEmbedder {
    /// `NomicBertModel::forward` is `&self`, but the tokenizer + device access are
    /// serialized behind a mutex for a uniform `&self` trait surface (mirrors
    /// [`crate::embedder::FastembedEmbedder`]).
    inner: Mutex<Inner>,
    device: Device,
    compute: Compute,
    dim: usize,
    prefix_doc: String,
    prefix_query: String,
    model_id: String,
}

struct Inner {
    model: NomicBertModel,
    tokenizer: Tokenizer,
}

impl CandleNomicEmbedder {
    /// Builds the candle nomic embedder on `compute`, loading F32 weights.
    ///
    /// `cache_dir` is where HF artifacts are fetched/read — by convention
    /// `{data_dir}/models/candle/` (kept apart from fastembed's ONNX cache, since
    /// the two engines download different files). Downloads ~547 MB on a cold cache.
    ///
    /// # Errors
    /// [`LensError::Model`] on Metal-device init failure, weight/tokenizer load
    /// failure, or an unsupported model id (only nomic is wired). Callers treat any
    /// error as "fall back to the CPU fastembed path".
    pub fn new(cache_dir: &Path, compute: Compute) -> Result<Self, LensError> {
        Self::new_with_spec(
            cache_dir,
            compute,
            crate::embedder::registry::resolve(DEFAULT_EMBED_MODEL_ID),
        )
    }

    /// Builds the candle nomic embedder for `spec`.
    ///
    /// Only `nomic-embed-text-v1.5` is currently wired; any other id is rejected
    /// with [`LensError::Model`] so the caller falls back to fastembed-CPU.
    pub fn new_with_spec(
        cache_dir: &Path,
        compute: Compute,
        spec: &EmbeddingModelSpec,
    ) -> Result<Self, LensError> {
        if spec.id != DEFAULT_EMBED_MODEL_ID {
            return Err(LensError::Model(format!(
                "candle backend currently wires only {DEFAULT_EMBED_MODEL_ID}; got {} \
                 (caller should fall back to fastembed-CPU)",
                spec.id
            )));
        }
        let device = candle_device(compute)?;
        let (config_path, tokenizer_path, weights_path) = fetch_artifacts(cache_dir)?;

        // Parse config.json; #[serde(default)] fills any absent field, and the
        // struct's Default IS nomic-v1.5 — so even a parse failure yields the
        // correct architecture. Unknown JSON fields are ignored by serde.
        //
        // COUPLING: this fallback correctness relies on `candle-transformers`'
        // `nomic_bert::Config::default()` matching nomic-v1.5 (vocab 30528, 12
        // layers, RoPE base 1000, swiglu, …), which holds for the pinned
        // candle-transformers 0.11.0. A future candle-transformers bump could change
        // that default; the fetched `config.json` normally supplies the real values,
        // so this only matters if BOTH the config read fails AND the default drifts.
        let config: Config = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        // SAFETY: `from_mmaped_safetensors` mmaps the file read-only via `memmap2`.
        // The `unsafe` contract requires the mapped file is not modified by any
        // process for the lifetime of the mapping (= the lifetime of the returned
        // `VarBuilder` and the `NomicBertModel` built from it, which hold
        // `Storage::Mmap` references for as long as this `CandleNomicEmbedder`
        // lives in the engine's embedder cache). Upheld because: (1) the file is in
        // `{data_dir}/models/candle/` (app-private); (2) the engine enforces one
        // process per data dir; (3) hf-hub writes content-addressed blob paths and
        // never overwrites in place; and (4) its bytes were sha256-verified against
        // `NOMIC_SAFETENSORS_SHA256` in `fetch_artifacts` before we mapped them.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| LensError::Model(format!("candle safetensors load failed: {e}")))?
        };
        let model = NomicBertModel::load(vb, &config)
            .map_err(|e| LensError::Model(format!("candle NomicBertModel load failed: {e}")))?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| LensError::Model(format!("candle tokenizer load failed: {e}")))?;
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| LensError::Model(format!("candle tokenizer config failed: {e}")))?;

        Ok(Self {
            inner: Mutex::new(Inner { model, tokenizer }),
            device,
            compute,
            dim: spec.dim,
            prefix_doc: spec.prefix_doc.to_string(),
            prefix_query: spec.prefix_query.to_string(),
            model_id: spec.id.to_string(),
        })
    }

    /// The compute device this embedder is pinned to.
    pub fn compute(&self) -> Compute {
        self.compute
    }

    /// Tokenize `texts`, run the NomicBERT forward pass on `self.device`, mean-pool
    /// with the attention mask, and L2-normalize. Returns one 768-f32 vector each.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LensError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let guard = self
            .inner
            .lock()
            .map_err(|e| LensError::Model(format!("candle mutex poisoned: {e}")))?;

        let to_embed: Vec<&str> = texts.iter().map(String::as_str).collect();
        let encodings = guard
            .tokenizer
            .encode_batch(to_embed, true)
            .map_err(|e| LensError::Model(format!("candle tokenize failed: {e}")))?;

        let batch = encodings.len();
        let seq = encodings[0].get_ids().len();
        let mut ids: Vec<u32> = Vec::with_capacity(batch * seq);
        let mut mask: Vec<u32> = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            ids.extend_from_slice(enc.get_ids());
            mask.extend_from_slice(enc.get_attention_mask());
        }

        let cd =
            |e: candle_core::Error, what: &str| LensError::Model(format!("candle {what}: {e}"));
        let input_ids =
            Tensor::from_vec(ids, (batch, seq), &self.device).map_err(|e| cd(e, "input_ids"))?;
        let attn_mask =
            Tensor::from_vec(mask, (batch, seq), &self.device).map_err(|e| cd(e, "attn_mask"))?;

        let hidden = guard
            .model
            .forward(&input_ids, None, Some(&attn_mask))
            .map_err(|e| cd(e, "forward"))?;
        let pooled = mean_pooling(&hidden, &attn_mask).map_err(|e| cd(e, "mean_pooling"))?;
        let normed = l2_normalize(&pooled).map_err(|e| cd(e, "l2_normalize"))?;
        let out: Vec<Vec<f32>> = normed.to_vec2().map_err(|e| cd(e, "to_vec2"))?;

        for (i, v) in out.iter().enumerate() {
            if v.len() != self.dim {
                return Err(LensError::Model(format!(
                    "candle vector {i} has dim {} (expected {})",
                    v.len(),
                    self.dim
                )));
            }
        }
        Ok(out)
    }
}

impl Embedder for CandleNomicEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LensError> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.prefix_doc))
            .collect();
        self.embed_batch(&prefixed)
    }

    fn embed_documents_owned(&self, mut texts: Vec<String>) -> Result<Vec<Vec<f32>>, LensError> {
        if !self.prefix_doc.is_empty() {
            for t in texts.iter_mut() {
                t.insert_str(0, &self.prefix_doc);
            }
        }
        self.embed_batch(&texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, LensError> {
        let prefixed = vec![format!("{}{text}", self.prefix_query)];
        let mut out = self.embed_batch(&prefixed)?;
        out.drain(..)
            .next()
            .ok_or_else(|| LensError::Model("candle returned empty batch for query".into()))
    }
}

/// Fetch (or read from cache) `config.json`, `tokenizer.json`, `model.safetensors`
/// for nomic-v1.5 at the PINNED [`NOMIC_HF_REVISION`], rooted at `cache_dir` (so the
/// ~547 MB weight file lands under the app data dir, not the user's global
/// `~/.cache/huggingface`). The weights are SHA-256-verified against
/// [`NOMIC_SAFETENSORS_SHA256`] before being returned (Kokoro-style integrity gate).
fn fetch_artifacts(
    cache_dir: &Path,
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf), LensError> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};

    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .map_err(|e| LensError::Model(format!("hf-hub api build failed: {e}")))?;
    // Pin the exact commit (not `main`) so the download is reproducible and the
    // sha256 check stays valid across upstream repo updates.
    let repo = api.repo(Repo::with_revision(
        NOMIC_HF_REPO.to_string(),
        RepoType::Model,
        NOMIC_HF_REVISION.to_string(),
    ));

    let get = |file: &str| {
        repo.get(file)
            .map_err(|e| LensError::Model(format!("hf-hub fetch {file} failed: {e}")))
    };
    let config = get("config.json")?;
    let tokenizer = get("tokenizer.json")?;
    let weights = get("model.safetensors")?;
    verify_sha256(&weights, NOMIC_SAFETENSORS_SHA256)?;
    Ok((config, tokenizer, weights))
}

/// Verifies `path`'s SHA-256 equals `expected` (lowercase hex). On mismatch the
/// (untrusted) file is removed so a subsequent run re-downloads it. Mirrors the
/// Kokoro model-download integrity check.
fn verify_sha256(path: &Path, expected: &str) -> Result<(), LensError> {
    use sha2::{Digest, Sha256};

    let bytes = std::fs::read(path)
        .map_err(|e| LensError::Model(format!("candle weights read for hash failed: {e}")))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        // Drop the tampered/corrupt artifact so the next attempt re-fetches it.
        let _ = std::fs::remove_file(path);
        return Err(LensError::Model(format!(
            "candle model.safetensors integrity check failed: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}
