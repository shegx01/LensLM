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

// NOTE: this is the ORIGINAL nomic repo (F32 safetensors + tokenizer.json),
// distinct from fastembed's Qdrant/…-onnx mirror — the two engines fetch different
// artifacts.
const NOMIC_HF_REPO: &str = "nomic-ai/nomic-embed-text-v1.5";

// Pinned commit (not `main`) so the download is deterministic and the SHA256 check
// stays valid across upstream updates.
const NOMIC_HF_REVISION: &str = "e9b6763023c676ca8431644204f50c2b100d9aab";

// SHA-256 of model.safetensors at NOMIC_HF_REVISION (HF LFS oid). Verified after
// fetch — same supply-chain gate as KOKORO_MODEL_SHA256. The ~547 MB file is
// mmapped into the process, so end-to-end verification is required.
const NOMIC_SAFETENSORS_SHA256: &str =
    "9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c";

/// Whether the candle backend is wired for `model_id`. Separate from
/// `accelerate_hint`: a model can be GPU-eligible but not yet wired here, in which
/// case the caller falls back to fastembed (expected, not an error).
pub fn candle_supports_model(model_id: &str) -> bool {
    model_id == DEFAULT_EMBED_MODEL_ID
}

/// hf-hub cache subdir for candle weights under `{data_dir}/models/candle/`,
/// or `None` for unsupported models. Same `models--{org}--{model}` scheme as fastembed.
pub fn candle_cache_subdir(model_id: &str) -> Option<String> {
    candle_supports_model(model_id).then(|| format!("models--{}", NOMIC_HF_REPO.replace('/', "--")))
}

// Truncation MUST be set explicitly: the serialized tokenizer.json bakes in
// neither truncation nor padding; an over-length input would crash the forward pass.

const MAX_TOKENS: usize = 2048;

/// Maps [`Compute`] to a candle [`Device`]. Hard error on Metal init failure;
/// the caller (`LensEngine::embedder_for`) is responsible for the CPU fallback.
fn candle_device(compute: Compute) -> Result<Device, LensError> {
    match compute {
        Compute::Cpu => Ok(Device::Cpu),
        Compute::Metal => Device::new_metal(0)
            .map_err(|e| LensError::Model(format!("candle Metal device init failed: {e}"))),
        // CUDA is interface-only (issue #91): the match must be total, but the
        // policy never routes a CUDA job here — a CUDA embedder is a separate impl.
        Compute::Cuda => Err(LensError::Model(
            "candle Metal backend cannot serve a CUDA device".into(),
        )),
    }
}

/// candle-backed NomicBERT embedder (768-dim), device-selectable.
pub struct CandleNomicEmbedder {
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
    /// Builds the candle nomic embedder on `compute` (F32 weights, ~547 MB download).
    ///
    /// # Errors
    /// [`LensError::Model`] on device init, weight/tokenizer load, or unsupported
    /// model id. Callers treat any error as "fall back to fastembed-CPU".
    pub fn new(cache_dir: &Path, compute: Compute) -> Result<Self, LensError> {
        Self::new_with_spec(
            cache_dir,
            compute,
            crate::embedder::registry::resolve(DEFAULT_EMBED_MODEL_ID),
        )
    }

    /// Builds the candle nomic embedder for `spec`. Only `nomic-embed-text-v1.5` is
    /// wired; other ids are rejected so the caller falls back to fastembed-CPU.
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

        // COUPLING: fallback relies on candle-transformers 0.11.0 nomic_bert::Config::default()
        // matching nomic-v1.5. A future bump could drift the default; the fetched
        // config.json normally wins, so this matters only if BOTH the read fails AND
        // the default drifts.
        let config: Config = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        // SAFETY: mmaps model.safetensors read-only. Contract: the file must not be
        // modified for the mmap's lifetime. Upheld: app-private dir; one process per
        // data dir; hf-hub writes content-addressed blobs and never overwrites;
        // sha256-verified in fetch_artifacts before mapping.
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

    pub fn compute(&self) -> Compute {
        self.compute
    }

    /// Tokenize, forward pass on `self.device`, mean-pool, L2-normalize.
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

/// Fetch (or read from cache) config, tokenizer, and safetensors for nomic-v1.5
/// at the pinned revision. SHA-256-verifies weights before returning.
fn fetch_artifacts(
    cache_dir: &Path,
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf), LensError> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};

    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .map_err(|e| LensError::Model(format!("hf-hub api build failed: {e}")))?;
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

/// Verifies `path`'s SHA-256 equals `expected`. Removes the file on mismatch
/// so the next run re-downloads it.
fn verify_sha256(path: &Path, expected: &str) -> Result<(), LensError> {
    use sha2::{Digest, Sha256};

    let bytes = std::fs::read(path)
        .map_err(|e| LensError::Model(format!("candle weights read for hash failed: {e}")))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        let _ = std::fs::remove_file(path);
        return Err(LensError::Model(format!(
            "candle model.safetensors integrity check failed: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}
