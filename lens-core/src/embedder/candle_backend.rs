//! STEP-0 SPIKE (issue #91) — candle NomicBERT embedding backend (CPU + Metal).
//!
//! A second [`Embedder`] implementation alongside [`FastembedEmbedder`], backed by
//! `candle-transformers`' native `nomic_bert` model. Runs the WHOLE forward pass on
//! one device — CPU or the Apple Metal GPU — which is the property the dead
//! ORT-CoreML-EP path lacked (it fragmented the graph across CoreML↔CPU boundaries;
//! see memory `issue-91-native-ml-seam`). Implementing the production [`Embedder`]
//! trait lets the spike reuse it unchanged for the cross-engine parity + recall gate.
//!
//! Weights load as `DType::F32` on BOTH devices: fp16 on Apple Silicon is only
//! ~1.1× faster but drops cross-engine cosine parity to ~0.998; F32 targets ~0.9999.
//!
//! Feature-gated (`native-ml-metal`, aarch64-apple-darwin only). Throwaway.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{Config, NomicBertModel, l2_normalize, mean_pooling};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::LensError;
use crate::embedder::Embedder;
use crate::embedder::registry::{DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, EmbeddingModelSpec};

/// HuggingFace repo the candle backend loads nomic weights from. NOTE this is the
/// ORIGINAL nomic repo (F32 safetensors + `tokenizer.json`), distinct from
/// fastembed's `Qdrant/…-onnx` mirror — the two engines fetch different artifacts.
const NOMIC_HF_REPO: &str = "nomic-ai/nomic-embed-text-v1.5";

/// Which device the candle forward pass runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandleCompute {
    /// Portable CPU path (candle's own CPU kernels, NOT ONNX).
    Cpu,
    /// Apple Metal GPU — the bulk-offload target under test.
    Metal,
}

impl CandleCompute {
    /// Human label for logs / bench rows.
    pub fn label(self) -> &'static str {
        match self {
            CandleCompute::Cpu => "candle-cpu",
            CandleCompute::Metal => "candle-metal",
        }
    }

    /// Resolves this choice to a concrete candle [`Device`]. `Metal` on a machine
    /// without a constructible Metal device is a hard error here (the spike wants
    /// to KNOW, not silently fall back — production would fall back per the policy).
    fn device(self) -> Result<Device, LensError> {
        match self {
            CandleCompute::Cpu => Ok(Device::Cpu),
            CandleCompute::Metal => Device::new_metal(0)
                .map_err(|e| LensError::Model(format!("candle Metal device init failed: {e}"))),
        }
    }
}

/// Max tokens per input. nomic-v1.5 trained at 2048; our chunks are ~512, so this
/// only guards a pathological input. Truncation MUST be set explicitly — the
/// serialized `tokenizer.json` bakes in neither truncation nor padding, and an
/// over-length input crashes the forward pass otherwise.
const MAX_TOKENS: usize = 2048;

/// candle-backed NomicBERT embedder (768-dim), device-selectable.
pub struct CandleNomicEmbedder {
    /// `NomicBertModel::forward` is `&self`, but tokenizer padding config mutation
    /// and the batch encode path keep this behind a mutex for a uniform `&self`
    /// trait surface and to serialize device access (mirrors `FastembedEmbedder`).
    inner: Mutex<Inner>,
    device: Device,
    compute: CandleCompute,
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
    /// `cache_dir` is where HF artifacts are fetched/read (a `models/candle`
    /// subdir under the app data dir by convention). Downloads on cold cache.
    pub fn new(cache_dir: &Path, compute: CandleCompute) -> Result<Self, LensError> {
        Self::new_with_spec(
            cache_dir,
            compute,
            crate::embedder::registry::resolve(DEFAULT_EMBED_MODEL_ID),
        )
    }

    /// Builds the candle nomic embedder for `spec` (currently only the nomic
    /// default is wired — other models are future work).
    pub fn new_with_spec(
        cache_dir: &Path,
        compute: CandleCompute,
        spec: &EmbeddingModelSpec,
    ) -> Result<Self, LensError> {
        if spec.id != DEFAULT_EMBED_MODEL_ID {
            return Err(LensError::Model(format!(
                "candle spike backend only wires {DEFAULT_EMBED_MODEL_ID}; got {}",
                spec.id
            )));
        }
        let device = compute.device()?;
        let (config_path, tokenizer_path, weights_path) = fetch_artifacts(cache_dir)?;

        // Parse config.json; #[serde(default)] fills any absent field, and the
        // struct's Default IS nomic-v1.5 — so a parse failure still yields the
        // correct architecture. Unknown JSON fields are ignored by serde.
        let config: Config = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        // SAFETY: from_mmaped_safetensors mmaps the file read-only; the file is a
        // trusted, checksummed model artifact fetched above and not mutated while
        // mapped.
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
    pub fn compute(&self) -> CandleCompute {
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

        let cd = |e: candle_core::Error, what: &str| LensError::Model(format!("candle {what}: {e}"));
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

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, LensError> {
        let prefixed = vec![format!("{}{text}", self.prefix_query)];
        let mut out = self.embed_batch(&prefixed)?;
        out.drain(..)
            .next()
            .ok_or_else(|| LensError::Model("candle returned empty batch for query".into()))
    }
}

/// Fetch (or read from cache) `config.json`, `tokenizer.json`, `model.safetensors`
/// for nomic-v1.5. Uses `hf-hub`'s blocking API rooted at `cache_dir` so the spike
/// controls where the ~547 MB weight file lands (under the app data dir, not the
/// user's global `~/.cache/huggingface`).
fn fetch_artifacts(
    cache_dir: &Path,
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf), LensError> {
    use hf_hub::api::sync::ApiBuilder;

    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .map_err(|e| LensError::Model(format!("hf-hub api build failed: {e}")))?;
    let repo = api.model(NOMIC_HF_REPO.to_string());

    let get = |file: &str| {
        repo.get(file)
            .map_err(|e| LensError::Model(format!("hf-hub fetch {file} failed: {e}")))
    };
    let config = get("config.json")?;
    let tokenizer = get("tokenizer.json")?;
    let weights = get("model.safetensors")?;
    Ok((config, tokenizer, weights))
}

/// Sanity-visible default dim (used only to keep the const referenced in
/// feature-off analyzers happy; the real dim comes from the spec).
#[allow(dead_code)]
const _NOMIC_DIM: usize = DEFAULT_EMBED_DIM;
