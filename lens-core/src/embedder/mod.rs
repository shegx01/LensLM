//! Text-embedding abstractions: the [`Embedder`] trait, [`FastembedEmbedder`]
//! (fastembed + ONNX), and the test-only [`CountingEmbedder`].
//!
//! Prefixes are PER-MODEL (see registry [`EmbeddingModelSpec`]); fastembed 5.17.2
//! does NOT apply them automatically — [`FastembedEmbedder`] does. The trait is
//! synchronous; every async ingest caller **MUST** use `tokio::task::spawn_blocking`.

use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use fastembed::{InitOptions, TextEmbedding};

use crate::LensError;

// issue #91: candle + Metal backend. Feature-gated to aarch64-apple-darwin.
#[cfg(feature = "native-ml-metal")]
pub mod candle_backend;
// issue #91: per-job compute-device policy + NativeAccelerator seam.
pub mod device;
pub mod ollama;
pub mod registry;

#[cfg(feature = "native-ml-metal")]
pub use candle_backend::{CandleNomicEmbedder, candle_cache_subdir, candle_supports_model};
pub use device::{
    Acceleration, Compute, NativeAccelerator, WorkloadKind, default_accelerator,
    gpu_accelerated_model_ids, gpu_embedding_active, select_compute,
};
pub use ollama::OllamaEmbedder;
pub use registry::{
    DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, EmbeddingBackend, EmbeddingModelSpec, REGISTRY,
    resolve, resolve_opt,
};

/// L2-normalizes `v` in place. Shared by [`OllamaEmbedder`] (server doesn't
/// normalize) and [`CountingEmbedder`]. Zero-norm guard: `1e-9` is above f32
/// rounding noise but below any real embedding norm — degenerate vectors left as-is.
pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Human-readable prefix convention stored in `embedding_index.prefix_convention`.
pub const PREFIX_CONVENTION: &str = "search_document/search_query";

/// A synchronous text-embedding seam. `Send + Sync` so it is held behind
/// `Arc<dyn Embedder>` in the engine's per-model cache. Sync because fastembed
/// is; all async ingest callers MUST use `tokio::task::spawn_blocking`.
pub trait Embedder: Send + Sync {
    fn model_id(&self) -> &str;

    fn dim(&self) -> usize;

    /// Embeds a batch of document texts, prepending the model's doc prefix.
    /// Returns one L2-normalized `Vec<f32>` per input, length [`Embedder::dim`].
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LensError>;

    /// Owned-input variant: applies the doc prefix in-place (one allocation per
    /// input instead of two on the hot path). Default impl borrows into
    /// `embed_documents` so existing impls stay unchanged.
    fn embed_documents_owned(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, LensError> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.embed_documents(&refs)
    }

    /// Embeds a single query text, prepending the model's query prefix.
    /// Returns one L2-normalized `Vec<f32>` of length [`Embedder::dim`].
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, LensError>;
}

/// Production embedder backed by `fastembed` + bundled onnxruntime.
///
/// Construction is expensive (~130 MB ONNX session init + one-time HF download).
/// Construct once and cache behind `Arc<dyn Embedder>` in the engine.
pub struct FastembedEmbedder {
    // fastembed::TextEmbedding::embed takes &mut self; Mutex provides interior
    // mutability behind the shared &self Embedder trait surface.
    inner: Mutex<TextEmbedding>,
    model_id: String,
    dim: usize,
    prefix_doc: String,
    prefix_query: String,
}

impl FastembedEmbedder {
    /// Builds a `FastembedEmbedder` for [`DEFAULT_EMBED_MODEL_ID`].
    ///
    /// # Errors
    /// [`LensError::Model`] on session init failure (download, corrupt weights, …).
    pub fn new(data_dir: &Path) -> Result<Self, LensError> {
        Self::new_with_spec(data_dir, resolve(DEFAULT_EMBED_MODEL_ID))
    }

    /// Builds a `FastembedEmbedder` for the model described by `spec`.
    ///
    /// # Errors
    /// [`LensError::Model`] on session init failure; [`LensError::Validation`] if
    /// `spec` has no fastembed variant (Ollama-only model).
    pub fn new_with_spec(cache_root: &Path, spec: &EmbeddingModelSpec) -> Result<Self, LensError> {
        // Defense-in-depth (issue #80): primary guard is in `embedder_for`; this
        // is the last line of defense for direct callers.
        if !spec.supports(EmbeddingBackend::Fastembed) || spec.fastembed_variant.is_none() {
            return Err(LensError::Validation(format!(
                "model {} does not support the fastembed backend",
                spec.id
            )));
        }
        let variant = spec
            .fastembed_variant
            .clone()
            .expect("fastembed_variant present: guarded by the check above");
        let cache_dir = cache_root.join("models").join("fastembed");
        let opts = InitOptions::new(variant).with_cache_dir(cache_dir);
        let inner = TextEmbedding::try_new(opts)
            .map_err(|e| LensError::Model(format!("fastembed init failed: {e}")))?;
        Ok(Self {
            inner: Mutex::new(inner),
            model_id: spec.id.to_string(),
            dim: spec.dim,
            prefix_doc: spec.prefix_doc.to_string(),
            prefix_query: spec.prefix_query.to_string(),
        })
    }

    /// Embeds already-prefixed strings and validates the output.
    fn embed_prefixed_documents(&self, prefixed: Vec<String>) -> Result<Vec<Vec<f32>>, LensError> {
        let prefixed_refs: Vec<&str> = prefixed.iter().map(String::as_str).collect();
        let result = self
            .inner
            .lock()
            .map_err(|e| LensError::Model(format!("fastembed mutex poisoned: {e}")))?
            .embed(prefixed_refs, None)
            .map_err(|e| LensError::Model(format!("fastembed embed_documents failed: {e}")))?;
        self.assert_normalized(&result)?;
        Ok(result)
    }

    /// Asserts every vector in `vecs` is L2-normalized (‖v‖ ≈ 1.0 ± 1e-3) and
    /// has the expected dimension. A canary — fastembed 5.17.2 normalizes
    /// unconditionally; this fires only if a future upgrade breaks that.
    fn assert_normalized(&self, vecs: &[Vec<f32>]) -> Result<(), LensError> {
        let expected = self.dim();
        for (i, v) in vecs.iter().enumerate() {
            if v.len() != expected {
                return Err(LensError::Model(format!(
                    "embedder returned vector {i} with dim {} (expected {expected})",
                    v.len()
                )));
            }
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if (norm - 1.0_f32).abs() > 1e-3 {
                return Err(LensError::Model(format!(
                    "embedder returned non-normalized vector {i}: ‖v‖ = {norm:.6} (expected 1.0 ± 1e-3)"
                )));
            }
        }
        Ok(())
    }
}

impl Embedder for FastembedEmbedder {
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
        self.embed_prefixed_documents(prefixed)
    }

    fn embed_documents_owned(&self, mut texts: Vec<String>) -> Result<Vec<Vec<f32>>, LensError> {
        if !self.prefix_doc.is_empty() {
            for t in texts.iter_mut() {
                t.insert_str(0, &self.prefix_doc);
            }
        }
        self.embed_prefixed_documents(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, LensError> {
        let prefixed = format!("{}{text}", self.prefix_query);
        let result = self
            .inner
            .lock()
            .map_err(|e| LensError::Model(format!("fastembed mutex poisoned: {e}")))?
            .embed(vec![prefixed.as_str()], None)
            .map_err(|e| LensError::Model(format!("fastembed embed_query failed: {e}")))?;
        self.assert_normalized(&result)?;
        result.into_iter().next().ok_or_else(|| {
            LensError::Model("fastembed returned empty batch for embed_query".into())
        })
    }
}

/// Deterministic, model-free embedder for tests. Verifies the cached-once and
/// concurrency-serialization invariants without a model download: `load_count`
/// increments on construction, `in_flight` tracks concurrent embed calls.
pub struct CountingEmbedder {
    pub load_count: Arc<AtomicUsize>,
    pub in_flight: Arc<AtomicUsize>,
    model_id: String,
    dim: usize,
    prefix_doc: String,
    prefix_query: String,
}

impl CountingEmbedder {
    /// Constructs a default (nomic, 768-dim) [`CountingEmbedder`], incrementing
    /// `load_count`. Share `Arc<AtomicUsize>` across instances to count loads.
    pub fn new(load_count: Arc<AtomicUsize>, in_flight: Arc<AtomicUsize>) -> Self {
        Self::new_with_dim(
            DEFAULT_EMBED_DIM,
            DEFAULT_EMBED_MODEL_ID,
            "search_document: ",
            "search_query: ",
            load_count,
            in_flight,
        )
    }

    /// Constructs a [`CountingEmbedder`] for an arbitrary model, incrementing
    /// `load_count`. Prefixes are applied so deterministic output matches
    /// the model's convention (as `FastembedEmbedder` does).
    pub fn new_with_dim(
        dim: usize,
        model_id: &str,
        prefix_doc: &str,
        prefix_query: &str,
        load_count: Arc<AtomicUsize>,
        in_flight: Arc<AtomicUsize>,
    ) -> Self {
        load_count.fetch_add(1, Ordering::SeqCst);
        Self {
            load_count,
            in_flight,
            model_id: model_id.to_string(),
            dim,
            prefix_doc: prefix_doc.to_string(),
            prefix_query: prefix_query.to_string(),
        }
    }

    /// FNV-1a-style hash spread across `self.dim` components, then L2-normalized.
    /// Stable across runs for the same input.
    fn deterministic_vector(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; self.dim];
        let bytes = text.as_bytes();
        for (i, val) in v.iter_mut().enumerate() {
            let mut h: u64 = 2166136261u64.wrapping_add(i as u64);
            for &b in bytes {
                h = h.wrapping_mul(16777619).wrapping_add(b as u64);
            }
            *val = ((h & 0xFFFF) as f32 / 32767.5) - 1.0;
        }
        l2_normalize(&mut v);
        v
    }
}

impl Embedder for CountingEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LensError> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        let result = texts
            .iter()
            .map(|t| self.deterministic_vector(&format!("{}{t}", self.prefix_doc)))
            .collect();
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(result)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, LensError> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        let result = self.deterministic_vector(&format!("{}{text}", self.prefix_query));
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::AtomicUsize};

    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-9 || nb < 1e-9 {
            return 0.0;
        }
        dot / (na * nb)
    }

    fn make_embedder() -> CountingEmbedder {
        let load = Arc::new(AtomicUsize::new(0));
        let in_flight = Arc::new(AtomicUsize::new(0));
        CountingEmbedder::new(load, in_flight)
    }

    #[test]
    fn constants_are_correct() {
        assert_eq!(DEFAULT_EMBED_MODEL_ID, "nomic-embed-text-v1.5");
        assert_eq!(DEFAULT_EMBED_DIM, 768);
        assert_eq!(PREFIX_CONVENTION, "search_document/search_query");
    }

    #[test]
    fn counting_embedder_model_id_and_dim() {
        let e = make_embedder();
        assert_eq!(e.model_id(), DEFAULT_EMBED_MODEL_ID);
        assert_eq!(e.dim(), DEFAULT_EMBED_DIM);
    }

    #[test]
    fn load_count_increments_on_construction() {
        let load = Arc::new(AtomicUsize::new(0));
        let in_flight = Arc::new(AtomicUsize::new(0));
        assert_eq!(load.load(Ordering::SeqCst), 0);
        let _e1 = CountingEmbedder::new(Arc::clone(&load), Arc::clone(&in_flight));
        assert_eq!(load.load(Ordering::SeqCst), 1);
        let _e2 = CountingEmbedder::new(Arc::clone(&load), Arc::clone(&in_flight));
        assert_eq!(load.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn in_flight_returns_to_zero_after_embed() {
        let load = Arc::new(AtomicUsize::new(0));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let e = CountingEmbedder::new(Arc::clone(&load), Arc::clone(&in_flight));

        e.embed_documents(&["hello", "world"]).unwrap();
        assert_eq!(in_flight.load(Ordering::SeqCst), 0);

        e.embed_query("test query").unwrap();
        assert_eq!(in_flight.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn embed_documents_returns_correct_count_and_dim() {
        let e = make_embedder();
        let texts = ["hello", "world", "foo"];
        let vecs = e.embed_documents(&texts).unwrap();
        assert_eq!(vecs.len(), 3);
        for v in &vecs {
            assert_eq!(v.len(), DEFAULT_EMBED_DIM);
        }
    }

    #[test]
    fn embed_documents_vectors_are_l2_normalized() {
        let e = make_embedder();
        let vecs = e.embed_documents(&["test text", "another text"]).unwrap();
        for v in &vecs {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "vector norm {norm:.6} not within 1e-3 of 1.0"
            );
        }
    }

    #[test]
    fn embed_query_returns_correct_dim_and_is_normalized() {
        let e = make_embedder();
        let v = e.embed_query("what is the meaning of life?").unwrap();
        assert_eq!(v.len(), DEFAULT_EMBED_DIM);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "query vector norm {norm:.6} not within 1e-3 of 1.0"
        );
    }

    #[test]
    fn doc_vector_differs_from_query_vector_for_same_text() {
        let e = make_embedder();
        let text = "some test sentence";
        let doc_vecs = e.embed_documents(&[text]).unwrap();
        let query_vec = e.embed_query(text).unwrap();
        assert_ne!(
            doc_vecs[0], query_vec,
            "embed_documents and embed_query should produce different vectors for the same text"
        );
    }

    #[test]
    fn self_similarity_is_one_for_same_document() {
        let e = make_embedder();
        let text = "the quick brown fox";
        let vecs = e.embed_documents(&[text, text]).unwrap();
        let sim = cosine(&vecs[0], &vecs[1]);
        assert!(
            sim > 0.999,
            "cosine self-similarity should be > 0.999, got {sim}"
        );
    }

    #[test]
    fn self_similarity_is_one_for_same_query() {
        let e = make_embedder();
        let q1 = e.embed_query("rust programming language").unwrap();
        let q2 = e.embed_query("rust programming language").unwrap();
        let sim = cosine(&q1, &q2);
        assert!(
            sim > 0.999,
            "cosine self-similarity for query should be > 0.999, got {sim}"
        );
    }

    #[test]
    fn unrelated_texts_have_lower_cosine_similarity_than_self() {
        let e = make_embedder();
        let texts = ["the quick brown fox", "quantum chromodynamics"];
        let vecs = e.embed_documents(&texts).unwrap();
        let cross_sim = cosine(&vecs[0], &vecs[1]);
        assert!(
            cross_sim < 0.999,
            "unrelated texts should have cosine < 0.999, got {cross_sim}"
        );
    }

    #[test]
    fn deterministic_vector_is_stable_across_calls() {
        let e = make_embedder();
        let v1 = e.deterministic_vector("stable input");
        let v2 = e.deterministic_vector("stable input");
        assert_eq!(v1, v2);
    }

    fn make_embedder_for(spec: &EmbeddingModelSpec) -> CountingEmbedder {
        CountingEmbedder::new_with_dim(
            spec.dim,
            spec.id,
            spec.prefix_doc,
            spec.prefix_query,
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        )
    }

    #[test]
    fn default_counting_embedder_is_768_nomic() {
        let e = make_embedder();
        assert_eq!(e.dim(), 768);
        assert_eq!(e.model_id(), "nomic-embed-text-v1.5");
        let v = e.embed_query("anything").unwrap();
        assert_eq!(v.len(), 768);
    }

    #[test]
    fn counting_embedder_384_returns_correct_length_and_dim() {
        let e = make_embedder_for(resolve("all-minilm"));
        assert_eq!(e.dim(), 384);
        assert_eq!(e.model_id(), "all-minilm");
        let docs = e.embed_documents(&["a", "b"]).unwrap();
        assert_eq!(docs.len(), 2);
        for v in &docs {
            assert_eq!(v.len(), 384);
        }
        assert_eq!(e.embed_query("q").unwrap().len(), 384);
    }

    #[test]
    fn counting_embedder_1024_returns_correct_length_and_dim() {
        let e = make_embedder_for(resolve("mxbai-embed-large"));
        assert_eq!(e.dim(), 1024);
        assert_eq!(e.model_id(), "mxbai-embed-large");
        let docs = e.embed_documents(&["x"]).unwrap();
        assert_eq!(docs[0].len(), 1024);
        assert_eq!(e.embed_query("y").unwrap().len(), 1024);
    }

    #[test]
    fn vectors_are_l2_normalized_at_384() {
        let e = make_embedder_for(resolve("all-minilm"));
        let v = e.embed_query("normalize me").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "384-dim query vector norm {norm:.6} not within 1e-3 of 1.0"
        );
        let docs = e.embed_documents(&["also me"]).unwrap();
        let dnorm: f32 = docs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (dnorm - 1.0).abs() < 1e-3,
            "384-dim doc vector norm {dnorm:.6} not within 1e-3 of 1.0"
        );
    }

    #[test]
    fn nomic_prefixes_make_doc_differ_from_query() {
        // nomic applies both a doc prefix and a (different) query prefix, so the
        // same raw text hashes differently across the two paths.
        let e = make_embedder_for(resolve("nomic-embed-text-v1.5"));
        let text = "shared sentence";
        let doc = e.embed_documents(&[text]).unwrap();
        let query = e.embed_query(text).unwrap();
        assert_ne!(doc[0], query);
    }

    #[test]
    fn mxbai_query_prefix_makes_doc_differ_from_query() {
        // mxbai: doc prefix is empty, query prefix is non-empty, so doc != query.
        let e = make_embedder_for(resolve("mxbai-embed-large"));
        let text = "shared sentence";
        let doc = e.embed_documents(&[text]).unwrap();
        let query = e.embed_query(text).unwrap();
        assert_ne!(doc[0], query);
        // The empty doc prefix means the doc vector equals the raw-text hash.
        assert_eq!(doc[0], e.deterministic_vector(text));
    }

    #[test]
    fn all_minilm_no_prefix_makes_doc_equal_query() {
        // all-minilm has empty prefixes on both sides, so the same raw text
        // produces identical doc and query vectors (symmetric encoder).
        let e = make_embedder_for(resolve("all-minilm"));
        let text = "shared sentence";
        let doc = e.embed_documents(&[text]).unwrap();
        let query = e.embed_query(text).unwrap();
        assert_eq!(doc[0], query);
        assert_eq!(doc[0], e.deterministic_vector(text));
    }

    #[test]
    fn new_with_dim_increments_load_count() {
        let load = Arc::new(AtomicUsize::new(0));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let _e = CountingEmbedder::new_with_dim(
            384,
            "all-minilm",
            "",
            "",
            Arc::clone(&load),
            Arc::clone(&in_flight),
        );
        assert_eq!(load.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn fastembed_new_with_spec_rejects_ollama_only_model() {
        let dir = std::env::temp_dir();
        // FastembedEmbedder is not Debug; use .err() instead of expect_err.
        let err = FastembedEmbedder::new_with_spec(&dir, resolve("qwen3-embedding:4b"))
            .err()
            .expect("an ollama-only model must be rejected by the fastembed backend guard");
        match err {
            LensError::Validation(msg) => {
                assert!(msg.contains("qwen3-embedding:4b"), "names the model: {msg}");
                assert!(msg.contains("fastembed"), "names the backend: {msg}");
            }
            other => panic!("expected LensError::Validation, got {other:?}"),
        }
    }

    #[test]
    #[ignore = "downloads ~130 MB fastembed weights; run explicitly with a warm cache"]
    fn fastembed_new_with_spec_selects_model_dim() {
        let dir = std::env::temp_dir();
        let e = FastembedEmbedder::new_with_spec(&dir, resolve("all-minilm")).unwrap();
        assert_eq!(e.dim(), 384);
        assert_eq!(e.model_id(), "all-minilm");
        let v = e.embed_query("hello").unwrap();
        assert_eq!(v.len(), 384);
    }

    #[test]
    #[ignore = "downloads ~90 MB all-minilm weights to OBSERVE the on-disk cache layout"]
    fn observe_fastembed_cache_layout() {
        let dir = tempfile::tempdir().unwrap();
        let _e = FastembedEmbedder::new_with_spec(dir.path(), resolve("all-minilm")).unwrap();
        let root = dir.path().join("models").join("fastembed");
        fn walk(p: &std::path::Path, depth: usize) {
            if let Ok(rd) = std::fs::read_dir(p) {
                for e in rd.flatten() {
                    let path = e.path();
                    let ty = if path.is_dir() { "DIR " } else { "FILE" };
                    eprintln!("{}{ty} {}", "  ".repeat(depth), path.display());
                    if path.is_dir() {
                        walk(&path, depth + 1);
                    }
                }
            }
        }
        eprintln!("=== OBSERVED fastembed cache root: {} ===", root.display());
        walk(&root, 0);
    }
}
