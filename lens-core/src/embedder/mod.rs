//! Text-embedding abstractions: the [`Embedder`] trait, the production
//! [`FastembedEmbedder`] backed by `fastembed` + bundled onnxruntime, and the
//! test-only [`CountingEmbedder`] that verifies the cached-once / concurrency
//! invariants without requiring a model download.
//!
//! ## Prefix convention
//!
//! Prefixes are PER-MODEL and live in the registry's [`EmbeddingModelSpec`]
//! (`prefix_doc`/`prefix_query`; empty = none). For example
//! `nomic-embed-text-v1.5` uses `"search_document: "` / `"search_query: "`, while
//! `all-minilm`/`bge-m3` use no prefix and `mxbai-embed-large` prefixes only the
//! query. `fastembed` 5.17.2 does **not** apply any of these automatically;
//! [`FastembedEmbedder`] applies its spec's prefixes (skipping empty ones).
//! [`PREFIX_CONVENTION`] records the nomic default specifically.
//!
//! ## Normalization
//!
//! `fastembed` 5.17.2 L2-normalizes output unconditionally for every model.
//! [`FastembedEmbedder`] adds a cheap **defensive** assert (`‖v‖ ≈ 1.0 ± 1e-3`)
//! after each call; the assert is not a correctness dependency — it is a canary
//! that fires if a future fastembed upgrade breaks that guarantee.
//!
//! ## Blocking contract
//!
//! `fastembed::TextEmbedding::embed` is **synchronous** and runs ~130 MB ONNX
//! inference.  The trait stays sync; callers on the async ingest path **MUST**
//! wrap every `embed_documents` / `embed_query` call in
//! [`tokio::task::spawn_blocking`].  A direct call on the Tokio runtime is a
//! defect: it blocks the worker thread for the duration of inference.

use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use fastembed::{InitOptions, TextEmbedding};

use crate::LensError;

// issue #91: the candle + Metal (Apple GPU) embedding backend. Feature-gated to
// aarch64-apple-darwin only (see Cargo.toml `native-ml-metal`).
#[cfg(feature = "native-ml-metal")]
pub mod candle_backend;
// issue #91: per-job compute-device selection policy + the polymorphic
// NativeAccelerator seam. Always compiled (the policy is pure + testable without
// the GPU feature); only the Metal accelerator impl is feature-gated.
pub mod device;
pub mod ollama;
pub mod registry;

#[cfg(feature = "native-ml-metal")]
pub use candle_backend::CandleNomicEmbedder;
pub use device::{
    Acceleration, Compute, NativeAccelerator, WorkloadKind, default_accelerator, select_compute,
};
pub use ollama::OllamaEmbedder;
pub use registry::{
    DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, EmbeddingBackend, EmbeddingModelSpec, REGISTRY,
    resolve, resolve_opt,
};

/// L2-normalizes `v` in place. The single normalization primitive shared by the
/// backends that must normalize themselves ([`OllamaEmbedder`], which talks to a
/// server that does NOT normalize) and by [`CountingEmbedder`]'s deterministic
/// test vectors — so the zero-norm guard and epsilon live in exactly one place.
///
/// Zero-norm guard in f32 space: a near-zero vector would divide by ~0 and
/// produce NaN/Inf; `1e-9` is comfortably above f32 rounding noise yet far below
/// any real unit-vector norm, so a genuine embedding always normalizes while a
/// degenerate zero vector is left untouched.
pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------
//
// `DEFAULT_EMBED_MODEL_ID` / `DEFAULT_EMBED_DIM` are re-exported from
// `registry` above (the single source of truth for the default coordinate).

/// Human-readable record of the prefix convention baked into [`Embedder`].
/// `"search_document/search_query"` matches the `embedding_index.prefix_convention`
/// column value written by the vector-store registry.
pub const PREFIX_CONVENTION: &str = "search_document/search_query";

// ---------------------------------------------------------------------------
// Embedder trait
// ---------------------------------------------------------------------------

/// A synchronous text-embedding seam.
///
/// ## Why sync?
///
/// `fastembed::TextEmbedding::embed` is synchronous.  The trait is therefore
/// sync, and the async ingest path **must** call it under
/// `tokio::task::spawn_blocking` — see module-level docs.
///
/// ## Object safety
///
/// The trait is `Send + Sync` so it can be held behind an `Arc<dyn Embedder>`
/// that is shared across threads and stored in the engine's keyed embedder cache
/// (one entry per `model_id`).
pub trait Embedder: Send + Sync {
    /// Returns the canonical model identifier, e.g. `"nomic-embed-text-v1.5"`.
    fn model_id(&self) -> &str;

    /// Returns the output vector dimension (model-dependent: 384, 768, or 1024).
    fn dim(&self) -> usize;

    /// Embeds a batch of document texts.
    ///
    /// Prepends the model's document prefix (from its registry spec; empty = no
    /// prefix) to each input before passing it to the underlying model. Returns
    /// one `Vec<f32>` per input, in order.
    ///
    /// Every returned vector has length [`Embedder::dim`] and is L2-normalized.
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LensError>;

    /// Owned-input variant of [`Embedder::embed_documents`] that avoids the
    /// borrow path's double copy.
    ///
    /// On the ingest hot path each chunk's text is already an owned `String`;
    /// `embed_documents(&[&str])` then clones every input again to prepend the
    /// model's document prefix. This variant takes ownership of the input strings
    /// so the prefix can be applied in place (a single allocation per input). The
    /// default impl just borrows back into [`Embedder::embed_documents`] so
    /// existing implementations keep working unchanged.
    ///
    /// Every returned vector has length [`Embedder::dim`] and is L2-normalized.
    fn embed_documents_owned(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, LensError> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.embed_documents(&refs)
    }

    /// Embeds a single query text.
    ///
    /// Prepends the model's query prefix (from its registry spec; empty = no
    /// prefix) to the input before passing it to the underlying model. Returns one
    /// L2-normalized `Vec<f32>` of length [`Embedder::dim`].
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, LensError>;
}

// ---------------------------------------------------------------------------
// FastembedEmbedder — production implementation
// ---------------------------------------------------------------------------

/// Production embedder backed by `fastembed` + bundled onnxruntime.
///
/// Wraps [`fastembed::TextEmbedding`] for any model described by an
/// [`EmbeddingModelSpec`] and applies that spec's document / query prefixes
/// (empty = none).
///
/// **Construction is expensive** (~130 MB ONNX session init, plus a one-time
/// model download from HuggingFace on first use).  Construct once and cache
/// behind an `Arc<dyn Embedder>` in the engine — see Decision D1 in the plan.
pub struct FastembedEmbedder {
    /// `fastembed::TextEmbedding::embed` takes `&mut self`, but the [`Embedder`]
    /// trait is `&self` (it is shared read-only behind `Arc<dyn Embedder>`). A
    /// `Mutex` provides the required interior mutability; embed calls are already
    /// serialized by the engine's single-permit ingest semaphore, so the lock is
    /// effectively uncontended.
    inner: Mutex<TextEmbedding>,
    /// Stable model id, copied from the spec this embedder was built from.
    model_id: String,
    /// Output dimension, copied from the spec.
    dim: usize,
    /// Document prefix from the spec (`""` = apply none).
    prefix_doc: String,
    /// Query prefix from the spec (`""` = apply none).
    prefix_query: String,
}

impl FastembedEmbedder {
    /// Builds a `FastembedEmbedder` for the default model
    /// ([`DEFAULT_EMBED_MODEL_ID`]).
    ///
    /// Convenience wrapper over [`FastembedEmbedder::new_with_spec`] that
    /// resolves the default spec from the registry.
    ///
    /// The ONNX session is initialized here (~130 MB).  On first call, the
    /// model weights are downloaded from HuggingFace into
    /// `{data_dir}/models/fastembed/` and cached there for subsequent runs.
    ///
    /// # Errors
    ///
    /// Returns [`LensError::Model`] if the fastembed session cannot be
    /// initialized (download failure, corrupt weights, onnxruntime error, …).
    pub fn new(data_dir: &Path) -> Result<Self, LensError> {
        Self::new_with_spec(data_dir, resolve(DEFAULT_EMBED_MODEL_ID))
    }

    /// Builds a `FastembedEmbedder` for the model described by `spec`.
    ///
    /// The fastembed variant, dimension, and prefix convention are all taken
    /// from `spec` (the registry is the single source of truth). Weights for
    /// `spec.fastembed_variant` are downloaded into `{data_dir}/models/fastembed/`
    /// on first use and cached there.
    ///
    /// # Errors
    ///
    /// Returns [`LensError::Model`] if the fastembed session cannot be
    /// initialized (download failure, corrupt weights, onnxruntime error, …).
    pub fn new_with_spec(data_dir: &Path, spec: &EmbeddingModelSpec) -> Result<Self, LensError> {
        // Defense-in-depth backend guard (issue #80): a spec that has no fastembed
        // variant (an Ollama-only model) can never be served by fastembed. Reject
        // it here with a clear error rather than unwrapping `None` — the primary
        // guard lives in `embedder_for`, this is the last line of defense.
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
        let cache_dir = data_dir.join("models").join("fastembed");
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

    /// Embeds a batch of already-prefixed document strings (the model's document
    /// prefix must already be applied) and validates the output. Shared by
    /// [`Embedder::embed_documents`] and [`Embedder::embed_documents_owned`].
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

    /// Asserts that every vector in `vecs` is L2-normalized (‖v‖ ≈ 1.0 ± 1e-3)
    /// and has this embedder's expected dimension (`self.dim()`).
    ///
    /// `fastembed` 5.17.2 normalizes unconditionally, so this should never fire.
    /// It is a cheap defensive canary, not a correctness dependency.
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

    /// Prepends this model's document prefix (`self.prefix_doc`, empty = none) to
    /// each text, embeds the batch, validates normalization + dimension, and
    /// returns the result.
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LensError> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.prefix_doc))
            .collect();
        self.embed_prefixed_documents(prefixed)
    }

    /// Owned-input variant: prefixes each input **in place** (`insert_str`) so the
    /// ingest hot path copies each chunk's text once instead of twice. A model
    /// with an empty document prefix skips the insert entirely.
    fn embed_documents_owned(&self, mut texts: Vec<String>) -> Result<Vec<Vec<f32>>, LensError> {
        if !self.prefix_doc.is_empty() {
            for t in texts.iter_mut() {
                t.insert_str(0, &self.prefix_doc);
            }
        }
        self.embed_prefixed_documents(texts)
    }

    /// Prepends this model's query prefix (`self.prefix_query`, empty = none) to
    /// the text, embeds it, validates normalization + dimension, and returns the
    /// single result vector.
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

// ---------------------------------------------------------------------------
// CountingEmbedder — test seam
// ---------------------------------------------------------------------------

/// A deterministic, model-free embedder for unit and integration tests.
///
/// ## Purpose
///
/// [`CountingEmbedder`] lets tests verify the **cached-once** and
/// **concurrency-serialization** ACs without downloading the real model:
///
/// - `load_count` is incremented by `1` in [`CountingEmbedder::new`], allowing
///   a test to assert that the real embedder was constructed exactly once even
///   when multiple ingest calls run in sequence or concurrency.
/// - `in_flight` is incremented on entry and decremented on exit of every
///   `embed_documents` / `embed_query` call, so a test can assert that the
///   peak in-flight count never exceeded `1` (single-permit semaphore AC).
///
/// ## Vector quality
///
/// Output vectors are **deterministic for a given input string** (derived from
/// a simple hash) and **L2-normalized to length 1.0**, so the normalization
/// defensive assert in `FastembedEmbedder` still fires if mis-wired.
///
/// ### Cosine self-similarity
///
/// `embed_documents(["x"])` and `embed_query("x")` hash the **prefixed** string
/// (`"search_document: x"` vs `"search_query: x"`), so they produce **different**
/// vectors — which satisfies the "doc ≠ query for the same string" AC.
/// Self-similarity (`embed_documents(["x"])` vs `embed_documents(["x"])`) is
/// exactly `1.0` by construction (identical deterministic hash).
///
/// ## Usage
///
/// ```rust,ignore
/// use std::sync::{Arc, atomic::AtomicUsize};
/// use lens_core::embedder::CountingEmbedder;
///
/// let load_count = Arc::new(AtomicUsize::new(0));
/// let in_flight  = Arc::new(AtomicUsize::new(0));
/// let embedder   = CountingEmbedder::new(Arc::clone(&load_count), Arc::clone(&in_flight));
///
/// // After constructing two embedders from the same counters, load_count == 2.
/// // After a single embed call, in_flight returns to 0.
/// ```
pub struct CountingEmbedder {
    /// Incremented once per construction.  Share the same `Arc` across multiple
    /// `CountingEmbedder` instances to count total model loads.
    pub load_count: Arc<AtomicUsize>,
    /// Instantaneous count of embed calls in progress.  Tests assert this never
    /// exceeds `1` (single-permit semaphore).
    pub in_flight: Arc<AtomicUsize>,
    /// Stable model id reported by [`Embedder::model_id`].
    model_id: String,
    /// Output dimension of generated vectors and reported by [`Embedder::dim`].
    dim: usize,
    /// Document prefix applied in [`Embedder::embed_documents`] (`""` = none).
    prefix_doc: String,
    /// Query prefix applied in [`Embedder::embed_query`] (`""` = none).
    prefix_query: String,
}

impl CountingEmbedder {
    /// Constructs a new default ([`DEFAULT_EMBED_MODEL_ID`], 768-dim,
    /// nomic prefixes) [`CountingEmbedder`], incrementing `load_count` by `1`.
    ///
    /// Pass shared `Arc<AtomicUsize>` values so multiple embedder instances (or
    /// engine-level caching wrappers) all write to the same counters.
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
    /// `load_count` by `1`.
    ///
    /// `dim` controls the length of every generated vector (and what
    /// [`Embedder::dim`] reports); `model_id` is echoed by
    /// [`Embedder::model_id`]; `prefix_doc` / `prefix_query` (empty = none) are
    /// applied in the embed calls so the deterministic output depends on the
    /// model's prefix convention, matching [`FastembedEmbedder`].
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

    /// Produces a deterministic, L2-normalized `self.dim`-length vector from an
    /// arbitrary string.
    ///
    /// Uses a simple FNV-1a-style hash spread across `self.dim` components, then
    /// L2-normalizes.  The result is stable across runs for the same input.
    fn deterministic_vector(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; self.dim];
        // Spread the bytes of the input across the output dimensions with a
        // simple hash mix to ensure distinct strings produce distinct vectors.
        let bytes = text.as_bytes();
        for (i, val) in v.iter_mut().enumerate() {
            // Combine dimension index with input bytes for per-dimension variety.
            let mut h: u64 = 2166136261u64.wrapping_add(i as u64);
            for &b in bytes {
                h = h.wrapping_mul(16777619).wrapping_add(b as u64);
            }
            // Map to [-1, 1] range before normalizing.
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

// ---------------------------------------------------------------------------
// Unit tests (no model download required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::AtomicUsize};

    use super::*;

    // Helper: cosine similarity of two equal-length vectors.
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

    // --- Trait constants ---

    #[test]
    fn constants_are_correct() {
        assert_eq!(DEFAULT_EMBED_MODEL_ID, "nomic-embed-text-v1.5");
        assert_eq!(DEFAULT_EMBED_DIM, 768);
        assert_eq!(PREFIX_CONVENTION, "search_document/search_query");
    }

    // --- CountingEmbedder structural tests ---

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

    // --- Output dimension and normalization (CountingEmbedder) ---

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

    // --- Prefix logic: doc ≠ query for same string ---

    #[test]
    fn doc_vector_differs_from_query_vector_for_same_text() {
        let e = make_embedder();
        let text = "some test sentence";
        let doc_vecs = e.embed_documents(&[text]).unwrap();
        let query_vec = e.embed_query(text).unwrap();
        // The prefixes differ so the hashes differ so the vectors differ.
        assert_ne!(
            doc_vecs[0], query_vec,
            "embed_documents and embed_query should produce different vectors for the same text"
        );
    }

    // --- Cosine self-similarity ---

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
        // Self-similarity is 1.0 by construction; cross should be strictly lower.
        // For the deterministic hash function this is virtually guaranteed for
        // distinct strings, but we use a lenient threshold to keep this test
        // stable across minor implementation changes.
        assert!(
            cross_sim < 0.999,
            "unrelated texts should have cosine < 0.999, got {cross_sim}"
        );
    }

    // --- Determinism ---

    #[test]
    fn deterministic_vector_is_stable_across_calls() {
        let e = make_embedder();
        let v1 = e.deterministic_vector("stable input");
        let v2 = e.deterministic_vector("stable input");
        assert_eq!(v1, v2);
    }

    // --- Step 2: parameterized model id / dim / prefixes (R4) ---

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

    /// Step 4 (issue #80): the construction-time backend guard rejects an
    /// Ollama-only spec BEFORE any ONNX/network work, with a clear error naming the
    /// model. Runs WITHOUT a download because the guard returns early — the
    /// defense-in-depth backstop under `embedder_for`'s primary guard.
    #[test]
    fn fastembed_new_with_spec_rejects_ollama_only_model() {
        let dir = std::env::temp_dir();
        // `FastembedEmbedder` is not `Debug`, so `.err()` (no Debug bound) instead
        // of `expect_err` (which requires the Ok type to be Debug).
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

    // --- Step 2: FastembedEmbedder real-model test (gated) ---

    // Constructing a real FastembedEmbedder downloads ~130 MB of ONNX weights
    // from HuggingFace, so it is #[ignore]d by default and only runs when
    // invoked explicitly (e.g. `cargo test -- --ignored` with a warm cache).
    // CountingEmbedder above covers the dim/prefix wiring deterministically.
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

    // R6 OBSERVE probe (M4 Phase 4b-B Step 5): downloads the smallest model
    // (all-minilm ~90 MB) and walks the cache dir so we can record the LITERAL
    // per-model subdir shape fastembed/hf-hub writes. #[ignore]d (network +
    // download); run explicitly with `cargo test -- --ignored observe_`.
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

    // -----------------------------------------------------------------------
    // NOTE: Tests that require the REAL fastembed model (FastembedEmbedder)
    // are NOT included here because they require a ~130 MB model download
    // from HuggingFace on first run (network-dependent, not hermetic).
    //
    // Those tests live in `lens-core/tests/ingest.rs` as integration tests
    // and are marked with `#[ignore]` or gated behind a feature/env var so
    // they only run in CI with a pre-warmed model cache.
    //
    // Integration tests that DO exercise FastembedEmbedder verify:
    //   - 768-dim L2-normalized output from the real ONNX session
    //   - cosine self-similarity > 0.999 for the same string
    //   - unrelated strings have strictly lower cosine similarity
    //   - embed_documents ≠ embed_query for the same text (prefix logic)
    // -----------------------------------------------------------------------
}
