//! Text-embedding abstractions: the [`Embedder`] trait, the production
//! [`FastembedEmbedder`] backed by `fastembed` + bundled onnxruntime, and the
//! test-only [`CountingEmbedder`] that verifies the cached-once / concurrency
//! invariants without requiring a model download.
//!
//! ## Prefix convention
//!
//! `nomic-embed-text-v1.5` requires caller-applied prefixes:
//! - `"search_document: "` for corpus text at ingest time.
//! - `"search_query: "` for query strings at retrieval time.
//!
//! `fastembed` 5.17.2 does **not** apply these automatically; [`FastembedEmbedder`]
//! applies them unconditionally.  See [`PREFIX_CONVENTION`] for the canonical
//! record.
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

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::LensError;

pub mod registry;

pub use registry::{DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL_ID, EmbeddingModelSpec, resolve};

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// Canonical model id for the default embedding model used in Phase 1.
///
/// Legacy alias of [`DEFAULT_EMBED_MODEL_ID`]; kept until Step 7 renames the
/// call sites. New code should reference [`DEFAULT_EMBED_MODEL_ID`].
pub const EMBED_MODEL_ID: &str = DEFAULT_EMBED_MODEL_ID;

/// Output dimension of [`EMBED_MODEL_ID`].
///
/// Legacy alias of [`DEFAULT_EMBED_DIM`]; kept until Step 7 renames the call
/// sites. New code should reference [`DEFAULT_EMBED_DIM`].
pub const EMBED_DIM: usize = DEFAULT_EMBED_DIM;

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
/// that is shared across threads and stored in the engine's `OnceCell`.
pub trait Embedder: Send + Sync {
    /// Returns the canonical model identifier, e.g. `"nomic-embed-text-v1.5"`.
    fn model_id(&self) -> &str;

    /// Returns the output vector dimension (e.g. `768`).
    fn dim(&self) -> usize;

    /// Embeds a batch of document texts.
    ///
    /// Prepends `"search_document: "` to each input before passing it to the
    /// underlying model.  Returns one `Vec<f32>` per input, in order.
    ///
    /// Every returned vector is length-[`EMBED_DIM`] and L2-normalized.
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LensError>;

    /// Owned-input variant of [`Embedder::embed_documents`] that avoids the
    /// borrow path's double copy.
    ///
    /// On the ingest hot path each chunk's text is already an owned `String`;
    /// `embed_documents(&[&str])` then clones every input again to prepend the
    /// `"search_document: "` prefix. This variant takes ownership of the input
    /// strings so the prefix can be applied in place (a single allocation per
    /// input). The default impl just borrows back into [`Embedder::embed_documents`]
    /// so existing implementations keep working unchanged.
    ///
    /// Every returned vector is length-[`EMBED_DIM`] and L2-normalized.
    fn embed_documents_owned(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, LensError> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.embed_documents(&refs)
    }

    /// Embeds a single query text.
    ///
    /// Prepends `"search_query: "` to the input before passing it to the
    /// underlying model.  Returns one L2-normalized `Vec<f32>` of length
    /// [`EMBED_DIM`].
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, LensError>;
}

// ---------------------------------------------------------------------------
// FastembedEmbedder — production implementation
// ---------------------------------------------------------------------------

/// Production embedder backed by `fastembed` + bundled onnxruntime.
///
/// Wraps [`fastembed::TextEmbedding`] for `nomic-embed-text-v1.5` (768d) and
/// applies the required `search_document:` / `search_query:` prefixes.
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
}

impl FastembedEmbedder {
    /// Builds a `FastembedEmbedder`.
    ///
    /// The ONNX session is initialized here (~130 MB).  On first call, the
    /// `nomic-embed-text-v1.5` weights are downloaded from HuggingFace into
    /// `{data_dir}/models/fastembed/` and cached there for subsequent runs.
    ///
    /// # Errors
    ///
    /// Returns [`LensError::Model`] if the fastembed session cannot be
    /// initialized (download failure, corrupt weights, onnxruntime error, …).
    pub fn new(data_dir: &Path) -> Result<Self, LensError> {
        let cache_dir = data_dir.join("models").join("fastembed");
        let opts = InitOptions::new(EmbeddingModel::NomicEmbedTextV15).with_cache_dir(cache_dir);
        let inner = TextEmbedding::try_new(opts)
            .map_err(|e| LensError::Model(format!("fastembed init failed: {e}")))?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Embeds a batch of already-prefixed document strings (the
    /// `"search_document: "` prefix must already be applied) and validates the
    /// output. Shared by [`Embedder::embed_documents`] and
    /// [`Embedder::embed_documents_owned`].
    fn embed_prefixed_documents(&self, prefixed: Vec<String>) -> Result<Vec<Vec<f32>>, LensError> {
        let prefixed_refs: Vec<&str> = prefixed.iter().map(String::as_str).collect();
        let result = self
            .inner
            .lock()
            .map_err(|e| LensError::Model(format!("fastembed mutex poisoned: {e}")))?
            .embed(prefixed_refs, None)
            .map_err(|e| LensError::Model(format!("fastembed embed_documents failed: {e}")))?;
        Self::assert_normalized(&result)?;
        Ok(result)
    }

    /// Asserts that every vector in `vecs` is L2-normalized (‖v‖ ≈ 1.0 ± 1e-3)
    /// and has the expected dimension.
    ///
    /// `fastembed` 5.17.2 normalizes unconditionally, so this should never fire.
    /// It is a cheap defensive canary, not a correctness dependency.
    fn assert_normalized(vecs: &[Vec<f32>]) -> Result<(), LensError> {
        for (i, v) in vecs.iter().enumerate() {
            if v.len() != EMBED_DIM {
                return Err(LensError::Model(format!(
                    "embedder returned vector {i} with dim {} (expected {EMBED_DIM})",
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
        EMBED_MODEL_ID
    }

    fn dim(&self) -> usize {
        EMBED_DIM
    }

    /// Prepends `"search_document: "` to each text, embeds the batch, validates
    /// normalization + dimension, and returns the result.
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LensError> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("search_document: {t}"))
            .collect();
        self.embed_prefixed_documents(prefixed)
    }

    /// Owned-input variant: prefixes each input **in place** (`insert_str`) so the
    /// ingest hot path copies each chunk's text once instead of twice.
    fn embed_documents_owned(&self, mut texts: Vec<String>) -> Result<Vec<Vec<f32>>, LensError> {
        for t in texts.iter_mut() {
            t.insert_str(0, "search_document: ");
        }
        self.embed_prefixed_documents(texts)
    }

    /// Prepends `"search_query: "` to the text, embeds it, validates normalization
    /// + dimension, and returns the single result vector.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, LensError> {
        let prefixed = format!("search_query: {text}");
        let result = self
            .inner
            .lock()
            .map_err(|e| LensError::Model(format!("fastembed mutex poisoned: {e}")))?
            .embed(vec![prefixed.as_str()], None)
            .map_err(|e| LensError::Model(format!("fastembed embed_query failed: {e}")))?;
        Self::assert_normalized(&result)?;
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
}

impl CountingEmbedder {
    /// Constructs a new [`CountingEmbedder`], incrementing `load_count` by `1`.
    ///
    /// Pass shared `Arc<AtomicUsize>` values so multiple embedder instances (or
    /// engine-level caching wrappers) all write to the same counters.
    pub fn new(load_count: Arc<AtomicUsize>, in_flight: Arc<AtomicUsize>) -> Self {
        load_count.fetch_add(1, Ordering::SeqCst);
        Self {
            load_count,
            in_flight,
        }
    }

    /// Produces a deterministic, L2-normalized 768-dim vector from an arbitrary
    /// string.
    ///
    /// Uses a simple FNV-1a-style hash spread across 768 components, then
    /// L2-normalizes.  The result is stable across runs for the same input.
    fn deterministic_vector(text: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; EMBED_DIM];
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
        // L2-normalize.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        v
    }
}

impl Embedder for CountingEmbedder {
    fn model_id(&self) -> &str {
        EMBED_MODEL_ID
    }

    fn dim(&self) -> usize {
        EMBED_DIM
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LensError> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        let result = texts
            .iter()
            .map(|t| Self::deterministic_vector(&format!("search_document: {t}")))
            .collect();
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(result)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, LensError> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        let result = Self::deterministic_vector(&format!("search_query: {text}"));
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
        assert_eq!(EMBED_MODEL_ID, "nomic-embed-text-v1.5");
        assert_eq!(EMBED_DIM, 768);
        assert_eq!(PREFIX_CONVENTION, "search_document/search_query");
    }

    // --- CountingEmbedder structural tests ---

    #[test]
    fn counting_embedder_model_id_and_dim() {
        let e = make_embedder();
        assert_eq!(e.model_id(), EMBED_MODEL_ID);
        assert_eq!(e.dim(), EMBED_DIM);
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
            assert_eq!(v.len(), EMBED_DIM);
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
        assert_eq!(v.len(), EMBED_DIM);
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
        let v1 = CountingEmbedder::deterministic_vector("stable input");
        let v2 = CountingEmbedder::deterministic_vector("stable input");
        assert_eq!(v1, v2);
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
