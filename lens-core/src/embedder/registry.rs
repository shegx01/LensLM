//! Embedding-model registry: the single source of truth that maps a stable
//! model id to its output dimension, the concrete `fastembed` variant, and the
//! caller-applied document/query prefixes.
//!
//! ## Why a registry?
//!
//! A notebook stores which embedding model it was indexed with (M4 Phase 4b).
//! Every read/write path needs to translate that stored id into the concrete
//! `fastembed::EmbeddingModel` to construct, the vector dimension to validate
//! against, and the prefix convention to apply at embed time. Centralizing that
//! mapping here keeps the four facts (id ↔ dim ↔ variant ↔ prefixes) in lock-step.
//!
//! ## Prefix convention (per model card / fastembed defaults)
//!
//! `fastembed` 5.17.2 does **not** inject task prefixes for any of these four
//! models — the caller is responsible. The registry is therefore the canonical
//! record of each model's prefix requirement:
//!
//! - **`nomic-embed-text-v1.5`** — requires explicit task prefixes
//!   (`"search_document: "` / `"search_query: "`). Matches the convention the
//!   pre-registry `FastembedEmbedder` hardcoded.
//! - **`mxbai-embed-large`** — asymmetric: documents are embedded raw, queries
//!   are prefixed with the model's retrieval instruction
//!   (`"Represent this sentence for searching relevant passages: "`).
//! - **`all-minilm`** (`all-MiniLM-L6-v2`) — no prefixes; symmetric sentence
//!   encoder.
//! - **`bge-m3`** — no prefixes; the model embeds queries and passages without
//!   task instructions in its default dense-retrieval mode.

use fastembed::EmbeddingModel;

/// Canonical model id for the default embedding model (Phase 1 carry-over).
pub const DEFAULT_EMBED_MODEL_ID: &str = "nomic-embed-text-v1.5";

/// Output dimension of [`DEFAULT_EMBED_MODEL_ID`].
pub const DEFAULT_EMBED_DIM: usize = 768;

/// Legacy alias historically used for the default model before the `-v1.5`
/// suffix was standardized. Resolves to the same entry as
/// [`DEFAULT_EMBED_MODEL_ID`].
const LEGACY_DEFAULT_ALIAS: &str = "nomic-embed-text";

/// Static description of one supported embedding model.
///
/// Holds everything a read/write path needs to construct the embedder and
/// validate its output: the stable [`id`](Self::id), the output
/// [`dim`](Self::dim), the concrete [`fastembed_variant`](Self::fastembed_variant),
/// and the caller-applied [`prefix_doc`](Self::prefix_doc) /
/// [`prefix_query`](Self::prefix_query). An empty prefix means "apply none".
pub struct EmbeddingModelSpec {
    /// Stable, storage-facing model id (the value persisted on a notebook).
    pub id: &'static str,
    /// Output vector dimension.
    pub dim: usize,
    /// Concrete `fastembed` variant to construct.
    pub fastembed_variant: EmbeddingModel,
    /// Prefix prepended to each document at ingest time (`""` = none).
    pub prefix_doc: &'static str,
    /// Prefix prepended to each query at retrieval time (`""` = none).
    pub prefix_query: &'static str,
}

/// The complete set of supported embedding models. The first entry is the
/// default and is what unknown / empty ids resolve to.
///
/// SYNC-CHECK: keep in sync with `src/lib/onboarding/system-check.ts`
/// `EMBEDDING_MODELS`. The ids and dims here must match the TS `EmbeddingModelSpec`
/// array; also keep `ALLOWED_EMBEDDING_MODELS` in `system_check.rs` in sync.
pub static REGISTRY: &[EmbeddingModelSpec] = &[
    EmbeddingModelSpec {
        id: "nomic-embed-text-v1.5",
        dim: 768,
        fastembed_variant: EmbeddingModel::NomicEmbedTextV15,
        prefix_doc: "search_document: ",
        prefix_query: "search_query: ",
    },
    EmbeddingModelSpec {
        id: "mxbai-embed-large",
        dim: 1024,
        fastembed_variant: EmbeddingModel::MxbaiEmbedLargeV1,
        prefix_doc: "",
        prefix_query: "Represent this sentence for searching relevant passages: ",
    },
    EmbeddingModelSpec {
        id: "all-minilm",
        dim: 384,
        fastembed_variant: EmbeddingModel::AllMiniLML6V2,
        prefix_doc: "",
        prefix_query: "",
    },
    EmbeddingModelSpec {
        id: "bge-m3",
        dim: 1024,
        fastembed_variant: EmbeddingModel::BGEM3,
        prefix_doc: "",
        prefix_query: "",
    },
];

/// Resolves a model id to its [`EmbeddingModelSpec`].
///
/// Matching is exact on [`EmbeddingModelSpec::id`]. The legacy alias
/// `"nomic-embed-text"` also resolves to the nomic entry. An unknown or empty
/// id falls back to the default ([`DEFAULT_EMBED_MODEL_ID`], the first registry
/// entry) so callers always get a usable spec.
pub fn resolve(id: &str) -> &'static EmbeddingModelSpec {
    resolve_opt(id).unwrap_or_else(default_spec)
}

/// Resolves a model id to its [`EmbeddingModelSpec`], returning `None` for a
/// genuinely-unknown id (rather than falling back to the default).
///
/// The legacy alias `"nomic-embed-text"` resolves to the nomic entry (so callers
/// can accept the frontend's Ollama-facing id and persist the canonical
/// `spec.id`). Use this — not [`resolve`] — when an unknown id must be REJECTED
/// (e.g. the `set_notebook_embedding_model` command, the `eval --model` flag).
pub fn resolve_opt(id: &str) -> Option<&'static EmbeddingModelSpec> {
    if id == LEGACY_DEFAULT_ALIAS {
        return Some(default_spec());
    }
    REGISTRY.iter().find(|spec| spec.id == id)
}

/// Returns the default spec (the first registry entry, the nomic model).
fn default_spec() -> &'static EmbeddingModelSpec {
    &REGISTRY[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variant_eq(a: &EmbeddingModel, b: &EmbeddingModel) -> bool {
        // EmbeddingModel derives Debug; compare via its debug form since it does
        // not implement PartialEq in fastembed 5.17.2.
        format!("{a:?}") == format!("{b:?}")
    }

    #[test]
    fn resolve_nomic() {
        let s = resolve("nomic-embed-text-v1.5");
        assert_eq!(s.id, "nomic-embed-text-v1.5");
        assert_eq!(s.dim, 768);
        assert!(variant_eq(
            &s.fastembed_variant,
            &EmbeddingModel::NomicEmbedTextV15
        ));
        assert_eq!(s.prefix_doc, "search_document: ");
        assert_eq!(s.prefix_query, "search_query: ");
    }

    #[test]
    fn resolve_mxbai() {
        let s = resolve("mxbai-embed-large");
        assert_eq!(s.id, "mxbai-embed-large");
        assert_eq!(s.dim, 1024);
        assert!(variant_eq(
            &s.fastembed_variant,
            &EmbeddingModel::MxbaiEmbedLargeV1
        ));
        assert_eq!(s.prefix_doc, "");
        assert_eq!(
            s.prefix_query,
            "Represent this sentence for searching relevant passages: "
        );
    }

    #[test]
    fn resolve_all_minilm() {
        let s = resolve("all-minilm");
        assert_eq!(s.id, "all-minilm");
        assert_eq!(s.dim, 384);
        assert!(variant_eq(
            &s.fastembed_variant,
            &EmbeddingModel::AllMiniLML6V2
        ));
        assert_eq!(s.prefix_doc, "");
        assert_eq!(s.prefix_query, "");
    }

    #[test]
    fn resolve_bge_m3() {
        let s = resolve("bge-m3");
        assert_eq!(s.id, "bge-m3");
        assert_eq!(s.dim, 1024);
        assert!(variant_eq(&s.fastembed_variant, &EmbeddingModel::BGEM3));
        assert_eq!(s.prefix_doc, "");
        assert_eq!(s.prefix_query, "");
    }

    #[test]
    fn resolve_unknown_falls_back_to_default() {
        let s = resolve("does-not-exist");
        assert_eq!(s.id, DEFAULT_EMBED_MODEL_ID);
        assert_eq!(s.dim, DEFAULT_EMBED_DIM);
    }

    #[test]
    fn resolve_empty_falls_back_to_default() {
        let s = resolve("");
        assert_eq!(s.id, DEFAULT_EMBED_MODEL_ID);
        assert_eq!(s.dim, DEFAULT_EMBED_DIM);
    }

    #[test]
    fn resolve_legacy_alias_maps_to_nomic() {
        let s = resolve("nomic-embed-text");
        assert_eq!(s.id, "nomic-embed-text-v1.5");
        assert_eq!(s.dim, 768);
        assert!(variant_eq(
            &s.fastembed_variant,
            &EmbeddingModel::NomicEmbedTextV15
        ));
    }

    #[test]
    fn default_consts_are_correct() {
        assert_eq!(DEFAULT_EMBED_MODEL_ID, "nomic-embed-text-v1.5");
        assert_eq!(DEFAULT_EMBED_DIM, 768);
    }

    #[test]
    fn registry_has_exactly_four_models() {
        assert_eq!(REGISTRY.len(), 4);
    }
}
