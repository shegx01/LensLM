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

/// The embedding *backend* that physically computes a notebook's vectors
/// (M4 Phase 4b-B). Orthogonal to the model id: the SAME registry model (e.g.
/// `nomic-embed-text-v1.5`/768) can be served by either `fastembed` (on-device
/// ONNX) or a local `ollama` server, and the two MUST live in physically
/// distinct vector tables (they are different numerical embeddings). The backend
/// is therefore a first-class axis of a notebook's embedding *coordinate*
/// alongside `(model, dim)`.
///
/// Strong-typed on purpose ([[strong-typing-no-stringly-domain]]): callers never
/// pass a raw `"fastembed"`/`"ollama"` string. [`as_str`](Self::as_str) /
/// [`from_str`](Self::from_str) / [`from_opt_str`](Self::from_opt_str) bridge the
/// storage/serde boundary, where an empty / NULL / unknown value resolves to the
/// [`Default`] (`Fastembed`) — the SAME empty-string-resolves-to-default pattern
/// the `embedding_model` config/notebook field uses via the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingBackend {
    /// On-device ONNX embeddings via the `fastembed` crate (the default; the only
    /// backend that existed before 4b-B). Weights are downloaded on construction.
    #[default]
    Fastembed,
    /// Embeddings computed by a local Ollama server (detect-only; the app never
    /// shells `ollama pull`). Loopback-bound for safety.
    Ollama,
}

impl EmbeddingBackend {
    /// The stable, storage-facing token for this backend (the value persisted on
    /// a notebook / in `embedding_index.backend` and matched in SQL).
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbeddingBackend::Fastembed => "fastembed",
            EmbeddingBackend::Ollama => "ollama",
        }
    }

    /// Parses a stored backend token. An empty or unknown value resolves to the
    /// [`Default`] (`Fastembed`) so callers always get a usable backend — the
    /// same forgiving resolution `embedding_model` uses via the registry.
    ///
    /// Deliberately INFALLIBLE (no `Result`): a NULL/empty/unknown stored value is
    /// a normal, expected case that resolves to the default, NOT a parse error.
    /// That is incompatible with [`std::str::FromStr`]'s fallible contract, so this
    /// is an inherent method (the `should_implement_trait` lint is suppressed).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "fastembed" => EmbeddingBackend::Fastembed,
            "ollama" => EmbeddingBackend::Ollama,
            // Empty / NULL-as-"" / unknown → the global default backend.
            _ => EmbeddingBackend::default(),
        }
    }

    /// Parses an optional stored backend token (a NULL column on a pre-migration
    /// row, or an absent config value). `None` / empty / unknown → the [`Default`]
    /// (`Fastembed`).
    pub fn from_opt_str(s: Option<&str>) -> Self {
        EmbeddingBackend::from_str(s.unwrap_or(""))
    }
}

/// Static description of one supported embedding model.
///
/// Holds everything a read/write path needs to construct the embedder and
/// validate its output: the stable [`id`](Self::id), the output
/// [`dim`](Self::dim), the concrete [`fastembed_variant`](Self::fastembed_variant),
/// the supported [`backends`](Self::backends), and the caller-applied
/// [`prefix_doc`](Self::prefix_doc) / [`prefix_query`](Self::prefix_query). An
/// empty prefix means "apply none".
pub struct EmbeddingModelSpec {
    /// Stable, storage-facing model id (the value persisted on a notebook).
    pub id: &'static str,
    /// Output vector dimension.
    pub dim: usize,
    /// Concrete `fastembed` variant to construct — `None` for a model that has NO
    /// fastembed ONNX variant (an Ollama-only model). Making this optional turns
    /// the fastembed/Ollama partition into a compile-enforced guard: every
    /// consumer that reaches for a fastembed variant MUST handle the `None` case.
    pub fastembed_variant: Option<EmbeddingModel>,
    /// The embedding backend(s) that can serve this model (M4 issue #80). A model
    /// is strictly partitioned: the original four are `[Fastembed]`, the curated
    /// Ollama catalog is `[Ollama]`. Strong-typed on the [`EmbeddingBackend`] enum
    /// ([[strong-typing-no-stringly-domain]]) — never a magic string.
    pub backends: &'static [EmbeddingBackend],
    /// Prefix prepended to each document at ingest time (`""` = none).
    pub prefix_doc: &'static str,
    /// Prefix prepended to each query at retrieval time (`""` = none).
    pub prefix_query: &'static str,
    /// The HuggingFace repo id (`{org}/{model}`) fastembed downloads this model's
    /// weights from — the SAME value as fastembed's internal `model_code` for
    /// [`fastembed_variant`](Self::fastembed_variant). Used by the on-disk cache
    /// check ([`crate::system_check::fastembed_weights_cached`]) to derive the
    /// per-model hf-hub subdirectory `models--{org}--{model}` under
    /// `{data_dir}/models/fastembed/`. Recorded here (not read from fastembed's
    /// private tables) so the registry stays the single source of truth. Empty
    /// (`""`) for an Ollama-only model that fastembed never downloads.
    pub hf_repo: &'static str,
}

impl EmbeddingModelSpec {
    /// A descriptive `doc/query` record of this model's prefix convention for the
    /// `embedding_index.prefix_convention` metadata column (`"none"` for an empty
    /// prefix). Metadata only — the prefixes actually APPLIED at embed time come
    /// from [`prefix_doc`](Self::prefix_doc)/[`prefix_query`](Self::prefix_query)
    /// directly, so this string is descriptive, not load-bearing.
    pub fn prefix_convention(&self) -> String {
        let label = |p: &str| if p.is_empty() { "none" } else { p.trim() }.to_string();
        format!("{}/{}", label(self.prefix_doc), label(self.prefix_query))
    }

    /// Whether this model can be served by backend `b` — the single check every
    /// spec↔backend meeting point uses (the construction-time and pre-dispatch
    /// guards in `embedder_for`, `FastembedEmbedder::new_with_spec`, and
    /// `OllamaEmbedder::new`). Strong-typed on the [`EmbeddingBackend`] enum.
    pub fn supports(&self, b: EmbeddingBackend) -> bool {
        self.backends.contains(&b)
    }

    /// The per-model hf-hub cache subdirectory name fastembed writes under
    /// `{data_dir}/models/fastembed/` — the OBSERVED shape `models--{org}--{model}`
    /// (every `/` in [`hf_repo`](Self::hf_repo) becomes `--`).
    ///
    /// OBSERVED EMPIRICALLY (M4 Phase 4b-B Step 5, R6 protocol): constructing a
    /// real `FastembedEmbedder::new_with_spec` for `all-minilm` into a temp
    /// data_dir produced `{data_dir}/models/fastembed/models--Qdrant--all-MiniLM-L6-v2-onnx/`
    /// (the standard `hf-hub` repo cache layout, with `snapshots/`, `blobs/`,
    /// `refs/` underneath). `all-minilm`'s [`hf_repo`](Self::hf_repo) is
    /// `Qdrant/all-MiniLM-L6-v2-onnx`, so the subdir is exactly
    /// `models--Qdrant--all-MiniLM-L6-v2-onnx` — confirming the
    /// `models--{repo-with-slashes-as-dashes}` rule used here for every model.
    ///
    /// Returns `None` when [`hf_repo`](Self::hf_repo) is empty — an Ollama-only
    /// model has no fastembed cache directory at all (formatting `models--` from an
    /// empty repo would produce the nonsense path `models--`). Callers that probe
    /// the on-disk fastembed cache therefore skip these models entirely.
    pub fn fastembed_cache_subdir(&self) -> Option<String> {
        if self.hf_repo.is_empty() {
            return None;
        }
        Some(format!("models--{}", self.hf_repo.replace('/', "--")))
    }
}

/// The complete set of supported embedding models. The first entry is the
/// default and is what unknown / empty ids resolve to.
///
/// SYNC-CHECK: keep in sync with `src/lib/embeddings/models.ts`
/// `EMBEDDING_MODELS` (re-exported by `src/lib/onboarding/system-check.ts`). The
/// **dims** AND the **backends** must match the TS `EmbeddingModelSpec` array.
/// Issue #80 splits the catalog by backend: the first four are fastembed-only
/// (`backends: [Fastembed]`), the last four are the curated Ollama-only catalog
/// (`backends: [Ollama]`, `fastembed_variant: None`, `hf_repo: ""`).
/// The **ids** intentionally differ for nomic: the TS/Ollama-facing id is the
/// alias `"nomic-embed-text"`, which [`LEGACY_DEFAULT_ALIAS`] bridges to the
/// canonical `"nomic-embed-text-v1.5"` here (the value persisted on a notebook).
/// `ALLOWED_EMBEDDING_MODELS` in `system_check.rs` is now DERIVED from this
/// `REGISTRY` at init (no longer a hand-maintained parallel list), so adding a
/// model here automatically extends the system-check allowlist.
pub static REGISTRY: &[EmbeddingModelSpec] = &[
    // ── Fastembed catalog (on-device ONNX) ────────────────────────────────────
    EmbeddingModelSpec {
        id: "nomic-embed-text-v1.5",
        dim: 768,
        fastembed_variant: Some(EmbeddingModel::NomicEmbedTextV15),
        backends: &[EmbeddingBackend::Fastembed],
        prefix_doc: "search_document: ",
        prefix_query: "search_query: ",
        hf_repo: "nomic-ai/nomic-embed-text-v1.5",
    },
    EmbeddingModelSpec {
        id: "mxbai-embed-large",
        dim: 1024,
        fastembed_variant: Some(EmbeddingModel::MxbaiEmbedLargeV1),
        backends: &[EmbeddingBackend::Fastembed],
        prefix_doc: "",
        prefix_query: "Represent this sentence for searching relevant passages: ",
        hf_repo: "mixedbread-ai/mxbai-embed-large-v1",
    },
    EmbeddingModelSpec {
        id: "all-minilm",
        dim: 384,
        fastembed_variant: Some(EmbeddingModel::AllMiniLML6V2),
        backends: &[EmbeddingBackend::Fastembed],
        prefix_doc: "",
        prefix_query: "",
        hf_repo: "Qdrant/all-MiniLM-L6-v2-onnx",
    },
    EmbeddingModelSpec {
        id: "bge-m3",
        dim: 1024,
        fastembed_variant: Some(EmbeddingModel::BGEM3),
        backends: &[EmbeddingBackend::Fastembed],
        prefix_doc: "",
        prefix_query: "",
        hf_repo: "BAAI/bge-m3",
    },
    // ── Ollama catalog (curated powerful models; issue #80) ────────────────────
    // Dims/prefixes pinned from ollama.com + HF/Google model cards (D4). Each dim
    // divides by 16 so IVF_PQ `num_sub_vectors = dim/16` is safe. `hf_repo: ""`
    // (fastembed never downloads these) and `fastembed_variant: None` make the
    // strict `[Ollama]`-only partition a compile-enforced fact.
    EmbeddingModelSpec {
        id: "embeddinggemma",
        dim: 768,
        fastembed_variant: None,
        backends: &[EmbeddingBackend::Ollama],
        prefix_doc: "title: none | text: ",
        prefix_query: "task: search result | query: ",
        hf_repo: "",
    },
    EmbeddingModelSpec {
        id: "qwen3-embedding:4b",
        dim: 2560,
        fastembed_variant: None,
        backends: &[EmbeddingBackend::Ollama],
        prefix_doc: "",
        prefix_query: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        hf_repo: "",
    },
    EmbeddingModelSpec {
        id: "nomic-embed-text-v2-moe",
        dim: 768,
        fastembed_variant: None,
        backends: &[EmbeddingBackend::Ollama],
        prefix_doc: "search_document: ",
        prefix_query: "search_query: ",
        hf_repo: "",
    },
    EmbeddingModelSpec {
        id: "snowflake-arctic-embed2",
        dim: 1024,
        fastembed_variant: None,
        backends: &[EmbeddingBackend::Ollama],
        prefix_doc: "",
        prefix_query: "query: ",
        hf_repo: "",
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
            s.fastembed_variant.as_ref().unwrap(),
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
            s.fastembed_variant.as_ref().unwrap(),
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
            s.fastembed_variant.as_ref().unwrap(),
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
        assert!(variant_eq(
            s.fastembed_variant.as_ref().unwrap(),
            &EmbeddingModel::BGEM3
        ));
        assert_eq!(s.prefix_doc, "");
        assert_eq!(s.prefix_query, "");
    }

    /// Step 1 (issue #80): every original spec is a fastembed-only model — it
    /// supports Fastembed, NOT Ollama, and carries a concrete fastembed variant.
    #[test]
    fn existing_specs_are_fastembed_only() {
        for id in [
            "nomic-embed-text-v1.5",
            "mxbai-embed-large",
            "all-minilm",
            "bge-m3",
        ] {
            let s = resolve(id);
            assert!(
                s.supports(EmbeddingBackend::Fastembed),
                "{id} supports fastembed"
            );
            assert!(!s.supports(EmbeddingBackend::Ollama), "{id} is NOT ollama");
            assert!(
                s.fastembed_variant.is_some(),
                "{id} has a fastembed variant"
            );
            assert!(
                s.fastembed_cache_subdir().is_some(),
                "{id} has a fastembed cache subdir (non-empty hf_repo)"
            );
        }
    }

    /// Step 2 (issue #80): each curated Ollama spec resolves with the verified
    /// dim/prefixes, supports ONLY Ollama, and has no fastembed variant / cache dir.
    #[test]
    fn ollama_specs_resolve_with_verified_data() {
        let gemma = resolve_opt("embeddinggemma").expect("embeddinggemma registered");
        assert_eq!(gemma.dim, 768);
        assert_eq!(gemma.prefix_doc, "title: none | text: ");
        assert_eq!(gemma.prefix_query, "task: search result | query: ");

        let qwen = resolve_opt("qwen3-embedding:4b").expect("qwen3-embedding:4b registered");
        assert_eq!(qwen.dim, 2560);
        assert_eq!(qwen.prefix_doc, "");
        assert!(
            qwen.prefix_query.starts_with("Instruct:"),
            "qwen query prefix is the instruction wrapper"
        );
        assert!(
            qwen.prefix_query.contains('\n'),
            "qwen query prefix contains a real newline before `Query: `"
        );

        let nomic2 =
            resolve_opt("nomic-embed-text-v2-moe").expect("nomic-embed-text-v2-moe registered");
        assert_eq!(nomic2.dim, 768);
        assert_eq!(nomic2.prefix_doc, "search_document: ");
        assert_eq!(nomic2.prefix_query, "search_query: ");

        let arctic =
            resolve_opt("snowflake-arctic-embed2").expect("snowflake-arctic-embed2 registered");
        assert_eq!(arctic.dim, 1024);
        assert_eq!(arctic.prefix_doc, "");
        assert_eq!(arctic.prefix_query, "query: ");

        for s in [gemma, qwen, nomic2, arctic] {
            assert!(
                s.supports(EmbeddingBackend::Ollama),
                "{} supports ollama",
                s.id
            );
            assert!(
                !s.supports(EmbeddingBackend::Fastembed),
                "{} not fastembed",
                s.id
            );
            assert!(
                s.fastembed_variant.is_none(),
                "{} has no fastembed variant",
                s.id
            );
            assert!(
                s.fastembed_cache_subdir().is_none(),
                "{} has no fastembed cache subdir (empty hf_repo)",
                s.id
            );
            assert_eq!(s.dim % 16, 0, "{} dim divides by 16 for IVF_PQ", s.id);
        }
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
            s.fastembed_variant.as_ref().unwrap(),
            &EmbeddingModel::NomicEmbedTextV15
        ));
    }

    #[test]
    fn default_consts_are_correct() {
        assert_eq!(DEFAULT_EMBED_MODEL_ID, "nomic-embed-text-v1.5");
        assert_eq!(DEFAULT_EMBED_DIM, 768);
    }

    #[test]
    fn registry_has_exactly_eight_models() {
        // 4 fastembed + 4 curated Ollama models (issue #80).
        assert_eq!(REGISTRY.len(), 8);
    }

    #[test]
    fn every_registry_dim_divides_by_16() {
        // IVF_PQ `num_sub_vectors = dim / 16` must divide the dim for every model.
        for s in REGISTRY {
            assert_eq!(s.dim % 16, 0, "{} dim {} must divide by 16", s.id, s.dim);
        }
    }

    #[test]
    fn embedding_backend_default_is_fastembed() {
        assert_eq!(EmbeddingBackend::default(), EmbeddingBackend::Fastembed);
    }

    #[test]
    fn embedding_backend_as_str_round_trips() {
        for b in [EmbeddingBackend::Fastembed, EmbeddingBackend::Ollama] {
            assert_eq!(EmbeddingBackend::from_str(b.as_str()), b);
        }
        assert_eq!(EmbeddingBackend::Fastembed.as_str(), "fastembed");
        assert_eq!(EmbeddingBackend::Ollama.as_str(), "ollama");
    }

    #[test]
    fn embedding_backend_from_str_known_and_unknown() {
        assert_eq!(
            EmbeddingBackend::from_str("fastembed"),
            EmbeddingBackend::Fastembed
        );
        assert_eq!(
            EmbeddingBackend::from_str("ollama"),
            EmbeddingBackend::Ollama
        );
        // Empty / unknown → the default backend (mirrors `embedding_model`).
        assert_eq!(EmbeddingBackend::from_str(""), EmbeddingBackend::Fastembed);
        assert_eq!(
            EmbeddingBackend::from_str("does-not-exist"),
            EmbeddingBackend::Fastembed
        );
    }

    #[test]
    fn embedding_backend_from_opt_str_handles_null() {
        // NULL column (pre-migration row) → default.
        assert_eq!(
            EmbeddingBackend::from_opt_str(None),
            EmbeddingBackend::Fastembed
        );
        assert_eq!(
            EmbeddingBackend::from_opt_str(Some("ollama")),
            EmbeddingBackend::Ollama
        );
        assert_eq!(
            EmbeddingBackend::from_opt_str(Some("")),
            EmbeddingBackend::Fastembed
        );
    }
}
