//! Embedding-model registry: maps a stable model id to dim, fastembed variant,
//! and caller-applied prefixes. fastembed 5.17.2 does NOT inject task prefixes;
//! the registry is the canonical record per model card:
//! nomic requires `"search_document: "`/`"search_query: "`, mxbai prefixes queries
//! only, all-minilm and bge-m3 use no prefixes.

use fastembed::EmbeddingModel;

pub const DEFAULT_EMBED_MODEL_ID: &str = "nomic-embed-text-v1.5";

pub const DEFAULT_EMBED_DIM: usize = 768;

/// Legacy alias for the default model before the `-v1.5` suffix was standardized.
const LEGACY_DEFAULT_ALIAS: &str = "nomic-embed-text";

/// The backend that physically computes a notebook's vectors. The same model
/// served by fastembed vs Ollama produces numerically different embeddings, so
/// the backend is a first-class axis of the embedding coordinate alongside
/// `(model, dim)`. Strong-typed ([[strong-typing-no-stringly-domain]]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingBackend {
    #[default]
    Fastembed,
    /// Local Ollama server (detect-only; loopback-bound).
    Ollama,
}

impl EmbeddingBackend {
    /// Storage token persisted in `embedding_index.backend`.
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbeddingBackend::Fastembed => "fastembed",
            EmbeddingBackend::Ollama => "ollama",
        }
    }

    /// Parses a stored backend token; empty/unknown resolves to the default.
    /// Deliberately INFALLIBLE (NULL is a normal case, not a parse error —
    /// incompatible with `std::str::FromStr`'s fallible contract).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "fastembed" => EmbeddingBackend::Fastembed,
            "ollama" => EmbeddingBackend::Ollama,
            // Empty / NULL-as-"" / unknown → the global default backend.
            _ => EmbeddingBackend::default(),
        }
    }

    /// `None` / empty / unknown → the [`Default`] (`Fastembed`).
    pub fn from_opt_str(s: Option<&str>) -> Self {
        EmbeddingBackend::from_str(s.unwrap_or(""))
    }
}

/// Static description of one supported embedding model.
///
/// Holds everything needed to construct the embedder and validate its output:
/// stable `id`, `dim`, `fastembed_variant`, supported `backends`, and prefixes.
pub struct EmbeddingModelSpec {
    pub id: &'static str,
    pub dim: usize,
    /// `None` for Ollama-only models (no fastembed ONNX variant).
    pub fastembed_variant: Option<EmbeddingModel>,
    pub backends: &'static [EmbeddingBackend],
    pub prefix_doc: &'static str,
    pub prefix_query: &'static str,
    /// HF repo id used by the on-disk cache check (`models--{org}--{model}`).
    /// Empty for Ollama-only models.
    pub hf_repo: &'static str,
    /// Whether this model benefits from GPU bulk embedding on Apple Silicon (issue
    /// #91). MEASURED 2026-07-03: all-minilm is a GPU loser (dispatch overhead >
    /// compute); nomic/mxbai/bge-m3 are eligible. `true` = "would benefit"; whether
    /// a GPU engine is actually wired at run time is resolved by the device layer.
    pub accelerate_hint: bool,
}

impl EmbeddingModelSpec {
    /// Descriptive prefix convention for the `embedding_index.prefix_convention`
    /// metadata column. Not load-bearing — actual prefixes come from the fields.
    pub fn prefix_convention(&self) -> String {
        let label = |p: &str| if p.is_empty() { "none" } else { p.trim() }.to_string();
        format!("{}/{}", label(self.prefix_doc), label(self.prefix_query))
    }

    /// Whether this model can be served by backend `b`.
    pub fn supports(&self, b: EmbeddingBackend) -> bool {
        self.backends.contains(&b)
    }

    /// hf-hub cache subdir under `{data_dir}/models/fastembed/`. Shape is
    /// `models--{org}--{model}` (every `/` in `hf_repo` becomes `--`), verified
    /// empirically (R6 probe). Returns `None` for Ollama-only models (empty `hf_repo`).
    pub fn fastembed_cache_subdir(&self) -> Option<String> {
        if self.hf_repo.is_empty() {
            return None;
        }
        Some(format!("models--{}", self.hf_repo.replace('/', "--")))
    }
}

/// Complete set of supported embedding models. First entry is the default.
///
/// SYNC-CHECK: keep `id`/`dim`/`backends` in sync with `src/lib/embeddings/models.ts`.
/// First four are fastembed-only; last four are the curated Ollama-only catalog.
/// `ALLOWED_EMBEDDING_MODELS` in `system_check.rs` is derived from this at init.
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
        // 137M/768d — GPU-eligible; candle-Metal is wired + parity-proven for it.
        accelerate_hint: true,
    },
    EmbeddingModelSpec {
        id: "mxbai-embed-large",
        dim: 1024,
        fastembed_variant: Some(EmbeddingModel::MxbaiEmbedLargeV1),
        backends: &[EmbeddingBackend::Fastembed],
        prefix_doc: "",
        prefix_query: "Represent this sentence for searching relevant passages: ",
        hf_repo: "mixedbread-ai/mxbai-embed-large-v1",
        // 335M/1024d — GPU-eligible; candle backend wiring is follow-up (falls back
        // to CPU until then).
        accelerate_hint: true,
    },
    EmbeddingModelSpec {
        id: "all-minilm",
        dim: 384,
        fastembed_variant: Some(EmbeddingModel::AllMiniLML6V2),
        backends: &[EmbeddingBackend::Fastembed],
        prefix_doc: "",
        prefix_query: "",
        hf_repo: "Qdrant/all-MiniLM-L6-v2-onnx",
        // 22M/384d — measured GPU LOSER (dispatch overhead > compute). CPU only.
        accelerate_hint: false,
    },
    EmbeddingModelSpec {
        id: "bge-m3",
        dim: 1024,
        fastembed_variant: Some(EmbeddingModel::BGEM3),
        backends: &[EmbeddingBackend::Fastembed],
        prefix_doc: "",
        prefix_query: "",
        hf_repo: "BAAI/bge-m3",
        // 567M/1024d — GPU-eligible; candle backend wiring is follow-up.
        accelerate_hint: true,
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
        // Ollama-only; device policy's backend gate (gate 1) forces CPU — false.
        accelerate_hint: false,
    },
    EmbeddingModelSpec {
        id: "qwen3-embedding:4b",
        dim: 2560,
        fastembed_variant: None,
        backends: &[EmbeddingBackend::Ollama],
        prefix_doc: "",
        prefix_query: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        hf_repo: "",
        // Ollama-only; device policy's backend gate (gate 1) forces CPU — false.
        accelerate_hint: false,
    },
    EmbeddingModelSpec {
        id: "nomic-embed-text-v2-moe",
        dim: 768,
        fastembed_variant: None,
        backends: &[EmbeddingBackend::Ollama],
        prefix_doc: "search_document: ",
        prefix_query: "search_query: ",
        hf_repo: "",
        // Ollama-only; device policy's backend gate (gate 1) forces CPU — false.
        accelerate_hint: false,
    },
    EmbeddingModelSpec {
        id: "snowflake-arctic-embed2",
        dim: 1024,
        fastembed_variant: None,
        backends: &[EmbeddingBackend::Ollama],
        prefix_doc: "",
        prefix_query: "query: ",
        hf_repo: "",
        // Ollama-only; device policy's backend gate (gate 1) forces CPU — false.
        accelerate_hint: false,
    },
];

/// Resolves a model id to its [`EmbeddingModelSpec`], falling back to the
/// default on unknown/empty ids. The legacy alias `"nomic-embed-text"` resolves
/// to the nomic entry.
pub fn resolve(id: &str) -> &'static EmbeddingModelSpec {
    resolve_opt(id).unwrap_or_else(default_spec)
}

/// Resolves a model id to its [`EmbeddingModelSpec`], returning `None` for an
/// unknown id. Use instead of [`resolve`] when an unknown id must be rejected.
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
        // EmbeddingModel lacks PartialEq in fastembed 5.17.2; compare via Debug.
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
        assert_eq!(REGISTRY.len(), 8);
    }

    #[test]
    fn every_registry_dim_divides_by_16() {
        for s in REGISTRY {
            assert_eq!(s.dim % 16, 0, "{} dim {} must divide by 16", s.id, s.dim);
        }
    }

    #[test]
    fn accelerate_hint_matches_measured_verdict() {
        assert!(resolve("nomic-embed-text-v1.5").accelerate_hint);
        assert!(resolve("mxbai-embed-large").accelerate_hint);
        assert!(resolve("bge-m3").accelerate_hint);
        assert!(!resolve("all-minilm").accelerate_hint);
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
        assert_eq!(EmbeddingBackend::from_str(""), EmbeddingBackend::Fastembed);
        assert_eq!(
            EmbeddingBackend::from_str("does-not-exist"),
            EmbeddingBackend::Fastembed
        );
    }

    #[test]
    fn embedding_backend_from_opt_str_handles_null() {
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
