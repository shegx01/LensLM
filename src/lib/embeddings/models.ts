// Single source of truth for the embedding-model catalog the Embeddings UI
// renders (onboarding + global Settings + per-notebook settings).
//
// SYNC-CHECK: keep in sync with lens-core/src/embedder/registry.rs REGISTRY
// (model specs — `id`, `dim`, `backends`, `hf_repo`/`ollamaName`) and
// ALLOWED_EMBEDDING_MODELS in lens-core/src/system_check.rs (the Ollama
// install allowlist). Adding or removing a model means editing both Rust files.
//
// The `id` here is the CANONICAL, storage-facing model id persisted on a
// notebook + in AppConfig.embedding_model and used by the fastembed on-disk
// cache check (`fastembed_models_cached`). Ollama reports a DIFFERENT name for
// nomic (the `-v1.5`-less alias), bridged by `ollamaName`.
//
// For Ollama-only models, `ollamaName` EQUALS `id` (including the `:tag` for
// qwen3-embedding:4b). The `backends` field partitions the catalog:
// fastembed-only models have `['fastembed']`, Ollama-only have `['ollama']`.

/** The two local embedding backends. Mirrors lens-core `EmbeddingBackend`. */
export type EmbeddingBackend = 'fastembed' | 'ollama';

export type EmbeddingModelId =
  | 'nomic-embed-text-v1.5'
  | 'mxbai-embed-large'
  | 'all-minilm'
  | 'bge-m3'
  | 'embeddinggemma'
  | 'qwen3-embedding:4b'
  | 'nomic-embed-text-v2-moe'
  | 'snowflake-arctic-embed2';

export interface EmbeddingModelSpec {
  /** Canonical, storage-facing id (persisted; matches the registry `id`). */
  id: EmbeddingModelId;
  /** Display label (the bold model id shown on the card). */
  label: string;
  /** Output vector dimension. */
  dims: number;
  /** Weight size on disk, in megabytes (rendered as MB or GB). */
  sizeMb: number;
  /** Speed tier copy. */
  speed: 'Very fast' | 'Fast' | 'Medium';
  /** One-line description (the design's per-model meta tail). */
  desc: string;
  /**
   * The model name Ollama reports via `/api/tags`.
   * For fastembed models, this is the Ollama alias (e.g. `nomic-embed-text`
   * without the `-v1.5` suffix). For Ollama-only models, this EQUALS `id`
   * (including the colon+tag for `qwen3-embedding:4b`).
   */
  ollamaName: string;
  /**
   * Which backends support this model. Mirrors the `backends` field in
   * lens-core's `EmbeddingModelSpec`. Use to filter the picker per provider.
   */
  backends: EmbeddingBackend[];
}

// Copy + dims verbatim from the plan's "Design facts" (build-ui-to-spec).
export const EMBEDDING_MODELS: EmbeddingModelSpec[] = [
  // ── fastembed-backed models ───────────────────────────────────────────────
  {
    id: 'nomic-embed-text-v1.5',
    label: 'nomic-embed-text-v1.5',
    dims: 768,
    sizeMb: 274,
    speed: 'Fast',
    desc: 'Best all-round local model',
    ollamaName: 'nomic-embed-text',
    backends: ['fastembed']
  },
  {
    id: 'mxbai-embed-large',
    label: 'mxbai-embed-large',
    dims: 1024,
    sizeMb: 670,
    speed: 'Medium',
    desc: 'Highest retrieval accuracy',
    ollamaName: 'mxbai-embed-large',
    backends: ['fastembed']
  },
  {
    id: 'all-minilm',
    label: 'all-minilm',
    dims: 384,
    sizeMb: 45,
    speed: 'Very fast',
    desc: 'Lightweight, minimal RAM',
    ollamaName: 'all-minilm',
    backends: ['fastembed']
  },
  {
    id: 'bge-m3',
    label: 'bge-m3',
    dims: 1024,
    sizeMb: 1200,
    speed: 'Medium',
    desc: 'Multilingual',
    ollamaName: 'bge-m3',
    backends: ['fastembed']
  },

  // ── Ollama-only models (curated powerful catalog — Issue #80) ──────────────
  // ollamaName === id for all Ollama-only entries (including colon+tag).
  {
    id: 'embeddinggemma',
    label: 'embeddinggemma',
    dims: 768,
    sizeMb: 622,
    speed: 'Fast',
    desc: "Google's Gemma embedding model",
    ollamaName: 'embeddinggemma',
    backends: ['ollama']
  },
  {
    id: 'qwen3-embedding:4b',
    label: 'qwen3-embedding:4b',
    dims: 2560,
    sizeMb: 2500,
    speed: 'Medium',
    desc: 'Qwen3 4B — high-dim multilingual retrieval',
    ollamaName: 'qwen3-embedding:4b',
    backends: ['ollama']
  },
  {
    id: 'nomic-embed-text-v2-moe',
    label: 'nomic-embed-text-v2-moe',
    dims: 768,
    sizeMb: 958,
    speed: 'Fast',
    desc: 'Nomic v2 MoE — strong multilingual embeddings',
    ollamaName: 'nomic-embed-text-v2-moe',
    backends: ['ollama']
  },
  {
    id: 'snowflake-arctic-embed2',
    label: 'snowflake-arctic-embed2',
    dims: 1024,
    sizeMb: 1200,
    speed: 'Fast',
    desc: 'Snowflake Arctic Embed 2 — enterprise retrieval',
    ollamaName: 'snowflake-arctic-embed2',
    backends: ['ollama']
  }
];

export const DEFAULT_EMBEDDING_MODEL: EmbeddingModelId = 'nomic-embed-text-v1.5';
export const DEFAULT_EMBEDDING_BACKEND: EmbeddingBackend = 'fastembed';

/**
 * The re-embed warning copy (verbatim from the design's onboarding embed step +
 * the per-notebook switch confirm). Single source of truth so the onboarding
 * panel and the notebook-mode confirm dialog can never drift.
 */
export const REEMBED_WARNING =
  "Embedding models are permanently linked to this notebook's vector index. " +
  'Switching models later requires re-embedding all sources from scratch. Choose carefully.';

/** Resolve a (possibly empty / legacy-alias) id to a known spec, defaulting. */
export function resolveModel(id: string): EmbeddingModelSpec {
  if (id === 'nomic-embed-text') return EMBEDDING_MODELS[0];
  return EMBEDDING_MODELS.find((m) => m.id === id) ?? EMBEDDING_MODELS[0];
}

/** Resolve an (empty / unknown) backend token to a known backend, defaulting. */
export function resolveBackend(token: string): EmbeddingBackend {
  return token === 'ollama' ? 'ollama' : 'fastembed';
}

/** Format a model's weight size: `45 MB`, `670 MB`, `1.2 GB`. */
export function formatSize(sizeMb: number): string {
  return sizeMb >= 1000 ? `${(sizeMb / 1000).toFixed(1)} GB` : `${sizeMb} MB`;
}

/** The design's per-model meta line: `768 dims · 274 MB · Fast · Best …`. */
export function modelMeta(m: EmbeddingModelSpec): string {
  return `${m.dims} dims · ${formatSize(m.sizeMb)} · ${m.speed} · ${m.desc}`;
}

/**
 * Whether an Ollama-reported model name (e.g. `"nomic-embed-text:latest"`)
 * matches a spec.
 *
 * D3 exact-tag rule: if `m.ollamaName` contains a colon (e.g. qwen3-embedding:4b),
 * require an EXACT match — `detected === m.ollamaName`. This prevents
 * `qwen3-embedding:0.6b` or `qwen3-embedding:8b` from spuriously matching.
 *
 * For bare names (no colon in ollamaName), keep the existing prefix rule:
 * `detected === m.ollamaName || detected.startsWith('<name>:')` — so
 * `nomic-embed-text:latest` still matches `nomic-embed-text`.
 */
export function ollamaMatches(detected: string, m: EmbeddingModelSpec): boolean {
  if (m.ollamaName.includes(':')) {
    // Exact tag match only — the colon in the name IS part of the tag spec.
    return detected === m.ollamaName;
  }
  return detected === m.ollamaName || detected.startsWith(`${m.ollamaName}:`);
}
