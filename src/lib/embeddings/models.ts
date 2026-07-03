// SYNC-CHECK: keep in sync with lens-core/src/embedder/registry.rs REGISTRY
// and ALLOWED_EMBEDDING_MODELS in lens-core/src/system_check.rs.
//
// `id` is the canonical storage-facing model id. `ollamaName` bridges the alias
// difference for nomic (Ollama drops the `-v1.5` suffix). For Ollama-only models
// `ollamaName === id` (including the `:tag` for qwen3-embedding:4b).

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
  label: string;
  /** Output vector dimension. */
  dims: number;
  /** Weight size on disk in MB. */
  sizeMb: number;
  speed: 'Very fast' | 'Fast' | 'Medium';
  desc: string;
  /**
   * Model name Ollama reports via `/api/tags`. Equals `id` for Ollama-only
   * models; differs for fastembed models (e.g. `nomic-embed-text` vs. `nomic-embed-text-v1.5`).
   */
  ollamaName: string;
  /** Which backends support this model. Mirrors lens-core `EmbeddingModelSpec.backends`. */
  backends: EmbeddingBackend[];
}

export const EMBEDDING_MODELS: EmbeddingModelSpec[] = [
  // fastembed-backed models
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

  // Ollama-only models (ollamaName === id for all entries)
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

/** Single source of truth for the re-embed warning — onboarding + per-notebook confirm. */
export const REEMBED_WARNING =
  "Embedding models are permanently linked to this notebook's vector index. " +
  'Switching models later requires re-embedding all sources from scratch. Choose carefully.';

/** Resolve a (possibly empty or legacy-alias) id to a known spec, defaulting to the first. */
export function resolveModel(id: string): EmbeddingModelSpec {
  if (id === 'nomic-embed-text') return EMBEDDING_MODELS[0];
  return EMBEDDING_MODELS.find((m) => m.id === id) ?? EMBEDDING_MODELS[0];
}

/** Resolve an (empty / unknown) backend token to a known backend, defaulting to fastembed. */
export function resolveBackend(token: string): EmbeddingBackend {
  return token === 'ollama' ? 'ollama' : 'fastembed';
}

/** Format a model's weight size: `45 MB`, `670 MB`, `1.2 GB`. */
export function formatSize(sizeMb: number): string {
  return sizeMb >= 1000 ? `${(sizeMb / 1000).toFixed(1)} GB` : `${sizeMb} MB`;
}

/** Per-model meta line: `768 dims · 274 MB · Fast · Best …`. */
export function modelMeta(m: EmbeddingModelSpec): string {
  return `${m.dims} dims · ${formatSize(m.sizeMb)} · ${m.speed} · ${m.desc}`;
}

/**
 * Whether an Ollama-reported model name matches a spec.
 *
 * If `ollamaName` contains a colon (e.g. `qwen3-embedding:4b`), require an
 * EXACT match to prevent `qwen3-embedding:0.6b` from spuriously matching.
 * For bare names, prefix match: `nomic-embed-text:latest` matches `nomic-embed-text`.
 */
export function ollamaMatches(detected: string, m: EmbeddingModelSpec): boolean {
  if (m.ollamaName.includes(':')) {
    return detected === m.ollamaName;
  }
  return detected === m.ollamaName || detected.startsWith(`${m.ollamaName}:`);
}
