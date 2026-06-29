// Single source of truth for the embedding-model catalog the Embeddings UI
// renders (onboarding + global Settings + per-notebook settings).
//
// SYNC-CHECK: keep in sync with lens-core/src/embedder/registry.rs REGISTRY
// (fastembed model specs — `id`, `dim`, `hf_repo`) and ALLOWED_EMBEDDING_MODELS
// in lens-core/src/system_check.rs (the Ollama install allowlist). Adding or
// removing a model means editing both Rust files too.
//
// The `id` here is the CANONICAL, storage-facing model id persisted on a
// notebook + in AppConfig.embedding_model and used by the fastembed on-disk
// cache check (`fastembed_models_cached`). Ollama reports a DIFFERENT name for
// nomic (the `-v1.5`-less alias), bridged by `ollamaName`.

/** The two local embedding backends. Mirrors lens-core `EmbeddingBackend`. */
export type EmbeddingBackend = 'fastembed' | 'ollama';

export type EmbeddingModelId =
  | 'nomic-embed-text-v1.5'
  | 'mxbai-embed-large'
  | 'all-minilm'
  | 'bge-m3';

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
   * The model name Ollama reports via `/api/tags` (without a `:tag` suffix).
   * Differs from `id` only for nomic (the runtime-facing alias).
   */
  ollamaName: string;
}

// Copy + dims verbatim from the plan's "Design facts" (build-ui-to-spec).
export const EMBEDDING_MODELS: EmbeddingModelSpec[] = [
  {
    id: 'nomic-embed-text-v1.5',
    label: 'nomic-embed-text-v1.5',
    dims: 768,
    sizeMb: 274,
    speed: 'Fast',
    desc: 'Best all-round local model',
    ollamaName: 'nomic-embed-text'
  },
  {
    id: 'mxbai-embed-large',
    label: 'mxbai-embed-large',
    dims: 1024,
    sizeMb: 670,
    speed: 'Medium',
    desc: 'Highest retrieval accuracy',
    ollamaName: 'mxbai-embed-large'
  },
  {
    id: 'all-minilm',
    label: 'all-minilm',
    dims: 384,
    sizeMb: 45,
    speed: 'Very fast',
    desc: 'Lightweight, minimal RAM',
    ollamaName: 'all-minilm'
  },
  {
    id: 'bge-m3',
    label: 'bge-m3',
    dims: 1024,
    sizeMb: 1200,
    speed: 'Medium',
    desc: 'Multilingual',
    ollamaName: 'bge-m3'
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
 * matches a spec — equals the bare `ollamaName` or starts with `"<name>:"`.
 */
export function ollamaMatches(detected: string, m: EmbeddingModelSpec): boolean {
  return detected === m.ollamaName || detected.startsWith(`${m.ollamaName}:`);
}
