// SYNC-CHECK: must match lens-core/src/notebooks.rs Source struct — update both together.

/** Source ingestion states. `needs_ocr`/`needs_js` are terminal-pending; `render_failed` is terminal for JS-render failures. */
export type SourceStatus =
  | 'pending'
  | 'queued'
  | 'parsing'
  | 'embedding'
  | 'indexed'
  | 'error'
  | 'needs_ocr'
  | 'needs_js'
  | 'render_failed';

// SYNC-CHECK: must match lens-core/src/parse.rs SourceKind enum
// `'file'` is the M1 legacy placeholder kind (SourceKind::File).
export type SourceKind =
  | 'text'
  | 'markdown'
  | 'pdf'
  | 'docx'
  | 'url'
  | 'json'
  | 'jsonl'
  | 'yaml'
  | 'xml'
  | 'rtf'
  | 'odt'
  | 'epub'
  | 'xlsx'
  | 'xls'
  | 'csv'
  | 'audio'
  | 'file';

/**
 * Known error kinds mirroring the Rust `LensError` enum variants.
 * Treat as an open set — unknown kinds are valid (forward-compat).
 */
export type LensErrorKind =
  | 'Validation'
  | 'Internal'
  | 'Io'
  | 'Parse'
  | 'Model'
  | 'Network'
  | 'Vector'
  | (string & Record<never, never>); // allow unknown future variants

/**
 * Structured ingest failure reason persisted in the `error_meta` DB column.
 * Mirrors the Rust `ErrorMeta` struct (serde JSON). `null` means the source
 * reached the `error` status before migration 0012 (crash-recovery rows).
 */
export interface ErrorMeta {
  kind: LensErrorKind;
  message: string;
  timestamp: string;
  attempt_count: number;
}

// SYNC-CHECK: must match lens-core/src/notebooks.rs Source struct (around line 82)
export interface Source {
  id: string;
  notebook_id: string;
  kind: SourceKind;
  title: string;
  status: SourceStatus;
  locator: string;
  selected: number;
  created_at: string;
  token_count: number | null;
  content_hash: string | null;
  /** SHA-256 of raw content (file bytes, text paste, or normalized URL) for dedup. `null` on pre-migration rows. */
  raw_content_hash: string | null;
  trashed_at: string | null;
  /** Enrichment lifecycle, SEPARATE from `status`. `null` ≡ `none` for pre-Phase-3 rows. */
  enrichment_status: string | null;
  /** JSON enrichment metadata; `null` until enriched. */
  enrichment_meta: string | null;
  /** SYNC-CHECK: must match lens-core/src/notebooks.rs `Source.force_js_render`.
   * SQLite integer boolean (`0`/`1`) — when set, always routes URL ingest through JS-render. */
  force_js_render: number;
  /** Parsed ingest failure reason. `null` for non-errored or pre-migration rows.
   * Gate error UI on `status === 'error'`, NOT on this field being non-null. */
  error_meta: ErrorMeta | null;
  /** whatlang ISO 639-3 language code detected at ingest (#194); `null` for
   * pre-migration rows and text where detection was unreliable/undetermined. */
  language?: string | null;
}

/** Return type of add-source IPC calls. `wasExisting = true` means a dedup hit; existing source returned. */
export interface AddSourceOutcome {
  source: Source;
  wasExisting: boolean;
}

// SYNC-CHECK: must match lens-core/src/notebooks.rs TrashedSource struct
/** Flattens `Source` and adds `notebook_title` from the parent notebook. */
export interface TrashedSource extends Source {
  notebook_title: string;
}

// SYNC-CHECK: must match lens-core/src/ingest.rs IngestProgress struct
export interface IngestProgress {
  phase: string;
  done: number;
  total: number | null;
}

/**
 * Adjacently-tagged ingest stream event `{type, data}`.
 * Mirrors `src-tauri/src/stream.rs StreamEvent<T>`. Unit variants have no `data` key.
 */
export type StreamEvent<T> =
  | { type: 'started' }
  | { type: 'chunk'; data: T }
  | { type: 'progress'; data: { done: number; total: number | null } }
  | { type: 'done' }
  | { type: 'failed'; data: { kind: string; message: string } };
