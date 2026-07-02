// SYNC-CHECK: must match lens-core/src/notebooks.rs Source struct — update both together.
//
// TypeScript mirrors of the Rust source structs. serde on the Rust side uses
// verbatim snake_case field names, so this shape must match exactly.

/** Constrained set of source ingestion states — mirrors the Rust
 * `SourceStatus` enum (lens-core/src/notebooks.rs). 'pending' is used by the
 * add_source (file) path before ingest begins; 'needs_ocr'/'needs_js' are the
 * terminal-pending states from the PDF/URL ingest gates; 'render_failed' is
 * the terminal state for a URL whose JS render attempt failed. */
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

/** Constrained set of source kinds — the exact `sources.kind` column values
 * returned across IPC. 'text'|'markdown'|'pdf'|'docx'|'url' mirror the Rust
 * `SourceKind` enum (lens-core/src/parse.rs) — the ingestable kinds. 'file' is
 * the legacy inert M1 placeholder written by the `add_source` path (kind =
 * "file", status = "pending") before M4 ingestion; it is NOT a `SourceKind`
 * variant on the Rust side but is a real persisted value, so it is included
 * here for the rows the backend can return. */
// SYNC-CHECK: must match lens-core/src/parse.rs SourceKind enum
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
  | 'file';

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
  /** Raw-content SHA-256 computed at add time for content dedup (#96/#100);
   * hashes file bytes, text-paste bytes, or the normalized URL. `null` for
   * pre-migration rows. */
  raw_content_hash: string | null;
  trashed_at: string | null;
  /** Enrichment lifecycle (none|pending|enriching|enriched|failed|skipped),
   * SEPARATE from `status`. `null` ≡ `none` for pre-Phase-3 rows. */
  enrichment_status: string | null;
  /** JSON enrichment metadata (composite cache key + budget/skip reason);
   * `null` until the source is enriched. */
  enrichment_meta: string | null;
}

/** Return type of all add-source IPC calls (add_file_source, add_source,
 * add_text_source, add_url_source — issues #96 + #100). Mirrors the Rust
 * `AddSourceOutcome` struct (serde camelCase). `wasExisting = true` means a
 * dedup hit: no new row was written and the existing source is returned. */
export interface AddSourceOutcome {
  source: Source;
  wasExisting: boolean;
}

// SYNC-CHECK: must match lens-core/src/notebooks.rs TrashedSource struct
/** A trashed source enriched with its parent notebook's title.
 * Mirrors the Rust `TrashedSource` struct (`notebooks.rs`) which flattens
 * `Source` via `#[serde(flatten)]` and adds `notebook_title`. */
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
 * One event in an ingest stream. Adjacently tagged: `{type, data}`.
 * Mirrors src-tauri/src/stream.rs StreamEvent<T> with
 * `#[serde(tag="type", content="data", rename_all="snake_case")]`.
 *
 * Unit variants (`started`, `done`) have no `data` key.
 * Data-carrying variants carry `data` with the payload.
 */
export type StreamEvent<T> =
  | { type: 'started' }
  | { type: 'chunk'; data: T }
  | { type: 'progress'; data: { done: number; total: number | null } }
  | { type: 'done' }
  | { type: 'failed'; data: { kind: string; message: string } };
