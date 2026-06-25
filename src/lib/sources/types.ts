// SYNC-CHECK: must match lens-core/src/notebooks.rs Source struct — update both together.
//
// TypeScript mirrors of the Rust source structs. serde on the Rust side uses
// verbatim snake_case field names, so this shape must match exactly.

/** Constrained set of source ingestion states — mirrors the Rust
 * `SourceStatus` enum (lens-core/src/notebooks.rs). 'pending' is used by the
 * add_source (file) path before ingest begins; 'needs_ocr'/'needs_js' are the
 * terminal-pending states from the PDF/URL ingest gates. */
export type SourceStatus =
  | 'pending'
  | 'queued'
  | 'parsing'
  | 'embedding'
  | 'indexed'
  | 'error'
  | 'needs_ocr'
  | 'needs_js';

/** Constrained set of source kinds — the exact `sources.kind` column values
 * returned across IPC. 'text'|'markdown'|'pdf'|'docx'|'url' mirror the Rust
 * `SourceKind` enum (lens-core/src/parse.rs) — the ingestable kinds. 'file' is
 * the legacy inert M1 placeholder written by the `add_source` path (kind =
 * "file", status = "pending") before M4 ingestion; it is NOT a `SourceKind`
 * variant on the Rust side but is a real persisted value, so it is included
 * here for the rows the backend can return. */
export type SourceKind = 'text' | 'markdown' | 'pdf' | 'docx' | 'url' | 'file';

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
  trashed_at: string | null;
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
