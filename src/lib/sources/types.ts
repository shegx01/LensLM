// SYNC-CHECK: must match lens-core/src/notebooks.rs Source struct — update both together.
//
// TypeScript mirrors of the Rust source structs. serde on the Rust side uses
// verbatim snake_case field names, so this shape must match exactly.

/** Constrained set of source ingestion states — mirrors the Rust status values.
 * 'pending' is used by the add_source (file) path before ingest begins. */
export type SourceStatus = 'pending' | 'queued' | 'parsing' | 'embedding' | 'indexed' | 'error';

// SYNC-CHECK: must match lens-core/src/notebooks.rs Source struct (around line 82)
export interface Source {
  id: string;
  notebook_id: string;
  kind: string;
  title: string;
  status: SourceStatus;
  locator: string;
  selected: number;
  created_at: string;
  token_count: number | null;
  content_hash: string | null;
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
