// SYNC-CHECK: must match lens-core/src/notebooks.rs InspectorChunk struct (around line 348)
// SYNC-CHECK: must match lens-core/src/notebooks.rs EmbeddingStats struct (around line 385)
//
// TypeScript mirrors of the dev/QA Embeddings Inspector Rust structs. serde on
// the Rust side uses verbatim snake_case field names (NO `#[serde(rename)]`), so
// these shapes must match exactly. `Option<T>` on the Rust side ⇒ `T | null`.

/** Chunk `kind` discriminant — mirrors the `chunk::kind` constants in
 * lens-core/src/chunk.rs (`PARENT`/`CHILD`/`SUMMARY`). `parent` = level 0,
 * `child` = level 1, `summary` = level 2. */
export type InspectorChunkKind = 'parent' | 'child' | 'summary';

// SYNC-CHECK: must match lens-core/src/notebooks.rs InspectorChunk struct (around line 348)
export interface InspectorChunk {
  id: string;
  parent_id: string | null;
  kind: InspectorChunkKind;
  level: number;
  section_path: string;
  text: string;
  block_type: string | null;
  char_start: number | null;
  char_end: number | null;
  source_anchor: string | null;
  embedding_text: string | null;
}

// SYNC-CHECK: must match lens-core/src/notebooks.rs EmbeddingStats struct (around line 385)
export interface EmbeddingStats {
  model: string;
  dim: number;
  status: string;
}

// SYNC-CHECK: must match src-tauri/src/commands/inspector.rs InspectorResponse struct
// NOTE: `stats` is an ARRAY — a notebook may have multiple active embedding-index
// rows (partial-unique `uq_embidx_active(notebook_id, model, dim)`), so the header
// renders one badge per entry.
export interface InspectorResponse {
  chunks: InspectorChunk[];
  stats: EmbeddingStats[];
}
