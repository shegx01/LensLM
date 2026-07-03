// SYNC-CHECK: must match lens-core/src/notebooks.rs InspectorChunk + EmbeddingStats structs.
// serde uses verbatim snake_case field names; `Option<T>` ⇒ `T | null`.

/** Mirrors `chunk::kind` constants in lens-core/src/chunk.rs (`PARENT`/`CHILD`/`SUMMARY`). */
export type InspectorChunkKind = 'parent' | 'child' | 'summary';

// SYNC-CHECK: must match lens-core/src/notebooks.rs InspectorChunk struct.
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

// SYNC-CHECK: must match lens-core/src/notebooks.rs EmbeddingStats struct.
export interface EmbeddingStats {
  model: string;
  dim: number;
  status: string;
}

// SYNC-CHECK: must match src-tauri/src/commands/inspector.rs InspectorResponse struct.
// `stats` is an ARRAY — a notebook may have multiple active embedding-index rows.
export interface InspectorResponse {
  chunks: InspectorChunk[];
  stats: EmbeddingStats[];
}
