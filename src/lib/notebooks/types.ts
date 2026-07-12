// SYNC-CHECK: must match lens-core/src/notebooks.rs Notebook struct — update both together.
// serde uses verbatim snake_case; `NotebookSummary` flattens `Notebook` + `source_count`.

/** Constrained set of notebook focus modes — mirrors the Rust `FocusMode` enum values. */
export type FocusMode = 'research' | 'coding' | 'notes';

// SYNC-CHECK: must match lens-core/src/notebooks.rs Notebook struct (around line 103)
export interface Notebook {
  id: string;
  title: string;
  description: string | null;
  focus_mode: FocusMode | null;
  /** `null` on pre-migration rows; backend resolves to global default. */
  embedding_model: string | null;
  /** `"fastembed"` | `"ollama"`. `null` on pre-migration rows; backend defaults to `"fastembed"`. */
  embedding_backend: string | null;
  created_at: string;
  updated_at: string;
  trashed_at: string | null;
  // SYNC-CHECK: must match lens-core/src/notebooks.rs Notebook.last_activity_at (nullable TEXT).
  last_activity_at: string | null;
  // SYNC-CHECK: must match lens-core/src/notebooks.rs Notebook.graph_retrieval_enabled (nullable).
  /** Raw per-notebook override; `null` = inherit the global default. NOT the effective
   *  value — read the effective bool via `getNotebookGraphRetrievalEnabled`. */
  graph_retrieval_enabled: boolean | null;
}

// SYNC-CHECK: must match lens-core/src/notebooks.rs NotebookSummary struct.
export interface NotebookSummary {
  id: string;
  title: string;
  description: string | null;
  focus_mode: FocusMode | null;
  /** `null` on pre-migration rows; backend resolves to global default. */
  embedding_model: string | null;
  /** `"fastembed"` | `"ollama"`. `null` on pre-migration rows; backend defaults to `"fastembed"`. */
  embedding_backend: string | null;
  created_at: string;
  updated_at: string;
  trashed_at: string | null;
  // SYNC-CHECK: must match lens-core/src/notebooks.rs Notebook.last_activity_at (nullable TEXT).
  last_activity_at: string | null;
  // SYNC-CHECK: must match lens-core/src/notebooks.rs Notebook.graph_retrieval_enabled (nullable).
  /** Raw per-notebook override; `null` = inherit the global default. */
  graph_retrieval_enabled: boolean | null;
  source_count: number;
}
