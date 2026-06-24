// SYNC-CHECK: must match lens-core/src/notebooks.rs Notebook struct — update both together.
//
// TypeScript mirrors of the Rust notebook structs. serde on the Rust side uses
// verbatim snake_case field names, so this shape must match exactly. M3 adds
// `source_count` to `NotebookSummary` (returned by list commands via JOIN+COUNT)
// while `Notebook` stays the plain row struct (returned by create/rename).

/** Constrained set of notebook focus modes — mirrors the Rust `FocusMode` enum values. */
export type FocusMode = 'research' | 'coding' | 'notes';

// SYNC-CHECK: must match lens-core/src/notebooks.rs Notebook struct (around line 103)
export interface Notebook {
  id: string;
  title: string;
  description: string | null;
  focus_mode: FocusMode | null;
  created_at: string;
  updated_at: string;
  trashed_at: string | null;
}

// SYNC-CHECK: must match lens-core/src/notebooks.rs NotebookSummary struct.
// The Rust struct uses `#[serde(flatten)]` on the inner Notebook, so the JSON
// representation is FLAT — all Notebook fields + source_count at the same level.
export interface NotebookSummary {
  id: string;
  title: string;
  description: string | null;
  focus_mode: FocusMode | null;
  created_at: string;
  updated_at: string;
  trashed_at: string | null;
  source_count: number;
}
