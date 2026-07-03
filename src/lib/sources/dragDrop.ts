// App-level Tauri v2 native drag-and-drop manager. One ref-counted listener; active = last-registered target.
// NO coordinate hit-test: Tauri's `position` is unreliable (tauri-apps/tauri#10744, ~28px macOS offset).
// Accepts drops window-wide; only one zone is mounted at a time (onboarding and modal are mutually exclusive).

import { isTauri } from '@tauri-apps/api/core';
import type { UnlistenFn } from '@tauri-apps/api/event';
import { showToast } from './toast.svelte.js';

// ---------------------------------------------------------------------------
// Accepted extensions (single source of truth)
// ---------------------------------------------------------------------------

/** Backend-accepted extensions (lowercase, no dot). Mirrors lens-core/src/notebooks.rs add_file_source. */
export const ACCEPTED_EXTENSIONS: ReadonlySet<string> = new Set([
  'pdf',
  'docx',
  'txt',
  'md',
  'markdown',
  'mdx',
  'json',
  'jsonl',
  'ndjson',
  'yaml',
  'yml',
  'xml',
  'rtf',
  'odt',
  'epub',
  'xlsx',
  'xls',
  'csv'
]);

/** Picker filter groups for `@tauri-apps/plugin-dialog` — three groups: Documents, Tabular, Structured. */
export const PICKER_FILTERS: Array<{ name: string; extensions: string[] }> = [
  {
    name: 'Documents',
    extensions: ['pdf', 'docx', 'txt', 'md', 'markdown', 'mdx', 'rtf', 'odt', 'epub']
  },
  { name: 'Tabular', extensions: ['xlsx', 'xls', 'csv'] },
  { name: 'Structured', extensions: ['json', 'jsonl', 'ndjson', 'yaml', 'yml', 'xml'] }
];

// ---------------------------------------------------------------------------
// Path partition utility
// ---------------------------------------------------------------------------

/** Lowercase extension (no dot) from a path; handles `/` and `\`; `''` when absent. */
function extractExtension(path: string): string {
  const lastSep = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
  const name = lastSep === -1 ? path : path.slice(lastSep + 1);
  const dot = name.lastIndexOf('.');
  if (dot === -1) return '';
  return name.slice(dot + 1).toLowerCase();
}

/** Split paths into { accepted, rejected } by extension.
 *  Returns rejected entries as { path, ext } for toast messaging. */
export function partitionPaths(paths: string[]): {
  accepted: string[];
  rejected: Array<{ path: string; ext: string }>;
} {
  const accepted: string[] = [];
  const rejected: Array<{ path: string; ext: string }> = [];
  for (const path of paths) {
    const ext = extractExtension(path);
    if (ext && ACCEPTED_EXTENSIONS.has(ext)) {
      accepted.push(path);
    } else {
      rejected.push({ path, ext });
    }
  }
  return { accepted, rejected };
}

// ---------------------------------------------------------------------------
// Drop target registry
// ---------------------------------------------------------------------------

export interface DropTarget {
  /** Called with accepted paths on drop; only called when `accepted.length > 0`. */
  onDrop: (paths: string[]) => void;
  /** Drive visual hover state. true = drag is over the window; false = left/dropped. */
  setHover: (hovering: boolean) => void;
}

// Module-level registry — push on register, splice on unregister.
// The ACTIVE target is the last-registered one (top of the stack).
const targets: DropTarget[] = [];

/** The currently active drop zone — the last-registered target, or null. */
function activeTarget(): DropTarget | null {
  return targets.length > 0 ? targets[targets.length - 1] : null;
}

// Track both the pending promise and resolved unlisten fn: teardown may fire before the promise resolves.
let listenerPromise: Promise<UnlistenFn> | null = null;
let unlistenFn: UnlistenFn | null = null;

/** Minimal shape of the drag-drop event payload we consume. The Tauri handler
 *  receives `Event<DragDropEvent>` — the data lives at `event.payload`. */
interface DragDropPayload {
  type: 'enter' | 'over' | 'drop' | 'leave';
  paths?: string[];
}

/** Drive hover state: the active target gets `setHover(hovering)`, every other
 *  registered target is forced to `setHover(false)`. */
function setActiveHover(hovering: boolean): void {
  const active = activeTarget();
  for (const target of targets) {
    target.setHover(hovering && target === active);
  }
}

/** The single global drag-drop handler. Receives `Event<DragDropEvent>`;
 *  all field access goes through `event.payload`. No coordinate hit-test —
 *  the active (last-registered) drop zone receives the drop. */
function handleDragDropEvent(event: { payload: DragDropPayload }): void {
  const payload = event.payload;

  switch (payload.type) {
    case 'enter':
    case 'over':
      setActiveHover(true);
      return;
    case 'leave':
      setActiveHover(false);
      return;
    case 'drop': {
      const active = activeTarget();
      setActiveHover(false);
      if (!active) return;
      const { accepted, rejected } = partitionPaths(payload.paths ?? []);
      if (accepted.length > 0) {
        active.onDrop(accepted);
      }
      if (rejected.length > 0) {
        const exts = rejected.map((r) => `.${r.ext}`).join(', ');
        showToast(`${rejected.length} file(s) skipped: ${exts} not supported`);
      }
      return;
    }
  }
}

/** Tear down the global listener, handling both resolved and pending states. */
function teardownListener(): void {
  if (unlistenFn) {
    unlistenFn();
    unlistenFn = null;
    listenerPromise = null;
  } else if (listenerPromise) {
    void listenerPromise.then((fn) => fn());
    listenerPromise = null;
  }
}

/** Register a drop target; returns an idempotent unregister function.
 *  Lazily initializes the global listener on first registration; tears it down when the last target leaves. */
export function registerDropTarget(target: DropTarget): () => void {
  if (!isTauri()) {
    return () => {};
  }

  targets.push(target);

  // First target: dynamically import webview so it's never evaluated under plain `vite dev`.
  if (targets.length === 1 && listenerPromise === null) {
    listenerPromise = import('@tauri-apps/api/webview').then((mod) =>
      mod.getCurrentWebview().onDragDropEvent(handleDragDropEvent)
    );
    void listenerPromise.then((fn) => {
      unlistenFn = fn;
    });
  }

  let removed = false;
  return () => {
    if (removed) return;
    removed = true;
    const idx = targets.indexOf(target);
    if (idx !== -1) targets.splice(idx, 1);
    if (targets.length === 0) teardownListener();
  };
}
