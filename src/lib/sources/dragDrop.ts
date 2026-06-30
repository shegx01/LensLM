// App-level Tauri v2 native drag-and-drop manager.
//
// Owns exactly ONE ref-counted `getCurrentWebview().onDragDropEvent` listener.
// Drop zones register `{ onDrop, setHover }`; a drop is routed to the currently
// ACTIVE drop zone (the last-registered target).
//
// We deliberately do NOT coordinate-hit-test the drop position against an
// element rect. Tauri's reported drop `position` is unreliable — it is
// documented as inaccurate while the devtools panel is open, and has a known
// per-platform cursor offset (e.g. ~28px on macOS, tauri-apps/tauri#10744).
// Gating acceptance on a position-vs-rect check silently drops valid files.
// The canonical Tauri pattern accepts the drop window-wide and uses enter/over/
// leave only to drive the visual hover state. Because only one drop zone is ever
// mounted at a time in this app (onboarding and the modal are mutually exclusive),
// "active = last-registered target" routes correctly without any hit-test.
//
// Pure helper `partitionPaths` carries zero Svelte/Tauri dependencies and is
// unit-testable in isolation. The Tauri webview API is imported DYNAMICALLY
// behind `isTauri()` so `vite dev` without a Tauri runtime never evaluates it.
//
// See `.omc/plans/issue95-drag-drop-ingest.md` for the full design.

import { isTauri } from '@tauri-apps/api/core';
import type { UnlistenFn } from '@tauri-apps/api/event';
import { showToast } from './toast.svelte.js';

// ---------------------------------------------------------------------------
// Accepted extensions (single source of truth)
// ---------------------------------------------------------------------------

/** Backend-accepted extensions, lowercase, no leading dot.
 *  Mirrors lens-core/src/notebooks.rs add_file_source exactly.
 *  Count: 18 (pdf, docx, txt, md, markdown, mdx, json, jsonl, ndjson,
 *  yaml, yml, xml, rtf, odt, epub, xlsx, xls, csv). */
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

/** Picker filter groups for @tauri-apps/plugin-dialog `open()`.
 *  Derived from ACCEPTED_EXTENSIONS — three groups: Documents + Tabular + Structured. */
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

/** Extract the lowercase extension (no leading dot) from a path.
 *  Handles both `/` and `\` separators. Returns `''` when there is no
 *  extension in the final path segment. */
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
  /** Called with the accepted file paths on a successful drop.
   *  Only called when accepted.length > 0. */
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

// Async listener state: `onDragDropEvent` returns Promise<UnlistenFn>. We track
// both the pending promise and the resolved unlisten fn so teardown can handle
// either state (listener may be torn down before the promise resolves).
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
      // Drive the hover highlight on the active zone. (`over` has no `paths`;
      // we never read paths here — filtering happens on `drop`.)
      setActiveHover(true);
      return;
    case 'leave':
      // Drag cancelled / left the window — clear hover everywhere.
      setActiveHover(false);
      return;
    case 'drop': {
      const active = activeTarget();
      setActiveHover(false);
      // No active drop zone (e.g. modal closed) — ignore the drop entirely.
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

/** Register a drop target. Returns an idempotent unregister function
 *  (safe to call multiple times — second call is a no-op).
 *  Lazily initializes the global onDragDropEvent listener on first registration.
 *  The listener is torn down when the last target unregisters (ref-counted). */
export function registerDropTarget(target: DropTarget): () => void {
  // Outside a Tauri runtime there is no native drag-drop — return a no-op
  // unregister immediately and never touch the webview API.
  if (!isTauri()) {
    return () => {};
  }

  targets.push(target);

  // First target: lazily wire up the single global listener via a dynamic
  // import so the webview module is never evaluated under plain `vite dev`.
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
