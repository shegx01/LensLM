// App-level Tauri v2 native drag-and-drop manager.
//
// Owns exactly ONE ref-counted `getCurrentWebview().onDragDropEvent` listener.
// Drop zones register `{ getRect, onDrop, setHover }`; on a native drop the
// manager DPR-corrects the physical position, hit-tests live rects, and
// dispatches to the last-registered (LIFO/topmost) containing target.
//
// Pure helpers (`partitionPaths`, `physicalToLogical`, `hitTest`) carry zero
// Svelte/Tauri dependencies and are unit-testable in isolation. The Tauri
// webview API is imported DYNAMICALLY behind `isTauri()` so `vite dev` without
// a Tauri runtime never evaluates it.
//
// See `.omc/plans/issue95-drag-drop-ingest.md` for the full design.

import { isTauri } from '@tauri-apps/api/core';
import type { UnlistenFn } from '@tauri-apps/api/event';
import { showToast } from './toast.svelte.js';

// ---------------------------------------------------------------------------
// Accepted extensions (single source of truth)
// ---------------------------------------------------------------------------

/** Backend-accepted extensions, lowercase, no leading dot.
 *  Mirrors lens-core/src/notebooks.rs:872-899 exactly.
 *  Count: 15 (pdf, docx, txt, md, markdown, mdx, json, jsonl, ndjson,
 *  yaml, yml, xml, rtf, odt, epub). */
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
  'epub'
]);

/** Picker filter groups for @tauri-apps/plugin-dialog `open()`.
 *  Derived from ACCEPTED_EXTENSIONS — two groups: Documents + Structured. */
export const PICKER_FILTERS: Array<{ name: string; extensions: string[] }> = [
  {
    name: 'Documents',
    extensions: ['pdf', 'docx', 'txt', 'md', 'markdown', 'mdx', 'rtf', 'odt', 'epub']
  },
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
  /** Return the element's current bounding rect (CSS px). Called at event time. */
  getRect: () => DOMRect;
  /** Called with the accepted file paths on a successful drop.
   *  Only called when accepted.length > 0. */
  onDrop: (paths: string[]) => void;
  /** Drive visual hover state. true = drag is over this zone; false = left/dropped. */
  setHover: (hovering: boolean) => void;
}

// Module-level registry — push on register, splice on unregister.
// LIFO resolution = iterate from end to start, return first containing target.
const targets: DropTarget[] = [];

// Async listener state: `onDragDropEvent` returns Promise<UnlistenFn>. We track
// both the pending promise and the resolved unlisten fn so teardown can handle
// either state (listener may be torn down before the promise resolves).
let listenerPromise: Promise<UnlistenFn> | null = null;
let unlistenFn: UnlistenFn | null = null;

/** Minimal shape of the drag-drop event payload we consume. The Tauri handler
 *  receives `Event<DragDropEvent>` — the data lives at `event.payload`. */
interface DragDropPayload {
  type: 'enter' | 'over' | 'drop' | 'leave';
  position?: { x: number; y: number };
  paths?: string[];
}

/** Apply hover state across all targets: the matched target (if any) gets
 *  `setHover(true)`, every other target gets `setHover(false)`. */
function applyHover(matched: DropTarget | null): void {
  for (const target of targets) {
    target.setHover(target === matched);
  }
}

/** The single global drag-drop handler. Receives `Event<DragDropEvent>`;
 *  all field access goes through `event.payload`. */
function handleDragDropEvent(event: { payload: DragDropPayload }): void {
  const payload = event.payload;

  if (payload.type === 'leave') {
    // No `position`, no `paths` — clear hover on every target.
    applyHover(null);
    return;
  }

  // `enter`, `over`, `drop` all carry `position` (physical pixels).
  const position = payload.position;
  if (!position) return;
  const { x, y } = physicalToLogical(position.x, position.y, window.devicePixelRatio);
  const matched = hitTest(targets, x, y);

  if (payload.type === 'enter' || payload.type === 'over') {
    // `over` has no `paths`; `enter` carries `paths` but they are not needed
    // (filtering happens on `drop`). Just drive hover state.
    applyHover(matched);
    return;
  }

  // payload.type === 'drop' — has `position` and `paths`.
  if (!matched) {
    // Drop outside any registered zone — ignore entirely.
    return;
  }
  applyHover(null);
  const { accepted, rejected } = partitionPaths(payload.paths ?? []);
  if (accepted.length > 0) {
    matched.onDrop(accepted);
  }
  if (rejected.length > 0) {
    const exts = rejected.map((r) => `.${r.ext}`).join(', ');
    showToast(`${rejected.length} file(s) skipped: ${exts} not supported`);
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

// ---------------------------------------------------------------------------
// Hit-test (exported for unit testing)
// ---------------------------------------------------------------------------

/** Convert physical pixel position to CSS px by dividing by devicePixelRatio.
 *  Exported for unit testing; not intended for external use. */
export function physicalToLogical(
  physX: number,
  physY: number,
  dpr: number
): { x: number; y: number } {
  // Guard against a 0 / NaN devicePixelRatio (theoretical in headless envs);
  // dividing by it would yield Infinity and silently break hit-testing.
  const safeDpr = dpr || 1;
  return { x: physX / safeDpr, y: physY / safeDpr };
}

/** Find the topmost (LIFO) registered target whose rect contains the point.
 *  Returns null if no target matches. Exported for unit testing. */
export function hitTest(
  targets: DropTarget[],
  logicalX: number,
  logicalY: number
): DropTarget | null {
  for (let i = targets.length - 1; i >= 0; i--) {
    const rect = targets[i].getRect();
    if (
      logicalX >= rect.left &&
      logicalX <= rect.right &&
      logicalY >= rect.top &&
      logicalY <= rect.bottom
    ) {
      return targets[i];
    }
  }
  return null;
}
