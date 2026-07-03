// Unit tests for the drag-drop manager (src/lib/sources/dragDrop.ts).
// physicalToLogical and hitTest were removed from dragDrop.ts; DropTarget no longer has getRect.

import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

// ---------------------------------------------------------------------------
// Mocks — configured before the module under test is (dynamically) imported.
// ---------------------------------------------------------------------------

let isTauriValue = false;
vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => isTauriValue
}));

let capturedHandler: ((event: unknown) => void) | null = null;
const unlistenSpy = vi.fn();
const onDragDropEventMock = vi.fn(async (handler: (event: unknown) => void) => {
  capturedHandler = handler;
  return unlistenSpy;
});
vi.mock('@tauri-apps/api/webview', () => ({
  getCurrentWebview: () => ({ onDragDropEvent: onDragDropEventMock })
}));

const showToastMock = vi.fn();
vi.mock('./toast.svelte.js', () => ({
  showToast: (message: string, duration?: number) => showToastMock(message, duration)
}));

import {
  ACCEPTED_EXTENSIONS,
  PICKER_FILTERS,
  partitionPaths,
  registerDropTarget,
  type DropTarget
} from './dragDrop.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTarget(): DropTarget & {
  onDrop: Mock<(paths: string[]) => void>;
  setHover: Mock<(hovering: boolean) => void>;
} {
  return {
    onDrop: vi.fn<(paths: string[]) => void>(),
    setHover: vi.fn<(hovering: boolean) => void>()
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  capturedHandler = null;
  isTauriValue = false;
});

afterEach(() => {
  isTauriValue = false;
});

// ---------------------------------------------------------------------------
// 1. partitionPaths — accepted extensions (incl. case-insensitivity)
// ---------------------------------------------------------------------------

describe('partitionPaths — accepted extensions', () => {
  it('classifies every accepted extension as accepted', () => {
    const paths = [...ACCEPTED_EXTENSIONS].map((ext) => `/docs/file.${ext}`);
    const { accepted, rejected } = partitionPaths(paths);
    expect(accepted).toHaveLength(ACCEPTED_EXTENSIONS.size);
    expect(rejected).toHaveLength(0);
  });

  it('is case-insensitive (.PDF, .Docx)', () => {
    const { accepted, rejected } = partitionPaths(['/a/file.PDF', '/b/file.Docx']);
    expect(accepted).toEqual(['/a/file.PDF', '/b/file.Docx']);
    expect(rejected).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// 2. partitionPaths — rejected extensions
// ---------------------------------------------------------------------------

describe('partitionPaths — rejected extensions', () => {
  it('classifies unsupported extensions as rejected with the correct ext', () => {
    const paths = [
      '/a/song.mp3',
      '/a/clip.mp4',
      '/a/deck.pptx',
      '/a/note.key',
      '/a/doc.pages',
      '/a/blob.unknown'
    ];
    const { accepted, rejected } = partitionPaths(paths);
    expect(accepted).toHaveLength(0);
    expect(rejected.map((r) => r.ext)).toEqual(['mp3', 'mp4', 'pptx', 'key', 'pages', 'unknown']);
  });
});

// ---------------------------------------------------------------------------
// 3. partitionPaths — mixed batch
// ---------------------------------------------------------------------------

describe('partitionPaths — mixed batch', () => {
  it('splits accepted and rejected entries', () => {
    const { accepted, rejected } = partitionPaths([
      '/a/file.pdf',
      '/b/song.mp3',
      '/c/doc.md',
      '/d/video.mov'
    ]);
    expect(accepted).toEqual(['/a/file.pdf', '/c/doc.md']);
    expect(rejected).toHaveLength(2);
    expect(rejected.map((r) => r.ext)).toEqual(['mp3', 'mov']);
  });
});

// ---------------------------------------------------------------------------
// 4. partitionPaths — no-extension file
// ---------------------------------------------------------------------------

describe('partitionPaths — no-extension file', () => {
  it('rejects a file with no extension and reports ext as empty string', () => {
    const { accepted, rejected } = partitionPaths(['/a/Makefile']);
    expect(accepted).toHaveLength(0);
    expect(rejected).toEqual([{ path: '/a/Makefile', ext: '' }]);
  });
});

// ---------------------------------------------------------------------------
// 5. ACCEPTED_EXTENSIONS — completeness
// ---------------------------------------------------------------------------

describe('ACCEPTED_EXTENSIONS', () => {
  it('contains exactly the 18 backend-accepted extensions', () => {
    expect(ACCEPTED_EXTENSIONS.size).toBe(18);
    const expected = [
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
    ];
    for (const ext of expected) {
      expect(ACCEPTED_EXTENSIONS.has(ext)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// 6. PICKER_FILTERS — structure
// ---------------------------------------------------------------------------

describe('PICKER_FILTERS', () => {
  it('has three groups (Documents + Tabular + Structured) covering all accepted extensions', () => {
    expect(PICKER_FILTERS).toHaveLength(3);
    const names = PICKER_FILTERS.map((g) => g.name);
    expect(names).toEqual(['Documents', 'Tabular', 'Structured']);
    const all = new Set(PICKER_FILTERS.flatMap((g) => g.extensions));
    expect(all).toEqual(new Set([...ACCEPTED_EXTENSIONS]));
  });
});

// ---------------------------------------------------------------------------
// 7. registerDropTarget — idempotent unregister (non-Tauri)
// ---------------------------------------------------------------------------

describe('registerDropTarget — non-Tauri', () => {
  it('returns an idempotent unregister function (safe to call twice)', () => {
    isTauriValue = false;
    const unregister = registerDropTarget(makeTarget());
    expect(typeof unregister).toBe('function');
    expect(() => {
      unregister();
      unregister();
    }).not.toThrow();
    expect(onDragDropEventMock).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// 8. Event-handler branching (enter/over/drop/leave) via mocked webview
// ---------------------------------------------------------------------------

describe('event-handler branching (Tauri)', () => {
  let target: ReturnType<typeof makeTarget>;
  let other: ReturnType<typeof makeTarget>;
  let unregister: () => void;
  let unregisterOther: () => void;

  beforeEach(async () => {
    isTauriValue = true;
    capturedHandler = null;

    other = makeTarget();
    target = makeTarget();
    unregisterOther = registerDropTarget(other);
    unregister = registerDropTarget(target);

    await vi.waitFor(() => expect(capturedHandler).not.toBeNull());
  });

  afterEach(() => {
    unregister();
    unregisterOther();
  });

  it("'enter': sets hover true on the active target and false on others", () => {
    capturedHandler!({ payload: { type: 'enter' } });
    expect(target.setHover).toHaveBeenLastCalledWith(true);
    expect(other.setHover).toHaveBeenLastCalledWith(false);
  });

  it("'over': sets hover true on the active target and false on others", () => {
    capturedHandler!({ payload: { type: 'over' } });
    expect(target.setHover).toHaveBeenLastCalledWith(true);
    expect(other.setHover).toHaveBeenLastCalledWith(false);
  });

  it("'leave': clears hover on all targets", () => {
    capturedHandler!({ payload: { type: 'leave' } });
    expect(target.setHover).toHaveBeenLastCalledWith(false);
    expect(other.setHover).toHaveBeenLastCalledWith(false);
  });

  it("'drop' with supported paths: calls active target's onDrop and clears hover on all", () => {
    capturedHandler!({ payload: { type: 'drop', paths: ['/a/file.pdf'] } });
    expect(target.onDrop).toHaveBeenCalledWith(['/a/file.pdf']);
    expect(target.setHover).toHaveBeenLastCalledWith(false);
    expect(other.setHover).toHaveBeenLastCalledWith(false);
  });

  it("'drop' with only unsupported paths: does NOT call onDrop and fires a toast", () => {
    capturedHandler!({ payload: { type: 'drop', paths: ['/x/a.mp3'] } });
    expect(target.onDrop).not.toHaveBeenCalled();
    expect(showToastMock).toHaveBeenCalledTimes(1);
    expect(showToastMock.mock.calls[0][0]).toContain('.mp3');
  });

  it("'drop' with no registered target: ignored entirely (no onDrop, no throw)", async () => {
    unregister();
    unregisterOther();

    // capturedHandler was captured before unregister; targets is now empty so activeTarget() returns null.
    expect(() => {
      capturedHandler!({ payload: { type: 'drop', paths: ['/a/file.pdf'] } });
    }).not.toThrow();
    expect(target.onDrop).not.toHaveBeenCalled();
    expect(other.onDrop).not.toHaveBeenCalled();

    unregister = () => {};
    unregisterOther = () => {};
  });

  it('active = last-registered: drop routes to the last-registered target, not the first', () => {
    capturedHandler!({ payload: { type: 'drop', paths: ['/a/file.pdf'] } });
    expect(target.onDrop).toHaveBeenCalledWith(['/a/file.pdf']);
    expect(other.onDrop).not.toHaveBeenCalled();
  });
});
