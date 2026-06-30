// Unit tests for the drag-drop manager (src/lib/sources/dragDrop.ts).
//
// Pure helpers (partitionPaths, physicalToLogical, hitTest) and the constants
// are tested directly. The registry + event-handler branching are tested by
// mocking `@tauri-apps/api/core` (isTauri) and `@tauri-apps/api/webview` (a
// fake getCurrentWebview whose onDragDropEvent captures the handler) per the
// Step 6 mock strategy. `./toast` is mocked so showToast is observable.

import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

// ---------------------------------------------------------------------------
// Mocks — configured before the module under test is (dynamically) imported.
// ---------------------------------------------------------------------------

// `isTauri` is toggled per test via this mutable flag.
let isTauriValue = false;
vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => isTauriValue
}));

// Capture the handler passed to onDragDropEvent and expose it to tests. The
// mock onDragDropEvent resolves a Promise<UnlistenFn> with a vi.fn() unlisten.
const capturedHandlers: Array<(event: unknown) => void> = [];
const unlistenSpy = vi.fn();
const onDragDropEventMock = vi.fn(async (handler: (event: unknown) => void) => {
  capturedHandlers.push(handler);
  return unlistenSpy;
});
vi.mock('@tauri-apps/api/webview', () => ({
  getCurrentWebview: () => ({ onDragDropEvent: onDragDropEventMock })
}));

// Observe rejection-toast calls.
const showToastMock = vi.fn();
vi.mock('./toast.svelte.js', () => ({
  showToast: (message: string, duration?: number) => showToastMock(message, duration)
}));

import {
  ACCEPTED_EXTENSIONS,
  PICKER_FILTERS,
  partitionPaths,
  physicalToLogical,
  hitTest,
  registerDropTarget,
  type DropTarget
} from './dragDrop.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeRect(left: number, top: number, right: number, bottom: number): DOMRect {
  return {
    left,
    top,
    right,
    bottom,
    x: left,
    y: top,
    width: right - left,
    height: bottom - top,
    toJSON() {
      return this;
    }
  } as DOMRect;
}

function makeTarget(rect: DOMRect): DropTarget & {
  onDrop: Mock<(paths: string[]) => void>;
  setHover: Mock<(hovering: boolean) => void>;
} {
  return {
    getRect: () => rect,
    onDrop: vi.fn<(paths: string[]) => void>(),
    setHover: vi.fn<(hovering: boolean) => void>()
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  capturedHandlers.length = 0;
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
      '/a/sheet.xlsx',
      '/a/data.csv',
      '/a/note.key',
      '/a/doc.pages',
      '/a/blob.unknown'
    ];
    const { accepted, rejected } = partitionPaths(paths);
    expect(accepted).toHaveLength(0);
    expect(rejected.map((r) => r.ext)).toEqual([
      'mp3',
      'mp4',
      'pptx',
      'xlsx',
      'csv',
      'key',
      'pages',
      'unknown'
    ]);
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
// 5-7. physicalToLogical — DPR correction
// ---------------------------------------------------------------------------

describe('physicalToLogical', () => {
  it('DPR=2 (Retina): divides by 2', () => {
    expect(physicalToLogical(400, 600, 2)).toEqual({ x: 200, y: 300 });
  });

  it('DPR=1 (standard): identity', () => {
    expect(physicalToLogical(400, 600, 1)).toEqual({ x: 400, y: 600 });
  });

  it('DPR=1.5 (Windows 150%): divides by 1.5', () => {
    expect(physicalToLogical(300, 450, 1.5)).toEqual({ x: 200, y: 300 });
  });
});

// ---------------------------------------------------------------------------
// 8-11. hitTest
// ---------------------------------------------------------------------------

describe('hitTest', () => {
  it('returns the target when the point is inside its rect', () => {
    const t = makeTarget(makeRect(100, 100, 300, 300));
    expect(hitTest([t], 200, 200)).toBe(t);
  });

  it('returns null when the point is outside all targets', () => {
    const t = makeTarget(makeRect(100, 100, 300, 300));
    expect(hitTest([t], 50, 50)).toBeNull();
  });

  it('resolves overlapping targets LIFO (last registered wins)', () => {
    const a = makeTarget(makeRect(0, 0, 500, 500));
    const b = makeTarget(makeRect(100, 100, 400, 400));
    // a registered first, b second; point inside both -> b (topmost)
    expect(hitTest([a, b], 200, 200)).toBe(b);
  });

  it('returns the containing target among non-overlapping targets', () => {
    const a = makeTarget(makeRect(0, 0, 100, 100));
    const b = makeTarget(makeRect(200, 200, 300, 300));
    expect(hitTest([a, b], 250, 250)).toBe(b);
  });
});

// ---------------------------------------------------------------------------
// 12. ACCEPTED_EXTENSIONS — completeness
// ---------------------------------------------------------------------------

describe('ACCEPTED_EXTENSIONS', () => {
  it('contains exactly the 15 backend-accepted extensions', () => {
    expect(ACCEPTED_EXTENSIONS.size).toBe(15);
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
      'epub'
    ];
    for (const ext of expected) {
      expect(ACCEPTED_EXTENSIONS.has(ext)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// 13. PICKER_FILTERS — structure
// ---------------------------------------------------------------------------

describe('PICKER_FILTERS', () => {
  it('has two groups (Documents + Structured) covering all accepted extensions', () => {
    expect(PICKER_FILTERS).toHaveLength(2);
    const names = PICKER_FILTERS.map((g) => g.name);
    expect(names).toEqual(['Documents', 'Structured']);
    const all = new Set(PICKER_FILTERS.flatMap((g) => g.extensions));
    expect(all).toEqual(new Set([...ACCEPTED_EXTENSIONS]));
  });
});

// ---------------------------------------------------------------------------
// 14. registerDropTarget — idempotent unregister (non-Tauri)
// ---------------------------------------------------------------------------

describe('registerDropTarget — non-Tauri', () => {
  it('returns an idempotent unregister function (safe to call twice)', () => {
    isTauriValue = false;
    const unregister = registerDropTarget(makeTarget(makeRect(0, 0, 100, 100)));
    expect(typeof unregister).toBe('function');
    expect(() => {
      unregister();
      unregister();
    }).not.toThrow();
    // No webview listener wired up when not under Tauri.
    expect(onDragDropEventMock).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// 15. Event-handler branching (enter/over/drop/leave) via mocked webview
// ---------------------------------------------------------------------------

describe('event-handler branching (Tauri)', () => {
  let target: ReturnType<typeof makeTarget>;
  let other: ReturnType<typeof makeTarget>;
  let unregister: () => void;
  let unregisterOther: () => void;
  let handler: (event: unknown) => void;
  const originalDpr = window.devicePixelRatio;

  beforeEach(async () => {
    isTauriValue = true;
    // DPR=1 so physical coords map 1:1 to logical for simple assertions.
    Object.defineProperty(window, 'devicePixelRatio', { value: 1, configurable: true });

    // `other` covers a disjoint region so it never matches our drop point.
    other = makeTarget(makeRect(1000, 1000, 1100, 1100));
    target = makeTarget(makeRect(100, 100, 300, 300));
    unregisterOther = registerDropTarget(other);
    unregister = registerDropTarget(target);

    // First registration wires the global listener (async import).
    await vi.waitFor(() => expect(capturedHandlers.length).toBeGreaterThan(0));
    handler = capturedHandlers[capturedHandlers.length - 1];
  });

  afterEach(() => {
    unregister();
    unregisterOther();
    Object.defineProperty(window, 'devicePixelRatio', {
      value: originalDpr,
      configurable: true
    });
  });

  it("'enter': sets hover true on the matched target and false on others", () => {
    handler({ payload: { type: 'enter', paths: ['/a/file.pdf'], position: { x: 200, y: 200 } } });
    expect(target.setHover).toHaveBeenLastCalledWith(true);
    expect(other.setHover).toHaveBeenLastCalledWith(false);
  });

  it("'over': drives hover state with no paths access", () => {
    handler({ payload: { type: 'over', position: { x: 200, y: 200 } } });
    expect(target.setHover).toHaveBeenLastCalledWith(true);
    expect(other.setHover).toHaveBeenLastCalledWith(false);
  });

  it("'drop': calls onDrop with accepted paths and clears hover on all", () => {
    handler({ payload: { type: 'drop', paths: ['/a/file.pdf'], position: { x: 200, y: 200 } } });
    expect(target.onDrop).toHaveBeenCalledWith(['/a/file.pdf']);
    expect(target.setHover).toHaveBeenLastCalledWith(false);
    expect(other.setHover).toHaveBeenLastCalledWith(false);
  });

  it("'drop' with only rejected files: does NOT call onDrop and fires a toast", () => {
    handler({ payload: { type: 'drop', paths: ['/a/song.mp3'], position: { x: 200, y: 200 } } });
    expect(target.onDrop).not.toHaveBeenCalled();
    expect(showToastMock).toHaveBeenCalledTimes(1);
    expect(showToastMock.mock.calls[0][0]).toContain('.mp3');
  });

  it("'drop' outside any zone: ignored entirely (no onDrop, no toast)", () => {
    handler({ payload: { type: 'drop', paths: ['/a/file.pdf'], position: { x: 5, y: 5 } } });
    expect(target.onDrop).not.toHaveBeenCalled();
    expect(other.onDrop).not.toHaveBeenCalled();
    expect(showToastMock).not.toHaveBeenCalled();
  });

  it("'leave': clears hover on all targets (no position, no paths)", () => {
    handler({ payload: { type: 'leave' } });
    expect(target.setHover).toHaveBeenLastCalledWith(false);
    expect(other.setHover).toHaveBeenLastCalledWith(false);
  });
});
