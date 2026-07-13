// Unit tests for notes export (#25 C4): markdown formatting + clipboard/file actions
// with the Tauri plugin APIs mocked (no real host).

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Note } from './types.js';

const { mockNotebookStore, mockNotesStore } = vi.hoisted(() => {
  let _notebooks: Array<{ id: string; title: string }> = [];
  let _groups: Array<{ sourceId: string | null; sourceTitle: string | null; notes: Note[] }> = [];
  let _manual: Note[] = [];
  return {
    mockNotebookStore: {
      get notebooks() {
        return _notebooks;
      },
      _setNotebooks(nbs: typeof _notebooks) {
        _notebooks = nbs;
      }
    },
    mockNotesStore: {
      groupedBySource: vi.fn(() => _groups),
      manualNotes: vi.fn(() => _manual),
      _setGroups(g: typeof _groups) {
        _groups = g;
      },
      _setManual(m: Note[]) {
        _manual = m;
      }
    }
  };
});

vi.mock('$lib/notebooks/notebooks-state.svelte.js', () => ({
  notebookStore: mockNotebookStore
}));

vi.mock('./notes-state.svelte.js', () => ({
  notesStore: mockNotesStore
}));

const isTauriMock = vi.fn(() => true);
vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => isTauriMock()
}));

const writeTextMock = vi.fn().mockResolvedValue(undefined);
vi.mock('@tauri-apps/plugin-clipboard-manager', () => ({
  writeText: (...args: unknown[]) => writeTextMock(...args)
}));

const saveMock = vi.fn();
vi.mock('@tauri-apps/plugin-dialog', () => ({
  save: (...args: unknown[]) => saveMock(...args)
}));

const writeTextFileMock = vi.fn().mockResolvedValue(undefined);
vi.mock('@tauri-apps/plugin-fs', () => ({
  writeTextFile: (...args: unknown[]) => writeTextFileMock(...args)
}));

import { formatNotesMarkdown, copyAllNotes, exportNotesToFile } from './export.js';

const NB = 'nb-001';

function makeNote(overrides?: Partial<Note>): Note {
  return {
    id: 'note-001',
    notebook_id: NB,
    origin: 'chat',
    content: 'an insight',
    citations: null,
    source_title: null,
    source_message_id: 'msg-001',
    created_at: '2026-07-12T00:00:00Z',
    updated_at: '2026-07-12T00:00:00Z',
    pinned: false,
    ...overrides
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  isTauriMock.mockReturnValue(true);
  mockNotebookStore._setNotebooks([{ id: NB, title: 'My Notebook' }]);
  mockNotesStore._setGroups([]);
  mockNotesStore._setManual([]);
});

afterEach(() => {
  mockNotebookStore._setNotebooks([]);
  mockNotesStore._setGroups([]);
  mockNotesStore._setManual([]);
});

describe('formatNotesMarkdown', () => {
  it('emits a title heading and both sections in store order', () => {
    mockNotesStore._setGroups([
      {
        sourceId: 'src-a',
        sourceTitle: 'Quarterly Report',
        notes: [
          makeNote({ id: 'n1', content: 'Revenue grew 34%.', source_title: 'Quarterly Report' })
        ]
      }
    ]);
    mockNotesStore._setManual([
      makeNote({
        id: 'm1',
        origin: 'manual',
        content: 'My own thought.',
        source_title: null,
        source_message_id: null
      })
    ]);

    const md = formatNotesMarkdown(NB);

    expect(md.startsWith('# My Notebook')).toBe(true);
    const insightsIdx = md.indexOf('## KEY INSIGHTS');
    const personalIdx = md.indexOf('## PERSONAL NOTES');
    expect(insightsIdx).toBeGreaterThan(-1);
    expect(personalIdx).toBeGreaterThan(insightsIdx);
    expect(md).toContain('Revenue grew 34%.');
    expect(md).toContain('*Source: Quarterly Report*');
    expect(md).toContain('My own thought.');
  });

  it('omits a section header when that section is empty', () => {
    mockNotesStore._setManual([makeNote({ id: 'm1', origin: 'manual', content: 'Solo note.' })]);

    const md = formatNotesMarkdown(NB);

    expect(md).not.toContain('## KEY INSIGHTS');
    expect(md).toContain('## PERSONAL NOTES');
  });

  it('falls back to "Notes" when the notebook is not found', () => {
    mockNotebookStore._setNotebooks([]);
    const md = formatNotesMarkdown(NB);
    expect(md.startsWith('# Notes')).toBe(true);
  });

  it('preserves store order (pinned-first) and emits raw fences as-is', () => {
    mockNotesStore._setManual([
      makeNote({ id: 'pinned-1', origin: 'manual', content: 'Pinned first.', pinned: true }),
      makeNote({
        id: 'plain-1',
        origin: 'manual',
        content: '```mermaid\ngraph TD; A-->B;\n```',
        pinned: false
      })
    ]);

    const md = formatNotesMarkdown(NB);
    const pinnedIdx = md.indexOf('Pinned first.');
    const fenceIdx = md.indexOf('```mermaid');
    expect(pinnedIdx).toBeGreaterThan(-1);
    expect(fenceIdx).toBeGreaterThan(pinnedIdx);
    expect(md).toContain('```mermaid\ngraph TD; A-->B;\n```');
  });

  it('appends no source attribution when source_title is absent', () => {
    mockNotesStore._setGroups([
      {
        sourceId: null,
        sourceTitle: null,
        notes: [makeNote({ id: 'n1', content: 'Uncited insight.', source_title: null })]
      }
    ]);
    const md = formatNotesMarkdown(NB);
    expect(md).toContain('Uncited insight.');
    expect(md).not.toContain('*Source:');
  });
});

describe('copyAllNotes', () => {
  it('writes the formatted markdown to the clipboard', async () => {
    mockNotesStore._setManual([makeNote({ id: 'm1', origin: 'manual', content: 'Clip me.' })]);

    await copyAllNotes(NB);

    expect(writeTextMock).toHaveBeenCalledTimes(1);
    expect(writeTextMock).toHaveBeenCalledWith(formatNotesMarkdown(NB));
  });

  it('is a no-op outside Tauri', async () => {
    isTauriMock.mockReturnValue(false);
    await copyAllNotes(NB);
    expect(writeTextMock).not.toHaveBeenCalled();
  });
});

describe('exportNotesToFile', () => {
  it('saves the dialog-chosen path with the formatted markdown', async () => {
    mockNotesStore._setManual([makeNote({ id: 'm1', origin: 'manual', content: 'Save me.' })]);
    saveMock.mockResolvedValue('/tmp/My Notebook.md');

    await exportNotesToFile(NB, 'md');

    expect(saveMock).toHaveBeenCalledTimes(1);
    const opts = saveMock.mock.calls[0][0];
    expect(opts.defaultPath).toBe('My Notebook.md');
    expect(opts.filters).toEqual([{ name: 'Markdown', extensions: ['md'] }]);
    expect(writeTextFileMock).toHaveBeenCalledWith('/tmp/My Notebook.md', formatNotesMarkdown(NB));
  });

  it('uses the .txt filter and extension when exporting as txt', async () => {
    saveMock.mockResolvedValue('/tmp/My Notebook.txt');

    await exportNotesToFile(NB, 'txt');

    const opts = saveMock.mock.calls[0][0];
    expect(opts.defaultPath).toBe('My Notebook.txt');
    expect(opts.filters).toEqual([{ name: 'Text', extensions: ['txt'] }]);
  });

  it('is a no-op when the user cancels the save dialog', async () => {
    saveMock.mockResolvedValue(null);

    await exportNotesToFile(NB, 'md');

    expect(writeTextFileMock).not.toHaveBeenCalled();
  });

  it('is a no-op outside Tauri', async () => {
    isTauriMock.mockReturnValue(false);
    await exportNotesToFile(NB, 'md');
    expect(saveMock).not.toHaveBeenCalled();
    expect(writeTextFileMock).not.toHaveBeenCalled();
  });
});
