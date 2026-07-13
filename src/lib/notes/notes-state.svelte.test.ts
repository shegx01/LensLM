// Store unit tests for notes-state.svelte.ts (IPC mocked, no Tauri host).

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./ipc.js', () => ({
  saveChatNote: vi.fn(),
  saveManualNote: vi.fn(),
  listNotes: vi.fn(),
  deleteNote: vi.fn().mockResolvedValue(undefined),
  updateNote: vi.fn(),
  setNotePinned: vi.fn()
}));

import {
  notesStore,
  resetNotesStore,
  hydrate,
  toggleSave,
  addManualNote,
  remove,
  editNote,
  setPinned,
  noteMatchesQuery
} from './notes-state.svelte.js';
import {
  saveChatNote,
  saveManualNote,
  listNotes,
  deleteNote,
  updateNote,
  setNotePinned
} from './ipc.js';
import type { Note } from './types.js';
import type { ChatMessage, Citation } from '$lib/chat/types.js';
import { makeChatMessage } from '$lib/chat/test-fixtures.js';

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
  resetNotesStore();
});

afterEach(() => {
  resetNotesStore();
});

describe('hydrate', () => {
  it('populates notes for a notebook', async () => {
    const note = makeNote();
    vi.mocked(listNotes).mockResolvedValue([note]);

    await hydrate(NB);

    expect(notesStore.notes(NB)).toEqual([note]);
  });
});

describe('hydrate generation guard', () => {
  it('a stale hydrate resolving after a newer toggleSave does not clobber the optimistic result', async () => {
    let resolveStaleList: (rows: Note[]) => void = () => {};
    vi.mocked(listNotes).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveStaleList = resolve;
        })
    );

    const staleHydrate = hydrate(NB);

    const message = makeChatMessage({ id: 'msg-010', role: 'assistant', content: 'insight' });
    const saved = makeNote({ id: 'note-010', source_message_id: 'msg-010' });
    vi.mocked(saveChatNote).mockResolvedValue(saved);
    await toggleSave(NB, message);

    // The stale listNotes call (started before the toggle) resolves last, with
    // a snapshot that predates the just-saved note.
    resolveStaleList([]);
    await staleHydrate;

    expect(notesStore.notes(NB)).toEqual([saved]);
  });

  it('a stale hydrate resolving after a newer hydrate does not clobber the fresh rows', async () => {
    let resolveFirst: (rows: Note[]) => void = () => {};
    vi.mocked(listNotes).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveFirst = resolve;
        })
    );
    const firstHydrate = hydrate(NB);

    const freshRows = [makeNote({ id: 'note-fresh' })];
    vi.mocked(listNotes).mockResolvedValueOnce(freshRows);
    await hydrate(NB);

    resolveFirst([makeNote({ id: 'note-stale' })]);
    await firstHydrate;

    expect(notesStore.notes(NB)).toEqual(freshRows);
  });
});

describe('toggleSave', () => {
  it('saves a new note and prepends it when not yet saved', async () => {
    const message: ChatMessage = makeChatMessage({
      id: 'msg-001',
      role: 'assistant',
      content: 'the answer'
    });
    const saved = makeNote({ id: 'note-001', source_message_id: 'msg-001' });
    vi.mocked(saveChatNote).mockResolvedValue(saved);

    await toggleSave(NB, message);

    expect(saveChatNote).toHaveBeenCalledWith(NB, 'the answer', null, 'msg-001');
    expect(notesStore.notes(NB)).toEqual([saved]);
    expect(notesStore.savedMessageIds(NB).has('msg-001')).toBe(true);
  });

  it('deletes the backing note and removes it when already saved (toggle off)', async () => {
    const existing = makeNote({ id: 'note-001', source_message_id: 'msg-001' });
    vi.mocked(listNotes).mockResolvedValue([existing]);
    await hydrate(NB);

    const message = makeChatMessage({ id: 'msg-001', role: 'assistant' });
    await toggleSave(NB, message);

    expect(deleteNote).toHaveBeenCalledWith('note-001');
    expect(notesStore.notes(NB)).toEqual([]);
    expect(notesStore.savedMessageIds(NB).has('msg-001')).toBe(false);
    expect(saveChatNote).not.toHaveBeenCalled();
  });

  it('toggles a NULL-citations message correctly, keying on source_message_id not citations', async () => {
    const message = makeChatMessage({ id: 'msg-002', role: 'assistant', citations: null });
    const saved = makeNote({ id: 'note-002', source_message_id: 'msg-002', citations: null });
    vi.mocked(saveChatNote).mockResolvedValue(saved);

    await toggleSave(NB, message);
    expect(saveChatNote).toHaveBeenCalledWith(NB, message.content, null, 'msg-002');
    expect(notesStore.savedMessageIds(NB).has('msg-002')).toBe(true);

    await toggleSave(NB, message);
    expect(deleteNote).toHaveBeenCalledWith('note-002');
    expect(notesStore.savedMessageIds(NB).has('msg-002')).toBe(false);
  });

  it('no-ops for a synthetic streaming-bubble id (belt-and-suspenders alongside the UI gate)', async () => {
    const message = makeChatMessage({
      id: 'turn-001-streaming',
      role: 'assistant',
      content: 'partial content'
    });

    await toggleSave(NB, message);

    expect(saveChatNote).not.toHaveBeenCalled();
    expect(deleteNote).not.toHaveBeenCalled();
    expect(notesStore.notes(NB)).toEqual([]);
  });

  it('two concurrent toggleSave calls for the same message create exactly one note', async () => {
    const message = makeChatMessage({
      id: 'msg-003',
      role: 'assistant',
      content: 'the answer'
    });
    const saved = makeNote({ id: 'note-003', source_message_id: 'msg-003' });
    let resolveSave: (note: Note) => void = () => {};
    vi.mocked(saveChatNote).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveSave = resolve;
        })
    );

    const first = toggleSave(NB, message);
    const second = toggleSave(NB, message);
    resolveSave(saved);
    await Promise.all([first, second]);

    expect(saveChatNote).toHaveBeenCalledTimes(1);
    expect(notesStore.notes(NB)).toEqual([saved]);
  });
});

describe('addManualNote', () => {
  it('saves an origin=manual note and prepends it', async () => {
    const saved = makeNote({
      id: 'm1',
      origin: 'manual',
      content: 'a personal thought',
      source_message_id: null
    });
    vi.mocked(saveManualNote).mockResolvedValue(saved);

    await addManualNote(NB, 'a personal thought');

    expect(saveManualNote).toHaveBeenCalledWith(NB, 'a personal thought');
    expect(notesStore.notes(NB)).toEqual([saved]);
  });

  it('prepends a manual note ahead of existing notes', async () => {
    const existing = makeNote({ id: 'note-001' });
    vi.mocked(listNotes).mockResolvedValue([existing]);
    await hydrate(NB);

    const saved = makeNote({ id: 'm1', origin: 'manual', source_message_id: null });
    vi.mocked(saveManualNote).mockResolvedValue(saved);
    await addManualNote(NB, 'newer');

    expect(notesStore.notes(NB).map((n) => n.id)).toEqual(['m1', 'note-001']);
  });

  it('no-ops (no ipc call) on empty/whitespace content', async () => {
    await addManualNote(NB, '   ');
    expect(saveManualNote).not.toHaveBeenCalled();
    expect(notesStore.notes(NB)).toEqual([]);
  });
});

describe('manualNotes', () => {
  it('returns only origin=manual notes, order preserved', async () => {
    const chat = makeNote({ id: 'c1', origin: 'chat' });
    const manual = makeNote({ id: 'm1', origin: 'manual', source_message_id: null });
    vi.mocked(listNotes).mockResolvedValue([manual, chat]);
    await hydrate(NB);

    expect(notesStore.manualNotes(NB).map((n) => n.id)).toEqual(['m1']);
    // groupedBySource stays chat-only so the sections never double-count.
    expect(notesStore.groupedBySource(NB).flatMap((g) => g.notes.map((n) => n.id))).toEqual(['c1']);
  });
});

describe('remove', () => {
  it('removes a note by id', async () => {
    const note = makeNote({ id: 'note-001' });
    vi.mocked(listNotes).mockResolvedValue([note]);
    await hydrate(NB);

    await remove(NB, 'note-001');

    expect(deleteNote).toHaveBeenCalledWith('note-001');
    expect(notesStore.notes(NB)).toEqual([]);
  });
});

describe('groupedBySource', () => {
  it('groups notes by the ordinal-1 citation source_id, newest-first preserved', async () => {
    const citationsA: Citation[] = [{ source_id: 'src-a', ordinal: 1, locators: [] }];
    const citationsB: Citation[] = [{ source_id: 'src-b', ordinal: 1, locators: [] }];
    const newest = makeNote({
      id: 'n3',
      content: 'newest',
      citations: JSON.stringify(citationsA),
      source_title: 'Source A',
      created_at: '2026-07-12T02:00:00Z'
    });
    const middle = makeNote({
      id: 'n2',
      content: 'middle',
      citations: JSON.stringify(citationsB),
      source_title: 'Source B',
      created_at: '2026-07-12T01:00:00Z'
    });
    const oldest = makeNote({
      id: 'n1',
      content: 'oldest',
      citations: JSON.stringify(citationsA),
      source_title: 'Source A',
      created_at: '2026-07-12T00:00:00Z'
    });
    // list_notes already returns newest-first; the store preserves order as-is.
    vi.mocked(listNotes).mockResolvedValue([newest, middle, oldest]);

    await hydrate(NB);
    const groups = notesStore.groupedBySource(NB);

    expect(groups).toHaveLength(2);
    expect(groups[0].sourceId).toBe('src-a');
    expect(groups[0].sourceTitle).toBe('Source A');
    expect(groups[0].notes.map((n) => n.id)).toEqual(['n3', 'n1']);
    expect(groups[1].sourceId).toBe('src-b');
    expect(groups[1].notes.map((n) => n.id)).toEqual(['n2']);
  });

  it('groups uncited notes under a null source key', async () => {
    const uncited = makeNote({ id: 'n1', citations: null, source_title: null });
    vi.mocked(listNotes).mockResolvedValue([uncited]);

    await hydrate(NB);
    const groups = notesStore.groupedBySource(NB);

    expect(groups).toHaveLength(1);
    expect(groups[0].sourceId).toBeNull();
    expect(groups[0].notes.map((n) => n.id)).toEqual(['n1']);
  });
});

describe('editNote', () => {
  it('replaces the note in place with the returned row (bumped updated_at, grounding preserved)', async () => {
    const original = makeNote({
      id: 'note-001',
      content: 'old',
      source_title: 'Src A',
      source_message_id: 'msg-001',
      citations: JSON.stringify([{ source_id: 'src-a', ordinal: 1, locators: [] }])
    });
    vi.mocked(listNotes).mockResolvedValue([original]);
    await hydrate(NB);

    const updated = makeNote({
      ...original,
      content: 'new content [1]',
      updated_at: '2026-07-13T00:00:00Z'
    });
    vi.mocked(updateNote).mockResolvedValue(updated);

    await editNote(NB, 'note-001', 'new content [1]');

    expect(updateNote).toHaveBeenCalledWith('note-001', 'new content [1]');
    const [note] = notesStore.notes(NB);
    expect(note.content).toBe('new content [1]');
    expect(note.updated_at).toBe('2026-07-13T00:00:00Z');
    // Grounding columns preserved from the returned row.
    expect(note.source_title).toBe('Src A');
    expect(note.source_message_id).toBe('msg-001');
    expect(note.citations).toBe(original.citations);
  });

  it('no-ops (no ipc call) on empty/whitespace content', async () => {
    const original = makeNote({ id: 'note-001', content: 'keep me' });
    vi.mocked(listNotes).mockResolvedValue([original]);
    await hydrate(NB);

    await editNote(NB, 'note-001', '   ');

    expect(updateNote).not.toHaveBeenCalled();
    expect(notesStore.notes(NB)[0].content).toBe('keep me');
  });

  it('a stale editNote resolving after a newer hydrate does not clobber', async () => {
    const original = makeNote({ id: 'note-001', content: 'v0' });
    vi.mocked(listNotes).mockResolvedValue([original]);
    await hydrate(NB);

    let resolveUpdate: (n: Note) => void = () => {};
    vi.mocked(updateNote).mockImplementation(
      () => new Promise((resolve) => (resolveUpdate = resolve))
    );
    const staleEdit = editNote(NB, 'note-001', 'v1');

    const freshRows = [makeNote({ id: 'note-002', content: 'fresh' })];
    vi.mocked(listNotes).mockResolvedValue(freshRows);
    await hydrate(NB);

    resolveUpdate(makeNote({ id: 'note-001', content: 'v1' }));
    await staleEdit;

    expect(notesStore.notes(NB)).toEqual(freshRows);
  });
});

describe('setPinned', () => {
  it('pins a note and re-sorts so pinned floats to the top', async () => {
    const older = makeNote({
      id: 'a',
      origin: 'manual',
      source_message_id: null,
      created_at: '2026-07-12T00:00:00Z'
    });
    const newer = makeNote({
      id: 'b',
      origin: 'manual',
      source_message_id: null,
      created_at: '2026-07-12T05:00:00Z'
    });
    // list_notes returns newest-first.
    vi.mocked(listNotes).mockResolvedValue([newer, older]);
    await hydrate(NB);
    expect(notesStore.notes(NB).map((n) => n.id)).toEqual(['b', 'a']);

    vi.mocked(setNotePinned).mockResolvedValue({ ...older, pinned: true });
    await setPinned(NB, 'a', true);

    // Pinned 'a' now floats above the newer-but-unpinned 'b'.
    expect(setNotePinned).toHaveBeenCalledWith('a', true);
    expect(notesStore.notes(NB).map((n) => n.id)).toEqual(['a', 'b']);
    expect(notesStore.notes(NB)[0].pinned).toBe(true);
  });

  it('unpinning re-sorts back to newest-first', async () => {
    const older = makeNote({
      id: 'a',
      origin: 'manual',
      source_message_id: null,
      pinned: true,
      created_at: '2026-07-12T00:00:00Z'
    });
    const newer = makeNote({
      id: 'b',
      origin: 'manual',
      source_message_id: null,
      created_at: '2026-07-12T05:00:00Z'
    });
    vi.mocked(listNotes).mockResolvedValue([older, newer]);
    await hydrate(NB);

    vi.mocked(setNotePinned).mockResolvedValue({ ...older, pinned: false });
    await setPinned(NB, 'a', false);

    expect(notesStore.notes(NB).map((n) => n.id)).toEqual(['b', 'a']);
  });

  it('manualNotes and groupedBySource surface pinned-first ordering', async () => {
    const manualOld = makeNote({
      id: 'm1',
      origin: 'manual',
      source_message_id: null,
      pinned: false,
      created_at: '2026-07-12T05:00:00Z'
    });
    const manualPinned = makeNote({
      id: 'm2',
      origin: 'manual',
      source_message_id: null,
      pinned: true,
      created_at: '2026-07-12T00:00:00Z'
    });
    const citations = JSON.stringify([{ source_id: 'src-a', ordinal: 1, locators: [] }]);
    const chatOld = makeNote({
      id: 'c1',
      origin: 'chat',
      citations,
      source_title: 'A',
      pinned: false,
      created_at: '2026-07-12T05:00:00Z'
    });
    const chatPinned = makeNote({
      id: 'c2',
      origin: 'chat',
      citations,
      source_title: 'A',
      pinned: true,
      created_at: '2026-07-12T00:00:00Z'
    });
    vi.mocked(listNotes).mockResolvedValue([manualOld, manualPinned, chatOld, chatPinned]);
    await hydrate(NB);

    expect(notesStore.manualNotes(NB).map((n) => n.id)).toEqual(['m2', 'm1']);
    expect(notesStore.groupedBySource(NB)[0].notes.map((n) => n.id)).toEqual(['c2', 'c1']);
  });
});

describe('noteMatchesQuery', () => {
  it('matches case-insensitive substrings of content', () => {
    const note = makeNote({ content: 'The Quarterly Revenue Report' });
    expect(noteMatchesQuery(note, 'revenue')).toBe(true);
    expect(noteMatchesQuery(note, 'REVENUE')).toBe(true);
    expect(noteMatchesQuery(note, 'nomatch')).toBe(false);
  });

  it('matches on source_title too', () => {
    const note = makeNote({ content: 'body', source_title: 'Annual Filing' });
    expect(noteMatchesQuery(note, 'annual')).toBe(true);
  });

  it('an empty/whitespace query matches everything', () => {
    const note = makeNote({ content: 'anything' });
    expect(noteMatchesQuery(note, '')).toBe(true);
    expect(noteMatchesQuery(note, '   ')).toBe(true);
  });
});
