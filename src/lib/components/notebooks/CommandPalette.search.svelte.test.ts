// Context-aware ⌘K search (C2): on the Notes tab of an open notebook the palette
// searches THAT notebook's notes and selecting one pushes a jump request through
// the notes-nav store. On the Chat tab it still searches notebook titles.

import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import CommandPalette from './CommandPalette.svelte';
import {
  notebookStore,
  resetNotebookStore,
  loadNotebooks
} from '$lib/notebooks/notebooks-state.svelte.js';
import { hydrate, resetNotesStore } from '$lib/notes/notes-state.svelte.js';
import { notesNav, resetNotesNav } from '$lib/notes/notes-nav.svelte.js';
import type { NotebookSummary } from '$lib/notebooks/types.js';
import type { Note } from '$lib/notes/types.js';

vi.mock('$lib/notebooks/ipc.js', () => ({
  listNotebooks: vi.fn(),
  createNotebook: vi.fn(),
  renameNotebook: vi.fn(),
  trashNotebook: vi.fn(),
  restoreNotebook: vi.fn(),
  listTrashed: vi.fn(),
  purgeNotebook: vi.fn(),
  touchNotebookActivity: vi.fn().mockResolvedValue(undefined)
}));

vi.mock('$lib/notes/ipc.js', () => ({
  saveChatNote: vi.fn(),
  saveManualNote: vi.fn(),
  listNotes: vi.fn(),
  deleteNote: vi.fn().mockResolvedValue(undefined),
  updateNote: vi.fn(),
  setNotePinned: vi.fn()
}));

import { listNotebooks } from '$lib/notebooks/ipc.js';
import { listNotes } from '$lib/notes/ipc.js';

const NB = 'nb-001';

function makeNotebook(overrides?: Partial<NotebookSummary>): NotebookSummary {
  return {
    id: NB,
    title: 'Alpha Research',
    description: null,
    focus_mode: 'research',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    trashed_at: null,
    last_activity_at: null,
    graph_retrieval_enabled: null,
    embedding_model: null,
    embedding_backend: null,
    source_count: 3,
    ...overrides
  };
}

function makeNote(overrides?: Partial<Note>): Note {
  return {
    id: 'note-001',
    notebook_id: NB,
    origin: 'manual',
    content: 'a note body',
    citations: null,
    source_title: null,
    source_message_id: null,
    created_at: '2026-07-12T00:00:00Z',
    updated_at: '2026-07-12T00:00:00Z',
    pinned: false,
    ...overrides
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  resetNotebookStore();
  resetNotesStore();
  resetNotesNav();
});

afterEach(() => {
  resetNotebookStore();
  resetNotesStore();
  resetNotesNav();
});

async function seedNotesTab(notes: Note[]) {
  vi.mocked(listNotebooks).mockResolvedValue([makeNotebook()]);
  await loadNotebooks();
  vi.mocked(listNotes).mockResolvedValue(notes);
  await hydrate(NB);
  notebookStore.activeNotebookId = NB;
  notebookStore.activeTab = 'notes';
  notebookStore.paletteOpen = true;
}

describe('notes-mode palette', () => {
  it('uses "Search notes" placeholder and a "Notes" section on the Notes tab', async () => {
    await seedNotesTab([makeNote({ id: 'n1', content: 'Revenue grew 34%' })]);
    render(CommandPalette);
    expect(screen.getByPlaceholderText('Search notes')).toBeInTheDocument();
    expect(screen.getByText('Notes')).toBeInTheDocument();
  });

  it('filters notes by case-insensitive substring on content + source_title', async () => {
    await seedNotesTab([
      makeNote({ id: 'n1', content: 'Revenue grew 34%' }),
      makeNote({ id: 'n2', content: 'unrelated musing' }),
      makeNote({
        id: 'n3',
        origin: 'chat',
        source_message_id: 'm3',
        content: 'body',
        source_title: 'Revenue Report'
      })
    ]);
    notebookStore.paletteQuery = 'revenue';
    render(CommandPalette);

    // n1 (content) and n3 (source_title) match; n2 does not.
    const rows = screen.getAllByRole('option');
    expect(rows).toHaveLength(2);
    expect(screen.getByText('Revenue grew 34%')).toBeInTheDocument();
    expect(screen.getByText('Revenue Report')).toBeInTheDocument();
    expect(screen.queryByText('unrelated musing')).toBeNull();
  });

  it('shows "No notes found" when nothing matches', async () => {
    await seedNotesTab([makeNote({ id: 'n1', content: 'alpha' })]);
    notebookStore.paletteQuery = 'zzz-none';
    render(CommandPalette);
    expect(screen.getByText('No notes found')).toBeInTheDocument();
  });

  it('selecting a note row pushes a jump request through notes-nav and closes', async () => {
    await seedNotesTab([makeNote({ id: 'note-42', content: 'jump target' })]);
    render(CommandPalette);

    await fireEvent.click(screen.getByRole('option', { name: /jump target/i }));

    expect(notesNav.request?.noteId).toBe('note-42');
    expect(notebookStore.paletteOpen).toBe(false);
  });

  it('Enter on the highlighted note fires the jump signal', async () => {
    await seedNotesTab([makeNote({ id: 'note-99', content: 'enter target' })]);
    render(CommandPalette);

    const panel = screen.getByRole('dialog');
    await fireEvent.keyDown(panel, { key: 'Enter' });

    expect(notesNav.request?.noteId).toBe('note-99');
  });

  it('falls back to notebook-title search on the Chat tab', async () => {
    await seedNotesTab([makeNote({ id: 'n1', content: 'a note' })]);
    notebookStore.activeTab = 'chat';
    render(CommandPalette);

    expect(screen.getByPlaceholderText('Search notebooks')).toBeInTheDocument();
    expect(screen.getByText('Notebooks')).toBeInTheDocument();
    expect(screen.getByText('Alpha Research')).toBeInTheDocument();
  });
});
