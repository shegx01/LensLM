// Component tests for NotesPane: KEY INSIGHTS + PERSONAL NOTES sections, the
// manual-note composer, the both-empty state, and per-card delete.

import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Note } from '$lib/notes/types.js';

const { mockNotesStore, addManualNote, remove } = vi.hoisted(() => {
  let _groups: Array<{ sourceId: string | null; sourceTitle: string | null; notes: Note[] }> = [];
  let _manual: Note[] = [];
  const mockNotesStore = {
    groupedBySource: vi.fn(() => _groups),
    manualNotes: vi.fn(() => _manual),
    _setGroups(g: typeof _groups) {
      _groups = g;
    },
    _setManual(m: Note[]) {
      _manual = m;
    }
  };
  return { mockNotesStore, addManualNote: vi.fn(), remove: vi.fn() };
});

vi.mock('$lib/notes/notes-state.svelte.js', () => ({
  notesStore: mockNotesStore,
  hydrate: vi.fn().mockResolvedValue(undefined),
  addManualNote,
  remove
}));

import NotesPane from './NotesPane.svelte';

function makeNote(overrides?: Partial<Note>): Note {
  return {
    id: 'note-001',
    notebook_id: 'nb-1',
    origin: 'chat',
    content: 'a saved insight',
    citations: null,
    source_title: null,
    source_message_id: 'msg-001',
    created_at: '2026-07-12T00:00:00Z',
    updated_at: '2026-07-12T00:00:00Z',
    ...overrides
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockNotesStore._setGroups([]);
  mockNotesStore._setManual([]);
});

afterEach(() => {
  mockNotesStore._setGroups([]);
  mockNotesStore._setManual([]);
});

describe('NotesPane', () => {
  it('shows the empty state only when both sections are empty', () => {
    render(NotesPane, { props: { notebookId: 'nb-1' } });
    expect(screen.getByText('No notes yet')).toBeInTheDocument();
    expect(screen.queryByText('KEY INSIGHTS')).not.toBeInTheDocument();
    expect(screen.queryByText('PERSONAL NOTES')).not.toBeInTheDocument();
  });

  it('renders both sections when insights and manual notes exist', () => {
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
      makeNote({ id: 'm1', origin: 'manual', content: 'My own thought.', source_message_id: null })
    ]);

    render(NotesPane, { props: { notebookId: 'nb-1' } });

    expect(screen.getByText('KEY INSIGHTS')).toBeInTheDocument();
    expect(screen.getByText('PERSONAL NOTES')).toBeInTheDocument();
    // Each note's text renders in its card body and again as its timeline label.
    expect(screen.getAllByText('Revenue grew 34%.').length).toBeGreaterThan(0);
    expect(screen.getAllByText('My own thought.').length).toBeGreaterThan(0);
    expect(screen.queryByText('No notes yet')).not.toBeInTheDocument();
  });

  it('adds a manual note via the composer and clears the input', async () => {
    render(NotesPane, { props: { notebookId: 'nb-1' } });
    const input = screen.getByLabelText<HTMLInputElement>('Add a note');
    await fireEvent.input(input, { target: { value: 'A new note' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Save note' }));
    expect(addManualNote).toHaveBeenCalledWith('nb-1', 'A new note');
    expect(input.value).toBe('');
  });

  it('does not save an empty/whitespace note', async () => {
    render(NotesPane, { props: { notebookId: 'nb-1' } });
    const saveBtn = screen.getByRole('button', { name: 'Save note' });
    expect(saveBtn).toBeDisabled();
    const input = screen.getByLabelText('Add a note');
    await fireEvent.input(input, { target: { value: '   ' } });
    expect(saveBtn).toBeDisabled();
    expect(addManualNote).not.toHaveBeenCalled();
  });

  it('deletes a manual note via its trash action', async () => {
    mockNotesStore._setManual([
      makeNote({ id: 'm1', origin: 'manual', content: 'Deletable.', source_message_id: null })
    ]);
    render(NotesPane, { props: { notebookId: 'nb-1' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Delete note' }));
    expect(remove).toHaveBeenCalledWith('nb-1', 'm1');
  });
});
