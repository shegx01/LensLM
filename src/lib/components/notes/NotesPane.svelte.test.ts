// Component tests for NotesPane: grouped KEY INSIGHTS cards + the empty state.

import { render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Note } from '$lib/notes/types.js';

const { mockNotesStore } = vi.hoisted(() => {
  let _groups: Array<{ sourceId: string | null; sourceTitle: string | null; notes: Note[] }> = [];
  const mockNotesStore = {
    groupedBySource: vi.fn(() => _groups),
    _setGroups(g: typeof _groups) {
      _groups = g;
    }
  };
  return { mockNotesStore };
});

vi.mock('$lib/notes/notes-state.svelte.js', () => ({
  notesStore: mockNotesStore,
  hydrate: vi.fn().mockResolvedValue(undefined)
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
});

afterEach(() => {
  mockNotesStore._setGroups([]);
});

describe('NotesPane', () => {
  it('shows the empty state when there are no notes', () => {
    render(NotesPane, { props: { notebookId: 'nb-1' } });
    expect(screen.getByText('No saved notes yet')).toBeInTheDocument();
  });

  it('renders the KEY INSIGHTS heading and grouped cards when notes exist', () => {
    mockNotesStore._setGroups([
      {
        sourceId: 'src-a',
        sourceTitle: 'Quarterly Report',
        notes: [
          makeNote({ id: 'n1', content: 'Revenue grew 34%.', source_title: 'Quarterly Report' })
        ]
      },
      {
        sourceId: null,
        sourceTitle: null,
        notes: [makeNote({ id: 'n2', content: 'An uncited note.', source_title: null })]
      }
    ]);

    render(NotesPane, { props: { notebookId: 'nb-1' } });

    expect(screen.getByText('KEY INSIGHTS')).toBeInTheDocument();
    expect(screen.getByText('Quarterly Report')).toBeInTheDocument();
    expect(screen.getByText('Revenue grew 34%.')).toBeInTheDocument();
    expect(screen.getByText('An uncited note.')).toBeInTheDocument();
    expect(screen.queryByText('No saved notes yet')).not.toBeInTheDocument();
  });
});
