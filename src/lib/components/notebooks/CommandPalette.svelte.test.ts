// CommandPalette.svelte — ⌘K palette component tests.
// IPC is mocked; store is reset between tests to prevent bleed.
// ⌘K open/close is driven via the store directly (AppShell owns the listener).

import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import CommandPalette from './CommandPalette.svelte';
import {
  notebookStore,
  resetNotebookStore,
  loadNotebooks
} from '$lib/notebooks/notebooks-state.svelte.js';

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

import { listNotebooks } from '$lib/notebooks/ipc.js';

import type { NotebookSummary } from '$lib/notebooks/types.js';

function makeNotebook(overrides?: Partial<NotebookSummary>): NotebookSummary {
  return {
    id: 'nb-001',
    title: 'Alpha Research',
    description: null,
    focus_mode: 'research',
    created_at: new Date(Date.now() - 7_200_000).toISOString(),
    updated_at: new Date(Date.now() - 7_200_000).toISOString(),
    trashed_at: null,
    last_activity_at: null,
    graph_retrieval_enabled: null,
    embedding_model: null,
    embedding_backend: null,
    source_count: 3,
    ...overrides
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  resetNotebookStore();
});

afterEach(() => {
  resetNotebookStore();
});

/** Seed the store's notebook list and open the palette. */
async function openWith(notebooks: NotebookSummary[]) {
  vi.mocked(listNotebooks).mockResolvedValue(notebooks);
  await loadNotebooks();
  notebookStore.paletteOpen = true;
}

describe('visibility', () => {
  it('renders nothing when paletteOpen is false', () => {
    render(CommandPalette);
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('renders the dialog when paletteOpen is true', async () => {
    await openWith([]);
    render(CommandPalette);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });
});

describe('search input', () => {
  it('has placeholder text exactly "Search notebooks"', async () => {
    await openWith([]);
    render(CommandPalette);
    expect(screen.getByPlaceholderText('Search notebooks')).toBeInTheDocument();
  });

  it('does NOT show a "SOURCES" section header', async () => {
    await openWith([makeNotebook()]);
    render(CommandPalette);
    // Exact uppercase section header — "3 sources" in subtitles is different.
    expect(screen.queryByText('SOURCES')).toBeNull();
  });

  it('does NOT show a "CHATS" section header', async () => {
    await openWith([makeNotebook()]);
    render(CommandPalette);
    expect(screen.queryByText('CHATS')).toBeNull();
    expect(screen.queryByText('Chats')).toBeNull();
  });

  it('typing updates paletteQuery on the store', async () => {
    await openWith([makeNotebook()]);
    render(CommandPalette);
    const input = screen.getByPlaceholderText('Search notebooks');
    await fireEvent.input(input, { target: { value: 'alpha' } });
    expect(notebookStore.paletteQuery).toBe('alpha');
  });
});

describe('result filtering', () => {
  it('shows all notebooks when query is empty', async () => {
    await openWith([
      makeNotebook({ id: 'nb-001', title: 'Alpha Research' }),
      makeNotebook({ id: 'nb-002', title: 'Beta Coding' })
    ]);
    render(CommandPalette);
    expect(screen.getByText('Alpha Research')).toBeInTheDocument();
    expect(screen.getByText('Beta Coding')).toBeInTheDocument();
  });

  it('shows only matching notebooks when paletteQuery is set', async () => {
    await openWith([
      makeNotebook({ id: 'nb-001', title: 'Alpha Research' }),
      makeNotebook({ id: 'nb-002', title: 'Beta Coding' })
    ]);
    notebookStore.paletteQuery = 'alpha';
    render(CommandPalette);
    // paletteResults filters by case-insensitive title substring.
    expect(screen.getByText('Alpha Research')).toBeInTheDocument();
    expect(screen.queryByText('Beta Coding')).toBeNull();
  });

  it('shows "No notebooks found" empty state when paletteResults is empty', async () => {
    await openWith([]);
    render(CommandPalette);
    expect(screen.getByText('No notebooks found')).toBeInTheDocument();
  });

  it('shows "No notebooks found" when query matches nothing', async () => {
    await openWith([makeNotebook({ title: 'Alpha Research' })]);
    notebookStore.paletteQuery = 'zzz-no-match';
    render(CommandPalette);
    expect(screen.getByText('No notebooks found')).toBeInTheDocument();
  });
});

describe('keyboard navigation', () => {
  it('ArrowDown moves the highlight from the first to the second result', async () => {
    await openWith([
      makeNotebook({ id: 'nb-001', title: 'Alpha' }),
      makeNotebook({ id: 'nb-002', title: 'Beta' })
    ]);
    render(CommandPalette);

    const panel = screen.getByRole('dialog');
    await fireEvent.keyDown(panel, { key: 'ArrowDown' });

    const rows = screen.getAllByRole('option');
    expect(rows[0]).toHaveAttribute('aria-selected', 'false');
    expect(rows[1]).toHaveAttribute('aria-selected', 'true');
  });

  it('ArrowUp wraps from the first result back to the last', async () => {
    await openWith([
      makeNotebook({ id: 'nb-001', title: 'Alpha' }),
      makeNotebook({ id: 'nb-002', title: 'Beta' })
    ]);
    render(CommandPalette);

    const panel = screen.getByRole('dialog');
    await fireEvent.keyDown(panel, { key: 'ArrowUp' });

    const rows = screen.getAllByRole('option');
    expect(rows[rows.length - 1]).toHaveAttribute('aria-selected', 'true');
  });

  it('Enter selects the highlighted notebook and closes the palette', async () => {
    await openWith([
      makeNotebook({ id: 'nb-001', title: 'Alpha' }),
      makeNotebook({ id: 'nb-002', title: 'Beta' })
    ]);
    render(CommandPalette);

    const panel = screen.getByRole('dialog');
    await fireEvent.keyDown(panel, { key: 'ArrowDown' });
    await fireEvent.keyDown(panel, { key: 'Enter' });

    expect(notebookStore.activeNotebookId).toBe('nb-002');
    expect(notebookStore.paletteOpen).toBe(false);
  });

  it('Enter on the first result (default highlight) selects it', async () => {
    await openWith([makeNotebook({ id: 'nb-001', title: 'Alpha' })]);
    render(CommandPalette);

    const panel = screen.getByRole('dialog');
    await fireEvent.keyDown(panel, { key: 'Enter' });

    expect(notebookStore.activeNotebookId).toBe('nb-001');
    expect(notebookStore.paletteOpen).toBe(false);
  });

  it('Escape closes the palette without selecting a notebook', async () => {
    await openWith([makeNotebook({ id: 'nb-001', title: 'Alpha' })]);
    render(CommandPalette);

    const panel = screen.getByRole('dialog');
    await fireEvent.keyDown(panel, { key: 'Escape' });

    expect(notebookStore.paletteOpen).toBe(false);
    expect(notebookStore.activeNotebookId).toBeNull();
  });

  it('closing the palette resets paletteQuery to ""', async () => {
    await openWith([makeNotebook()]);
    notebookStore.paletteQuery = 'alpha';
    render(CommandPalette);

    const panel = screen.getByRole('dialog');
    await fireEvent.keyDown(panel, { key: 'Escape' });

    // The store setter auto-resets paletteQuery when paletteOpen is set to false.
    expect(notebookStore.paletteQuery).toBe('');
  });
});

describe('mouse interaction', () => {
  it('clicking a result row selects that notebook and closes the palette', async () => {
    await openWith([makeNotebook({ id: 'nb-001', title: 'Alpha Research' })]);
    render(CommandPalette);

    const row = screen.getByRole('option', { name: /Alpha Research/i });
    await fireEvent.click(row);

    expect(notebookStore.activeNotebookId).toBe('nb-001');
    expect(notebookStore.paletteOpen).toBe(false);
  });

  it('clicking the Esc button closes the palette', async () => {
    await openWith([makeNotebook()]);
    render(CommandPalette);

    const escBtn = screen.getByRole('button', { name: /close search/i });
    await fireEvent.click(escBtn);
    expect(notebookStore.paletteOpen).toBe(false);
  });

  it('clicking the backdrop (not the panel) closes the palette', async () => {
    await openWith([makeNotebook()]);
    render(CommandPalette);

    const backdrop = document.querySelector('[role="presentation"]') as HTMLElement;
    expect(backdrop).not.toBeNull();
    // Simulate a backdrop click where target === currentTarget.
    await fireEvent.click(backdrop);
    expect(notebookStore.paletteOpen).toBe(false);
  });
});

describe('accessibility', () => {
  it('dialog has role="dialog" and aria-modal="true"', async () => {
    await openWith([]);
    render(CommandPalette);
    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
  });

  it('search input has aria-label "Search notebooks"', async () => {
    await openWith([]);
    render(CommandPalette);
    expect(screen.getByRole('combobox', { name: 'Search notebooks' })).toBeInTheDocument();
  });

  it('each result row has role="option"', async () => {
    await openWith([
      makeNotebook({ id: 'nb-001', title: 'Alpha' }),
      makeNotebook({ id: 'nb-002', title: 'Beta' })
    ]);
    render(CommandPalette);
    const rows = screen.getAllByRole('option');
    expect(rows).toHaveLength(2);
  });

  it('the first result row is aria-selected by default', async () => {
    await openWith([makeNotebook({ id: 'nb-001', title: 'Alpha' })]);
    render(CommandPalette);
    const row = screen.getByRole('option');
    expect(row).toHaveAttribute('aria-selected', 'true');
  });

  it('the input aria-activedescendant points to the highlighted result row', async () => {
    const nb = makeNotebook({ id: 'nb-001', title: 'Alpha' });
    await openWith([nb]);
    render(CommandPalette);
    const input = screen.getByRole('combobox');
    expect(input).toHaveAttribute('aria-activedescendant', 'palette-result-nb-001');
  });
});

describe('scope guard — notebooks-only (M3)', () => {
  it('does not render a SOURCES section header', async () => {
    await openWith([makeNotebook()]);
    render(CommandPalette);
    expect(screen.queryByText('SOURCES')).toBeNull();
  });

  it('does not render a CHATS section header', async () => {
    await openWith([makeNotebook()]);
    render(CommandPalette);
    expect(screen.queryByText('CHATS')).toBeNull();
  });
});

describe('footer hint bar', () => {
  it('renders all three keyboard hint segments', async () => {
    await openWith([]);
    render(CommandPalette);
    expect(screen.getByText('↑↓ navigate')).toBeInTheDocument();
    expect(screen.getByText('↵ open')).toBeInTheDocument();
    expect(screen.getByText('⌘ anywhere')).toBeInTheDocument();
  });
});
