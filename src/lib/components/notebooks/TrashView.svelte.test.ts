// TrashView component tests.
//
// Covers:
//   - renders trashed notebook rows (title, source count, trashed-at time)
//   - shows empty state when trashedNotebooks is empty
//   - Restore button calls restoreNotebookAction with the correct id
//   - "Delete forever" button opens a confirm dialog
//   - confirming the dialog calls purgeNotebookAction with the correct id
//   - canceling the dialog does NOT call purgeNotebookAction
//   - back affordance sets viewMode to 'notebook'
//
// Mocks the $lib/notebooks barrel so no Tauri IPC occurs.
// The real bits-ui Dialog component is used for the confirm flow; its portal
// renders in `document.body` so we query there as normal.

import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// ── Hoisted mock refs ─────────────────────────────────────────────────────────

const { storeProxy, mockRestoreAction, mockPurgeAction, mockLoadTrashed, mockResetStore } =
  vi.hoisted(() => {
    const state = {
      trashedNotebooks: [] as import('$lib/notebooks/types.js').NotebookSummary[],
      viewMode: 'trash' as 'notebook' | 'trash'
    };

    return {
      storeProxy: state,
      mockRestoreAction: vi.fn().mockResolvedValue(undefined),
      mockPurgeAction: vi.fn().mockResolvedValue(undefined),
      mockLoadTrashed: vi.fn().mockResolvedValue(undefined),
      mockResetStore: vi.fn()
    };
  });

// Mock the entire notebooks barrel — no real IPC / store
vi.mock('$lib/notebooks/index.js', () => ({
  get notebookStore() {
    return {
      get trashedNotebooks() {
        return storeProxy.trashedNotebooks;
      },
      get viewMode() {
        return storeProxy.viewMode;
      },
      set viewMode(v: 'notebook' | 'trash') {
        storeProxy.viewMode = v;
      }
    };
  },
  loadTrashed: mockLoadTrashed,
  restoreNotebookAction: mockRestoreAction,
  purgeNotebookAction: mockPurgeAction,
  resetNotebookStore: mockResetStore,
  // Passthrough utilities
  notebookAccentClass: (_id: string) => 'nb-purple',
  formatRelativeTime: (_iso: string) => '2d ago',
  formatSourceCount: (count: number) => (count === 1 ? '1 source' : `${count} sources`)
}));

import TrashView from './TrashView.svelte';
import type { NotebookSummary } from '$lib/notebooks/types.js';

// ── Fixtures ──────────────────────────────────────────────────────────────────

function makeNotebook(overrides?: Partial<NotebookSummary>): NotebookSummary {
  return {
    id: 'nb-trash-001',
    title: 'Old Research Notes',
    description: null,
    focus_mode: null,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-05-01T00:00:00Z',
    trashed_at: '2026-06-01T00:00:00Z',
    source_count: 4,
    ...overrides
  };
}

// ── Setup / teardown ─────────────────────────────────────────────────────────

beforeEach(() => {
  storeProxy.trashedNotebooks = [];
  storeProxy.viewMode = 'trash';
  mockRestoreAction.mockClear();
  mockPurgeAction.mockClear();
  mockLoadTrashed.mockClear();
});

afterEach(() => {
  vi.clearAllMocks();
});

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('TrashView — row rendering', () => {
  it('renders a trashed notebook title', () => {
    storeProxy.trashedNotebooks = [makeNotebook()];
    render(TrashView);
    expect(screen.getByText('Old Research Notes')).toBeInTheDocument();
  });

  it('renders source count (plural)', () => {
    storeProxy.trashedNotebooks = [makeNotebook({ source_count: 4 })];
    render(TrashView);
    expect(screen.getByText(/4 sources/i)).toBeInTheDocument();
  });

  it('renders source count (singular)', () => {
    storeProxy.trashedNotebooks = [makeNotebook({ source_count: 1 })];
    render(TrashView);
    expect(screen.getByText(/1 source/i)).toBeInTheDocument();
  });

  it('renders trashed relative time in subtitle', () => {
    storeProxy.trashedNotebooks = [makeNotebook()];
    render(TrashView);
    // formatRelativeTime is mocked to return '2d ago'
    expect(screen.getByText(/trashed 2d ago/i)).toBeInTheDocument();
  });

  it('renders multiple trashed notebook rows', () => {
    storeProxy.trashedNotebooks = [
      makeNotebook({ id: 'nb-001', title: 'Alpha Notes' }),
      makeNotebook({ id: 'nb-002', title: 'Beta Research' })
    ];
    render(TrashView);
    expect(screen.getByText('Alpha Notes')).toBeInTheDocument();
    expect(screen.getByText('Beta Research')).toBeInTheDocument();
  });
});

describe('TrashView — empty state', () => {
  it('renders "Trash is empty" when trashedNotebooks is empty', () => {
    storeProxy.trashedNotebooks = [];
    render(TrashView);
    expect(screen.getByText('Trash is empty')).toBeInTheDocument();
  });

  it('does NOT render the empty state when there are trashed notebooks', () => {
    storeProxy.trashedNotebooks = [makeNotebook()];
    render(TrashView);
    expect(screen.queryByText('Trash is empty')).not.toBeInTheDocument();
  });
});

describe('TrashView — Restore action', () => {
  it('clicking Restore calls restoreNotebookAction with the correct id', async () => {
    storeProxy.trashedNotebooks = [makeNotebook({ id: 'nb-trash-001', title: 'Old Notes' })];
    render(TrashView);
    const restoreBtn = screen.getByRole('button', { name: /restore Old Notes/i });
    await fireEvent.click(restoreBtn);
    await waitFor(() => expect(mockRestoreAction).toHaveBeenCalledWith('nb-trash-001'));
  });

  it('Restore does NOT call purgeNotebookAction', async () => {
    storeProxy.trashedNotebooks = [makeNotebook()];
    render(TrashView);
    const restoreBtn = screen.getByRole('button', { name: /restore/i });
    await fireEvent.click(restoreBtn);
    await waitFor(() => expect(mockRestoreAction).toHaveBeenCalled());
    expect(mockPurgeAction).not.toHaveBeenCalled();
  });
});

describe('TrashView — Delete forever (confirm dialog)', () => {
  it('clicking "Delete forever" opens the confirm dialog', async () => {
    storeProxy.trashedNotebooks = [makeNotebook({ title: 'Old Research Notes' })];
    render(TrashView);
    const deleteBtn = screen.getByRole('button', {
      name: /delete Old Research Notes forever/i
    });
    await fireEvent.click(deleteBtn);
    // Dialog should be visible — bits-ui renders into a portal in document.body
    await waitFor(() => expect(screen.getByRole('dialog')).toBeInTheDocument());
  });

  it('confirm dialog contains the notebook title', async () => {
    storeProxy.trashedNotebooks = [makeNotebook({ title: 'Old Research Notes' })];
    render(TrashView);
    await fireEvent.click(
      screen.getByRole('button', { name: /delete Old Research Notes forever/i })
    );
    const dialog = await waitFor(() => screen.getByRole('dialog'));
    expect(dialog).toHaveTextContent(/Old Research Notes/);
  });

  it('confirming the dialog calls purgeNotebookAction with the correct id', async () => {
    storeProxy.trashedNotebooks = [makeNotebook({ id: 'nb-trash-001', title: 'Old Notes' })];
    render(TrashView);

    await fireEvent.click(screen.getByRole('button', { name: /delete Old Notes forever/i }));
    await waitFor(() => screen.getByRole('dialog'));

    const confirmBtn =
      (document.querySelector('[data-confirm-purge-btn]') as HTMLElement) ??
      screen.getByRole('button', { name: /delete forever/i });
    await fireEvent.click(confirmBtn);

    await waitFor(() => expect(mockPurgeAction).toHaveBeenCalledWith('nb-trash-001'));
  });

  it('canceling the dialog does NOT call purgeNotebookAction', async () => {
    storeProxy.trashedNotebooks = [makeNotebook({ title: 'Old Research Notes' })];
    render(TrashView);

    await fireEvent.click(
      screen.getByRole('button', { name: /delete Old Research Notes forever/i })
    );
    await waitFor(() => screen.getByRole('dialog'));

    const cancelBtn = document.querySelector('[data-cancel-btn]') as HTMLElement;
    await fireEvent.click(cancelBtn);

    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(mockPurgeAction).not.toHaveBeenCalled();
  });

  it('canceling the dialog closes it without action', async () => {
    storeProxy.trashedNotebooks = [makeNotebook()];
    render(TrashView);

    await fireEvent.click(screen.getByRole('button', { name: /delete .* forever/i }));
    await waitFor(() => screen.getByRole('dialog'));

    const cancelBtn = document.querySelector('[data-cancel-btn]') as HTMLElement;
    await fireEvent.click(cancelBtn);

    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  });
});

describe('TrashView — back affordance', () => {
  it('clicking the back button sets viewMode to "notebook"', async () => {
    render(TrashView);
    const backBtn = screen.getByRole('button', { name: /back to notebooks/i });
    await fireEvent.click(backBtn);
    expect(storeProxy.viewMode).toBe('notebook');
  });
});

describe('TrashView — mount behavior', () => {
  it('calls loadTrashed on mount', () => {
    render(TrashView);
    expect(mockLoadTrashed).toHaveBeenCalledOnce();
  });
});
