// NotebooksSidebar component tests.
//
// Covers: renders rows with title/count/time, search trigger opens palette,
// collapse toggle flips sidebar state, "Sign out" is NOT present.
// Mocks the $lib/notebooks module and ThemeSwitcher to isolate the component.

import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// ── Hoisted mock state ────────────────────────────────────────────────────────
const { storeProxy, mockOpenTrash, mockSelectNotebook, mockResetStore } = vi.hoisted(() => {
  // Mutable store state that tests can manipulate directly
  const state = {
    notebooks: [] as import('$lib/notebooks/types.js').NotebookSummary[],
    trashedNotebooks: [] as import('$lib/notebooks/types.js').NotebookSummary[],
    trashCount: 0,
    activeNotebookId: null as string | null,
    sidebarCollapsed: false,
    paletteOpen: false,
    viewMode: 'notebook' as 'notebook' | 'trash'
  };

  return {
    storeProxy: state,
    mockOpenTrash: vi.fn().mockResolvedValue(undefined),
    mockSelectNotebook: vi.fn(),
    mockResetStore: vi.fn()
  };
});

vi.mock('$lib/notebooks/index.js', () => ({
  get notebookStore() {
    return {
      get notebooks() {
        return storeProxy.notebooks;
      },
      get trashCount() {
        return storeProxy.trashCount;
      },
      get activeNotebookId() {
        return storeProxy.activeNotebookId;
      },
      get sidebarCollapsed() {
        return storeProxy.sidebarCollapsed;
      },
      set sidebarCollapsed(v: boolean) {
        storeProxy.sidebarCollapsed = v;
      },
      get paletteOpen() {
        return storeProxy.paletteOpen;
      },
      set paletteOpen(v: boolean) {
        storeProxy.paletteOpen = v;
      },
      get viewMode() {
        return storeProxy.viewMode;
      }
    };
  },
  openTrash: mockOpenTrash,
  selectNotebook: mockSelectNotebook,
  resetNotebookStore: mockResetStore,
  notebookAccentClass: () => 'nb-purple',
  formatRelativeTime: () => '1w ago',
  formatSourceCount: (count: number) => (count === 1 ? '1 source' : `${count} sources`),
  getInitials: (name: string) =>
    name
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((w) => w[0].toUpperCase())
      .join('') || '?',
  renameNotebookAction: vi.fn().mockResolvedValue(undefined),
  trashNotebookAction: vi.fn().mockResolvedValue(undefined)
}));

// Stub ThemeSwitcher to avoid mode-watcher IPC in tests
vi.mock('$lib/components/ThemeSwitcher.svelte', () => ({
  default: function ThemeSwitcherStub() {}
}));

import NotebooksSidebar from './NotebooksSidebar.svelte';
import type { NotebookSummary } from '$lib/notebooks/types.js';

// ── Fixtures ──────────────────────────────────────────────────────────────────

function makeNotebook(id: string, title: string, sourceCount = 2): NotebookSummary {
  return {
    id,
    title,
    description: null,
    focus_mode: null,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-06-01T00:00:00Z',
    trashed_at: null,
    source_count: sourceCount
  };
}

// ── Setup / teardown ─────────────────────────────────────────────────────────

beforeEach(() => {
  storeProxy.notebooks = [];
  storeProxy.trashCount = 0;
  storeProxy.activeNotebookId = null;
  storeProxy.sidebarCollapsed = false;
  storeProxy.paletteOpen = false;
  storeProxy.viewMode = 'notebook';
  mockOpenTrash.mockClear();
  mockSelectNotebook.mockClear();
});

afterEach(() => {
  vi.clearAllMocks();
});

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('NotebooksSidebar (expanded)', () => {
  it('renders all notebook titles', () => {
    storeProxy.notebooks = [
      makeNotebook('nb-1', 'Alpha Notebook'),
      makeNotebook('nb-2', 'Beta Notebook')
    ];
    render(NotebooksSidebar);
    expect(screen.getByText('Alpha Notebook')).toBeInTheDocument();
    expect(screen.getByText('Beta Notebook')).toBeInTheDocument();
  });

  it('renders source count and relative time for each row', () => {
    storeProxy.notebooks = [makeNotebook('nb-1', 'Alpha Notebook', 3)];
    render(NotebooksSidebar);
    expect(screen.getByText(/3 sources/i)).toBeInTheDocument();
    expect(screen.getByText(/1w ago/i)).toBeInTheDocument();
  });

  it('search trigger button is present and opens the command palette', async () => {
    render(NotebooksSidebar);
    const trigger = screen.getByRole('button', { name: /search notebooks/i });
    await fireEvent.click(trigger);
    expect(storeProxy.paletteOpen).toBe(true);
  });

  it('collapse toggle flips sidebarCollapsed to true', async () => {
    render(NotebooksSidebar);
    const collapseBtn = screen.getByRole('button', { name: /collapse sidebar/i });
    await fireEvent.click(collapseBtn);
    expect(storeProxy.sidebarCollapsed).toBe(true);
  });

  it('onnewnotebook callback is invoked on "New notebook" click', async () => {
    const onNew = vi.fn();
    render(NotebooksSidebar, { props: { onnewnotebook: onNew } });
    const newBtn = screen.getByRole('button', { name: /new notebook/i });
    await fireEvent.click(newBtn);
    expect(onNew).toHaveBeenCalledOnce();
  });

  it('trash entry calls openTrash', async () => {
    render(NotebooksSidebar);
    const trashBtn = screen.getByRole('button', { name: /trash/i });
    await fireEvent.click(trashBtn);
    await waitFor(() => expect(mockOpenTrash).toHaveBeenCalled());
  });

  it('does NOT render a "Sign out" button', () => {
    render(NotebooksSidebar);
    expect(screen.queryByRole('button', { name: /sign out/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/sign out/i)).not.toBeInTheDocument();
  });

  it('renders trash count badge when trashCount > 0', () => {
    storeProxy.trashCount = 4;
    render(NotebooksSidebar);
    expect(screen.getByText('4')).toBeInTheDocument();
  });

  it('shows empty state message when notebooks list is empty', () => {
    storeProxy.notebooks = [];
    render(NotebooksSidebar);
    expect(screen.getByText(/no notebooks yet/i)).toBeInTheDocument();
  });
});

describe('NotebooksSidebar (collapsed)', () => {
  beforeEach(() => {
    storeProxy.sidebarCollapsed = true;
  });

  it('renders expand button in collapsed state', () => {
    render(NotebooksSidebar);
    expect(screen.getByRole('button', { name: /expand sidebar/i })).toBeInTheDocument();
  });

  it('collapse toggle flips sidebarCollapsed to false (expanding)', async () => {
    render(NotebooksSidebar);
    const expandBtn = screen.getByRole('button', { name: /expand sidebar/i });
    await fireEvent.click(expandBtn);
    expect(storeProxy.sidebarCollapsed).toBe(false);
  });

  it('icon-only search trigger opens palette', async () => {
    render(NotebooksSidebar);
    const searchBtn = screen.getByRole('button', { name: /search notebooks/i });
    await fireEvent.click(searchBtn);
    expect(storeProxy.paletteOpen).toBe(true);
  });

  it('renders notebook color icons (collapsed NotebookRow) for each notebook', () => {
    storeProxy.notebooks = [makeNotebook('nb-1', 'Alpha'), makeNotebook('nb-2', 'Beta')];
    render(NotebooksSidebar);
    // Collapsed rows render as buttons with the notebook title as aria-label
    expect(screen.getByRole('button', { name: 'Alpha' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Beta' })).toBeInTheDocument();
  });

  it('does NOT render "Sign out" in collapsed mode', () => {
    render(NotebooksSidebar);
    expect(screen.queryByText(/sign out/i)).not.toBeInTheDocument();
  });
});
