// NotebooksSidebar component tests.
//
// Covers: renders rows with title/count/time, search trigger opens palette,
// collapse toggle flips sidebar state, "Sign out" is NOT present.
// Mocks the $lib/notebooks module, mode-watcher, and $lib/theme to isolate the
// component (the brand-row theme-cycle button is inlined and uses these).

import { render, screen, fireEvent, waitFor, within } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { storeProxy, mockOpenTrash, mockSelectNotebook, mockResetStore } = vi.hoisted(() => {
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
  notebookColorClass: () => 'nb-purple',
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

// Mock mode-watcher + $lib/theme so the inlined brand-row theme-cycle button
// is deterministic and never performs IPC/localStorage writes in tests.
const mockUserPrefersMode = vi.hoisted(() => ({
  current: 'system' as 'light' | 'dark' | 'system'
}));
const mockSetMode = vi.hoisted(() => vi.fn());
vi.mock('mode-watcher', () => ({
  userPrefersMode: mockUserPrefersMode,
  setMode: mockSetMode
}));
vi.mock('$lib/theme/index.js', () => ({
  persistTheme: vi.fn()
}));

import NotebooksSidebar from './NotebooksSidebar.svelte';
import type { NotebookSummary } from '$lib/notebooks/types.js';

function makeNotebook(id: string, title: string, sourceCount = 2): NotebookSummary {
  return {
    id,
    title,
    description: null,
    focus_mode: null,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-06-01T00:00:00Z',
    trashed_at: null,
    last_activity_at: null,
    graph_retrieval_enabled: null,
    embedding_model: null,
    embedding_backend: null,
    source_count: sourceCount
  };
}

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
  // Collapsed layout is driven by the `collapsed` prop (AppShell supplies the
  // effective state; the rail is button-only, no hover). Pass `collapsed={true}`.
  it('renders expand button in collapsed state', () => {
    render(NotebooksSidebar, { props: { collapsed: true } });
    expect(screen.getByRole('button', { name: 'Expand sidebar' })).toBeInTheDocument();
  });

  it('collapse toggle flips sidebarCollapsed to false (expanding)', async () => {
    storeProxy.sidebarCollapsed = true;
    render(NotebooksSidebar, { props: { collapsed: true } });
    const expandBtn = screen.getByRole('button', { name: 'Expand sidebar' });
    await fireEvent.click(expandBtn);
    expect(storeProxy.sidebarCollapsed).toBe(false);
  });

  it('icon-only search trigger opens palette', async () => {
    render(NotebooksSidebar, { props: { collapsed: true } });
    const searchBtn = screen.getByRole('button', { name: /search notebooks/i });
    await fireEvent.click(searchBtn);
    expect(storeProxy.paletteOpen).toBe(true);
  });

  it('renders notebook color icons (collapsed NotebookRow) for each notebook', () => {
    storeProxy.notebooks = [makeNotebook('nb-1', 'Alpha'), makeNotebook('nb-2', 'Beta')];
    render(NotebooksSidebar, { props: { collapsed: true } });
    expect(screen.getByRole('button', { name: 'Alpha' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Beta' })).toBeInTheDocument();
  });

  it('does NOT render "Sign out" in collapsed mode', () => {
    render(NotebooksSidebar, { props: { collapsed: true } });
    expect(screen.queryByText(/sign out/i)).not.toBeInTheDocument();
  });

  it('collapsed footer exposes Settings, theme, Embeddings Inspector (DEV), and account controls', () => {
    // The collapsed footer replaces the AccountFooter popup with an icon stack.
    // Scope the queries to the footer so the (display:none) brand theme button
    // can't create a duplicate match.
    const { container } = render(NotebooksSidebar, {
      props: { collapsed: true, userName: 'Jamie Doe' }
    });
    const footer = container.querySelector('[data-collapsed-footer]') as HTMLElement;
    expect(footer).not.toBeNull();
    const scoped = within(footer);
    expect(scoped.getByRole('button', { name: /^settings$/i })).toBeInTheDocument();
    // ThemeCycleButton's accessible name always begins with "Theme: …".
    expect(scoped.getByRole('button', { name: /^theme:/i })).toBeInTheDocument();
    // Embeddings Inspector is DEV-only; vitest runs with import.meta.env.DEV true.
    expect(scoped.getByRole('button', { name: /embeddings inspector/i })).toBeInTheDocument();
    expect(scoped.getByRole('button', { name: /account:/i })).toBeInTheDocument();
  });

  it('clicking the collapsed account avatar expands the rail', async () => {
    storeProxy.sidebarCollapsed = true;
    const { container } = render(NotebooksSidebar, {
      props: { collapsed: true, userName: 'Jamie Doe' }
    });
    const footer = container.querySelector('[data-collapsed-footer]') as HTMLElement;
    await fireEvent.click(within(footer).getByRole('button', { name: /account:/i }));
    expect(storeProxy.sidebarCollapsed).toBe(false);
  });
});

describe('NotebooksSidebar (active row)', () => {
  // happy-dom cannot measure layout (offsetTop = 0), so the sliding-indicator
  // POSITION is verified visually, not here. Instead assert the active row's
  // aria-pressed state and that clicking a row selects it.
  it('marks the active notebook row with aria-pressed=true', () => {
    storeProxy.notebooks = [makeNotebook('nb-1', 'Alpha'), makeNotebook('nb-2', 'Beta')];
    storeProxy.activeNotebookId = 'nb-2';
    render(NotebooksSidebar);
    expect(screen.getByRole('button', { name: 'Beta' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'Alpha' })).toHaveAttribute('aria-pressed', 'false');
  });

  it('clicking a notebook row selects it', async () => {
    storeProxy.notebooks = [makeNotebook('nb-1', 'Alpha')];
    render(NotebooksSidebar);
    await fireEvent.click(screen.getByRole('button', { name: 'Alpha' }));
    expect(mockSelectNotebook).toHaveBeenCalledWith('nb-1');
  });
});

describe('NotebooksSidebar (collapsed prop fallback)', () => {
  // Without the `collapsed` prop, layout falls back to the store's
  // `sidebarCollapsed` — preserves existing direct usage.
  it('falls back to store sidebarCollapsed when collapsed prop is omitted', () => {
    storeProxy.sidebarCollapsed = true;
    render(NotebooksSidebar);
    expect(screen.getByRole('button', { name: 'Expand sidebar' })).toBeInTheDocument();
  });

  it('store-expanded layout when prop omitted and sidebarCollapsed is false', () => {
    storeProxy.sidebarCollapsed = false;
    render(NotebooksSidebar);
    expect(screen.getByText('Notebooks')).toBeInTheDocument();
  });

  it('prop overrides the store (collapsed=false while store is collapsed)', () => {
    storeProxy.sidebarCollapsed = true;
    render(NotebooksSidebar, { props: { collapsed: false } });
    expect(screen.getByText('Notebooks')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /collapse sidebar/i })).toBeInTheDocument();
  });
});
