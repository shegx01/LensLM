import { render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AppShell from './AppShell.svelte';
import { notebookStore, resetNotebookStore } from '$lib/notebooks/notebooks-state.svelte.js';

// AppShell mounts NotebooksSidebar (loads notebooks via the store) + CommandPalette
// + NotebookCreateDialog. Mock the IPC layer so the store's loadNotebooks() resolves
// without a Tauri host, and stub @tauri-apps/api/core (isTauri/invoke).
vi.mock('$lib/notebooks/ipc.js', () => ({
  listNotebooks: vi.fn().mockResolvedValue([]),
  createNotebook: vi.fn(),
  renameNotebook: vi.fn(),
  trashNotebook: vi.fn(),
  restoreNotebook: vi.fn(),
  listTrashed: vi.fn().mockResolvedValue([]),
  purgeNotebook: vi.fn()
}));

vi.mock('$lib/sources/ipc.js', () => ({
  listSources: vi.fn().mockResolvedValue([]),
  addTextSource: vi.fn(),
  addFileSource: vi.fn(),
  ingestSource: vi.fn(),
  setSourceSelected: vi.fn()
}));

vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => false,
  invoke: vi.fn()
}));

beforeEach(() => {
  vi.clearAllMocks();
  resetNotebookStore();
});

afterEach(() => {
  resetNotebookStore();
});

describe('AppShell.svelte', () => {
  it('renders the sidebar, the centre empty state, and the right rail', () => {
    render(AppShell);
    // Left rail: NotebooksSidebar renders the "Notebooks" section label + search trigger.
    expect(screen.getByText('Notebooks')).toBeInTheDocument();
    expect(screen.getByLabelText(/search notebooks/i)).toBeInTheDocument();
    // Centre: empty state (no active notebook).
    expect(screen.getByText('Your workspace')).toBeInTheDocument();
    expect(screen.getByText(/select or create a notebook/i)).toBeInTheDocument();
    // Right rail: SourcesRail now renders "Sources" heading.
    expect(screen.getByText('Sources')).toBeInTheDocument();
  });

  it('uses semantic landmarks for the regions', () => {
    const { container } = render(AppShell);
    // Two <aside> rails + one <main> workspace.
    expect(container.querySelectorAll('aside')).toHaveLength(2);
    expect(container.querySelector('main')).not.toBeNull();
  });

  it('grid uses the expanded left column when sidebar is not collapsed', () => {
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    expect(grid.className).toContain('grid-cols-[256px_1fr_320px]');
  });

  it('grid uses the 104px collapsed left column when sidebarCollapsed is true', async () => {
    notebookStore.sidebarCollapsed = true;
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    expect(grid.className).toContain('grid-cols-[104px_1fr_320px]');
  });

  it('collapsed rail shows the icon-only layout (no hover behaviour)', () => {
    notebookStore.sidebarCollapsed = true;
    render(AppShell);
    // Collapsed: icon rail → "Expand sidebar" affordance, no "Notebooks" label.
    expect(screen.getByRole('button', { name: /expand sidebar/i })).toBeInTheDocument();
    expect(screen.queryByText('Notebooks')).not.toBeInTheDocument();
  });

  it('the rail is always a normal-flow aside (no overlay / no hover flyout)', () => {
    notebookStore.sidebarCollapsed = true;
    const { container } = render(AppShell);
    const rail = container.querySelector('[data-sidebar-rail]') as HTMLElement;
    expect(rail.tagName).toBe('ASIDE');
    expect(rail.className).not.toContain('absolute');
    // No flyout overlay element exists at all anymore.
    expect(container.querySelector('[data-sidebar-flyout]')).toBeNull();
  });

  it('expanded sidebar renders the full layout in normal flow', () => {
    notebookStore.sidebarCollapsed = false;
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    expect(grid.className).toContain('grid-cols-[256px_1fr_320px]');
    expect(screen.getByText('Notebooks')).toBeInTheDocument();
  });
});
