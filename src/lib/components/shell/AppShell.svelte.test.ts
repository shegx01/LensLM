import { render, screen, fireEvent } from '@testing-library/svelte';
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
    // Right rail: M4 seam unchanged.
    expect(screen.getByText(/sources & studio/i)).toBeInTheDocument();
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

  it('grid uses the 88px collapsed left column when sidebarCollapsed is true', async () => {
    notebookStore.sidebarCollapsed = true;
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    expect(grid.className).toContain('grid-cols-[88px_1fr_320px]');
  });

  it('left aside is the floating rail that carries the hover handlers', () => {
    notebookStore.sidebarCollapsed = true;
    const { container } = render(AppShell);
    // The floating left rail is the aside carrying the m-2 gutter class — it is
    // the element wired with onpointerenter/leave (asserted behaviorally below).
    const leftAside = container.querySelector('aside.m-2') as HTMLElement;
    expect(leftAside).not.toBeNull();
  });

  it('hovering a collapsed rail expands the grid to 256px, leaving collapses back', async () => {
    notebookStore.sidebarCollapsed = true;
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    const leftAside = container.querySelector('aside.m-2') as HTMLElement;

    expect(grid.className).toContain('grid-cols-[88px_1fr_320px]');

    await fireEvent.pointerEnter(leftAside);
    expect(grid.className).toContain('grid-cols-[256px_1fr_320px]');
    // Persisted state is unchanged by hover.
    expect(notebookStore.sidebarCollapsed).toBe(true);

    await fireEvent.pointerLeave(leftAside);
    expect(grid.className).toContain('grid-cols-[88px_1fr_320px]');
    expect(notebookStore.sidebarCollapsed).toBe(true);
  });

  it('hover does not change width when sidebar is already expanded', async () => {
    notebookStore.sidebarCollapsed = false;
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    const leftAside = container.querySelector('aside.m-2') as HTMLElement;

    await fireEvent.pointerEnter(leftAside);
    expect(grid.className).toContain('grid-cols-[256px_1fr_320px]');
    await fireEvent.pointerLeave(leftAside);
    expect(grid.className).toContain('grid-cols-[256px_1fr_320px]');
  });
});
