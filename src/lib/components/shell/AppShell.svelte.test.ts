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

  it('collapsed rail renders the floating flyout panel that carries the hover handlers', () => {
    notebookStore.sidebarCollapsed = true;
    const { container } = render(AppShell);
    // The collapsed rail is an absolutely-positioned overlay panel (z-50) inside a
    // reserved relative cell — it floats over the centre without reflowing it.
    const flyout = container.querySelector('[data-sidebar-flyout]') as HTMLElement;
    expect(flyout).not.toBeNull();
    expect(flyout.className).toContain('absolute');
    expect(flyout.className).toContain('z-50');
  });

  it('collapsed rail shows the icon layout, hover swaps to the expanded layout', async () => {
    notebookStore.sidebarCollapsed = true;
    const { container } = render(AppShell);
    const flyout = container.querySelector('[data-sidebar-flyout]') as HTMLElement;

    // Collapsed: icon rail → "Expand sidebar" affordance, no "Notebooks" label.
    expect(screen.getByRole('button', { name: /expand sidebar/i })).toBeInTheDocument();
    expect(screen.queryByText('Notebooks')).not.toBeInTheDocument();

    await fireEvent.pointerEnter(flyout);
    // Expanded layout swaps in: the "Notebooks" section label + collapse button.
    expect(screen.getByText('Notebooks')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /collapse sidebar/i })).toBeInTheDocument();

    await fireEvent.pointerLeave(flyout);
    expect(screen.getByRole('button', { name: /expand sidebar/i })).toBeInTheDocument();
    expect(screen.queryByText('Notebooks')).not.toBeInTheDocument();
  });

  it('hovering the collapsed rail does NOT change the grid column width (no reflow)', async () => {
    notebookStore.sidebarCollapsed = true;
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    const flyout = container.querySelector('[data-sidebar-flyout]') as HTMLElement;

    expect(grid.className).toContain('grid-cols-[88px_1fr_320px]');

    await fireEvent.pointerEnter(flyout);
    // The grid column width is UNCHANGED on hover — only the flyout panel widens,
    // floating over the centre content.
    expect(grid.className).toContain('grid-cols-[88px_1fr_320px]');
    expect(grid.className).not.toContain('grid-cols-[256px_1fr_320px]');
    // Persisted state is unchanged by hover.
    expect(notebookStore.sidebarCollapsed).toBe(true);

    await fireEvent.pointerLeave(flyout);
    expect(grid.className).toContain('grid-cols-[88px_1fr_320px]');
    expect(notebookStore.sidebarCollapsed).toBe(true);
  });

  it('the collapsed flyout panel widens on hover (88→256 overlay) while the grid stays 88px', async () => {
    notebookStore.sidebarCollapsed = true;
    const { container } = render(AppShell);
    const flyout = container.querySelector('[data-sidebar-flyout]') as HTMLElement;

    // Panel is the 72px icon rail (88px cell − 2×8px gutter) when not hovered.
    expect(flyout.className).toContain('w-[72px]');

    await fireEvent.pointerEnter(flyout);
    // On hover it widens to the 240px expanded panel (256px layout − gutter).
    expect(flyout.className).toContain('w-[240px]');

    await fireEvent.pointerLeave(flyout);
    expect(flyout.className).toContain('w-[72px]');
  });

  it('expanded sidebar renders in normal flow with no overlay flyout', () => {
    notebookStore.sidebarCollapsed = false;
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    expect(grid.className).toContain('grid-cols-[256px_1fr_320px]');
    // No absolute flyout overlay in the expanded state.
    expect(container.querySelector('[data-sidebar-flyout]')).toBeNull();
    // The rail is a normal-flow aside.
    const rail = container.querySelector('[data-sidebar-rail]') as HTMLElement;
    expect(rail.tagName).toBe('ASIDE');
    expect(rail.className).not.toContain('absolute');
  });
});
