import { render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AppShell from './AppShell.svelte';
import { resetNotebookStore } from '$lib/notebooks/notebooks-state.svelte.js';

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
});
