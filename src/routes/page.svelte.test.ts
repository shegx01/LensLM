import { render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Page from './+page.svelte';
import { resetNotebookStore } from '$lib/notebooks/notebooks-state.svelte.js';

// +page.svelte renders <AppShell />, which mounts the notebooks sidebar + palette.
// Mock the IPC layer + Tauri core so the store loads cleanly without a host.
vi.mock('$lib/notebooks/ipc.js', () => ({
  listNotebooks: vi.fn().mockResolvedValue([]),
  createNotebook: vi.fn(),
  renameNotebook: vi.fn(),
  trashNotebook: vi.fn(),
  restoreNotebook: vi.fn(),
  listTrashed: vi.fn().mockResolvedValue([]),
  purgeNotebook: vi.fn(),
  touchNotebookActivity: vi.fn().mockResolvedValue(undefined)
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

describe('+page.svelte', () => {
  it('renders the app shell (not the old Hello World placeholder)', async () => {
    render(Page);
    // The shell replaced the Hello World landing.
    expect(screen.queryByRole('heading', { name: /hello world/i })).not.toBeInTheDocument();
    // Left rail sidebar and right rail are always present.
    expect(screen.getByText('Notebooks')).toBeInTheDocument();
    expect(screen.getByText('Sources')).toBeInTheDocument();
    // Empty state renders after loading completes (gated on !loading to prevent flash).
    await waitFor(() => {
      expect(screen.getByText('Your workspace')).toBeInTheDocument();
    });
  });
});
