import { render, screen } from '@testing-library/svelte';
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

describe('+page.svelte', () => {
  it('renders the app shell (not the old Hello World placeholder)', () => {
    render(Page);
    // The shell replaced the Hello World landing.
    expect(screen.queryByRole('heading', { name: /hello world/i })).not.toBeInTheDocument();
    // Left rail sidebar, centre workspace, right rail are all present.
    expect(screen.getByText('Notebooks')).toBeInTheDocument();
    expect(screen.getByText('Sources')).toBeInTheDocument();
    expect(screen.getByText('Your workspace')).toBeInTheDocument();
  });
});
