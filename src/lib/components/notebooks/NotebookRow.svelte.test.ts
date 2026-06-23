// NotebookRow component tests.
//
// Covers: renders title/count/time, click selects, dblclick enters rename,
// Enter commits, Esc cancels, trash icon calls trashNotebookAction.
// Mocks the $lib/notebooks module so no Tauri IPC occurs.

import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// ── Hoisted mock refs ─────────────────────────────────────────────────────────
const { mockSelectNotebook, mockRenameAction, mockTrashAction, mockResetStore } = vi.hoisted(
  () => ({
    mockSelectNotebook: vi.fn(),
    mockRenameAction: vi.fn().mockResolvedValue(undefined),
    mockTrashAction: vi.fn().mockResolvedValue(undefined),
    mockResetStore: vi.fn()
  })
);

// Mock the entire notebooks barrel so no real IPC / store is used
vi.mock('$lib/notebooks/index.js', () => ({
  notebookStore: { activeNotebookId: null, sidebarCollapsed: false },
  selectNotebook: mockSelectNotebook,
  renameNotebookAction: mockRenameAction,
  trashNotebookAction: mockTrashAction,
  resetNotebookStore: mockResetStore,
  // Passthrough utilities — use real implementations
  notebookAccentClass: (id: string) => `nb-purple`,
  formatRelativeTime: (_iso: string) => '1h ago'
}));

import NotebookRow from './NotebookRow.svelte';
import type { NotebookSummary } from '$lib/notebooks/types.js';

// ── Fixtures ──────────────────────────────────────────────────────────────────

function makeNotebook(overrides?: Partial<NotebookSummary>): NotebookSummary {
  return {
    id: 'nb-001',
    title: 'Test Notebook',
    description: null,
    focus_mode: null,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-06-01T00:00:00Z',
    trashed_at: null,
    source_count: 3,
    ...overrides
  };
}

// ── Setup / teardown ─────────────────────────────────────────────────────────

beforeEach(() => {
  mockSelectNotebook.mockClear();
  mockRenameAction.mockClear();
  mockTrashAction.mockClear();
});

afterEach(() => {
  vi.clearAllMocks();
});

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('NotebookRow (expanded)', () => {
  it('renders notebook title', () => {
    render(NotebookRow, { props: { notebook: makeNotebook(), active: false } });
    expect(screen.getByText('Test Notebook')).toBeInTheDocument();
  });

  it('renders source count (plural)', () => {
    render(NotebookRow, { props: { notebook: makeNotebook({ source_count: 3 }), active: false } });
    expect(screen.getByText(/3 sources/i)).toBeInTheDocument();
  });

  it('renders source count (singular)', () => {
    render(NotebookRow, { props: { notebook: makeNotebook({ source_count: 1 }), active: false } });
    expect(screen.getByText(/1 source/i)).toBeInTheDocument();
  });

  it('renders relative time', () => {
    render(NotebookRow, { props: { notebook: makeNotebook(), active: false } });
    expect(screen.getByText(/1h ago/i)).toBeInTheDocument();
  });

  it('clicking the row calls selectNotebook with the notebook id', async () => {
    render(NotebookRow, { props: { notebook: makeNotebook(), active: false } });
    const row = screen.getByRole('button', { name: /^Test Notebook$/i });
    await fireEvent.click(row);
    expect(mockSelectNotebook).toHaveBeenCalledWith('nb-001');
  });

  it('double-clicking the title reveals an input for renaming', async () => {
    render(NotebookRow, { props: { notebook: makeNotebook(), active: false } });
    const title = screen.getByText('Test Notebook');
    await fireEvent.dblClick(title);
    const input = screen.getByRole('textbox', { name: /rename notebook/i });
    expect(input).toBeInTheDocument();
    expect((input as HTMLInputElement).value).toBe('Test Notebook');
  });

  it('Enter in rename input calls renameNotebookAction and hides input', async () => {
    render(NotebookRow, { props: { notebook: makeNotebook(), active: false } });
    const title = screen.getByText('Test Notebook');
    await fireEvent.dblClick(title);
    const input = screen.getByRole('textbox', { name: /rename notebook/i });
    await fireEvent.input(input, { target: { value: 'Renamed Notebook' } });
    await fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() =>
      expect(mockRenameAction).toHaveBeenCalledWith('nb-001', 'Renamed Notebook')
    );
    await waitFor(() =>
      expect(screen.queryByRole('textbox', { name: /rename notebook/i })).not.toBeInTheDocument()
    );
  });

  it('Esc in rename input cancels without calling renameNotebookAction', async () => {
    render(NotebookRow, { props: { notebook: makeNotebook(), active: false } });
    const title = screen.getByText('Test Notebook');
    await fireEvent.dblClick(title);
    const input = screen.getByRole('textbox', { name: /rename notebook/i });
    await fireEvent.input(input, { target: { value: 'Changed Name' } });
    await fireEvent.keyDown(input, { key: 'Escape' });
    expect(mockRenameAction).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(screen.queryByRole('textbox', { name: /rename notebook/i })).not.toBeInTheDocument()
    );
  });

  it('clicking the trash icon calls trashNotebookAction', async () => {
    render(NotebookRow, { props: { notebook: makeNotebook(), active: false } });
    const trashBtn = screen.getByRole('button', { name: /delete Test Notebook/i });
    await fireEvent.click(trashBtn);
    await waitFor(() => expect(mockTrashAction).toHaveBeenCalledWith('nb-001'));
  });
});

describe('NotebookRow (collapsed)', () => {
  it('renders as a compact button with the notebook title as aria-label', () => {
    render(NotebookRow, { props: { notebook: makeNotebook(), active: false, collapsed: true } });
    expect(screen.getByRole('button', { name: /Test Notebook/i })).toBeInTheDocument();
  });

  it('clicking collapsed row calls selectNotebook', async () => {
    render(NotebookRow, { props: { notebook: makeNotebook(), active: false, collapsed: true } });
    const btn = screen.getByRole('button', { name: /Test Notebook/i });
    await fireEvent.click(btn);
    expect(mockSelectNotebook).toHaveBeenCalledWith('nb-001');
  });
});
