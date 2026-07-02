import { render, screen, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { baseAppConfig } from '$lib/test-fixtures.js';
import { notebookStore, loadNotebooks } from '$lib/notebooks/index.js';
import type { NotebookSummary } from '$lib/notebooks/types.js';
import PreferencesShell from './PreferencesShell.svelte';
import NotebookSettingsSheet from './NotebookSettingsSheet.svelte';

function makeNotebook(id: string, title: string): NotebookSummary {
  return {
    id,
    title,
    description: null,
    focus_mode: null,
    embedding_model: null,
    embedding_backend: null,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    trashed_at: null,
    last_activity_at: null,
    source_count: 2
  };
}

let notebookRows: NotebookSummary[] = [];

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
  notebookRows = [];
  mockIPC((cmd) => {
    if (cmd === 'get_config') return baseAppConfig();
    if (cmd === 'fastembed_models_cached') return [];
    if (cmd === 'list_ollama_models') return [];
    if (cmd === 'get_notebook_embedding_model')
      return { model_id: 'nomic-embed-text-v1.5', dim: 768, backend: 'fastembed', status: 'none' };
    if (cmd === 'list_notebooks') return notebookRows;
  });
});

afterEach(() => {
  clearMocks();
  notebookStore.settingsOpen = false;
  notebookStore.notebookSettingsOpen = false;
  notebookStore.activeNotebookId = null;
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('PreferencesShell', () => {
  it('renders the Preferences left-nav with Embeddings live and the rest stubbed', async () => {
    notebookStore.settingsOpen = true;
    render(PreferencesShell);

    expect(await screen.findByText('Preferences')).toBeInTheDocument();
    // The full nav order is present.
    for (const label of [
      'General',
      'AI Model',
      'Embeddings',
      'Storage',
      'Privacy',
      'Shortcuts',
      'About'
    ]) {
      expect(screen.getByRole('button', { name: new RegExp(label, 'i') })).toBeInTheDocument();
    }
    // General and Embeddings are live (stub: false); the rest carry "Soon" + aria-disabled.
    const general = screen.getByRole('button', { name: /general/i });
    expect(general).not.toHaveAttribute('aria-disabled', 'true');
    // The live Embeddings section renders by default (default active = 'embeddings').
    expect(screen.getByRole('heading', { name: 'Embeddings' })).toBeInTheDocument();
  });

  it('is hidden when settingsOpen is false', () => {
    notebookStore.settingsOpen = false;
    render(PreferencesShell);
    expect(screen.queryByText('Preferences')).not.toBeInTheDocument();
  });
});

describe('NotebookSettingsSheet', () => {
  it('titles the sheet with the active notebook and mounts the Embeddings section', async () => {
    // Seed an active notebook in the store via the real load path.
    notebookRows = [makeNotebook('nb-x', 'Quarterly Review')];
    await loadNotebooks();
    notebookStore.activeNotebookId = 'nb-x';
    notebookStore.notebookSettingsOpen = true;

    render(NotebookSettingsSheet);

    expect(await screen.findByRole('heading', { name: 'Notebook settings' })).toBeInTheDocument();
    expect(screen.getByText('Quarterly Review')).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Embeddings' })).toBeInTheDocument()
    );
  });
});
