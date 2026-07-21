import { render, screen, waitFor, fireEvent } from '@testing-library/svelte';
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
    graph_retrieval_enabled: null,
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
    const general = screen.getByRole('button', { name: /general/i });
    expect(general).not.toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByRole('heading', { name: 'Embeddings' })).toBeInTheDocument();
  });

  it('mounts TtsConfigPanel under the Text-to-Speech nav item (#194)', async () => {
    notebookStore.settingsOpen = true;
    render(PreferencesShell);

    const ttsNav = screen.getByRole('button', { name: /text-to-speech/i });
    expect(ttsNav).not.toHaveAttribute('aria-disabled', 'true');
    await fireEvent.click(ttsNav);

    // The panel's own heading (an h2, distinct from the nav button) proves it mounted.
    expect(await screen.findByRole('heading', { name: /text-to-speech/i })).toBeInTheDocument();
    expect(screen.getByText(/choose the voice engine/i)).toBeInTheDocument();
  });

  it('is hidden when settingsOpen is false', () => {
    notebookStore.settingsOpen = false;
    render(PreferencesShell);
    expect(screen.queryByText('Preferences')).not.toBeInTheDocument();
  });
});

describe('NotebookSettingsSheet', () => {
  it('titles the sheet with the active notebook and mounts the Embeddings section', async () => {
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
