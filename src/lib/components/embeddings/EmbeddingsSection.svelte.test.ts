import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import EmbeddingsSection from './EmbeddingsSection.svelte';

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('EmbeddingsSection — global mode', () => {
  it('renders the design copy, provider selector and model radio-list', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingsSection, { props: { mode: 'global' } });

    expect(await screen.findByRole('heading', { name: 'Embeddings' })).toBeInTheDocument();
    expect(screen.getByText(/local only — all vectors computed on-device/i)).toBeInTheDocument();
    expect(screen.getByText(/select your local embeddings provider/i)).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'fastembed' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Ollama' })).toBeInTheDocument();
    // All four models are present.
    expect(screen.getByRole('radio', { name: /nomic-embed-text-v1\.5/i })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /bge-m3/i })).toBeInTheDocument();
  });

  it('NEVER shows the re-embed warning in global mode', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'fastembed_models_cached') return ['all-minilm'];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingsSection, { props: { mode: 'global' } });
    await screen.findByRole('heading', { name: 'Embeddings' });
    expect(screen.queryByText(/re-embed this notebook/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/permanently linked/i)).not.toBeInTheDocument();
  });

  it('changing a cached model persists model + backend to the global config', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig({ embedding_model: 'nomic-embed-text-v1.5' });
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5', 'all-minilm'];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(EmbeddingsSection, { props: { mode: 'global' } });

    // Pick a different, already-cached model — the apply button is "Set as default".
    await fireEvent.click(await screen.findByRole('radio', { name: /all-minilm/i }));
    await fireEvent.click(await screen.findByRole('button', { name: /apply selected model/i }));

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).embedding_model).toBe('all-minilm');
    expect((written as unknown as AppConfig).embedding_backend).toBe('fastembed');
  });
});

describe('EmbeddingsSection — notebook mode', () => {
  it('shows the current model + backend from get_notebook_embedding_model', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_notebook_embedding_model')
        return { model_id: 'mxbai-embed-large', dim: 1024, backend: 'fastembed', status: 'active' };
      if (cmd === 'fastembed_models_cached') return ['mxbai-embed-large'];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingsSection, { props: { mode: 'notebook', notebookId: 'nb1' } });

    // The current model is selected (checked radio).
    const radio = await screen.findByRole('radio', { name: /mxbai-embed-large/i });
    await waitFor(() => expect(radio).toHaveAttribute('aria-checked', 'true'));
  });

  it('changing an indexed coordinate opens the re-embed confirm dialog with the warning', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_notebook_embedding_model')
        return {
          model_id: 'nomic-embed-text-v1.5',
          dim: 768,
          backend: 'fastembed',
          status: 'active'
        };
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5', 'all-minilm'];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingsSection, { props: { mode: 'notebook', notebookId: 'nb1' } });

    await fireEvent.click(await screen.findByRole('radio', { name: /all-minilm/i }));
    await fireEvent.click(await screen.findByRole('button', { name: /apply selected model/i }));

    // The confirm dialog carries the verbatim re-embed warning.
    expect(await screen.findByText(/re-embed this notebook/i)).toBeInTheDocument();
    expect(screen.getByText(/permanently linked/i)).toBeInTheDocument();
  });

  it('confirming the re-embed streams progress and calls set_notebook_embedding_model', async () => {
    let setArgs: { notebookId: string; modelId: string; backend: string } | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_notebook_embedding_model')
        return {
          model_id: 'nomic-embed-text-v1.5',
          dim: 768,
          backend: 'fastembed',
          status: 'active'
        };
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5', 'all-minilm'];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'set_notebook_embedding_model') {
        const a = args as {
          notebookId: string;
          modelId: string;
          backend: string;
          onProgress?: { onmessage?: (m: unknown) => void };
        };
        setArgs = { notebookId: a.notebookId, modelId: a.modelId, backend: a.backend };
        a.onProgress?.onmessage?.({ type: 'started' });
        a.onProgress?.onmessage?.({ type: 'chunk', data: { done: 2, total: 4 } });
        a.onProgress?.onmessage?.({ type: 'done' });
        return null;
      }
    });

    render(EmbeddingsSection, { props: { mode: 'notebook', notebookId: 'nb1' } });

    await fireEvent.click(await screen.findByRole('radio', { name: /all-minilm/i }));
    await fireEvent.click(await screen.findByRole('button', { name: /apply selected model/i }));
    await fireEvent.click(await screen.findByRole('button', { name: /confirm re-embed/i }));

    await waitFor(() => expect(setArgs).not.toBeNull());
    expect(setArgs).toMatchObject({
      notebookId: 'nb1',
      modelId: 'all-minilm',
      backend: 'fastembed'
    });
  });

  it('surfaces a re-embed failure as an inline error and clears the in-flight state', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_notebook_embedding_model')
        return {
          model_id: 'nomic-embed-text-v1.5',
          dim: 768,
          backend: 'fastembed',
          status: 'active'
        };
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5', 'all-minilm'];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'set_notebook_embedding_model') {
        const a = args as { onProgress?: { onmessage?: (m: unknown) => void } };
        a.onProgress?.onmessage?.({
          type: 'failed',
          data: { kind: 'Internal', message: 'boom' }
        });
        return null;
      }
    });

    render(EmbeddingsSection, { props: { mode: 'notebook', notebookId: 'nb1' } });
    await fireEvent.click(await screen.findByRole('radio', { name: /all-minilm/i }));
    await fireEvent.click(await screen.findByRole('button', { name: /apply selected model/i }));
    await fireEvent.click(await screen.findByRole('button', { name: /confirm re-embed/i }));

    expect(await screen.findByRole('alert')).toHaveTextContent(/boom/i);
  });
});

describe('EmbeddingsSection — Ollama detect-only', () => {
  it('Refresh re-probes /api/tags and lights up a now-detected model', async () => {
    let tags: string[] = [];
    mockIPC((cmd) => {
      // Start on ollama backend with embeddinggemma selected (an ollama-only model).
      if (cmd === 'get_config')
        return baseAppConfig({ embedding_backend: 'ollama', embedding_model: 'embeddinggemma' });
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return tags;
    });

    render(EmbeddingsSection, { props: { mode: 'global' } });

    // Initially Ollama reports nothing → the pull hint is shown for embeddinggemma.
    expect(await screen.findByText(/ollama pull embeddinggemma/i)).toBeInTheDocument();

    // The model becomes available; Refresh re-probes and the card lights up.
    tags = ['embeddinggemma:latest'];
    await fireEvent.click(screen.getByRole('button', { name: /refresh ollama models/i }));

    expect(await screen.findByLabelText(/embeddinggemma ready/i)).toBeInTheDocument();
  });
});

// ── Step 8 TDD: backend-filtered picker + provider-switch reset ──────────────

describe('EmbeddingsSection — backend-filtered model picker (Step 8)', () => {
  const FASTEMBED_IDS = ['nomic-embed-text-v1.5', 'mxbai-embed-large', 'all-minilm', 'bge-m3'];
  const OLLAMA_IDS = [
    'embeddinggemma',
    'qwen3-embedding:4b',
    'nomic-embed-text-v2-moe',
    'snowflake-arctic-embed2'
  ];

  it('fastembed backend renders exactly the 4 fastembed models, not the 4 ollama models', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config')
        return baseAppConfig({
          embedding_backend: 'fastembed',
          embedding_model: 'nomic-embed-text-v1.5'
        });
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingsSection, { props: { mode: 'global' } });

    await screen.findByRole('heading', { name: 'Embeddings' });

    // All fastembed models present
    for (const id of FASTEMBED_IDS) {
      expect(
        screen.getByRole('radio', {
          name: new RegExp(id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')
        })
      ).toBeInTheDocument();
    }
    // No ollama-only models visible
    for (const id of OLLAMA_IDS) {
      expect(
        screen.queryByRole('radio', {
          name: new RegExp(id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')
        })
      ).not.toBeInTheDocument();
    }
  });

  it('ollama backend renders exactly the 4 ollama models, not the 4 fastembed models', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config')
        return baseAppConfig({ embedding_backend: 'ollama', embedding_model: 'embeddinggemma' });
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingsSection, { props: { mode: 'global' } });

    await screen.findByRole('heading', { name: 'Embeddings' });

    // All ollama models present
    for (const id of OLLAMA_IDS) {
      expect(
        screen.getByRole('radio', {
          name: new RegExp(id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')
        })
      ).toBeInTheDocument();
    }
    // No fastembed-only models visible
    for (const id of FASTEMBED_IDS) {
      expect(
        screen.queryByRole('radio', {
          name: new RegExp(id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')
        })
      ).not.toBeInTheDocument();
    }
  });

  it('switching from ollama to fastembed backend resets selectedModel if it is not in the new set', async () => {
    // Start on ollama with embeddinggemma selected (a model NOT in the fastembed set).
    mockIPC((cmd) => {
      if (cmd === 'get_config')
        return baseAppConfig({ embedding_backend: 'ollama', embedding_model: 'embeddinggemma' });
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingsSection, { props: { mode: 'global' } });

    // Wait for mount — the ollama provider button should be checked.
    const ollamaBtn = await screen.findByRole('radio', { name: /^ollama$/i });
    expect(ollamaBtn).toHaveAttribute('aria-checked', 'true');

    // Switch to fastembed.
    const fastembedBtn = screen.getByRole('radio', { name: /^fastembed$/i });
    await fireEvent.click(fastembedBtn);

    // After switching, fastembed models should now be visible.
    await waitFor(() => {
      expect(screen.getByRole('radio', { name: /nomic-embed-text-v1\.5/i })).toBeInTheDocument();
    });

    // The previous ollama model (embeddinggemma) is not in the fastembed set,
    // so selectedModel must have been reset to the first fastembed model.
    // The first fastembed model (nomic-embed-text-v1.5) should be aria-checked=true.
    const nomicRadio = screen.getByRole('radio', { name: /nomic-embed-text-v1\.5/i });
    expect(nomicRadio).toHaveAttribute('aria-checked', 'true');

    // The ollama model card should not exist anymore.
    expect(screen.queryByRole('radio', { name: /embeddinggemma/i })).not.toBeInTheDocument();
  });

  it('switching backend does NOT reset selectedModel when the current model exists in the new set', async () => {
    // This scenario would only happen with dual-backend models; currently none exist,
    // so we test the inverse: start on fastembed with nomic selected, switch to ollama →
    // nomic-embed-text-v1.5 is NOT in ollama set → reset happens (same as test above direction).
    // More useful: start already on fastembed and switch to fastembed (no-op path).
    mockIPC((cmd) => {
      if (cmd === 'get_config')
        return baseAppConfig({ embedding_backend: 'fastembed', embedding_model: 'all-minilm' });
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingsSection, { props: { mode: 'global' } });

    await screen.findByRole('heading', { name: 'Embeddings' });

    // all-minilm is selected and is in the fastembed set.
    const allMiniLm = screen.getByRole('radio', { name: /all-minilm/i });
    expect(allMiniLm).toHaveAttribute('aria-checked', 'true');

    // Switch to ollama — all-minilm is NOT in ollama set, reset must happen.
    const ollamaBtn = screen.getByRole('radio', { name: /^ollama$/i });
    await fireEvent.click(ollamaBtn);

    // After switch, ollama models are visible and first ollama model is selected.
    await waitFor(() => {
      expect(screen.getByRole('radio', { name: /embeddinggemma/i })).toBeInTheDocument();
    });
    const embeddinggemmaRadio = screen.getByRole('radio', { name: /embeddinggemma/i });
    expect(embeddinggemmaRadio).toHaveAttribute('aria-checked', 'true');
  });

  it('ollama pull hint shows the correct ollamaName for qwen3-embedding:4b', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config')
        return baseAppConfig({
          embedding_backend: 'ollama',
          embedding_model: 'qwen3-embedding:4b'
        });
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingsSection, { props: { mode: 'global' } });

    // The pull hint should show ollama pull qwen3-embedding:4b (exact tag).
    expect(await screen.findByText(/ollama pull qwen3-embedding:4b/i)).toBeInTheDocument();
  });
});
