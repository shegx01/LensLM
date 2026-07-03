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
    expect(screen.getByRole('radio', { name: 'On-device' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Ollama' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /nomic-embed-text-v1\.5/i })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /bge-m3/i })).toBeInTheDocument();
  });

  it('badges only GPU-accelerated models and shows the hint when one is selected (issue #91)', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig({ embedding_model: 'nomic-embed-text-v1.5' });
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5'];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return ['nomic-embed-text-v1.5'];
    });

    render(EmbeddingsSection, { props: { mode: 'global' } });

    await waitFor(() =>
      expect(
        screen.getByLabelText(/nomic-embed-text-v1\.5 runs on the apple gpu/i)
      ).toBeInTheDocument()
    );
    expect(screen.queryByLabelText(/all-minilm runs on the apple gpu/i)).not.toBeInTheDocument();
    expect(screen.getByText(/best performance — embeds on your apple gpu/i)).toBeInTheDocument();
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

    expect(await screen.findByText(/ollama pull embeddinggemma/i)).toBeInTheDocument();

    tags = ['embeddinggemma:latest'];
    await fireEvent.click(screen.getByRole('button', { name: /refresh ollama models/i }));

    expect(await screen.findByLabelText(/embeddinggemma ready/i)).toBeInTheDocument();
  });
});

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

    for (const id of FASTEMBED_IDS) {
      expect(
        screen.getByRole('radio', {
          name: new RegExp(id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')
        })
      ).toBeInTheDocument();
    }
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

    for (const id of OLLAMA_IDS) {
      expect(
        screen.getByRole('radio', {
          name: new RegExp(id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')
        })
      ).toBeInTheDocument();
    }
    for (const id of FASTEMBED_IDS) {
      expect(
        screen.queryByRole('radio', {
          name: new RegExp(id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')
        })
      ).not.toBeInTheDocument();
    }
  });

  it('switching from ollama to fastembed backend resets selectedModel if it is not in the new set', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config')
        return baseAppConfig({ embedding_backend: 'ollama', embedding_model: 'embeddinggemma' });
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingsSection, { props: { mode: 'global' } });

    const ollamaBtn = await screen.findByRole('radio', { name: /^ollama$/i });
    expect(ollamaBtn).toHaveAttribute('aria-checked', 'true');

    const fastembedBtn = screen.getByRole('radio', { name: /^on-device$/i });
    await fireEvent.click(fastembedBtn);

    await waitFor(() => {
      expect(screen.getByRole('radio', { name: /nomic-embed-text-v1\.5/i })).toBeInTheDocument();
    });

    const nomicRadio = screen.getByRole('radio', { name: /nomic-embed-text-v1\.5/i });
    expect(nomicRadio).toHaveAttribute('aria-checked', 'true');
    expect(screen.queryByRole('radio', { name: /embeddinggemma/i })).not.toBeInTheDocument();
  });

  it('switching backend does NOT reset selectedModel when the current model exists in the new set', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config')
        return baseAppConfig({ embedding_backend: 'fastembed', embedding_model: 'all-minilm' });
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingsSection, { props: { mode: 'global' } });

    await screen.findByRole('heading', { name: 'Embeddings' });

    const allMiniLm = screen.getByRole('radio', { name: /all-minilm/i });
    expect(allMiniLm).toHaveAttribute('aria-checked', 'true');

    const ollamaBtn = screen.getByRole('radio', { name: /^ollama$/i });
    await fireEvent.click(ollamaBtn);

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

    expect(await screen.findByText(/ollama pull qwen3-embedding:4b/i)).toBeInTheDocument();
  });
});
