import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import EmbeddingConfigPanel from './EmbeddingConfigPanel.svelte';

const baseConfig = baseAppConfig;

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('EmbeddingConfigPanel — Step 9 rework (backend-aware)', () => {
  it('shows the provider selector + helper note + re-embed warning', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingConfigPanel, {
      props: { oncheck: vi.fn().mockResolvedValue(undefined), oncollapse: vi.fn() }
    });

    expect(await screen.findByText(/select your local embeddings provider/i)).toBeInTheDocument();
    // On-device provider (labeled "On-device"; "· Apple GPU" on Apple Silicon,
    // issue #91). Tests run outside Tauri so the GPU signal is false.
    expect(screen.getByRole('radio', { name: 'On-device' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Ollama' })).toBeInTheDocument();
    expect(screen.getByText(/Ollama must be installed if chosen/i)).toBeInTheDocument();
    expect(screen.getByText(/permanently linked/i)).toBeInTheDocument();
  });

  it('lights up a fastembed card as Ready from on-disk cache detection (not Ollama tags)', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      // fastembed-cached weights for the default model — must light up even with
      // Ollama down (the regression the rework fixes: detection was Ollama-only).
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5'];
      if (cmd === 'list_ollama_models') return [];
    });

    render(EmbeddingConfigPanel, {
      props: { oncheck: vi.fn().mockResolvedValue(undefined), oncollapse: vi.fn() }
    });

    expect(await screen.findByLabelText(/nomic-embed-text-v1\.5 ready/i)).toBeInTheDocument();
  });

  it('fastembed Install warms the model then persists model + backend as the global default', async () => {
    let written: AppConfig | null = null;
    let warmed: string | null = null;
    const oncheck = vi.fn().mockResolvedValue(undefined);
    const oncollapse = vi.fn();
    let cached: string[] = [];
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return cached;
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'warm_fastembed_model') {
        warmed = (args as { model: string }).model;
        cached = [warmed]; // the next refresh sees it cached
        return null;
      }
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(EmbeddingConfigPanel, { props: { oncheck, oncollapse } });

    const install = await screen.findByRole('button', { name: /install nomic-embed-text-v1\.5/i });
    await fireEvent.click(install);

    await waitFor(() => expect(warmed).toBe('nomic-embed-text-v1.5'));
    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).embedding_model).toBe('nomic-embed-text-v1.5');
    expect((written as unknown as AppConfig).embedding_backend).toBe('fastembed');
    await waitFor(() => expect(oncheck).toHaveBeenCalledOnce());
    expect(oncollapse).toHaveBeenCalledOnce();
  });

  it('Ollama backend is detect-only: a missing model shows the pull hint, never pulls', async () => {
    const oncheck = vi.fn().mockResolvedValue(undefined);
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return []; // nothing detected
    });

    render(EmbeddingConfigPanel, { props: { oncheck, oncollapse: vi.fn() } });

    await fireEvent.click(await screen.findByRole('radio', { name: 'Ollama' }));

    // Detect-only affordances: a Refresh action + a pull hint (no Install button).
    // After switching to ollama, the first ollama model (embeddinggemma) is auto-selected.
    expect(
      await screen.findByRole('button', { name: /refresh ollama models/i })
    ).toBeInTheDocument();
    expect(screen.getByText(/ollama pull embeddinggemma/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /install nomic/i })).not.toBeInTheDocument();
  });
});
