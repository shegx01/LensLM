import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import EmbeddingConfigPanel from './EmbeddingConfigPanel.svelte';

function baseConfig(): AppConfig {
  return {
    theme: 'dark',
    accent: 'purple',
    models: [],
    endpoints: {},
    voices: { host: '', guest: '' },
    tts: { provider: '', api_key: '' },
    paths: { data_dir: '' },
    tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
    onboarding_complete: false,
    embedding_model: ''
  };
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('EmbeddingConfigPanel — persists embedding_model on install success', () => {
  it('writes the selected model to AppConfig.embedding_model after install resolves', async () => {
    let written: AppConfig | null = null;
    const oncheck = vi.fn().mockResolvedValue(undefined);
    const oncollapse = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
      if (cmd === 'install_embedding_model') {
        // Drive the progress channel to completion, then resolve.
        const ch = (args as { onProgress?: { onmessage?: (m: unknown) => void } }).onProgress;
        ch?.onmessage?.({ status: 'success', completed: 100, total: 100 });
        return null;
      }
    });

    render(EmbeddingConfigPanel, { props: { oncheck, oncollapse } });
    // Default selection is nomic-embed-text.
    await fireEvent.click(screen.getByRole('button', { name: /install nomic-embed-text/i }));

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).embedding_model).toBe('nomic-embed-text');
    await waitFor(() => expect(oncheck).toHaveBeenCalledOnce());
    expect(oncollapse).toHaveBeenCalledOnce();
  });
});

describe('EmbeddingConfigPanel — active model + switching', () => {
  it('gives the active model (config.embedding_model) the green "Active" treatment', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        return { ...baseConfig(), embedding_model: 'mxbai-embed-large' };
      }
      // Runtime reports the active model installed.
      if (cmd === 'detect_llm')
        return { reachable: true, version: 'Ollama 0.3.2', models: ['mxbai-embed-large:latest'] };
    });

    render(EmbeddingConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });

    // The active model gets the "Active" badge with green styling.
    const badge = await screen.findByLabelText(/active embedding model/i);
    expect(badge).toBeInTheDocument();
    expect(badge.className).toMatch(/text-green-primary/);

    // The active card's wrapper carries the green border (not the purple --primary).
    const card = badge.closest('div.rounded-lg');
    expect(card?.className).toMatch(/border-green-primary/);
    expect(card?.className).not.toMatch(/border-primary\b/);
  });

  it('"Set as active" on an installed model calls set_config with the new embedding_model', async () => {
    let written: AppConfig | null = null;
    const oncheck = vi.fn().mockResolvedValue(undefined);
    const oncollapse = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') {
        return { ...baseConfig(), embedding_model: 'nomic-embed-text' };
      }
      // Both nomic (active) and all-minilm (installed, not active) are present.
      if (cmd === 'detect_llm')
        return {
          reachable: true,
          version: 'Ollama 0.3.2',
          models: ['nomic-embed-text:latest', 'all-minilm:latest']
        };
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(EmbeddingConfigPanel, { props: { oncheck, oncollapse } });

    // The installed-not-active model exposes a "Set … as active" button
    // (its accessible name includes the model name via aria-label).
    const setActive = await screen.findByRole('button', { name: /as active/i });
    await fireEvent.click(setActive);

    await waitFor(() => expect(written).not.toBeNull());
    // No re-download: only embedding_model changes to the installed model.
    expect((written as unknown as AppConfig).embedding_model).toBe('all-minilm');
    await waitFor(() => expect(oncheck).toHaveBeenCalledOnce());
    expect(oncollapse).toHaveBeenCalledOnce();
  });
});
