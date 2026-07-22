import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import OnboardingEmbeddingPicker from './OnboardingEmbeddingPicker.svelte';

const baseConfig = baseAppConfig;

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('OnboardingEmbeddingPicker — inline, both backends', () => {
  it('renders provider tabs, the focused default model, and quick-switch pills — no Choose/expand', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
    });

    render(OnboardingEmbeddingPicker, { props: { oncheck: vi.fn().mockResolvedValue(undefined) } });

    expect(await screen.findByRole('radio', { name: 'On-device' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Ollama' })).toBeInTheDocument();
    // Focused default model (full id in the focus card).
    expect(screen.getByText('nomic-embed-text-v1.5')).toBeInTheDocument();
    // Quick-switch pills use short labels.
    expect(screen.getByRole('radio', { name: 'all-minilm' })).toBeInTheDocument();
    // Inline: nothing to expand.
    expect(screen.queryByRole('button', { name: /choose/i })).not.toBeInTheDocument();
  });

  it('header reads "Needs a model" when the persisted default is not cached', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
    });

    render(OnboardingEmbeddingPicker, { props: { oncheck: vi.fn().mockResolvedValue(undefined) } });

    expect(await screen.findByText('Needs a model')).toBeInTheDocument();
    // fastembed default not cached → the focused model offers Install.
    expect(
      screen.getByRole('button', { name: /install nomic-embed-text-v1\.5/i })
    ).toBeInTheDocument();
  });

  it('lights up a fastembed model as Ready from on-disk cache (not Ollama tags)', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5'];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
    });

    render(OnboardingEmbeddingPicker, { props: { oncheck: vi.fn().mockResolvedValue(undefined) } });

    expect(await screen.findByLabelText(/nomic-embed-text-v1\.5 ready/i)).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: /install nomic-embed-text-v1\.5/i })
    ).not.toBeInTheDocument();
  });

  it('fastembed Install warms the model then persists model + backend as the global default', async () => {
    let written: AppConfig | null = null;
    let warmed: string | null = null;
    const oncheck = vi.fn().mockResolvedValue(undefined);
    let cached: string[] = [];
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return cached;
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
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

    render(OnboardingEmbeddingPicker, { props: { oncheck } });

    const install = await screen.findByRole('button', {
      name: /install nomic-embed-text-v1\.5/i
    });
    await fireEvent.click(install);

    await waitFor(() => expect(warmed).toBe('nomic-embed-text-v1.5'));
    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).embedding_model).toBe('nomic-embed-text-v1.5');
    expect((written as unknown as AppConfig).embedding_backend).toBe('fastembed');
    await waitFor(() => expect(oncheck).toHaveBeenCalled());
  });

  it('selecting an already-cached model persists it as the default (reactive, no Save button)', async () => {
    let written: AppConfig | null = null;
    const oncheck = vi.fn().mockResolvedValue(undefined);
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5', 'all-minilm'];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(OnboardingEmbeddingPicker, { props: { oncheck } });

    // Wait for the on-disk cache probe to land (the default shows Ready) before
    // clicking — otherwise the pill is present but its cached state isn't known yet.
    await screen.findByLabelText(/nomic-embed-text-v1\.5 ready/i);
    await fireEvent.click(screen.getByRole('radio', { name: 'all-minilm' }));

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).embedding_model).toBe('all-minilm');
    expect((written as unknown as AppConfig).embedding_backend).toBe('fastembed');
  });

  it('Ollama is detect-only: nothing detected shows Refresh + pull hint, never a download button', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return []; // Ollama down / nothing pulled
      if (cmd === 'gpu_accelerated_models') return [];
    });

    render(OnboardingEmbeddingPicker, { props: { oncheck: vi.fn().mockResolvedValue(undefined) } });

    await fireEvent.click(await screen.findByRole('radio', { name: 'Ollama' }));

    expect(
      await screen.findByRole('button', { name: /refresh ollama models/i })
    ).toBeInTheDocument();
    expect(screen.getByText(/ollama pull embeddinggemma/i)).toBeInTheDocument();
    expect(screen.getByText(/never downloads them/i)).toBeInTheDocument();
    // Detect-only: no install/download affordance for Ollama.
    expect(screen.queryByRole('button', { name: /install /i })).not.toBeInTheDocument();
  });

  it('persists Ollama backend when a detected Ollama model is selected', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return ['embeddinggemma'];
      if (cmd === 'gpu_accelerated_models') return [];
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(OnboardingEmbeddingPicker, {
      props: { oncheck: vi.fn().mockResolvedValue(undefined) }
    });

    await fireEvent.click(await screen.findByRole('radio', { name: 'Ollama' }));
    // embeddinggemma is the focused Ollama default; selecting its pill persists it.
    await fireEvent.click(await screen.findByRole('radio', { name: 'embeddinggemma' }));

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).embedding_model).toBe('embeddinggemma');
    expect((written as unknown as AppConfig).embedding_backend).toBe('ollama');
  });
});
