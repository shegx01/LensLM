import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import type { CheckResult } from '$lib/onboarding/system-check.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import OnboardingEmbeddingPicker from './OnboardingEmbeddingPicker.svelte';

const baseConfig = baseAppConfig;

/** The authoritative embedding gate row the parent hands down. */
function embRow(status: 'pass' | 'fail'): CheckResult {
  return {
    id: 'embedding_model',
    label: 'Embedding model',
    status,
    detail: status === 'pass' ? 'Embedding model installed' : 'No embedding model installed',
    action: status === 'pass' ? null : 'choose'
  };
}

function renderPicker(status: 'pass' | 'fail', oncheck = vi.fn().mockResolvedValue(undefined)) {
  render(OnboardingEmbeddingPicker, { props: { result: embRow(status), oncheck } });
  return { oncheck };
}

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

    renderPicker('fail');

    expect(await screen.findByRole('radio', { name: 'On-device' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Ollama' })).toBeInTheDocument();
    // Focused default model (full id in the focus card).
    expect(screen.getByText('nomic-embed-text-v1.5')).toBeInTheDocument();
    // Quick-switch pills use the short label from the catalog.
    expect(screen.getByRole('radio', { name: 'all-minilm' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /choose/i })).not.toBeInTheDocument();
  });

  it('header tracks the authoritative gate, not the local probe', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return []; // nothing cached locally
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
    });

    // Gate says pass → header reads Ready even though the local cache is empty.
    renderPicker('pass');
    expect(await screen.findByText('Ready')).toBeInTheDocument();
    expect(screen.queryByText('Needs a model')).not.toBeInTheDocument();
  });

  it('header reads "Needs a model" when the gate is failing', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
    });

    renderPicker('fail');
    expect(await screen.findByText('Needs a model')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /install nomic-embed-text-v1\.5/i })
    ).toBeInTheDocument();
  });

  it('lights up a fastembed model as Ready from on-disk cache (independent of the gate)', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5'];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
    });

    renderPicker('fail');
    expect(await screen.findByLabelText(/nomic-embed-text-v1\.5 ready/i)).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: /install nomic-embed-text-v1\.5/i })
    ).not.toBeInTheDocument();
  });

  it('fastembed Install warms the model then persists model + backend as the global default', async () => {
    let written: AppConfig | null = null;
    let warmed: string | null = null;
    let cached: string[] = [];
    const { oncheck } = renderPickerWithMocks((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return cached;
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
      if (cmd === 'warm_fastembed_model') {
        warmed = (args as { model: string }).model;
        cached = [warmed];
        return null;
      }
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    const install = await screen.findByRole('button', { name: /install nomic-embed-text-v1\.5/i });
    await fireEvent.click(install);

    await waitFor(() => expect(warmed).toBe('nomic-embed-text-v1.5'));
    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).embedding_model).toBe('nomic-embed-text-v1.5');
    expect((written as unknown as AppConfig).embedding_backend).toBe('fastembed');
    await waitFor(() => expect(oncheck).toHaveBeenCalled());
  });

  it('surfaces an error and does NOT persist when the install fails', async () => {
    let written: AppConfig | null = null;
    renderPickerWithMocks((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
      if (cmd === 'warm_fastembed_model') throw new Error('download failed');
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    await fireEvent.click(
      await screen.findByRole('button', { name: /install nomic-embed-text-v1\.5/i })
    );

    expect(await screen.findByRole('alert')).toHaveTextContent(/download failed/i);
    expect(written).toBeNull();
  });

  it('selecting an already-cached model persists it as the default (reactive, no Save button)', async () => {
    let written: AppConfig | null = null;
    renderPickerWithMocks((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5', 'all-minilm'];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    // Wait for the cache probe to land before clicking (the default shows Ready).
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
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
    });

    renderPicker('fail');
    await fireEvent.click(await screen.findByRole('radio', { name: 'Ollama' }));

    expect(
      await screen.findByRole('button', { name: /refresh ollama models/i })
    ).toBeInTheDocument();
    expect(screen.getByText(/ollama pull embeddinggemma/i)).toBeInTheDocument();
    expect(screen.getByText(/never downloads them/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /install /i })).not.toBeInTheDocument();
  });

  it('Ollama Refresh re-runs the gate so the footer reflects new detection', async () => {
    const { oncheck } = renderPickerWithMocks((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
    }, 'fail');

    await fireEvent.click(await screen.findByRole('radio', { name: 'Ollama' }));
    oncheck.mockClear();
    await fireEvent.click(await screen.findByRole('button', { name: /refresh ollama models/i }));

    await waitFor(() => expect(oncheck).toHaveBeenCalled());
  });

  it('persists Ollama backend when a detected Ollama model is selected', async () => {
    let written: AppConfig | null = null;
    renderPickerWithMocks((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return ['embeddinggemma'];
      if (cmd === 'gpu_accelerated_models') return [];
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    await fireEvent.click(await screen.findByRole('radio', { name: 'Ollama' }));
    // Pills use the catalog short label; embeddinggemma → "gemma".
    await fireEvent.click(await screen.findByRole('radio', { name: 'gemma' }));

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).embedding_model).toBe('embeddinggemma');
    expect((written as unknown as AppConfig).embedding_backend).toBe('ollama');
  });

  it('restores a persisted Ollama default on mount — preselects the Ollama tab', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config')
        return baseConfig({ embedding_backend: 'ollama', embedding_model: 'embeddinggemma' });
      if (cmd === 'fastembed_models_cached') return [];
      if (cmd === 'list_ollama_models') return ['embeddinggemma'];
      if (cmd === 'gpu_accelerated_models') return [];
    });

    renderPicker('pass');

    await waitFor(() =>
      expect(screen.getByRole('radio', { name: 'Ollama' })).toHaveAttribute('aria-checked', 'true')
    );
    expect(screen.getByText('embeddinggemma')).toBeInTheDocument();
  });
});

/** Render with a custom IPC handler + capture the oncheck spy. */
function renderPickerWithMocks(
  handler: Parameters<typeof mockIPC>[0],
  status: 'pass' | 'fail' = 'fail'
) {
  const oncheck = vi.fn().mockResolvedValue(undefined);
  mockIPC(handler);
  render(OnboardingEmbeddingPicker, { props: { result: embRow(status), oncheck } });
  return { oncheck };
}
