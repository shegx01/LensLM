import { render, screen, waitFor, fireEvent } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CheckResult } from '$lib/onboarding/system-check.js';
import SystemCheck from './SystemCheck.svelte';
import { resetChatProvider } from '$lib/models/chat-provider.svelte.js';
import { baseAppConfig } from '$lib/test-fixtures.js';

// Both readiness gates passing. TTS is no longer a readiness gate (moved to Settings, #194).
const ALL_PASS: CheckResult[] = [
  {
    id: 'llm_runtime',
    label: 'LLM runtime',
    status: 'pass',
    detail: 'Local LLM reachable',
    action: 'configure'
  },
  {
    id: 'embedding_model',
    label: 'Embedding model',
    status: 'pass',
    detail: 'Embedding model installed',
    action: 'choose'
  }
];

/** Both gates failing (fresh machine). */
function allFail(): CheckResult[] {
  return ALL_PASS.map((r) => ({ ...r, status: 'fail' as const }));
}

/** Only embedding_model failing — the footer must stay blocked (embedding is required). */
function embeddingFail(): CheckResult[] {
  return ALL_PASS.map((r) =>
    r.id === 'embedding_model'
      ? {
          ...r,
          status: 'fail' as const,
          detail: 'No embedding model installed',
          action: 'choose' as const
        }
      : r
  );
}

/** Only llm_runtime failing — the LLM never blocks, so the footer must ENABLE. */
function llmFail(): CheckResult[] {
  return ALL_PASS.map((r) => (r.id === 'llm_runtime' ? { ...r, status: 'fail' as const } : r));
}

function skipButton(): HTMLElement {
  return screen.getByRole('button', { name: /skip for now/i });
}
function saveButton(): HTMLElement {
  return screen.getByRole('button', { name: /save & continue/i });
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  resetChatProvider();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('SystemCheck', () => {
  it('renders the title, the Local AI picker, and the embedding row', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
      if (cmd === 'list_ollama_models') return [];
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    expect(screen.getByText('System check')).toBeInTheDocument();
    // llm_runtime now renders the always-visible Local AI picker (not a "LLM runtime" row).
    await waitFor(() => expect(screen.getByText('Local AI')).toBeInTheDocument());
    expect(screen.getByText('Embedding model')).toBeInTheDocument();
    expect(screen.queryByText('Text-to-speech')).not.toBeInTheDocument();
  });

  it('enables BOTH footer buttons when embedding passes', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
      if (cmd === 'list_ollama_models') return [];
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() => expect(skipButton()).not.toBeDisabled());
    expect(saveButton()).not.toBeDisabled();
  });

  it('LLM never blocks: embedding-pass + LLM-fail still enables both buttons', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return llmFail();
      if (cmd === 'list_ollama_models') return [];
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('Embedding model')).toBeInTheDocument());
    expect(skipButton()).not.toBeDisabled();
    expect(saveButton()).not.toBeDisabled();
  });

  it('embedding is required: embedding-fail disables BOTH buttons (even with LLM pass)', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return embeddingFail();
      if (cmd === 'list_ollama_models') return [];
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('Embedding model')).toBeInTheDocument());
    expect(skipButton()).toBeDisabled();
    expect(saveButton()).toBeDisabled();
  });

  it('disables both buttons when all gates fail', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return allFail();
      if (cmd === 'list_ollama_models') return [];
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('Embedding model')).toBeInTheDocument());
    expect(skipButton()).toBeDisabled();
    expect(saveButton()).toBeDisabled();
  });

  it('advances (TTS no longer blocks) filtering out a failing text_to_speech row', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') {
        return [
          ...ALL_PASS,
          {
            id: 'text_to_speech',
            label: 'Text-to-speech',
            status: 'fail',
            detail: 'No text-to-speech engine configured',
            action: 'choose'
          }
        ];
      }
      if (cmd === 'list_ollama_models') return [];
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('Local AI')).toBeInTheDocument());
    expect(screen.queryByText('Text-to-speech')).not.toBeInTheDocument();
    expect(skipButton()).not.toBeDisabled();
  });

  it('Skip advances WITHOUT persisting', async () => {
    const onadvance = vi.fn();
    const setConfig = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'run_system_check') return ALL_PASS;
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
    });
    render(SystemCheck, { props: { onadvance } });
    await waitFor(() => expect(skipButton()).not.toBeDisabled());
    await fireEvent.click(skipButton());
    await waitFor(() => expect(onadvance).toHaveBeenCalledOnce());
    expect(setConfig).not.toHaveBeenCalled();
  });

  it('Save & continue persists the picked local model then advances', async () => {
    const onadvance = vi.fn();
    const setConfigs: Array<Record<string, unknown>> = [];
    mockIPC((cmd, args) => {
      if (cmd === 'run_system_check') return ALL_PASS;
      if (cmd === 'get_config') {
        return {
          theme: 'dark',
          accent: 'purple',
          models: [],
          endpoints: {},
          voices: { host: '', guest: '' },
          tts: { provider: '', api_key: '' },
          enrichment: { enabled: true, coref_strategy: 'none', cloud_consent: false },
          paths: { data_dir: '' },
          tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
          onboarding_complete: false,
          embedding_model: ''
        };
      }
      if (cmd === 'set_config') {
        setConfigs.push((args as { config: Record<string, unknown> }).config);
        return null;
      }
      if (cmd === 'list_ollama_models') return ['llama3.2:3b'];
      if (cmd === 'has_chat_provider') return true;
    });

    render(SystemCheck, { props: { onadvance } });
    const select = (await waitFor(() => {
      const el = screen.getByLabelText('Model', { selector: '#onboarding-llm-model' });
      expect(el.tagName).toBe('SELECT');
      return el;
    })) as HTMLSelectElement;
    await fireEvent.change(select, { target: { value: 'llama3.2:3b' } });

    await fireEvent.click(saveButton());

    await waitFor(() => expect(onadvance).toHaveBeenCalledOnce());
    expect(
      setConfigs.some(
        (c) =>
          Array.isArray(c.models) &&
          (c.models as Array<{ provider: string }>).some((m) => m.provider === 'ollama')
      )
    ).toBe(true);
  });

  it('shows an inline error and blocks both buttons when the check itself fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') throw new Error('probe boom');
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() =>
      expect(screen.getByText(/could not run the system check/i)).toBeInTheDocument()
    );
    expect(skipButton()).toBeDisabled();
    expect(saveButton()).toBeDisabled();
  });

  it('a failed re-check keeps the pickers mounted instead of dead-ending the screen', async () => {
    let calls = 0;
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') {
        calls += 1;
        if (calls === 1) return embeddingFail();
        throw new Error('transient probe failure'); // the post-persist re-check fails
      }
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5', 'all-minilm'];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
      if (cmd === 'set_config') return null;
      if (cmd === 'has_chat_provider') return true;
    });

    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await screen.findByText('Embedding model');
    // Wait for the cache probe so the pill is a ready (persistable) selection.
    await screen.findByLabelText(/nomic-embed-text-v1\.5 ready/i);

    // Reactive persist → onchange → re-check, which throws.
    await fireEvent.click(screen.getByRole('radio', { name: 'all-minilm' }));

    // The screen must NOT collapse into the full-page error; pickers stay mounted.
    await waitFor(() => expect(calls).toBeGreaterThanOrEqual(2));
    expect(screen.getByText('Embedding model')).toBeInTheDocument();
    expect(screen.queryByText(/could not run the system check/i)).not.toBeInTheDocument();
  });

  it('shows "0 of 2" when all gates fail and "2 of 2" when all pass', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return allFail();
      if (cmd === 'list_ollama_models') return [];
    });
    const { unmount } = render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('0 of 2 checks passed')).toBeInTheDocument());
    unmount();

    clearMocks();
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
      if (cmd === 'list_ollama_models') return [];
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('2 of 2 checks passed')).toBeInTheDocument());
  });

  it('outer <main> is a data-tauri-drag-region with the Card marked no-drag', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
      if (cmd === 'list_ollama_models') return [];
    });
    const { container } = render(SystemCheck, { props: { onadvance: vi.fn() } });
    const main = container.querySelector('main[data-tauri-drag-region]') as HTMLElement;
    expect(main).not.toBeNull();
    const noDrag = main.querySelector('[style*="-webkit-app-region: no-drag"]');
    expect(noDrag).not.toBeNull();
  });

  it('ThemeCycleButton wrapper carries -webkit-app-region: no-drag', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
      if (cmd === 'list_ollama_models') return [];
    });
    const { container } = render(SystemCheck, { props: { onadvance: vi.fn() } });
    const themeWrapper = container.querySelector('.absolute.top-4.right-4') as HTMLElement;
    expect(themeWrapper).not.toBeNull();
    expect(themeWrapper.getAttribute('style') ?? '').toContain('-webkit-app-region: no-drag');
  });
});
