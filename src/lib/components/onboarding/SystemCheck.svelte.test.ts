import { render, screen, waitFor, fireEvent } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CheckResult } from '$lib/onboarding/system-check.js';
import SystemCheck from './SystemCheck.svelte';
import { resetChatProvider } from '$lib/models/chat-provider.svelte.js';
import { baseAppConfig } from '$lib/test-fixtures.js';

// SystemCheck now renders ONE gate per screen: `gate="llm"` (Local AI, first step,
// never blocks) and `gate="embedding"` (embedding model, REQUIRED, second step).
// TTS is no longer a readiness gate (moved to Settings, #194).
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

/** embedding_model failing — the Embedding step's Continue must stay disabled. */
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

function skipButton(): HTMLElement {
  return screen.getByRole('button', { name: /skip for now/i });
}
function continueButton(): HTMLElement {
  return screen.getByRole('button', { name: /continue/i });
}
function backButton(): HTMLElement {
  return screen.getByRole('button', { name: /back/i });
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  resetChatProvider();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('SystemCheck — Local AI step (gate="llm")', () => {
  it('renders the Local AI picker with Skip + Continue and no Back', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_ollama_models') return [];
    });
    render(SystemCheck, { props: { gate: 'llm', onadvance: vi.fn() } });
    expect(screen.getByText('System check')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Local AI')).toBeInTheDocument());
    // The embedding gate is a separate step — its picker must NOT be on this screen.
    expect(screen.queryByText('Embedding model')).not.toBeInTheDocument();
    expect(skipButton()).toBeInTheDocument();
    expect(continueButton()).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /back/i })).not.toBeInTheDocument();
  });

  it('is self-contained: never calls run_system_check', async () => {
    const runCheck = vi.fn();
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') {
        runCheck();
        return ALL_PASS;
      }
      if (cmd === 'list_ollama_models') return [];
    });
    render(SystemCheck, { props: { gate: 'llm', onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('Local AI')).toBeInTheDocument());
    expect(runCheck).not.toHaveBeenCalled();
  });

  it('both buttons are enabled even when the local LLM is unreachable', async () => {
    // detect_llm defaults to unreachable; the LLM never blocks onboarding.
    mockIPC((cmd) => {
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'detect_llm') return { reachable: false, version: null, models: [] };
    });
    render(SystemCheck, { props: { gate: 'llm', onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('Local AI')).toBeInTheDocument());
    expect(skipButton()).not.toBeDisabled();
    expect(continueButton()).not.toBeDisabled();
  });

  it('Skip advances WITHOUT persisting', async () => {
    const onadvance = vi.fn();
    const setConfig = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
    });
    render(SystemCheck, { props: { gate: 'llm', onadvance } });
    await waitFor(() => expect(skipButton()).not.toBeDisabled());
    await fireEvent.click(skipButton());
    await waitFor(() => expect(onadvance).toHaveBeenCalledOnce());
    expect(setConfig).not.toHaveBeenCalled();
  });

  it('Continue persists the picked local model then advances', async () => {
    const onadvance = vi.fn();
    const setConfigs: Array<Record<string, unknown>> = [];
    mockIPC((cmd, args) => {
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

    render(SystemCheck, { props: { gate: 'llm', onadvance } });
    const select = (await waitFor(() => {
      const el = screen.getByLabelText('Model', { selector: '#onboarding-llm-model' });
      expect(el.tagName).toBe('SELECT');
      return el;
    })) as HTMLSelectElement;
    await fireEvent.change(select, { target: { value: 'llama3.2:3b' } });

    await fireEvent.click(continueButton());

    await waitFor(() => expect(onadvance).toHaveBeenCalledOnce());
    expect(
      setConfigs.some(
        (c) =>
          Array.isArray(c.models) &&
          (c.models as Array<{ provider: string }>).some((m) => m.provider === 'ollama')
      )
    ).toBe(true);
  });
});

describe('SystemCheck — Embedding step (gate="embedding")', () => {
  /** The full set of IPC the embedding picker needs to initialise. */
  function embeddingMocks(checks: CheckResult[], calls?: { n: number }) {
    return (cmd: string) => {
      if (cmd === 'run_system_check') {
        if (calls) calls.n += 1;
        return checks;
      }
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5', 'all-minilm'];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
      if (cmd === 'set_config') return null;
      if (cmd === 'has_chat_provider') return true;
    };
  }

  it('renders the embedding row with Back + Continue and no Skip', async () => {
    mockIPC(embeddingMocks(ALL_PASS));
    render(SystemCheck, { props: { gate: 'embedding', onadvance: vi.fn(), onback: vi.fn() } });
    await waitFor(() => expect(screen.getByText('Embedding model')).toBeInTheDocument());
    // The Local AI picker belongs to the previous step.
    expect(screen.queryByText('Local AI')).not.toBeInTheDocument();
    expect(backButton()).toBeInTheDocument();
    expect(continueButton()).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /skip for now/i })).not.toBeInTheDocument();
  });

  it('Continue is enabled once the embedding gate passes', async () => {
    mockIPC(embeddingMocks(ALL_PASS));
    render(SystemCheck, { props: { gate: 'embedding', onadvance: vi.fn(), onback: vi.fn() } });
    await waitFor(() => expect(continueButton()).not.toBeDisabled());
  });

  it('embedding required: a failing gate keeps Continue disabled but Back usable', async () => {
    const onback = vi.fn();
    mockIPC(embeddingMocks(embeddingFail()));
    render(SystemCheck, { props: { gate: 'embedding', onadvance: vi.fn(), onback } });
    await waitFor(() => expect(screen.getByText('Embedding model')).toBeInTheDocument());
    expect(continueButton()).toBeDisabled();
    expect(backButton()).not.toBeDisabled();
    await fireEvent.click(backButton());
    expect(onback).toHaveBeenCalledOnce();
  });

  it('Continue advances once ready (embedding picker persists on its own)', async () => {
    const onadvance = vi.fn();
    mockIPC(embeddingMocks(ALL_PASS));
    render(SystemCheck, { props: { gate: 'embedding', onadvance, onback: vi.fn() } });
    await waitFor(() => expect(continueButton()).not.toBeDisabled());
    await fireEvent.click(continueButton());
    await waitFor(() => expect(onadvance).toHaveBeenCalledOnce());
  });

  it('shows an inline error and disables Continue when the check itself fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') throw new Error('probe boom');
    });
    render(SystemCheck, { props: { gate: 'embedding', onadvance: vi.fn(), onback: vi.fn() } });
    await waitFor(() =>
      expect(screen.getByText(/could not run the system check/i)).toBeInTheDocument()
    );
    expect(continueButton()).toBeDisabled();
    // Back must still let the user retreat — a check error never traps them.
    expect(backButton()).not.toBeDisabled();
  });

  it('a failed re-check keeps the picker mounted instead of dead-ending the screen', async () => {
    let n = 0;
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') {
        n += 1;
        if (n === 1) return embeddingFail();
        throw new Error('transient probe failure'); // the post-persist re-check fails
      }
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5', 'all-minilm'];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
      if (cmd === 'set_config') return null;
      if (cmd === 'has_chat_provider') return true;
    });

    render(SystemCheck, { props: { gate: 'embedding', onadvance: vi.fn(), onback: vi.fn() } });
    await screen.findByText('Embedding model');
    // Wait for the cache probe so the pill is a ready (persistable) selection.
    await screen.findByLabelText(/nomic-embed-text-v1\.5 ready/i);

    // Reactive persist → onchange → re-check, which throws.
    await fireEvent.click(screen.getByRole('radio', { name: 'all-minilm' }));

    // The screen must NOT collapse into the full-page error; the picker stays mounted.
    await waitFor(() => expect(n).toBeGreaterThanOrEqual(2));
    expect(screen.getByText('Embedding model')).toBeInTheDocument();
    expect(screen.queryByText(/could not run the system check/i)).not.toBeInTheDocument();
  });
});

describe('SystemCheck — chrome (shared)', () => {
  it('outer <main> is a data-tauri-drag-region with the Card marked no-drag', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_ollama_models') return [];
    });
    const { container } = render(SystemCheck, { props: { gate: 'llm', onadvance: vi.fn() } });
    const main = container.querySelector('main[data-tauri-drag-region]') as HTMLElement;
    expect(main).not.toBeNull();
    const noDrag = main.querySelector('[style*="-webkit-app-region: no-drag"]');
    expect(noDrag).not.toBeNull();
  });

  it('ThemeCycleButton wrapper carries -webkit-app-region: no-drag', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_ollama_models') return [];
    });
    const { container } = render(SystemCheck, { props: { gate: 'llm', onadvance: vi.fn() } });
    const themeWrapper = container.querySelector('.absolute.top-4.right-4') as HTMLElement;
    expect(themeWrapper).not.toBeNull();
    expect(themeWrapper.getAttribute('style') ?? '').toContain('-webkit-app-region: no-drag');
  });
});
