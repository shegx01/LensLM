import { render, screen, waitFor, fireEvent } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CheckResult } from '$lib/onboarding/system-check.js';
import SystemCheck from './SystemCheck.svelte';

// All three readiness gates passing — the only state in which Continue enables.
const ALL_PASS: CheckResult[] = [
  {
    id: 'llm_runtime',
    label: 'LLM runtime',
    status: 'pass',
    detail: 'Ollama 0.3.2 detected',
    action: 'configure'
  },
  {
    id: 'embedding_model',
    label: 'Embedding model',
    status: 'pass',
    detail: 'Embedding model installed',
    action: 'choose'
  },
  {
    id: 'text_to_speech',
    label: 'Text-to-speech',
    status: 'pass',
    detail: 'Kokoro audio engine ready',
    action: 'choose'
  }
];

/** All three gates failing (fresh machine). */
function allFail(): CheckResult[] {
  return ALL_PASS.map((r) => ({ ...r, status: 'fail' as const }));
}

/** Exactly one gate (text_to_speech) failing — Continue must stay disabled. */
function ttsFail(): CheckResult[] {
  return ALL_PASS.map((r) =>
    r.id === 'text_to_speech'
      ? {
          ...r,
          status: 'fail' as const,
          detail: 'No text-to-speech engine configured',
          action: 'choose' as const
        }
      : r
  );
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('SystemCheck', () => {
  it('renders the System check title and the three rows returned by runSystemCheck', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
    });
    render(SystemCheck, { props: { oncomplete: vi.fn() } });
    expect(screen.getByText('System check')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('LLM runtime')).toBeInTheDocument());
    expect(screen.getByText('Embedding model')).toBeInTheDocument();
    expect(screen.getByText('Text-to-speech')).toBeInTheDocument();
  });

  it('enables Continue only when all three gates pass', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
    });
    render(SystemCheck, { props: { oncomplete: vi.fn() } });
    const cont = screen.getByRole('button', { name: 'Continue to setup' });
    await waitFor(() => expect(cont).not.toBeDisabled());
  });

  it('disables Continue when even one gate fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ttsFail();
    });
    render(SystemCheck, { props: { oncomplete: vi.fn() } });
    await waitFor(() => expect(screen.getByText('Text-to-speech')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Continue to setup' })).toBeDisabled();
  });

  it('disables Continue when all gates fail', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return allFail();
    });
    render(SystemCheck, { props: { oncomplete: vi.fn() } });
    await waitFor(() => expect(screen.getByText('LLM runtime')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Continue to setup' })).toBeDisabled();
  });

  it('persists then calls oncomplete when Continue is clicked', async () => {
    const oncomplete = vi.fn();
    const setConfig = vi.fn();
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
          paths: { data_dir: '' },
          tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
          onboarding_complete: false,
          embedding_model: ''
        };
      }
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
    });
    render(SystemCheck, { props: { oncomplete } });
    const cont = screen.getByRole('button', { name: 'Continue to setup' });
    await waitFor(() => expect(cont).not.toBeDisabled());
    await fireEvent.click(cont);
    await waitFor(() => expect(oncomplete).toHaveBeenCalledOnce());
    // The flag was persisted (RMW) before oncomplete fired.
    expect(setConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({ onboarding_complete: true })
      })
    );
  });

  it('surfaces an inline error and does NOT call oncomplete when persistence fails', async () => {
    const oncomplete = vi.fn();
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
      if (cmd === 'get_config') throw new Error('disk full');
    });
    render(SystemCheck, { props: { oncomplete } });
    const cont = screen.getByRole('button', { name: 'Continue to setup' });
    await waitFor(() => expect(cont).not.toBeDisabled());
    await fireEvent.click(cont);
    await waitFor(() => expect(screen.getByText(/could not save your setup/i)).toBeInTheDocument());
    expect(oncomplete).not.toHaveBeenCalled();
  });

  it('shows an inline error and blocks Continue when the check itself fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') throw new Error('probe boom');
    });
    render(SystemCheck, { props: { oncomplete: vi.fn() } });
    await waitFor(() =>
      expect(screen.getByText(/could not run the system check/i)).toBeInTheDocument()
    );
    expect(screen.getByRole('button', { name: 'Continue to setup' })).toBeDisabled();
  });

  it('shows "0 of 3" when all gates fail and "3 of 3" when all pass', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return allFail();
    });
    const { unmount } = render(SystemCheck, { props: { oncomplete: vi.fn() } });
    await waitFor(() => expect(screen.getByText('0 of 3 checks passed')).toBeInTheDocument());
    unmount();

    clearMocks();
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
    });
    render(SystemCheck, { props: { oncomplete: vi.fn() } });
    await waitFor(() => expect(screen.getByText('3 of 3 checks passed')).toBeInTheDocument());
  });
});
