import { render, screen, waitFor, fireEvent } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CheckResult } from '$lib/onboarding/system-check.js';
import SystemCheck from './SystemCheck.svelte';

// Both readiness gates passing — the only state in which Continue enables.
// TTS is no longer a readiness gate here (moved to Settings, #194).
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

/** Exactly one gate (embedding_model) failing — Continue must stay disabled. */
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

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('SystemCheck', () => {
  it('renders the System check title and the two rows returned by runSystemCheck', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    expect(screen.getByText('System check')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('LLM runtime')).toBeInTheDocument());
    expect(screen.getByText('Embedding model')).toBeInTheDocument();
    expect(screen.queryByText('Text-to-speech')).not.toBeInTheDocument();
  });

  it('enables Continue only when both gates pass', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    const cont = screen.getByRole('button', { name: 'Continue to setup' });
    await waitFor(() => expect(cont).not.toBeDisabled());
  });

  it('disables Continue when even one gate fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return embeddingFail();
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('Embedding model')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Continue to setup' })).toBeDisabled();
  });

  it('advances (TTS no longer blocks onboarding) even when the backend still reports a failing text_to_speech row', async () => {
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
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('LLM runtime')).toBeInTheDocument());
    expect(screen.queryByText('Text-to-speech')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Continue to setup' })).not.toBeDisabled();
  });

  it('disables Continue when all gates fail', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return allFail();
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('LLM runtime')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Continue to setup' })).toBeDisabled();
  });

  it('advances (does NOT persist) when Continue is clicked', async () => {
    const onadvance = vi.fn();
    const setConfig = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'run_system_check') return ALL_PASS;
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
    });
    render(SystemCheck, { props: { onadvance } });
    const cont = screen.getByRole('button', { name: 'Continue to setup' });
    await waitFor(() => expect(cont).not.toBeDisabled());
    await fireEvent.click(cont);
    await waitFor(() => expect(onadvance).toHaveBeenCalledOnce());
    // This step no longer persists onboarding_complete — it only advances.
    expect(setConfig).not.toHaveBeenCalled();
  });

  it('shows an inline error and blocks Continue when the check itself fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') throw new Error('probe boom');
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() =>
      expect(screen.getByText(/could not run the system check/i)).toBeInTheDocument()
    );
    expect(screen.getByRole('button', { name: 'Continue to setup' })).toBeDisabled();
  });

  it('shows "0 of 2" when all gates fail and "2 of 2" when all pass', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return allFail();
    });
    const { unmount } = render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('0 of 2 checks passed')).toBeInTheDocument());
    unmount();

    clearMocks();
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
    });
    render(SystemCheck, { props: { onadvance: vi.fn() } });
    await waitFor(() => expect(screen.getByText('2 of 2 checks passed')).toBeInTheDocument());
  });

  it('outer <main> is a data-tauri-drag-region with the Card marked no-drag', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
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
    });
    const { container } = render(SystemCheck, { props: { onadvance: vi.fn() } });
    const themeWrapper = container.querySelector('.absolute.top-4.right-4') as HTMLElement;
    expect(themeWrapper).not.toBeNull();
    expect(themeWrapper.getAttribute('style') ?? '').toContain('-webkit-app-region: no-drag');
  });
});
