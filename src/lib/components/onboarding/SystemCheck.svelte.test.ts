import { render, screen, waitFor, fireEvent } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CheckResult } from '$lib/onboarding/system-check.js';
import SystemCheck from './SystemCheck.svelte';

const ALL_PASS: CheckResult[] = [
  {
    id: 'local_backend',
    label: 'Local backend',
    status: 'pass',
    detail: 'In-process engine ready',
    action: null
  },
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
    status: 'pending',
    detail: 'Set up when you add your first source',
    action: 'choose'
  },
  {
    id: 'vector_database',
    label: 'Vector database',
    status: 'pending',
    detail: 'Built-in · set up automatically when you add your first source',
    action: null
  },
  {
    id: 'disk_permissions',
    label: 'Disk permissions',
    status: 'pass',
    detail: '~/Library/Application Support/Lens',
    action: null
  }
];

function withDiskFail(): CheckResult[] {
  return ALL_PASS.map((r) =>
    r.id === 'disk_permissions'
      ? {
          ...r,
          status: 'fail' as const,
          detail: 'Cannot write to app data directory',
          action: 'retry' as const
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
  it('renders the System check title and all rows from runSystemCheck (+ synthetic TTS row)', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
    });
    render(SystemCheck, { props: { oncomplete: vi.fn() } });
    expect(screen.getByText('System check')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Local backend')).toBeInTheDocument());
    expect(screen.getByText('Vector database')).toBeInTheDocument();
    // Synthetic TTS row is always appended by the component
    expect(screen.getByText('Text-to-speech')).toBeInTheDocument();
  });

  it('enables Continue when no blocking check fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
    });
    render(SystemCheck, { props: { oncomplete: vi.fn() } });
    const cont = screen.getByRole('button', { name: 'Continue to setup' });
    await waitFor(() => expect(cont).not.toBeDisabled());
  });

  it('disables Continue when disk_permissions fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return withDiskFail();
    });
    render(SystemCheck, { props: { oncomplete: vi.fn() } });
    await waitFor(() => expect(screen.getByText('Disk permissions')).toBeInTheDocument());
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
          paths: { data_dir: '' },
          tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
          onboarding_complete: false
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

  it('shows the "{ready} of {total} checks passed" summary (design footer)', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
    });
    render(SystemCheck, { props: { oncomplete: vi.fn() } });
    // ALL_PASS = 3 pass (backend, llm, disk) + 2 pending (embedding, vector) from IPC
    // + 1 pending synthetic TTS row = 3 of 6 total.
    await waitFor(() => expect(screen.getByText('3 of 6 checks passed')).toBeInTheDocument());
  });

  it("a failed row's Retry action re-runs the system check", async () => {
    let calls = 0;
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') {
        calls += 1;
        return withDiskFail();
      }
    });
    render(SystemCheck, { props: { oncomplete: vi.fn() } });
    await waitFor(() => expect(screen.getByText('Disk permissions')).toBeInTheDocument());
    expect(calls).toBe(1);
    // The design has NO footer Retry; re-check is the failed row's per-row action.
    await fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(calls).toBe(2));
  });
});
