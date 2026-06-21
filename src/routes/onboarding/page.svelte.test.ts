import { render, screen, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CheckResult } from '$lib/onboarding/system-check.js';

// goto is not available in the test environment (no router); stub it.
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

import Onboarding from './+page.svelte';

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

describe('/onboarding', () => {
  it('renders the System check title and all rows from runSystemCheck', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
    });
    render(Onboarding);
    expect(screen.getByText('System check')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Local backend')).toBeInTheDocument());
    expect(screen.getByText('Vector database')).toBeInTheDocument();
  });

  it('enables Continue when no blocking check fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return ALL_PASS;
    });
    render(Onboarding);
    const cont = screen.getByRole('button', { name: 'Continue' });
    await waitFor(() => expect(cont).not.toBeDisabled());
  });

  it('disables Continue when disk_permissions fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') return withDiskFail();
    });
    render(Onboarding);
    await waitFor(() => expect(screen.getByText('Disk permissions')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled();
  });
});
