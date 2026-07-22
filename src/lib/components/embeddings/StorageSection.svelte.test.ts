import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig, StorageStats } from '$lib/theme/types.js';
import StorageSection from './StorageSection.svelte';

vi.mock('@tauri-apps/plugin-opener', () => ({
  revealItemInDir: vi.fn().mockResolvedValue(undefined)
}));

vi.mock('@tauri-apps/plugin-clipboard-manager', () => ({
  writeText: vi.fn().mockResolvedValue(undefined)
}));

vi.mock('@tauri-apps/plugin-dialog', () => ({
  open: vi.fn()
}));

beforeEach(async () => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
  // Default to "user cancelled the picker" so tests that don't exercise a
  // move/offload flow are unaffected by a leftover resolved value.
  const { open } = await import('@tauri-apps/plugin-dialog');
  vi.mocked(open).mockResolvedValue(null);
});

afterEach(() => {
  clearMocks();
  vi.clearAllMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

const DATA_DIR = '/Users/test/Library/Application Support/LensLM';

function config(
  pathOverrides: { cache_dir?: string | null } = {},
  storageOverrides: { cache_quota_bytes?: number | null } = {}
): Partial<AppConfig> {
  return {
    paths: { data_dir: DATA_DIR, cache_dir: null, ...pathOverrides },
    storage: { cache_quota_bytes: null, ...storageOverrides }
  };
}

function stats(reclaimable: number): StorageStats {
  const db = 1_048_576; // 1.0 MB
  const vectors = 2_097_152; // 2.0 MB
  const sources = 1_572_864; // 1.5 MB
  const audio = 524_288; // 512.0 KB
  const corpus = db + vectors + sources + audio; // 5_242_880 -> 5.0 MB
  return {
    db_bytes: db,
    vectors_bytes: vectors,
    sources_bytes: sources,
    audio_bytes: audio,
    corpus_bytes: corpus,
    reclaimable_cache_bytes: reclaimable,
    retained_bytes: 274_000_000,
    total_bytes: corpus + reclaimable + 274_000_000
  };
}

describe('StorageSection', () => {
  it('renders the data directory path on mount', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'get_storage_stats') return stats(1_500_000_000);
    });

    render(StorageSection);

    await waitFor(() => expect(screen.getByText(DATA_DIR)).toBeInTheDocument());
  });

  it('renders both the reclaimable cache and corpus figures', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'get_storage_stats') return stats(1_500_000_000);
    });

    render(StorageSection);

    await waitFor(() => expect(screen.getByText(/1\.4 GB/)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText(/5\.0 MB/)).toBeInTheDocument());
  });

  it('renders the per-bucket usage breakdown', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'get_storage_stats') return stats(0);
    });

    render(StorageSection);

    await waitFor(() => expect(screen.getByText('Database')).toBeInTheDocument());
    expect(screen.getByText('Vectors')).toBeInTheDocument();
    expect(screen.getByText('Sources')).toBeInTheDocument();
    expect(screen.getByText('Audio')).toBeInTheDocument();

    expect(screen.getByText(/1\.0 MB/)).toBeInTheDocument();
    expect(screen.getByText(/2\.0 MB/)).toBeInTheDocument();
    expect(screen.getByText(/1\.5 MB/)).toBeInTheDocument();
    expect(screen.getByText(/512\.0 KB/)).toBeInTheDocument();
  });

  it('clears the model cache and refetches stats after confirming', async () => {
    let clearCalled = false;
    let statsCallCount = 0;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'get_storage_stats') {
        statsCallCount += 1;
        return statsCallCount === 1 ? stats(1_500_000_000) : stats(0);
      }
      if (cmd === 'clear_model_cache') {
        clearCalled = true;
        return 1_500_000_000;
      }
    });

    render(StorageSection);

    await waitFor(() => expect(screen.getByText(/1\.4 GB/)).toBeInTheDocument());

    const clearButton = await screen.findByRole('button', {
      name: /clear reclaimable model cache/i
    });
    await fireEvent.click(clearButton);

    const confirmButton = await screen.findByRole('button', { name: /confirm clear model cache/i });
    await fireEvent.click(confirmButton);

    await waitFor(() => expect(clearCalled).toBe(true));
    await waitFor(() => expect(statsCallCount).toBe(2));
    await waitFor(() => expect(screen.getByText(/^0 B$/)).toBeInTheDocument());
  });

  it('copies the path and reveals it in Finder', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'get_storage_stats') return stats(0);
    });

    const { writeText } = await import('@tauri-apps/plugin-clipboard-manager');
    const { revealItemInDir } = await import('@tauri-apps/plugin-opener');

    render(StorageSection);
    await waitFor(() => expect(screen.getByText(DATA_DIR)).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /copy data folder path/i }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(DATA_DIR));

    await fireEvent.click(screen.getByRole('button', { name: /reveal data folder in finder/i }));
    await waitFor(() => expect(revealItemInDir).toHaveBeenCalledWith(DATA_DIR));
  });

  it('moves the data folder and forces a mandatory restart (no "Later")', async () => {
    const { open } = await import('@tauri-apps/plugin-dialog');
    vi.mocked(open).mockResolvedValue('/Volumes/External/LensLM');

    let relocateArgs: unknown;
    let restartCalled = false;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'get_storage_stats') return stats(0);
      if (cmd === 'relocate_data_dir') {
        relocateArgs = args;
        return undefined;
      }
      if (cmd === 'restart_app') {
        restartCalled = true;
        return undefined;
      }
    });

    render(StorageSection);
    await waitFor(() => expect(screen.getByText(DATA_DIR)).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /move data folder/i }));
    await waitFor(() => expect(open).toHaveBeenCalled());

    const confirmButton = await screen.findByRole('button', { name: /confirm move data folder/i });
    await fireEvent.click(confirmButton);

    await waitFor(() => expect(relocateArgs).toEqual({ new_path: '/Volumes/External/LensLM' }));
    const restartButton = await screen.findByRole('button', { name: /^restart now$/i });
    // The relocate restart is mandatory — no dismiss/"Later" affordance.
    expect(screen.queryByRole('button', { name: /^later$/i })).not.toBeInTheDocument();

    await fireEvent.click(restartButton);
    await waitFor(() => expect(restartCalled).toBe(true));
  });

  it('surfaces an error inline when relocate_data_dir fails', async () => {
    const { open } = await import('@tauri-apps/plugin-dialog');
    vi.mocked(open).mockResolvedValue('/Volumes/External/LensLM');

    mockIPC((cmd) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'get_storage_stats') return stats(0);
      if (cmd === 'relocate_data_dir') throw new Error('disk full');
    });

    render(StorageSection);
    await waitFor(() => expect(screen.getByText(DATA_DIR)).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /move data folder/i }));
    const confirmButton = await screen.findByRole('button', { name: /confirm move data folder/i });
    await fireEvent.click(confirmButton);

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
  });

  it('moves the model cache, calls offload_cache, and offers an optional restart', async () => {
    const { open } = await import('@tauri-apps/plugin-dialog');
    vi.mocked(open).mockResolvedValue('/Volumes/External/models');

    let offloadArgs: unknown;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'get_storage_stats') return stats(0);
      if (cmd === 'offload_cache') {
        offloadArgs = args;
        return 900_000_000;
      }
    });

    render(StorageSection);
    await waitFor(() => expect(screen.getByText('Default (with your data)')).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /move cache/i }));

    await waitFor(() => expect(offloadArgs).toEqual({ new_path: '/Volumes/External/models' }));
    await waitFor(() => expect(screen.getByText('/Volumes/External/models')).toBeInTheDocument());
    // Offload's restart is optional — both actions are offered.
    await screen.findByRole('button', { name: /^restart now$/i });
    expect(screen.getByRole('button', { name: /^later$/i })).toBeInTheDocument();
  });

  it('keeps the offload result when the post-move stats refetch fails', async () => {
    const { open } = await import('@tauri-apps/plugin-dialog');
    vi.mocked(open).mockResolvedValue('/Volumes/External/models');

    let statsCallCount = 0;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'get_storage_stats') {
        statsCallCount += 1;
        if (statsCallCount > 1) throw new Error('stats unavailable');
        return stats(0);
      }
      if (cmd === 'offload_cache') return 900_000_000;
    });

    render(StorageSection);
    await waitFor(() => expect(screen.getByText('Default (with your data)')).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /move cache/i }));

    await waitFor(() => expect(screen.getByText('/Volumes/External/models')).toBeInTheDocument());
    await waitFor(() => expect(statsCallCount).toBeGreaterThan(1));
    // The refetch failure is swallowed — the successful move is not reported as an error.
    expect(screen.queryByText(/could not move the model cache/i)).not.toBeInTheDocument();
  });

  it('shows reset-to-default when offloaded and calls reset_cache_location', async () => {
    let resetCalled = false;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config({ cache_dir: '/Volumes/External/models' });
      if (cmd === 'get_storage_stats') return stats(0);
      if (cmd === 'reset_cache_location') {
        resetCalled = true;
        return 900_000_000;
      }
    });

    render(StorageSection);
    const resetButton = await screen.findByRole('button', { name: /reset to default/i });
    await fireEvent.click(resetButton);

    await waitFor(() => expect(resetCalled).toBe(true));
    await waitFor(() => expect(screen.getByText('Default (with your data)')).toBeInTheDocument());
  });

  it('persists the cache quota in bytes via set_config on blur', async () => {
    let setConfigArg: AppConfig | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'get_storage_stats') return stats(0);
      if (cmd === 'set_config') {
        setConfigArg = (args as { config: AppConfig }).config;
        return undefined;
      }
    });

    render(StorageSection);
    const quotaInput = await screen.findByRole('spinbutton', {
      name: /model cache limit in gigabytes/i
    });

    await fireEvent.input(quotaInput, { target: { value: '2' } });
    await fireEvent.blur(quotaInput);

    await waitFor(() => expect(setConfigArg?.storage?.cache_quota_bytes).toBe(2_000_000_000));
  });

  it('clearing the quota input persists a null (no-limit) quota', async () => {
    let setConfigArg: AppConfig | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return config({}, { cache_quota_bytes: 1_000_000_000 });
      if (cmd === 'get_storage_stats') return stats(0);
      if (cmd === 'set_config') {
        setConfigArg = (args as { config: AppConfig }).config;
        return undefined;
      }
    });

    render(StorageSection);
    const quotaInput = await screen.findByRole('spinbutton', {
      name: /model cache limit in gigabytes/i
    });
    await waitFor(() => expect(quotaInput).toHaveValue(1));

    await fireEvent.input(quotaInput, { target: { value: '' } });
    await fireEvent.blur(quotaInput);

    await waitFor(() => expect(setConfigArg?.storage?.cache_quota_bytes).toBeNull());
  });

  it('shows an over-limit warning and reclaims via the same confirm flow', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config({}, { cache_quota_bytes: 1_000_000_000 });
      if (cmd === 'get_storage_stats') return stats(1_500_000_000);
    });

    render(StorageSection);

    await waitFor(() => expect(screen.getByText(/exceeds your limit/i)).toBeInTheDocument());

    const reclaimButton = screen.getByRole('button', { name: /reclaim now/i });
    await fireEvent.click(reclaimButton);

    await screen.findByRole('button', { name: /confirm clear model cache/i });
  });

  it('does not render an over-limit warning when under the quota', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config({}, { cache_quota_bytes: 5_000_000_000 });
      if (cmd === 'get_storage_stats') return stats(1_500_000_000);
    });

    render(StorageSection);

    await waitFor(() => expect(screen.getByText(/1\.4 GB/)).toBeInTheDocument());
    expect(screen.queryByText(/exceeds your limit/i)).not.toBeInTheDocument();
  });
});
