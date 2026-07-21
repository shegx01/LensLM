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

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  vi.clearAllMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

const DATA_DIR = '/Users/test/Library/Application Support/LensLM';

function config(): Partial<AppConfig> {
  return { paths: { data_dir: DATA_DIR } };
}

function stats(reclaimable: number): StorageStats {
  return {
    corpus_bytes: 5_242_880,
    reclaimable_cache_bytes: reclaimable,
    retained_bytes: 274_000_000,
    total_bytes: 5_242_880 + reclaimable + 274_000_000
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
});
