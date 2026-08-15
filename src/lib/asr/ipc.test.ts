import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { listWhisperModels, whisperModelDownloaded, downloadWhisperModel } from './ipc.js';

type ProgressChannel = {
  onmessage: (m: { received: number; total: number | null; done: boolean }) => void;
};

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('listWhisperModels', () => {
  it('returns the registry rows from list_whisper_models', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_whisper_models') {
        return [
          { id: 'tiny', approx_mb: 75, is_default: false },
          { id: 'base', approx_mb: 142, is_default: true },
          { id: 'small', approx_mb: 466, is_default: false }
        ];
      }
    });

    const models = await listWhisperModels();
    expect(models).toHaveLength(3);
    expect(models.find((m) => m.id === 'base')?.is_default).toBe(true);
  });

  it('returns [] outside a Tauri host', async () => {
    delete (globalThis as { isTauri?: boolean }).isTauri;
    await expect(listWhisperModels()).resolves.toEqual([]);
  });
});

describe('whisperModelDownloaded', () => {
  it('invokes whisper_model_downloaded with the model id', async () => {
    let receivedArgs: unknown;
    mockIPC((cmd, args) => {
      if (cmd === 'whisper_model_downloaded') {
        receivedArgs = args;
        return true;
      }
    });

    await expect(whisperModelDownloaded('base')).resolves.toBe(true);
    expect(receivedArgs).toEqual({ model: 'base' });
  });

  it('returns false outside a Tauri host', async () => {
    delete (globalThis as { isTauri?: boolean }).isTauri;
    await expect(whisperModelDownloaded('base')).resolves.toBe(false);
  });
});

describe('downloadWhisperModel', () => {
  it('emits the known percentage when total is present, then 100 on done', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'download_whisper_model') {
        const ch = (args as { onProgress: ProgressChannel }).onProgress;
        ch.onmessage({ received: 50, total: 200, done: false });
        ch.onmessage({ received: 142_000_000, total: 142_000_000, done: true });
        return null;
      }
    });

    const calls: (number | null)[] = [];
    await downloadWhisperModel('base', (pct) => calls.push(pct));
    expect(calls).toEqual([25, 100]);
  });

  it('emits null while the total is unknown', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'download_whisper_model') {
        const ch = (args as { onProgress: ProgressChannel }).onProgress;
        ch.onmessage({ received: 0, total: null, done: false });
        return null;
      }
    });

    const calls: (number | null)[] = [];
    await downloadWhisperModel('base', (pct) => calls.push(pct));
    expect(calls).toEqual([null]);
  });

  it('is a no-op outside a Tauri host', async () => {
    delete (globalThis as { isTauri?: boolean }).isTauri;
    const calls: (number | null)[] = [];
    await expect(downloadWhisperModel('base', (pct) => calls.push(pct))).resolves.toBeUndefined();
    expect(calls).toEqual([]);
  });
});
