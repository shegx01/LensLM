import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { TtsConfig } from '$lib/theme/types.js';
import {
  runSystemCheck,
  downloadTtsModel,
  prepareQwenModel,
  cancelPrepare,
  nextTtsConfig
} from './system-check.js';

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

describe('runSystemCheck', () => {
  it('filters out the legacy text_to_speech row — onboarding gates on LLM + embeddings only', async () => {
    mockIPC((cmd) => {
      if (cmd === 'run_system_check') {
        return [
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
          },
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

    const results = await runSystemCheck();
    expect(results).toHaveLength(2);
    expect(results.map((r) => r.id).sort()).toEqual(['embedding_model', 'llm_runtime']);
  });

  it('returns [] outside a Tauri host', async () => {
    delete (globalThis as { isTauri?: boolean }).isTauri;
    const results = await runSystemCheck();
    expect(results).toEqual([]);
  });
});

describe('shared progress channel factory (downloadTtsModel / prepareQwenModel)', () => {
  it('downloadTtsModel emits null while total is unknown, then 100 on done', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'download_tts_model') {
        const ch = (args as { onProgress: ProgressChannel }).onProgress;
        ch.onmessage({ received: 0, total: null, done: false });
        ch.onmessage({ received: 1, total: null, done: false });
        ch.onmessage({ received: 100, total: 100, done: true });
        return null;
      }
    });

    const calls: (number | null)[] = [];
    await downloadTtsModel('orpheus', 'orpheus', (pct) => calls.push(pct));
    expect(calls).toEqual([null, null, 100]);
  });

  it('downloadTtsModel emits the known percentage when total is present', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'download_tts_model') {
        const ch = (args as { onProgress: ProgressChannel }).onProgress;
        ch.onmessage({ received: 50, total: 200, done: false });
        return null;
      }
    });

    const calls: (number | null)[] = [];
    await downloadTtsModel('orpheus', 'orpheus', (pct) => calls.push(pct));
    expect(calls).toEqual([25]);
  });

  it('prepareQwenModel emits null while total is unknown, then 100 on done', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'prepare_qwen_model') {
        const ch = (args as { onProgress: ProgressChannel }).onProgress;
        ch.onmessage({ received: 10, total: null, done: false });
        ch.onmessage({ received: 10, total: 10, done: true });
        return null;
      }
    });

    const calls: (number | null)[] = [];
    await prepareQwenModel((pct) => calls.push(pct));
    expect(calls).toEqual([null, 100]);
  });
});

describe('cancelPrepare', () => {
  it('invokes cancel_prepare', async () => {
    let invoked = false;
    mockIPC((cmd) => {
      if (cmd === 'cancel_prepare') {
        invoked = true;
        return true;
      }
    });

    await cancelPrepare();
    expect(invoked).toBe(true);
  });

  it('swallows an unregistered-command error (cfg-gated off this platform)', async () => {
    mockIPC((cmd) => {
      if (cmd === 'cancel_prepare') {
        throw new Error('command cancel_prepare not found');
      }
    });

    await expect(cancelPrepare()).resolves.toBeUndefined();
  });

  it('is a no-op outside a Tauri host', async () => {
    delete (globalThis as { isTauri?: boolean }).isTauri;
    await expect(cancelPrepare()).resolves.toBeUndefined();
  });
});

describe('nextTtsConfig — cloud kind selection (#40)', () => {
  const prev: TtsConfig = {
    version: 1,
    backend: 'orpheus',
    model: '',
    cloud: null
  };

  it('defaults to open_ai_compatible when no cloudKind is given', () => {
    const next = nextTtsConfig(prev, { provider: 'cloud', apiKey: 'sk-x', baseUrl: 'https://x' });
    expect(next.backend).toEqual({ cloud: 'open_ai_compatible' });
    expect(next.cloud).toEqual({
      kind: 'open_ai_compatible',
      api_key: 'sk-x',
      base_url: 'https://x'
    });
  });

  it('writes eleven_labs into both backend and cloud.kind when selected', () => {
    const next = nextTtsConfig(prev, {
      provider: 'cloud',
      apiKey: 'sk-11labs',
      baseUrl: 'https://api.elevenlabs.io',
      cloudKind: 'eleven_labs'
    });
    expect(next.backend).toEqual({ cloud: 'eleven_labs' });
    expect(next.cloud).toEqual({
      kind: 'eleven_labs',
      api_key: 'sk-11labs',
      base_url: 'https://api.elevenlabs.io'
    });
  });

  it('writes google_cloud into both backend and cloud.kind when selected', () => {
    const next = nextTtsConfig(prev, {
      provider: 'cloud',
      apiKey: 'sk-google',
      baseUrl: 'https://generativelanguage.googleapis.com',
      cloudKind: 'google_cloud'
    });
    expect(next.backend).toEqual({ cloud: 'google_cloud' });
    expect(next.cloud).toEqual({
      kind: 'google_cloud',
      api_key: 'sk-google',
      base_url: 'https://generativelanguage.googleapis.com'
    });
  });

  it('switching to a local provider preserves the previously-saved cloud config', () => {
    const cloudPrev: TtsConfig = {
      version: 1,
      backend: { cloud: 'eleven_labs' },
      model: '',
      cloud: { kind: 'eleven_labs', api_key: 'sk-keep-me', base_url: 'https://api.elevenlabs.io' }
    };

    const next = nextTtsConfig(cloudPrev, { provider: 'orpheus', apiKey: '' });

    expect(next.backend).toBe('orpheus');
    expect(next.cloud).toEqual(cloudPrev.cloud);
  });
});
