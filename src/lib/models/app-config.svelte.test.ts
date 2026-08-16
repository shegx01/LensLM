import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { baseAppConfig } from '$lib/test-fixtures.js';
import {
  appConfigStore,
  ensureLoaded,
  refreshConfig,
  persist,
  resetConfig
} from './app-config.svelte.js';

const DEFAULT_ENRICHMENT = { enabled: false, coref_strategy: 'llm_inline', cloud_consent: false };

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
  resetConfig();
});

describe('appConfigStore unloaded-state contract', () => {
  it('reports safe defaults before any load has happened', () => {
    expect(appConfigStore.models).toEqual([]);
    expect(appConfigStore.enrichment).toEqual(DEFAULT_ENRICHMENT);
    expect(appConfigStore.asr).toBeNull();
    expect(appConfigStore.audioCloudConsent).toBe(false);
    expect(appConfigStore.ttsCloudConsent).toBe(false);
    expect(appConfigStore.loadError).toBeNull();
    expect(appConfigStore.staleError).toBeNull();
    expect(appConfigStore.persistError).toBeNull();
  });
});

describe('appConfigStore.ttsCloudConsent', () => {
  it('reads tts_cloud_consent as withheld when a loaded config omits the key', async () => {
    const { tts_cloud_consent: _omitted, ...withoutConsent } = baseAppConfig({
      tts_cloud_consent: true
    });
    mockIPC((cmd) => {
      if (cmd === 'get_config') return withoutConsent;
    });

    await ensureLoaded();

    expect(appConfigStore.ttsCloudConsent).toBe(false);
  });

  it('reads a persisted tts_cloud_consent independently of audio_cloud_consent', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config')
        return baseAppConfig({ tts_cloud_consent: true, audio_cloud_consent: false });
    });

    await ensureLoaded();

    expect(appConfigStore.ttsCloudConsent).toBe(true);
    expect(appConfigStore.audioCloudConsent).toBe(false);
  });
});

describe('ensureLoaded', () => {
  it('is load-once: N concurrent callers share exactly one get_config call', async () => {
    let getConfigCalls = 0;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        getConfigCalls += 1;
        return baseAppConfig();
      }
    });

    await Promise.all([1, 2, 3, 4, 5].map(() => ensureLoaded()));

    expect(getConfigCalls).toBe(1);
  });

  it('is a no-op once cfg is already populated', async () => {
    let getConfigCalls = 0;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        getConfigCalls += 1;
        return baseAppConfig();
      }
    });

    await ensureLoaded();
    await ensureLoaded();

    expect(getConfigCalls).toBe(1);
  });
});

describe('refreshConfig', () => {
  it('forces a reload even when cfg is already populated', async () => {
    let current = baseAppConfig({ audio_cloud_consent: false });
    mockIPC((cmd) => {
      if (cmd === 'get_config') return current;
    });

    await ensureLoaded();
    expect(appConfigStore.audioCloudConsent).toBe(false);

    current = baseAppConfig({ audio_cloud_consent: true });
    await refreshConfig();

    expect(appConfigStore.audioCloudConsent).toBe(true);
  });

  it('leaves cfg untouched on failure instead of wiping it to defaults', async () => {
    // ActiveModelSection/ProvidersSection both force a reload on mount and after every
    // credential/model write; wiping to [] on a transient re-fetch would blank a
    // working provider/model list even though nothing about it actually changed.
    let shouldFail = false;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        if (shouldFail) throw new Error('transient reload failure');
        return baseAppConfig({
          audio_cloud_consent: true,
          models: [
            {
              provider: 'openai',
              base_url: '',
              model: 'gpt-4o',
              context: 128000,
              temperature: 0.7,
              api_key: 'x'
            }
          ]
        });
      }
    });

    await ensureLoaded();
    expect(appConfigStore.models).toHaveLength(1);

    shouldFail = true;
    await refreshConfig();

    expect(appConfigStore.models).toHaveLength(1);
    expect(appConfigStore.audioCloudConsent).toBe(true);
  });
});

describe('persist', () => {
  it('keeps the optimistically-mutated value and surfaces persistError (not loadError) when the re-read fails, never falling back to defaults', async () => {
    let getConfigCalls = 0;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        getConfigCalls += 1;
        if (getConfigCalls === 3) throw new Error('reread failed');
        return baseAppConfig({ audio_cloud_consent: false });
      }
      if (cmd === 'set_config') return null;
    });

    await ensureLoaded();
    expect(appConfigStore.audioCloudConsent).toBe(false);

    await persist((cfg) => ({ ...cfg, audio_cloud_consent: true }));

    expect(appConfigStore.audioCloudConsent).toBe(true);
    expect(appConfigStore.persistError).not.toBeNull();
    expect(appConfigStore.loadError).toBeNull();
  });

  it('surfaces the real Tauri {kind,message} rejection message on re-read failure, not a generic fallback', async () => {
    let getConfigCalls = 0;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        getConfigCalls += 1;
        if (getConfigCalls === 3) throw { kind: 'Io', message: 'disk unavailable' };
        return baseAppConfig();
      }
      if (cmd === 'set_config') return null;
    });

    await ensureLoaded();
    await persist((cfg) => ({ ...cfg, audio_cloud_consent: true }));

    expect(appConfigStore.persistError).toBe('disk unavailable');
  });

  it('does not let a persist() re-read failure leak into loadError for a later, unrelated ensureLoaded() caller', async () => {
    let getConfigCalls = 0;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        getConfigCalls += 1;
        if (getConfigCalls === 3) throw new Error('reread failed');
        return baseAppConfig();
      }
      if (cmd === 'set_config') return null;
    });

    await ensureLoaded();
    await persist((cfg) => ({ ...cfg, audio_cloud_consent: true }));
    expect(appConfigStore.persistError).not.toBeNull();

    // A second panel mounts and calls ensureLoaded(); cfg is already populated so this
    // is a no-op. It must not see the earlier, unrelated write's re-read failure.
    await ensureLoaded();
    expect(appConfigStore.loadError).toBeNull();
  });
});

describe('loadError', () => {
  it('is set when the initial load fails, and getters fall back to unloaded defaults', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') throw new Error('engine unreachable');
    });

    await ensureLoaded();

    expect(appConfigStore.loadError).toBe('engine unreachable');
    expect(appConfigStore.staleError).toBeNull();
    expect(appConfigStore.models).toEqual([]);
    expect(appConfigStore.enrichment).toEqual(DEFAULT_ENRICHMENT);
    expect(appConfigStore.asr).toBeNull();
    expect(appConfigStore.audioCloudConsent).toBe(false);
  });

  it('surfaces the real Tauri {kind,message} rejection message on initial load failure, not a generic fallback', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') throw { kind: 'Internal', message: 'engine not started' };
    });

    await ensureLoaded();

    expect(appConfigStore.loadError).toBe('engine not started');
  });

  it('clears once a later load succeeds', async () => {
    let shouldFail = true;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        if (shouldFail) throw new Error('engine unreachable');
        return baseAppConfig();
      }
    });

    await ensureLoaded();
    expect(appConfigStore.loadError).not.toBeNull();

    shouldFail = false;
    await refreshConfig();

    expect(appConfigStore.loadError).toBeNull();
  });
});

describe('staleError', () => {
  // The mechanism under test: load()'s `cfg === null ? loadError : staleError` branch. Every
  // case here pins down one side of that branch so deleting or inverting it fails a test.
  it('is set (not loadError) when a forced reload fails while a snapshot is already populated', async () => {
    let shouldFail = false;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        if (shouldFail) throw new Error('transient reload failure');
        return baseAppConfig();
      }
    });

    await ensureLoaded();
    expect(appConfigStore.loadError).toBeNull();
    expect(appConfigStore.staleError).toBeNull();

    shouldFail = true;
    await refreshConfig();

    expect(appConfigStore.staleError).toBe('transient reload failure');
    expect(appConfigStore.loadError).toBeNull();
  });

  it('is NOT set by an initial load failure with no snapshot yet — that is loadError', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') throw new Error('engine unreachable');
    });

    await ensureLoaded();

    expect(appConfigStore.loadError).toBe('engine unreachable');
    expect(appConfigStore.staleError).toBeNull();
  });

  it('surfaces the real Tauri {kind,message} rejection message on a stale-reload failure, not a generic fallback', async () => {
    let shouldFail = false;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        if (shouldFail) throw { kind: 'Io', message: 'disk unavailable' };
        return baseAppConfig();
      }
    });

    await ensureLoaded();
    shouldFail = true;
    await refreshConfig();

    expect(appConfigStore.staleError).toBe('disk unavailable');
  });

  it('clears once a later reload succeeds', async () => {
    let shouldFail = false;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        if (shouldFail) throw new Error('transient');
        return baseAppConfig();
      }
    });

    await ensureLoaded();
    shouldFail = true;
    await refreshConfig();
    expect(appConfigStore.staleError).not.toBeNull();

    shouldFail = false;
    await refreshConfig();
    expect(appConfigStore.staleError).toBeNull();
  });

  it('is left untouched by a persist() re-read failure — that is persistError, not staleError', async () => {
    let getConfigCalls = 0;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        getConfigCalls += 1;
        if (getConfigCalls === 3) throw new Error('reread failed');
        return baseAppConfig();
      }
      if (cmd === 'set_config') return null;
    });

    await ensureLoaded();
    await persist((cfg) => ({ ...cfg, audio_cloud_consent: true }));

    expect(appConfigStore.persistError).not.toBeNull();
    expect(appConfigStore.staleError).toBeNull();
  });
});

describe('resetConfig', () => {
  it('clears cfg to null so the next ensureLoaded() performs a real reload, not a re-served default', async () => {
    let getConfigCalls = 0;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        getConfigCalls += 1;
        return baseAppConfig({ audio_cloud_consent: true });
      }
    });

    await ensureLoaded();
    expect(getConfigCalls).toBe(1);
    expect(appConfigStore.audioCloudConsent).toBe(true);

    resetConfig();
    expect(appConfigStore.audioCloudConsent).toBe(false);

    await ensureLoaded();
    expect(getConfigCalls).toBe(2);
  });

  it('clears a pending staleError so it cannot leak into the next reload cycle', async () => {
    let shouldFail = false;
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        if (shouldFail) throw new Error('transient');
        return baseAppConfig();
      }
    });

    await ensureLoaded();
    shouldFail = true;
    await refreshConfig();
    expect(appConfigStore.staleError).not.toBeNull();

    resetConfig();

    expect(appConfigStore.staleError).toBeNull();
  });
});
