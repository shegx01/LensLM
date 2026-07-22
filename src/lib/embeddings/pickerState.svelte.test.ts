import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { baseAppConfig } from '$lib/test-fixtures.js';
import { EmbeddingPickerState } from './pickerState.svelte.js';

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('EmbeddingPickerState', () => {
  it('does NOT persist a fastembed default when the warm leaves the cache empty', async () => {
    // A warm that reports success without populating the on-disk cache must not
    // persist an unusable default (guards against a green install + failing gate).
    let setCalled = false;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'fastembed_models_cached') return []; // never populates
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
      if (cmd === 'warm_fastembed_model') return null; // "success"
      if (cmd === 'set_config') {
        setCalled = true;
        return null;
      }
    });

    const picker = new EmbeddingPickerState({ mode: 'global' });
    await picker.init();
    await picker.install();

    expect(setCalled).toBe(false);
    expect(picker.actionError).toMatch(/did not finish installing/i);
    expect(picker.installing).toBe(false);
  });

  it('persists once the warm populates the cache', async () => {
    let written: { embedding_model?: string } | null = null;
    let cached: string[] = [];
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'fastembed_models_cached') return cached;
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
      if (cmd === 'warm_fastembed_model') {
        cached = ['nomic-embed-text-v1.5'];
        return null;
      }
      if (cmd === 'set_config') {
        written = (args as { config: { embedding_model?: string } }).config;
        return null;
      }
    });

    const picker = new EmbeddingPickerState({ mode: 'global' });
    await picker.init();
    await picker.install();

    expect(written).not.toBeNull();
    expect(written!.embedding_model).toBe('nomic-embed-text-v1.5');
  });

  it('blocks selection changes while a write is in flight', () => {
    const picker = new EmbeddingPickerState({ mode: 'global' });
    picker.installing = true;
    picker.pickBackend('ollama');
    expect(picker.backend).toBe('fastembed'); // guarded by `busy`
  });

  it('opens the confirm dialog for an indexed notebook coordinate change instead of re-embedding immediately', async () => {
    let reembedCalled = false;
    mockIPC((cmd) => {
      if (cmd === 'get_notebook_embedding_model')
        return {
          model_id: 'nomic-embed-text-v1.5',
          dim: 768,
          backend: 'fastembed',
          status: 'active'
        };
      if (cmd === 'fastembed_models_cached') return ['nomic-embed-text-v1.5', 'all-minilm'];
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'gpu_accelerated_models') return [];
      if (cmd === 'set_notebook_embedding_model') {
        reembedCalled = true;
        return null;
      }
    });

    const picker = new EmbeddingPickerState({ mode: 'notebook', notebookId: 'nb-1' });
    await picker.init();
    picker.pickModel('all-minilm'); // dirty change on an indexed coordinate
    await picker.commit();

    expect(picker.confirmOpen).toBe(true);
    expect(reembedCalled).toBe(false);
  });
});
