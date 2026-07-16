import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { runSystemCheck } from './system-check.js';

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
