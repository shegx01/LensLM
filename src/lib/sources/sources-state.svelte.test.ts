// Reactivity tests for the sources store. Separate from sources-state.test.ts because
// runes only compile in a `.svelte.` module, and these assertions are about whether a
// mutation notifies readers at all — not about the value it leaves behind.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';
import { listSources, ingestSource } from './ipc.js';
import { ingest, loadSources, resetSourcesStore, sourcesStore } from './sources-state.svelte.js';
import type { Source } from './types.js';

vi.mock('./ipc.js', () => ({
  listSources: vi.fn(),
  ingestSource: vi.fn(),
  retryIngestSource: vi.fn(),
  retryAllFailedSources: vi.fn(),
  setSourceSelected: vi.fn(),
  trashSource: vi.fn(),
  restoreSource: vi.fn()
}));

vi.mock('$lib/notebooks/notebooks-state.svelte.js', () => ({
  notebookStore: { activeNotebookId: 'nb-001' },
  refreshTrashedSources: vi.fn()
}));

function makeSource(overrides: Partial<Source> = {}): Source {
  return {
    id: 'src-001',
    notebook_id: 'nb-001',
    kind: 'audio',
    title: 'Call.mp3',
    uri: null,
    status: 'queued',
    token_count: null,
    selected: 1,
    created_at: '2026-01-01T00:00:00Z',
    error_meta: null,
    ...overrides
  } as Source;
}

beforeEach(() => {
  vi.mocked(listSources).mockResolvedValue([makeSource()]);
});

afterEach(() => {
  resetSourcesStore();
  vi.clearAllMocks();
});

describe('effectiveBackendFor reactivity', () => {
  // A plain `$state(new Map())` never tracks `.set()`, so a reader watching only the
  // marker sees nothing. Asserting on a reader that also watches `sources` would pass
  // either way, because the accompanying status write re-runs it.
  it('notifies a reader that watches only the marker, with no other state change', async () => {
    await loadSources('nb-001');

    let handler: ((e: unknown) => void) | null = null;
    vi.mocked(ingestSource).mockImplementation(async (_id, onProgress) => {
      handler = onProgress as (e: unknown) => void;
    });

    const seen: (string | null)[] = [];
    const stop = $effect.root(() => {
      $effect(() => {
        seen.push(sourcesStore.effectiveBackendFor('src-001'));
      });
    });
    flushSync();
    expect(seen).toEqual([null]);

    const done = ingest('src-001');
    (handler as unknown as (e: unknown) => void)({
      type: 'chunk',
      data: {
        phase: 'transcribing',
        done: 100,
        total: 100,
        effective_backend: 'local_whisper (fallback)'
      }
    });
    await done;
    flushSync();
    stop();

    expect(seen).toEqual([null, 'local_whisper (fallback)']);
  });
});
