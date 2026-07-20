// Store unit tests for audio-state.svelte.ts.
//
// The IPC + sibling-store modules are mocked so tests run without a Tauri host
// and without the module-level auto-refresh `$effect` firing (activeNotebookId
// is pinned to `null`). `resetAudioOverviewStore()` runs in afterEach to prevent
// cross-test bleed from module-level `$state` globals — same pattern as
// sources-state.test.ts.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./audio-ipc.js', () => ({
  synthesizeOverview: vi.fn(),
  getAudioOverviewStatus: vi.fn(),
  isOverviewGenerating: vi.fn(),
  cancelSynthesis: vi.fn()
}));

// A separate mutable mock object (not a direct re-export) so tests can assign
// `.sources` freely — the real module types it as a getter-only property.
const { mockSourcesStore } = vi.hoisted(() => {
  let _sources: Array<{ selected: number }> = [];
  return {
    mockSourcesStore: {
      get sources() {
        return _sources;
      },
      get selectedCount() {
        return _sources.filter((s) => s.selected === 1).length;
      },
      _setSources(s: Array<{ selected: number }>) {
        _sources = s;
      }
    }
  };
});

vi.mock('./sources-state.svelte.js', () => ({
  sourcesStore: mockSourcesStore
}));

vi.mock('$lib/notebooks/notebooks-state.svelte.js', () => ({
  notebookStore: {
    get activeNotebookId() {
      return null;
    }
  }
}));

vi.mock('$lib/onboarding/system-check.js', () => ({
  isTtsReady: vi.fn()
}));

import {
  audioOverviewStore,
  hydrateOverview,
  generateOverview,
  cancelOverview,
  resetAudioOverviewStore
} from './audio-state.svelte.js';
import {
  synthesizeOverview,
  getAudioOverviewStatus,
  isOverviewGenerating,
  cancelSynthesis,
  type AudioOverviewRecord,
  type TtsPhase
} from './audio-ipc.js';
import type { StreamEvent } from './types.js';
import { isTtsReady } from '$lib/onboarding/system-check.js';

type Progress = (e: StreamEvent<TtsPhase>) => void;

function record(overrides?: Partial<AudioOverviewRecord>): AudioOverviewRecord {
  return {
    path: '/data/notebooks/nb-001/overview.wav',
    generated_at: '2026-07-01T00:00:00Z',
    status: 'ready',
    source_set_hash: 'hash-1',
    ...overrides
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  resetAudioOverviewStore();
  mockSourcesStore._setSources([]);
  vi.mocked(isOverviewGenerating).mockResolvedValue(false);
  vi.mocked(getAudioOverviewStatus).mockResolvedValue(null);
  vi.mocked(isTtsReady).mockResolvedValue(true);
});

afterEach(() => {
  resetAudioOverviewStore();
});

// ---------------------------------------------------------------------------
// resetAudioOverviewStore
// ---------------------------------------------------------------------------

describe('resetAudioOverviewStore', () => {
  it('resets all fields to initial values', async () => {
    vi.mocked(getAudioOverviewStatus).mockResolvedValue(record());
    await hydrateOverview('nb-001');
    expect(audioOverviewStore.overviewStatus).toBe('ready');

    resetAudioOverviewStore();

    expect(audioOverviewStore.overviewStatus).toBe('none');
    expect(audioOverviewStore.phase).toBe('idle');
    expect(audioOverviewStore.overviewPath).toBeNull();
    expect(audioOverviewStore.generatedAt).toBeNull();
    expect(audioOverviewStore.errorMessage).toBeNull();
    expect(audioOverviewStore.modelReady).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// hydrateOverview
// ---------------------------------------------------------------------------

describe('hydrateOverview', () => {
  it('populates status from a persisted ready record', async () => {
    vi.mocked(getAudioOverviewStatus).mockResolvedValue(record({ status: 'ready' }));

    await hydrateOverview('nb-001');

    expect(audioOverviewStore.overviewStatus).toBe('ready');
    expect(audioOverviewStore.overviewPath).toBe('/data/notebooks/nb-001/overview.wav');
    expect(audioOverviewStore.generatedAt).toBe('2026-07-01T00:00:00Z');
  });

  it('surfaces stale/failed/missing statuses as-is (status is authoritative)', async () => {
    vi.mocked(getAudioOverviewStatus).mockResolvedValue(record({ status: 'stale' }));
    await hydrateOverview('nb-001');
    expect(audioOverviewStore.overviewStatus).toBe('stale');

    vi.mocked(getAudioOverviewStatus).mockResolvedValue(record({ status: 'missing' }));
    await hydrateOverview('nb-001');
    expect(audioOverviewStore.overviewStatus).toBe('missing');

    vi.mocked(getAudioOverviewStatus).mockResolvedValue(record({ status: 'failed' }));
    await hydrateOverview('nb-001');
    expect(audioOverviewStore.overviewStatus).toBe('failed');
  });

  it('sets status to "none" when no record was ever generated', async () => {
    vi.mocked(getAudioOverviewStatus).mockResolvedValue(null);

    await hydrateOverview('nb-001');

    expect(audioOverviewStore.overviewStatus).toBe('none');
    expect(audioOverviewStore.overviewPath).toBeNull();
  });

  it('overrides to "generating" when a run is in flight, even with a persisted record', async () => {
    vi.mocked(getAudioOverviewStatus).mockResolvedValue(record({ status: 'ready' }));
    vi.mocked(isOverviewGenerating).mockResolvedValue(true);

    await hydrateOverview('nb-001');

    expect(audioOverviewStore.overviewStatus).toBe('generating');
    expect(audioOverviewStore.phase).toBe('starting');
  });

  it('populates modelReady from isTtsReady', async () => {
    vi.mocked(isTtsReady).mockResolvedValue(false);

    await hydrateOverview('nb-001');

    expect(audioOverviewStore.modelReady).toBe(false);
  });

  it('reopen-with-in-flight run polls until it settles, then re-hydrates to ready', async () => {
    vi.useFakeTimers();
    try {
      // In flight on open (poll probe #1 still true, #2 false); the persisted record
      // only becomes readable once the run finishes.
      vi.mocked(isOverviewGenerating)
        .mockResolvedValueOnce(true)
        .mockResolvedValueOnce(true)
        .mockResolvedValue(false);
      vi.mocked(getAudioOverviewStatus)
        .mockResolvedValueOnce(null)
        .mockResolvedValue(record({ status: 'ready' }));

      await hydrateOverview('nb-001');
      expect(audioOverviewStore.overviewStatus).toBe('generating');

      await vi.advanceTimersByTimeAsync(1500); // probe #2 → still generating, reschedules
      expect(audioOverviewStore.overviewStatus).toBe('generating');

      await vi.advanceTimersByTimeAsync(1500); // probe → false → re-hydrate → ready
      expect(audioOverviewStore.overviewStatus).toBe('ready');
      expect(audioOverviewStore.overviewPath).toBe('/data/notebooks/nb-001/overview.wav');
    } finally {
      vi.useRealTimers();
    }
  });

  it('a notebook switch cancels an in-flight poll (no stale write)', async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(isOverviewGenerating).mockResolvedValue(true);
      vi.mocked(getAudioOverviewStatus).mockResolvedValue(null);

      await hydrateOverview('nb-001');
      expect(audioOverviewStore.overviewStatus).toBe('generating');

      // A later hydrate (notebook switch) bumps the generation token; the pending poll
      // must not probe/write for the superseded notebook.
      vi.mocked(isOverviewGenerating).mockResolvedValue(false);
      vi.mocked(getAudioOverviewStatus).mockResolvedValue(record({ status: 'ready' }));
      await hydrateOverview('nb-002');
      expect(audioOverviewStore.overviewStatus).toBe('ready');

      const probeCalls = vi.mocked(isOverviewGenerating).mock.calls.length;
      await vi.advanceTimersByTimeAsync(1500);
      expect(vi.mocked(isOverviewGenerating).mock.calls.length).toBe(probeCalls);
      expect(audioOverviewStore.overviewStatus).toBe('ready');
    } finally {
      vi.useRealTimers();
    }
  });
});

// ---------------------------------------------------------------------------
// generateOverview
// ---------------------------------------------------------------------------

describe('generateOverview', () => {
  it('sets status to generating immediately on call', async () => {
    let resolvePromise: (path: string) => void = () => {};
    vi.mocked(synthesizeOverview).mockImplementation(
      () => new Promise((resolve) => (resolvePromise = resolve))
    );

    const promise = generateOverview('nb-001', 'medium');
    expect(audioOverviewStore.overviewStatus).toBe('generating');
    expect(audioOverviewStore.phase).toBe('starting');

    resolvePromise('/data/notebooks/nb-001/overview.wav');
    await promise;
  });

  it('maps TtsPhase chunk events to the phase/turn/total fields', async () => {
    let capturedOnProgress: Progress = () => {};
    vi.mocked(synthesizeOverview).mockImplementation(async (_id, _len, onProgress) => {
      capturedOnProgress = onProgress as Progress;
      return new Promise(() => {}); // never resolves within this test
    });

    void generateOverview('nb-001', 'short');
    await Promise.resolve();

    capturedOnProgress({ type: 'chunk', data: { synthesizing: { turn: 2, total: 6 } } });
    expect(audioOverviewStore.phase).toBe('synthesizing');
    expect(audioOverviewStore.turn).toBe(2);
    expect(audioOverviewStore.total).toBe(6);

    capturedOnProgress({ type: 'chunk', data: 'stitching' });
    expect(audioOverviewStore.phase).toBe('stitching');
    expect(audioOverviewStore.turn).toBeNull();

    capturedOnProgress({ type: 'chunk', data: 'encoding' });
    expect(audioOverviewStore.phase).toBe('encoding');
  });

  it('on success, sets status ready with the resolved path and a fresh generatedAt', async () => {
    vi.mocked(synthesizeOverview).mockResolvedValue('/data/notebooks/nb-001/overview.wav');

    await generateOverview('nb-001', 'long');

    expect(audioOverviewStore.overviewStatus).toBe('ready');
    expect(audioOverviewStore.overviewPath).toBe('/data/notebooks/nb-001/overview.wav');
    expect(audioOverviewStore.generatedAt).toBeTruthy();
    expect(audioOverviewStore.phase).toBe('idle');
  });

  it('on a genuine failure, sets status failed with the error message', async () => {
    vi.mocked(synthesizeOverview).mockRejectedValue({ kind: 'Tts', message: 'no backend' });
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await generateOverview('nb-001', 'medium');

    expect(audioOverviewStore.overviewStatus).toBe('failed');
    expect(audioOverviewStore.errorMessage).toBe('no backend');
    consoleSpy.mockRestore();
  });

  it('[M2] on Cancelled, re-hydrates instead of surfacing a failed state', async () => {
    vi.mocked(synthesizeOverview).mockRejectedValue({
      kind: 'Cancelled',
      message: 'overview generation cancelled'
    });
    // The re-hydrate call after cancel finds the prior persisted ready row untouched.
    vi.mocked(getAudioOverviewStatus).mockResolvedValue(record({ status: 'ready' }));

    await generateOverview('nb-001', 'medium');

    expect(audioOverviewStore.overviewStatus).toBe('ready');
    expect(audioOverviewStore.errorMessage).toBeNull();
  });

  it('[M2] on Cancelled with no prior record, returns to "none" (never a bare error)', async () => {
    vi.mocked(synthesizeOverview).mockRejectedValue({ kind: 'Cancelled', message: 'cancelled' });
    vi.mocked(getAudioOverviewStatus).mockResolvedValue(null);

    await generateOverview('nb-001', 'medium');

    expect(audioOverviewStore.overviewStatus).toBe('none');
  });
});

// ---------------------------------------------------------------------------
// cancelOverview
// ---------------------------------------------------------------------------

describe('cancelOverview', () => {
  it('calls cancelSynthesis with the notebookId', async () => {
    vi.mocked(cancelSynthesis).mockResolvedValue(true);

    await cancelOverview('nb-001');

    expect(cancelSynthesis).toHaveBeenCalledWith('nb-001');
  });

  it('swallows a cancelSynthesis failure (logs, does not throw)', async () => {
    vi.mocked(cancelSynthesis).mockRejectedValue(new Error('IPC error'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await expect(cancelOverview('nb-001')).resolves.toBeUndefined();

    expect(consoleSpy).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// canGenerate — derived gating
// ---------------------------------------------------------------------------

describe('canGenerate', () => {
  it('is false with zero selected sources, even when the model is ready', async () => {
    mockSourcesStore._setSources([{ selected: 0 }]);
    vi.mocked(isTtsReady).mockResolvedValue(true);
    await hydrateOverview('nb-001');

    expect(audioOverviewStore.canGenerate).toBe(false);
  });

  it('is false when the TTS model is not ready, even with sources selected', async () => {
    mockSourcesStore._setSources([{ selected: 1 }]);
    vi.mocked(isTtsReady).mockResolvedValue(false);
    await hydrateOverview('nb-001');

    expect(audioOverviewStore.canGenerate).toBe(false);
  });

  it('is false while a generation is already in flight', async () => {
    mockSourcesStore._setSources([{ selected: 1 }]);
    vi.mocked(isTtsReady).mockResolvedValue(true);
    vi.mocked(isOverviewGenerating).mockResolvedValue(true);
    await hydrateOverview('nb-001');

    expect(audioOverviewStore.overviewStatus).toBe('generating');
    expect(audioOverviewStore.canGenerate).toBe(false);
  });

  it('is true with ≥1 selected source, a ready model, and no run in flight', async () => {
    mockSourcesStore._setSources([{ selected: 0 }, { selected: 1 }]);
    vi.mocked(isTtsReady).mockResolvedValue(true);
    await hydrateOverview('nb-001');

    expect(audioOverviewStore.canGenerate).toBe(true);
  });
});
