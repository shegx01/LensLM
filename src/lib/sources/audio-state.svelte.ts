// Per-notebook Audio Overview lifecycle store (Svelte 5 runes, module singleton).
// Mirrors sources-state.svelte.ts's auto-refresh-on-active-notebook pattern. Hydrates
// from the persisted + disk-reconciled backend record on notebook open — `status` is
// AUTHORITATIVE, this module never recomputes a source-set hash itself.

import {
  synthesizeOverview,
  getAudioOverviewStatus,
  isOverviewGenerating,
  cancelSynthesis,
  type AudioOverviewStatus,
  type Length,
  type TtsPhase
} from './audio-ipc.js';
import type { StreamEvent } from './types.js';
import { sourcesStore } from './sources-state.svelte.js';
import { notebookStore } from '$lib/notebooks/notebooks-state.svelte.js';
import { isTtsReady } from '$lib/onboarding/system-check.js';

export type OverviewState = 'none' | 'generating' | AudioOverviewStatus;
export type OverviewPhase = 'idle' | 'starting' | 'synthesizing' | 'stitching' | 'encoding';

let overviewStatus = $state<OverviewState>('none');
let phase = $state<OverviewPhase>('idle');
let turn = $state<number | null>(null);
let total = $state<number | null>(null);
let overviewPath = $state<string | null>(null);
let generatedAt = $state<string | null>(null);
let errorMessage = $state<string | null>(null);
let modelReady = $state(false);

// A stale hydrate resolving after a newer one (fast notebook switching) must not
// clobber the current notebook's state — same guard shape as chat-state's streamGeneration.
let hydrateGeneration = 0;

const IN_FLIGHT_POLL_MS = 1500;

export const audioOverviewStore = {
  get overviewStatus() {
    return overviewStatus;
  },
  get phase() {
    return phase;
  },
  get turn() {
    return turn;
  },
  get total() {
    return total;
  },
  get overviewPath() {
    return overviewPath;
  },
  get generatedAt() {
    return generatedAt;
  },
  get errorMessage() {
    return errorMessage;
  },
  get modelReady() {
    return modelReady;
  },
  /** Generate is allowed with a ready TTS model, ≥1 selected source, and no run already in flight. */
  get canGenerate(): boolean {
    return modelReady && overviewStatus !== 'generating' && sourcesStore.selectedCount > 0;
  }
};

function resetOverviewState(): void {
  overviewStatus = 'none';
  phase = 'idle';
  turn = null;
  total = null;
  overviewPath = null;
  generatedAt = null;
  errorMessage = null;
}

/** Normalizes a Tauri invoke rejection into `{kind, message}` — mirrors chat-state.svelte.ts's toLensError. */
function toLensError(err: unknown): { kind: string; message: string } {
  if (err && typeof err === 'object' && 'kind' in err && 'message' in err) {
    return err as { kind: string; message: string };
  }
  return { kind: 'Internal', message: err instanceof Error ? err.message : String(err) };
}

/** Hydrate from the persisted record + in-flight generation probe (notebook-open/restart). */
export async function hydrateOverview(notebookId: string): Promise<void> {
  const gen = ++hydrateGeneration;
  resetOverviewState();
  try {
    const [record, generating, ready] = await Promise.all([
      getAudioOverviewStatus(notebookId),
      isOverviewGenerating(notebookId),
      isTtsReady()
    ]);
    if (gen !== hydrateGeneration) return; // superseded by a later hydrate/notebook switch

    modelReady = ready;
    if (generating) {
      overviewStatus = 'generating';
      phase = 'starting';
      // A run started elsewhere (e.g. before a notebook switch) emits no progress
      // to this session; poll until it settles, then re-hydrate the persisted record.
      pollInFlightOverview(notebookId, gen);
      return;
    }
    if (record) {
      overviewStatus = record.status;
      overviewPath = record.path;
      generatedAt = record.generated_at;
    }
  } catch (err) {
    console.error('hydrateOverview: failed for notebook', notebookId, err);
  }
}

/**
 * Light poll for an out-of-session in-flight run (see hydrateOverview). Re-schedules
 * itself only while `gen` is still the latest hydrate and the local status is still
 * 'generating', so a notebook switch (which bumps `hydrateGeneration`) cancels it with
 * no overlapping polls or stale writes. Re-hydrates once the run settles.
 */
function pollInFlightOverview(notebookId: string, gen: number): void {
  setTimeout(() => {
    if (gen !== hydrateGeneration || overviewStatus !== 'generating') return;
    void (async () => {
      let stillGenerating: boolean;
      try {
        stillGenerating = await isOverviewGenerating(notebookId);
      } catch (err) {
        console.error('pollInFlightOverview: probe failed for notebook', notebookId, err);
        return;
      }
      if (gen !== hydrateGeneration || overviewStatus !== 'generating') return;
      if (stillGenerating) {
        pollInFlightOverview(notebookId, gen);
      } else {
        await hydrateOverview(notebookId);
      }
    })();
  }, IN_FLIGHT_POLL_MS);
}

function applyPhaseEvent(data: TtsPhase): void {
  if (data === 'stitching' || data === 'encoding') {
    phase = data;
    turn = null;
    total = null;
    return;
  }
  phase = 'synthesizing';
  turn = data.synthesizing.turn;
  total = data.synthesizing.total;
}

/**
 * Kicks off generation. Resolves once the run reaches a terminal state (ready,
 * failed, or — on cancel — re-hydrated back to the prior persisted state).
 */
export async function generateOverview(notebookId: string, length: Length): Promise<void> {
  const gen = ++hydrateGeneration;
  overviewStatus = 'generating';
  phase = 'starting';
  turn = null;
  total = null;
  errorMessage = null;

  try {
    const path = await synthesizeOverview(notebookId, length, (e: StreamEvent<TtsPhase>) => {
      if (e.type === 'chunk') applyPhaseEvent(e.data);
    });
    if (gen !== hydrateGeneration) return;
    overviewStatus = 'ready';
    overviewPath = path;
    generatedAt = new Date().toISOString();
    phase = 'idle';
  } catch (err) {
    if (gen !== hydrateGeneration) return;
    phase = 'idle';
    const lensErr = toLensError(err);
    if (lensErr.kind === 'Cancelled') {
      // [M2] No row was written on cancel — re-hydrate so a regenerate-over-ready
      // returns to its prior persisted state instead of a bare "none".
      await hydrateOverview(notebookId);
      return;
    }
    overviewStatus = 'failed';
    errorMessage = lensErr.message;
    console.error('generateOverview: failed for notebook', notebookId, err);
  }
}

/** Cancels the in-flight generation; the awaited `generateOverview` call settles the state. */
export async function cancelOverview(notebookId: string): Promise<void> {
  try {
    await cancelSynthesis(notebookId);
  } catch (err) {
    console.error('cancelOverview: failed for notebook', notebookId, err);
  }
}

/** Reset all state. Call in `afterEach` to prevent cross-test bleed from module-level `$state` globals. */
export function resetAudioOverviewStore(): void {
  hydrateGeneration++;
  modelReady = false;
  resetOverviewState();
}

/** Dispose fn from `$effect.root`. Exposed so tests can stop the auto-refresh effect. */
export const disposeAudioAutoRefresh: () => void = $effect.root(() => {
  $effect(() => {
    const id = notebookStore.activeNotebookId;
    if (id) {
      void hydrateOverview(id);
    } else {
      hydrateGeneration++;
      modelReady = false;
      resetOverviewState();
    }
  });
});
