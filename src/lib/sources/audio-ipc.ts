// Typed IPC wrappers for the Audio Overview Tauri commands (#29). Mirrors the
// Channel-streaming pattern in chat/ipc.ts and sources/ipc.ts. All guards with `isTauri()`.

import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
import type { StreamEvent } from './types.js';

/** Requested script length. Wire strings are snake_case (lowercase); mirrors lens-core `dialogue::Length`. */
export type Length = 'short' | 'medium' | 'long';

// SYNC-CHECK: wire strings mirror lens-core `dialogue::OverviewFormat` (serde snake_case).
export type OverviewFormat = 'deep_dive' | 'brief' | 'critique' | 'debate';

// SYNC-CHECK: mirrors lens-core `dialogue::Speaker` (serde snake_case).
export type TranscriptSpeaker = 'host' | 'guest';

/** One spoken line of the persisted dialogue script — powers the Transcript tab. */
export interface TranscriptTurn {
  speaker: TranscriptSpeaker;
  text: string;
}

/** The persisted `DialogueScript` (extra per-turn fields like emotion/source_ids are ignored here). */
export interface TranscriptScript {
  turns: TranscriptTurn[];
}

/** Parameters the setup modal supplies to a generate run. */
export interface OverviewSetup {
  length: Length;
  format: OverviewFormat;
  /** Human-readable target language (e.g. "Spanish"); omit for the default. */
  language?: string;
  /** Free-text steer; omit or blank for none. */
  focus?: string;
}

// SYNC-CHECK: mirrors lens-core AudioOverviewStatus (see that module for the status model).
export type AudioOverviewStatus = 'ready' | 'failed' | 'stale' | 'missing';

// SYNC-CHECK: must match lens-core/src/audio_overview.rs AudioOverviewRecord (serde field names).
export interface AudioOverviewRecord {
  path: string;
  generated_at: string;
  status: AudioOverviewStatus;
  source_set_hash: string;
  script: TranscriptScript | null;
}

// SYNC-CHECK: must match lens-core/src/tts/mod.rs TtsPhase (externally tagged, snake_case).
export type TtsPhase = { synthesizing: { turn: number; total: number } } | 'stitching' | 'encoding';

/**
 * Generates + synthesizes the per-notebook Audio Overview at `length`, streaming
 * `StreamEvent<TtsPhase>` progress over `onProgress`. Resolves with the absolute WAV
 * path — feed it directly to `convertFileSrc()` for immediate playback; do NOT
 * re-fetch status for the just-finished run. A user cancel rejects with
 * `LensError{kind:"Cancelled"}` — treat that as "return to idle", not an error.
 */
export async function synthesizeOverview(
  notebookId: string,
  setup: OverviewSetup,
  onProgress: (e: StreamEvent<TtsPhase>) => void
): Promise<string> {
  if (!isTauri()) throw new Error('synthesizeOverview: not running under Tauri');
  const channel = new Channel<StreamEvent<TtsPhase>>();
  channel.onmessage = onProgress;
  const focus = setup.focus?.trim();
  return invoke<string>('synthesize_overview', {
    notebookId,
    length: setup.length,
    format: setup.format,
    language: setup.language ?? null,
    focus: focus ? focus : null,
    onProgress: channel
  });
}

/**
 * One cheap LLM call over the selected source titles that proposes a short focus
 * phrase for the Focus field. Returns an empty string outside Tauri.
 */
export async function suggestOverviewFocus(notebookId: string): Promise<string> {
  if (!isTauri()) return '';
  return invoke<string>('suggest_overview_focus', { notebookId });
}

/**
 * Hydration-only: the persisted (and disk-reconciled) overview record, or `null`
 * when no overview was ever generated. `status` is authoritative — never recompute
 * a source-set hash on the frontend.
 */
export async function getAudioOverviewStatus(
  notebookId: string
): Promise<AudioOverviewRecord | null> {
  if (!isTauri()) return null;
  return invoke<AudioOverviewRecord | null>('get_audio_overview_status', { notebookId });
}

/** Detects a run started before a notebook switch. */
export async function isOverviewGenerating(notebookId: string): Promise<boolean> {
  if (!isTauri()) return false;
  return invoke<boolean>('is_overview_generating', { notebookId });
}

/** Covers either the dialogue or the TTS phase. */
export async function cancelSynthesis(notebookId: string): Promise<boolean> {
  if (!isTauri()) return false;
  return invoke<boolean>('cancel_synthesis', { notebookId });
}
