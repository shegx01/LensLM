// Thin IPC wrappers for the Local Whisper ASR commands. Mirrors the TTS wrappers in
// $lib/onboarding/system-check.ts (same DownloadProgress channel shape/semantics).

import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
import type { DownloadProgress } from '$lib/onboarding/system-check.js';

// SYNC-CHECK: must match src-tauri/src/commands/system.rs WhisperModelInfo (plain
// derive(Serialize), no rename_all -> snake_case keys, as declared below).
// `is_default` is computed on the DTO, not a registry field.
export interface WhisperModelInfo {
  id: string;
  approx_mb: number;
  is_default: boolean;
}

/** The Whisper model registry (tiny / base / small) with size labels. Returns `[]` outside Tauri. */
export async function listWhisperModels(): Promise<WhisperModelInfo[]> {
  if (!isTauri()) return [];
  return invoke<WhisperModelInfo[]>('list_whisper_models');
}

/** Whether the given Whisper model is already on disk. Returns `false` outside Tauri. */
export async function whisperModelDownloaded(model: string): Promise<boolean> {
  if (!isTauri()) return false;
  return invoke<boolean>('whisper_model_downloaded', { model });
}

/** Clamp a 0..1 ratio to an integer 0..100 percentage. */
function toPct(received: number, total: number | null): number | null {
  if (total === null || total <= 0) return null;
  return Math.min(100, Math.max(0, Math.round((received / total) * 100)));
}

/**
 * Download a Whisper ggml model, streaming 0–100% progress (`null` while the total is
 * unknown). No-op outside Tauri. Mirrors `downloadTtsModel` in system-check.ts.
 */
export async function downloadWhisperModel(
  model: string,
  onProgress: (pct: number | null) => void
): Promise<void> {
  if (!isTauri()) return;
  const channel = new Channel<DownloadProgress>();
  channel.onmessage = (p) => {
    if (p.done) {
      onProgress(100);
      return;
    }
    onProgress(toPct(p.received, p.total));
  };
  // `on_progress`, NOT `onProgress`: the command declares `rename_all = "snake_case"`, so
  // Tauri looks up the snake_case key and a camelCase one fails with `missing required key`.
  await invoke<void>('download_whisper_model', { model, on_progress: channel });
}
