// Static ASR catalog data: engine rows, cloud presets, language options, and the
// per-engine capability matrix. Pure data + lookups — no Tauri calls.

import type { AppleAsrAvailability } from '$lib/asr/ipc.js';
import type { AsrLang, CloudAsrProvider } from '$lib/theme/types.js';

/** UI engine row id. Distinct from the wire `AsrConfig.backend` token — see `asrBackendToken`. */
export type AsrEngineId = 'automatic' | 'apple_native' | 'local_whisper' | 'cloud';

export interface AsrEngineCatalogEntry {
  id: AsrEngineId;
  label: string;
  description: string;
}

// SYNC-CHECK: tokens must match lens-core/src/config.rs AsrConfig.backend doc and
// AsrBackend::from_opt_str (asr/mod.rs) — "" | "apple_native" | "local_whisper" |
// "cloud". Automatic has no engine of its own; "" is router-resolved (router.rs:30-53).
export const ASR_ENGINE_CATALOG: AsrEngineCatalogEntry[] = [
  {
    id: 'automatic',
    label: 'Automatic',
    description: 'Prefers on-device Apple transcription where supported, otherwise Local Whisper.'
  },
  {
    id: 'apple_native',
    label: 'Apple (on-device)',
    description: 'On-device transcription via the macOS Speech framework. Private, no network.'
  },
  {
    id: 'local_whisper',
    label: 'Local Whisper',
    description: 'On-device transcription via whisper.cpp. Works on any platform.'
  },
  {
    id: 'cloud',
    label: 'Cloud',
    description: 'Sends audio to a cloud provider for transcription. Requires consent.'
  }
];

/**
 * Why the Apple row is unavailable, or `null` when it is available. Blockers read
 * differently because they call for different actions; a `null` availability means
 * the probe has not answered, which is not itself a claim about the device.
 */
export function appleAsrUnavailableReason(
  availability: AppleAsrAvailability | null
): string | null {
  if (availability === null) return 'Unavailable — availability could not be determined.';
  if (availability === 'available') return null;
  if (availability === 'not_built') {
    return 'Unavailable — this build of LensLM does not include the Apple speech bridge.';
  }
  const cause = availability.unsupported;
  if (cause === 'not_apple_silicon') return 'Unavailable — needs an Apple silicon Mac.';
  if (cause === 'version_probe_failed') {
    return "Unavailable — this Mac's macOS version could not be read.";
  }
  const { found, required } = cause.macos_too_old;
  return `Unavailable — needs macOS ${required} or later; this Mac runs macOS ${found}.`;
}

/** Maps a UI engine id to the wire `AsrConfig.backend` token (`""` for Automatic). */
export function asrBackendToken(id: AsrEngineId): string {
  return id === 'automatic' ? '' : id;
}

/**
 * Maps a persisted `AsrConfig.backend` token to its UI engine id. Any token
 * `AsrBackend::from_opt_str` does not recognize (including `""`) resolves the
 * same way `None` does at the router — treat it as Automatic.
 */
export function asrEngineIdFromBackend(backend: string): AsrEngineId {
  switch (backend) {
    case 'apple_native':
      return 'apple_native';
    case 'local_whisper':
      return 'local_whisper';
    case 'cloud':
      return 'cloud';
    default:
      return 'automatic';
  }
}

export interface CloudAsrPreset {
  base_url: string;
  model: string;
}

// SYNC-CHECK: base_url/model must match what each adapter actually requests —
// openai_compat.rs posts to `{base_url}/v1/audio/transcriptions`; deepgram.rs
// posts to `{base_url}/v1/listen` — and lens-core/src/asr/cloud/mod.rs's
// default_base_url/default_model, which AppConfig::normalize fills blanks with.
export const CLOUD_ASR_PRESETS = {
  open_ai_compatible: { base_url: 'https://api.openai.com', model: 'whisper-1' },
  deepgram: { base_url: 'https://api.deepgram.com', model: 'nova-3' }
} satisfies Record<CloudAsrProvider, CloudAsrPreset>;

export interface AsrLanguageOption {
  /** `null` = auto-detect. */
  value: AsrLang | null;
  label: string;
}

// Exact Rust variant tokens (theme/types.ts AsrLang) — never hand-lowercase these.
export const ASR_LANGUAGE_OPTIONS: AsrLanguageOption[] = [
  { value: null, label: 'Auto-detect' },
  { value: 'En', label: 'English' },
  { value: 'De', label: 'German' },
  { value: 'Fr', label: 'French' },
  { value: 'Es', label: 'Spanish' },
  { value: 'It', label: 'Italian' },
  { value: 'Pt', label: 'Portuguese' },
  { value: 'Nl', label: 'Dutch' },
  { value: 'Ru', label: 'Russian' },
  { value: 'Zh', label: 'Chinese' },
  { value: 'Ja', label: 'Japanese' },
  { value: 'Ko', label: 'Korean' }
];

/** Mirrors `AsrConfig.language`'s free-text hatch (`{ Other: string }`) for a code not in `ASR_LANGUAGE_OPTIONS`. */
export type AsrLanguageValue = AsrLang | { Other: string } | null;

export function isOtherAsrLanguage(value: AsrLanguageValue): value is { Other: string } {
  return typeof value === 'object' && value !== null && 'Other' in value;
}

export type AsrCapabilityMode = 'honoured' | 'ignored' | 'reroutes';

export interface AsrCapabilityMatrixEntry {
  language: AsrCapabilityMode;
  translate: AsrCapabilityMode;
}

/** Per-engine tri-state capability; Automatic resolves to one of these at runtime, so it has no row of its own. */
export const ASR_CAPABILITY_MATRIX: Record<
  Exclude<AsrEngineId, 'automatic'>,
  AsrCapabilityMatrixEntry
> = {
  // language: src-tauri/src/asr/mod.rs:278 (lang_to_bcp47); translate reroute
  // chain is cited once, on appleTranslateRerouteNotice below.
  apple_native: { language: 'honoured', translate: 'reroutes' },
  // whisper.rs::resolve_transcribe_params forwards both — both honoured.
  local_whisper: { language: 'honoured', translate: 'honoured' },
  // deepgram.rs:52, openai_compat.rs:53-54 (language honoured); neither reads config.translate.
  cloud: { language: 'honoured', translate: 'ignored' }
};

export function asrCapability(
  engine: Exclude<AsrEngineId, 'automatic'>,
  field: 'language' | 'translate'
): AsrCapabilityMode {
  return ASR_CAPABILITY_MATRIX[engine][field];
}

/**
 * Notice copy for enabling translate while Apple is the resolved engine: the
 * reroute (lib.rs:2099-2104) sends it through the LocalWhisper arm (lib.rs:2158),
 * which errors out if no Whisper model is on disk (lib.rs:2343-2348).
 */
export function appleTranslateRerouteNotice(whisperModelDownloaded: boolean): string {
  const base = 'Enabling translate reroutes Apple transcription to Local Whisper.';
  return whisperModelDownloaded
    ? base
    : `${base} No Whisper model is downloaded yet, so transcription will fail until one is.`;
}

/**
 * Notice copy for enabling translate under Automatic: whichever engine the
 * router resolves to, translate lands on Local Whisper either way (directly,
 * or via the Apple reroute above), so the same no-model failure applies.
 */
export function automaticTranslateNotice(whisperModelDownloaded: boolean): string {
  const base =
    'Automatic prefers on-device Apple transcription where supported, otherwise Local Whisper — enabling translate always routes through Local Whisper either way.';
  return whisperModelDownloaded
    ? base
    : `${base} No Whisper model is downloaded yet, so translate will fail until one is.`;
}
