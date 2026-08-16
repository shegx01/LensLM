import { describe, expect, it } from 'vitest';
import {
  ASR_CAPABILITY_MATRIX,
  ASR_ENGINE_CATALOG,
  ASR_LANGUAGE_OPTIONS,
  CLOUD_ASR_PRESETS,
  appleTranslateRerouteNotice,
  asrBackendToken,
  asrCapability,
  asrEngineIdFromBackend,
  isOtherAsrLanguage
} from './catalog.js';
import type { AsrEngineId } from './catalog.js';
import type { AsrLang } from '$lib/theme/types.js';

describe('ASR_ENGINE_CATALOG', () => {
  it('has exactly the four contract engine rows', () => {
    expect(ASR_ENGINE_CATALOG.map((e) => e.id)).toEqual([
      'automatic',
      'apple_native',
      'local_whisper',
      'cloud'
    ]);
  });

  it("Automatic's description names no Cloud leg and no hardware prediction", () => {
    const automatic = ASR_ENGINE_CATALOG.find((e) => e.id === 'automatic')!;
    expect(automatic.description.toLowerCase()).not.toContain('cloud');
    expect(automatic.description.toLowerCase()).not.toMatch(/apple silicon|macos \d/);
  });

  it('Apple row states an unavailable reason naming both possible causes', () => {
    const apple = ASR_ENGINE_CATALOG.find((e) => e.id === 'apple_native')!;
    expect(apple.unavailableReason).toMatch(/apple silicon/i);
    expect(apple.unavailableReason).toMatch(/macos 26/i);
  });
});

describe('backend token mapping', () => {
  const cases: Array<[AsrEngineId, string]> = [
    ['automatic', ''],
    ['apple_native', 'apple_native'],
    ['local_whisper', 'local_whisper'],
    ['cloud', 'cloud']
  ];

  it.each(cases)('asrBackendToken(%s) -> %j', (id, token) => {
    expect(asrBackendToken(id)).toBe(token);
  });

  it.each(cases)('asrEngineIdFromBackend(%j) -> %s', (id, token) => {
    expect(asrEngineIdFromBackend(token)).toBe(id);
  });

  it('treats an unrecognized token the same as the router treats None: as Automatic', () => {
    expect(asrEngineIdFromBackend('bogus')).toBe('automatic');
  });
});

// The adapters' actual request paths (`{base_url}/v1/audio/transcriptions`,
// `{base_url}/v1/listen`) are proven end-to-end against real wiremock requests
// in lens-core/tests/asr_capability_matrix_sync_check.rs, not by reading Rust
// source text here — a source-text match can't tell a live path from a comment.
describe('CLOUD_ASR_PRESETS', () => {
  it("OpenAI-compatible base_url has no path that would double up with the adapter's own suffix", () => {
    expect(CLOUD_ASR_PRESETS.open_ai_compatible.base_url).not.toContain('/v1');
  });

  it("Deepgram base_url has no path that would double up with the adapter's own suffix", () => {
    expect(CLOUD_ASR_PRESETS.deepgram.base_url).not.toContain('/v1');
  });
});

describe('ASR_LANGUAGE_OPTIONS', () => {
  it('is auto-detect plus the 11 named Rust tokens, in PascalCase', () => {
    const values = ASR_LANGUAGE_OPTIONS.map((o) => o.value);
    const named = values.slice(1) as AsrLang[];
    expect(named).toEqual(['En', 'De', 'Fr', 'Es', 'It', 'Pt', 'Nl', 'Ru', 'Zh', 'Ja', 'Ko']);
    for (const v of named) {
      expect(v[0]).toBe(v[0].toUpperCase());
    }
  });

  it('every option has a human label', () => {
    for (const option of ASR_LANGUAGE_OPTIONS) {
      expect(option.label.length).toBeGreaterThan(0);
    }
  });
});

describe('isOtherAsrLanguage', () => {
  it('narrows the free-text hatch', () => {
    expect(isOtherAsrLanguage({ Other: 'xx' })).toBe(true);
    expect(isOtherAsrLanguage('En')).toBe(false);
    expect(isOtherAsrLanguage(null)).toBe(false);
  });
});

// These literal cells are cross-checked against real engine behaviour in
// lens-core/tests/asr_capability_matrix_sync_check.rs (parses this same object
// literal); pinning them here too catches an accidental TS-only edit early.
describe('ASR_CAPABILITY_MATRIX', () => {
  it('apple_native honours language and reroutes translate', () => {
    expect(ASR_CAPABILITY_MATRIX.apple_native.language).toBe('honoured');
    expect(ASR_CAPABILITY_MATRIX.apple_native.translate).toBe('reroutes');
  });

  it('local_whisper honours both language and translate', () => {
    expect(ASR_CAPABILITY_MATRIX.local_whisper.language).toBe('honoured');
    expect(ASR_CAPABILITY_MATRIX.local_whisper.translate).toBe('honoured');
  });

  it('cloud honours language but ignores translate', () => {
    expect(ASR_CAPABILITY_MATRIX.cloud.language).toBe('honoured');
    expect(ASR_CAPABILITY_MATRIX.cloud.translate).toBe('ignored');
  });
});

describe('asrCapability', () => {
  it('reads the exact [engine][field] cell for every combination', () => {
    for (const engine of ['apple_native', 'local_whisper', 'cloud'] as const) {
      for (const field of ['language', 'translate'] as const) {
        expect(asrCapability(engine, field)).toBe(ASR_CAPABILITY_MATRIX[engine][field]);
      }
    }
  });
});

describe('appleTranslateRerouteNotice', () => {
  it('names the reroute to Local Whisper regardless of model presence', () => {
    expect(appleTranslateRerouteNotice(true)).toContain(
      'reroutes Apple transcription to Local Whisper'
    );
    expect(appleTranslateRerouteNotice(false)).toContain(
      'reroutes Apple transcription to Local Whisper'
    );
  });

  it('warns of failure only when no Whisper model is downloaded', () => {
    expect(appleTranslateRerouteNotice(true)).not.toMatch(/fail/i);
    expect(appleTranslateRerouteNotice(false)).toMatch(/fail/i);
  });
});
