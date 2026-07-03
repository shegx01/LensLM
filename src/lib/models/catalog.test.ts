// Unit tests for cloud model-picker ordering + text-capability filter in catalog.ts.

import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ModelInfo } from './types.js';

vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => true,
  invoke: vi.fn()
}));

import { invoke } from '@tauri-apps/api/core';
import { listCloudModelOptions, formatCompact, formatUsd } from './catalog.js';

/** Minimal `ModelInfo` defaulting to text-capable modalities. */
function model(overrides: Partial<ModelInfo> & { name: string }): ModelInfo {
  return {
    id: overrides.name,
    reasoning: false,
    reasoning_options: [],
    tool_call: false,
    temperature: false,
    modalities: { input: ['text'], output: ['text'] },
    context_limit: null,
    output_limit: null,
    open_weights: false,
    last_updated: null,
    release_date: null,
    ...overrides
  };
}

beforeEach(() => {
  vi.mocked(invoke).mockReset();
});

describe('listCloudModelOptions ordering', () => {
  it('orders by last_updated desc, with release_date then label tiebreaks, undated last', async () => {
    vi.mocked(invoke).mockResolvedValue({
      // dated, oldest last_updated
      older: model({ name: 'Older', last_updated: '2025-01-01', release_date: '2025-01-01' }),
      // undated entirely → must sort last
      undatedB: model({ name: 'Zeta Undated' }),
      // newest last_updated → must sort first
      newest: model({ name: 'Newest', last_updated: '2025-11-24', release_date: '2025-10-01' }),
      // another undated → alphabetical among undated
      undatedA: model({ name: 'Alpha Undated' }),
      // same last_updated as `tieOlderRelease`; newer release_date wins
      tieNewerRelease: model({
        name: 'Tie Newer Release',
        last_updated: '2025-06-01',
        release_date: '2025-06-01'
      }),
      tieOlderRelease: model({
        name: 'Tie Older Release',
        last_updated: '2025-06-01',
        release_date: '2025-05-01'
      })
    } satisfies Record<string, ModelInfo>);

    const options = await listCloudModelOptions('anthropic');

    expect(options.map((o) => o.label)).toEqual([
      'Newest', // 2025-11-24
      'Tie Newer Release', // 2025-06-01, release 2025-06-01
      'Tie Older Release', // 2025-06-01, release 2025-05-01
      'Older', // 2025-01-01
      'Alpha Undated', // undated, alphabetical
      'Zeta Undated'
    ]);
  });

  it('falls back to label asc when last_updated matches and release_date matches', async () => {
    vi.mocked(invoke).mockResolvedValue({
      b: model({ name: 'Bravo', last_updated: '2025-06-01', release_date: '2025-06-01' }),
      a: model({ name: 'Alpha', last_updated: '2025-06-01', release_date: '2025-06-01' })
    } satisfies Record<string, ModelInfo>);

    const options = await listCloudModelOptions('openai');
    expect(options.map((o) => o.label)).toEqual(['Alpha', 'Bravo']);
  });

  it('treats a model with only release_date as dated (sorts ahead of fully undated)', async () => {
    vi.mocked(invoke).mockResolvedValue({
      undated: model({ name: 'Undated' }),
      releaseOnly: model({ name: 'Release Only', release_date: '2025-03-01' })
    } satisfies Record<string, ModelInfo>);

    const options = await listCloudModelOptions('google');
    expect(options.map((o) => o.label)).toEqual(['Release Only', 'Undated']);
  });
});

describe('listCloudModelOptions text-capability filter', () => {
  it('keeps text-in/text-out models and drops non-text models', async () => {
    vi.mocked(invoke).mockResolvedValue({
      chat: model({ name: 'Chat', modalities: { input: ['text'], output: ['text'] } }),
      // Multimodal-in but still text-out → a usable chat model, kept.
      vision: model({ name: 'Vision', modalities: { input: ['text', 'image'], output: ['text'] } }),
      // Audio/TTS-only output → not a chat model, dropped.
      tts: model({ name: 'Speech', modalities: { input: ['text'], output: ['audio'] } }),
      // Image-input only (no text input) → dropped.
      imageOnly: model({ name: 'Image In', modalities: { input: ['image'], output: ['text'] } }),
      // Embedding-style: text-in but no text-out → dropped.
      embed: model({ name: 'Embed', modalities: { input: ['text'], output: [] } })
    } satisfies Record<string, ModelInfo>);

    const options = await listCloudModelOptions('openai');
    expect(options.map((o) => o.label)).toEqual(['Chat', 'Vision']);
  });

  it('excludes a model with missing/empty modalities (strict: undated AND non-text)', async () => {
    vi.mocked(invoke).mockResolvedValue({
      // Empty modalities → not text-capable, excluded.
      bare: model({ name: 'Bare', modalities: { input: [], output: [] } }),
      // Missing modalities object entirely → excluded (tolerant of malformed entry).
      malformed: model({ name: 'Malformed', modalities: undefined as never }),
      keep: model({ name: 'Keep', modalities: { input: ['text'], output: ['text'] } })
    } satisfies Record<string, ModelInfo>);

    const options = await listCloudModelOptions('anthropic');
    expect(options.map((o) => o.label)).toEqual(['Keep']);
  });

  it('keeps the newest text-capable model FIRST after filtering', async () => {
    vi.mocked(invoke).mockResolvedValue({
      newestText: model({
        name: 'Newest Text',
        last_updated: '2025-12-01',
        modalities: { input: ['text'], output: ['text'] }
      }),
      // Newer date, but audio-only output → filtered out, must NOT win the slot.
      newestAudio: model({
        name: 'Newest Audio',
        last_updated: '2025-12-31',
        modalities: { input: ['text'], output: ['audio'] }
      }),
      olderText: model({
        name: 'Older Text',
        last_updated: '2025-01-01',
        modalities: { input: ['text'], output: ['text'] }
      })
    } satisfies Record<string, ModelInfo>);

    const options = await listCloudModelOptions('openai');
    expect(options.map((o) => o.label)).toEqual(['Newest Text', 'Older Text']);
  });
});

describe('formatCompact', () => {
  it('formats K/M/B magnitude boundaries', () => {
    expect(formatCompact(8_000)).toBe('8K');
    expect(formatCompact(200_000)).toBe('200K');
    expect(formatCompact(1_000_000)).toBe('1M');
    expect(formatCompact(1_000_000_000)).toBe('1B');
  });

  it('keeps up to 2 significant decimals and trims trailing zeros', () => {
    expect(formatCompact(1_050_000)).toBe('1.05M');
    expect(formatCompact(128_000)).toBe('128K');
    expect(formatCompact(2_000_000)).toBe('2M');
    expect(formatCompact(1_050_000_000)).toBe('1.05B');
  });

  it('renders values below 1000 as-is', () => {
    expect(formatCompact(0)).toBe('0');
    expect(formatCompact(512)).toBe('512');
  });
});

describe('formatUsd', () => {
  it('drops a trailing .0 and keeps cents when present', () => {
    expect(formatUsd(5)).toBe('5');
    expect(formatUsd(25)).toBe('25');
    expect(formatUsd(2.5)).toBe('2.5');
    expect(formatUsd(0.5)).toBe('0.5');
  });
});
