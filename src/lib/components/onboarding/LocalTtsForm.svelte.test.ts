import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import type {
  TtsEngineCatalogEntry,
  TtsModelStatus,
  TtsVoice
} from '$lib/onboarding/system-check.js';
import LocalTtsForm from './LocalTtsForm.svelte';

const DEFAULT_PRESET_VOICES: TtsVoice[] = [
  { id: 'leo', name: 'Leo', gender: 'male' },
  { id: 'tara', name: 'Tara', gender: 'female' }
];

/** 3-engine catalog matching the shape used by the panel integration tests. */
function catalogFixture(overrides?: { qwenAvailable?: boolean }): TtsEngineCatalogEntry[] {
  const qwenAvailable = overrides?.qwenAvailable ?? false;
  return [
    {
      id: 'orpheus',
      platform: 'cross_platform',
      needs_key: false,
      available: true,
      unavailable_reason: null,
      multilingual: false,
      supported_languages: ['english'],
      preset_voices: DEFAULT_PRESET_VOICES,
      model_size_bytes: 2_300_000_000,
      language_capability_label: 'English only',
      required_model_ids: ['orpheus', 'snac']
    },
    {
      id: 'qwen3_local',
      platform: 'apple_silicon',
      needs_key: false,
      available: qwenAvailable,
      unavailable_reason: qwenAvailable ? null : 'Requires Apple Silicon',
      multilingual: false,
      supported_languages: ['chinese', 'english'],
      preset_voices: DEFAULT_PRESET_VOICES,
      model_size_bytes: 4_500_000_000,
      language_capability_label: '10 languages',
      required_model_ids: []
    },
    {
      id: 'cloud',
      platform: 'cross_platform',
      needs_key: true,
      available: false,
      unavailable_reason: 'Requires an API key',
      multilingual: true,
      supported_languages: [],
      preset_voices: [],
      model_size_bytes: null,
      language_capability_label: 'Multilingual (cloud)',
      required_model_ids: []
    }
  ];
}

function renderLocal(): { unmount: () => void } {
  return render(LocalTtsForm, { props: { catalog: [], active: true } });
}

type ProgressChannel = {
  onmessage: (m: { received: number; total: number | null; done: boolean }) => void;
};

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

/** The corrected engine-level aggregation of per-model tri-states (Orpheus needs
 *  `orpheus` + `snac`): Complete iff both Complete; Partial iff not-all-complete
 *  but at least one Partial; Absent otherwise — notably `{complete, absent}` is
 *  Absent (plain "Download"), NOT a re-download prompt. */
describe('LocalTtsForm — engine status aggregation (corrected tri-state rule)', () => {
  type Matrix = { orpheus: TtsModelStatus; snac: TtsModelStatus };

  async function mountWith(m: Matrix): Promise<void> {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'tts_model_status') {
        return (args as { model: string }).model === 'orpheus' ? m.orpheus : m.snac;
      }
    });
    renderLocal();
    // Wait for the engine radios (proves the catalog fetch resolved) before asserting.
    await screen.findByRole('radio', { name: /orpheus/i });
  }

  it('{complete, complete} → complete (voice engine ready)', async () => {
    await mountWith({ orpheus: 'complete', snac: 'complete' });
    await waitFor(() => expect(screen.getByText(/voice engine ready/i)).toBeInTheDocument());
    expect(
      screen.queryByRole('button', { name: /download voice engine/i })
    ).not.toBeInTheDocument();
  });

  it('{complete, partial} → partial (re-download)', async () => {
    await mountWith({ orpheus: 'complete', snac: 'partial' });
    expect(
      await screen.findByRole('button', { name: /model incomplete.*re-download/i })
    ).toBeInTheDocument();
  });

  it('{partial, absent} → partial (re-download)', async () => {
    await mountWith({ orpheus: 'partial', snac: 'absent' });
    expect(
      await screen.findByRole('button', { name: /model incomplete.*re-download/i })
    ).toBeInTheDocument();
  });

  it('{complete, absent} → absent (plain Download, NOT re-download) — the divergent case', async () => {
    await mountWith({ orpheus: 'complete', snac: 'absent' });
    expect(
      await screen.findByRole('button', { name: /download voice engine/i })
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /re-download/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/voice engine ready/i)).not.toBeInTheDocument();
  });

  it('{absent, absent} → absent (plain Download)', async () => {
    await mountWith({ orpheus: 'absent', snac: 'absent' });
    expect(
      await screen.findByRole('button', { name: /download voice engine/i })
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /re-download/i })).not.toBeInTheDocument();
  });
});

describe('LocalTtsForm — status probe count (AC-5)', () => {
  it('probes each required model exactly once per engine switch (2 for Orpheus, 1 for Qwen), no repeats', async () => {
    let probes: { engine: string; model: string }[] = [];
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: true });
      if (cmd === 'tts_model_status') {
        const a = args as { engine: string; model: string };
        probes.push({ engine: a.engine, model: a.model });
        return a.engine === 'qwen3_local' ? 'absent' : 'complete';
      }
      if (cmd === 'set_config') return null;
    });

    render(LocalTtsForm, { props: { catalog: [], active: true } });

    // Mount probes Orpheus (the default) once per required model.
    const qwenRadio = await screen.findByRole('radio', { name: /qwen3-tts/i });
    await waitFor(() =>
      expect(
        probes
          .filter((p) => p.engine === 'orpheus')
          .map((p) => p.model)
          .sort()
      ).toEqual(['orpheus', 'snac'])
    );

    // One engine switch → Qwen: exactly one probe, the empty-model sentinel.
    probes = [];
    await fireEvent.click(qwenRadio);
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /download voice engine/i })).toBeInTheDocument()
    );
    expect(probes).toEqual([{ engine: 'qwen3_local', model: '' }]);

    // Switch back to Orpheus → exactly one probe per required model, no repeats.
    probes = [];
    await fireEvent.click(screen.getByRole('radio', { name: /orpheus/i }));
    await waitFor(() => expect(screen.getByText(/voice engine ready/i)).toBeInTheDocument());
    expect(probes.map((p) => p.model).sort()).toEqual(['orpheus', 'snac']);
    expect(new Set(probes.map((p) => p.model)).size).toBe(probes.length);
  });
});

describe('LocalTtsForm — post-download re-check', () => {
  it('offers re-download when a finished download fails its presence re-check', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'tts_model_status') return 'absent';
      if (cmd === 'download_tts_model') {
        const ch = (args as { onProgress?: { onmessage?: (m: unknown) => void } }).onProgress;
        ch?.onmessage?.({ received: 100, total: 100, done: true });
        return null;
      }
      if (cmd === 'set_config') return null;
    });

    render(LocalTtsForm, { props: { catalog: [], active: true } });
    await screen.findByRole('radio', { name: /orpheus/i });
    await fireEvent.click(screen.getByRole('button', { name: /download voice engine/i }));

    expect(
      await screen.findByRole('button', { name: /model incomplete.*re-download/i })
    ).toBeInTheDocument();
    expect(screen.queryByText(/voice engine ready/i)).not.toBeInTheDocument();
  });
});

/** Guards the persist path independently of the parent shell. */
describe('LocalTtsForm — reactive persist', () => {
  it('persists default voices after a genuinely-complete download (no Save button)', async () => {
    let written: AppConfig | null = null;
    let onDisk = false;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'tts_model_status') return onDisk ? 'complete' : 'absent';
      if (cmd === 'download_tts_model') {
        onDisk = true;
        const ch = (args as { onProgress?: { onmessage?: (m: unknown) => void } }).onProgress;
        ch?.onmessage?.({ received: 100, total: 100, done: true });
        return null;
      }
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(LocalTtsForm, { props: { catalog: [], active: true } });
    await screen.findByRole('radio', { name: /orpheus/i });
    await fireEvent.click(screen.getByRole('button', { name: /download voice engine/i }));

    await waitFor(() => expect(screen.getByText(/voice engine ready/i)).toBeInTheDocument());
    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).voices).toEqual({ host: 'leo', guest: 'tara' });
    expect((written as unknown as AppConfig).tts.backend).toBe('orpheus');
  });
});

describe('LocalTtsForm — indeterminate progress (null pct)', () => {
  it('qwen3_local: null pct flips downloadIndeterminate and isDownloading stays true', async () => {
    let progressCh: ProgressChannel | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: true });
      if (cmd === 'tts_model_status') return 'absent';
      if (cmd === 'prepare_qwen_model') {
        progressCh = (args as { onProgress: ProgressChannel }).onProgress;
        return new Promise(() => {}); // keep the download in flight
      }
    });

    renderLocal();
    const qwenRadio = await screen.findByRole('radio', { name: /qwen3-tts/i });
    await fireEvent.click(qwenRadio);
    await fireEvent.click(await screen.findByRole('button', { name: /download voice engine/i }));
    await waitFor(() => expect(progressCh).toBeDefined());

    progressCh?.onmessage({ received: 1, total: null, done: false });

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /downloading/i })).toBeInTheDocument()
    );
    expect(screen.queryByText(/% downloaded/)).not.toBeInTheDocument();
    expect(screen.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow');
  });

  it('Orpheus composite loop treats a null pct as an indeterminate phase, not a silent low value', async () => {
    let secondCh: ProgressChannel | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'tts_model_status') return 'absent';
      if (cmd === 'download_tts_model') {
        const a = args as { model: string; onProgress: ProgressChannel };
        if (a.model === 'orpheus') {
          a.onProgress.onmessage({ received: 100, total: 100, done: true });
          return null;
        }
        // 'snac' (second model): report 40% then hold, so the composite reaches
        // 70% before the null tick below — a regression to `null/100 === 0`
        // would silently drop this to 50%, not crash.
        a.onProgress.onmessage({ received: 40, total: 100, done: false });
        secondCh = a.onProgress;
        return new Promise(() => {});
      }
    });

    renderLocal();
    await screen.findByRole('radio', { name: /orpheus/i });
    await fireEvent.click(screen.getByRole('button', { name: /download voice engine/i }));

    await waitFor(() => expect(screen.getByText(/70% downloaded/)).toBeInTheDocument());

    secondCh?.onmessage({ received: 0, total: null, done: false });

    await waitFor(() => expect(screen.queryByText(/% downloaded/)).not.toBeInTheDocument());
    expect(screen.queryByText(/50% downloaded/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /downloading/i })).toBeInTheDocument();
  });
});

describe('LocalTtsForm — cancel on unmount (engine-guarded)', () => {
  it('invokes cancel_prepare on unmount mid-download for qwen3_local', async () => {
    let cancelInvoked = false;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: true });
      if (cmd === 'tts_model_status') return 'absent';
      if (cmd === 'prepare_qwen_model') return new Promise(() => {});
      if (cmd === 'cancel_prepare') {
        cancelInvoked = true;
        return true;
      }
    });

    const { unmount } = renderLocal();
    const qwenRadio = await screen.findByRole('radio', { name: /qwen3-tts/i });
    await fireEvent.click(qwenRadio);
    await fireEvent.click(await screen.findByRole('button', { name: /download voice engine/i }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /downloading/i })).toBeInTheDocument()
    );

    unmount();
    await waitFor(() => expect(cancelInvoked).toBe(true));
  });

  it('does NOT invoke cancel_prepare on unmount mid-download for Orpheus (no cancel path)', async () => {
    let cancelInvoked = false;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'tts_model_status') return 'absent';
      if (cmd === 'download_tts_model') return new Promise(() => {});
      if (cmd === 'cancel_prepare') {
        cancelInvoked = true;
        return true;
      }
    });

    const { unmount } = renderLocal();
    await screen.findByRole('radio', { name: /orpheus/i });
    await fireEvent.click(screen.getByRole('button', { name: /download voice engine/i }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /downloading/i })).toBeInTheDocument()
    );

    unmount();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(cancelInvoked).toBe(false);
  });
});

describe('LocalTtsForm — cancellation is not surfaced as a download failure', () => {
  it('a Cancelled error from prepare_qwen_model resets to idle without an error alert', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: true });
      if (cmd === 'tts_model_status') return 'absent';
      if (cmd === 'prepare_qwen_model') {
        throw { kind: 'Cancelled', message: 'prepare cancelled' };
      }
    });

    renderLocal();
    const qwenRadio = await screen.findByRole('radio', { name: /qwen3-tts/i });
    await fireEvent.click(qwenRadio);
    await fireEvent.click(await screen.findByRole('button', { name: /download voice engine/i }));

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /download voice engine/i })).toBeInTheDocument()
    );
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByText(/download failed/i)).not.toBeInTheDocument();
  });
});
