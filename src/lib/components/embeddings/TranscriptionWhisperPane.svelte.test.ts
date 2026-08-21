import { render, screen, fireEvent, waitFor, within } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import { resetConfig } from '$lib/models/app-config.svelte.js';
import TranscriptionWhisperPane, { resetActiveDownloads } from './TranscriptionWhisperPane.svelte';

type ProgressChannel = {
  onmessage: (m: { received: number; total: number | null; done: boolean }) => void;
};

const MODELS = [
  { id: 'tiny', approx_mb: 74, is_default: false },
  { id: 'base', approx_mb: 141, is_default: true },
  { id: 'small', approx_mb: 465, is_default: false }
];

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
  resetConfig();
  resetActiveDownloads();
});

/** Wires the IPC surface this pane depends on. `downloaded` seeds
 *  `whisper_model_downloaded` per model id; defaults to all-false. */
function mount(opts: {
  whisperModel?: string;
  downloaded?: Record<string, boolean>;
  onDownloadChannel?: (ch: ProgressChannel, model: string) => void;
  setConfigSpy?: (cfg: AppConfig) => void;
}) {
  const downloaded = opts.downloaded ?? {};
  mockIPC((cmd, args) => {
    if (cmd === 'get_config') {
      return baseAppConfig({
        asr: {
          backend: '',
          whisper_model: opts.whisperModel ?? 'base',
          language: null,
          translate: false,
          cloud_provider: null,
          cloud_base_url: '',
          cloud_model: '',
          cloud_api_key: '',
          apple_min_confidence: 0.5
        }
      });
    }
    if (cmd === 'set_config') {
      opts.setConfigSpy?.((args as { config: AppConfig }).config);
      return null;
    }
    if (cmd === 'list_whisper_models') return MODELS;
    if (cmd === 'whisper_model_downloaded') {
      return downloaded[(args as { model: string }).model] ?? false;
    }
    if (cmd === 'download_whisper_model') {
      const ch = (args as { on_progress: ProgressChannel }).on_progress;
      opts.onDownloadChannel?.(ch, (args as { model: string }).model);
      return null;
    }
  });

  const onPresenceChange = vi.fn();
  const result = render(TranscriptionWhisperPane, { props: { onPresenceChange } });
  return { ...result, onPresenceChange };
}

/** Rows are looked up by role+name since the radio's accessible name is the bare
 *  model id (`aria-label={m.id}`); the Download button/"Downloaded" badge live in
 *  a sibling slot, so `.parentElement` is the shared row containing both. */
function rowFor(modelId: string): HTMLElement {
  return screen.getByRole('radio', { name: modelId }).parentElement as HTMLElement;
}

describe('TranscriptionWhisperPane', () => {
  it('lists tiny/base/small with size labels and marks the default model', async () => {
    mount({ downloaded: {} });

    await screen.findByRole('radio', { name: 'tiny' });
    const tinyRow = rowFor('tiny');
    const baseRow = rowFor('base');
    const smallRow = rowFor('small');

    expect(within(tinyRow).getByText(/74 ?mb/i)).toBeInTheDocument();
    expect(within(baseRow).getByText(/141 ?mb/i)).toBeInTheDocument();
    expect(within(smallRow).getByText(/465 ?mb/i)).toBeInTheDocument();

    expect(within(baseRow).getByText(/recommended/i)).toBeInTheDocument();
    expect(within(tinyRow).queryByText(/recommended/i)).not.toBeInTheDocument();
    expect(within(smallRow).queryByText(/recommended/i)).not.toBeInTheDocument();
  });

  it('reflects per-model downloaded presence', async () => {
    mount({ downloaded: { tiny: false, base: true, small: false } });

    await waitFor(() =>
      expect(within(rowFor('base')).getByText(/downloaded/i)).toBeInTheDocument()
    );
    expect(
      within(rowFor('tiny')).getByRole('button', { name: /download tiny/i })
    ).toBeInTheDocument();
  });

  it('the radio has no interactive descendant, so assistive tech is never pruned from the Download action', async () => {
    mount({ downloaded: {} });
    await screen.findByRole('radio', { name: 'tiny' });

    const tinyRadio = screen.getByRole('radio', { name: 'tiny' });
    expect(tinyRadio.querySelector('button, [role="button"]')).toBeNull();
    expect(
      within(rowFor('tiny')).getByRole('button', { name: /download tiny/i })
    ).toBeInTheDocument();
  });

  it('download shows determinate progress, then re-probes and flips presence on completion', async () => {
    let resolveDownload: (() => void) | undefined;
    let probeCount = 0;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') {
        const model = (args as { model: string }).model;
        if (model !== 'tiny') return false;
        probeCount += 1;
        return probeCount > 1;
      }
      if (cmd === 'download_whisper_model') {
        const ch = (args as { on_progress: ProgressChannel }).on_progress;
        // Fire the tick synchronously (mirrors a real streaming command) and hold
        // the invoke promise open until the test resolves it, like the terminal
        // event in production — see `resolveDownload` below.
        ch.onmessage({ received: 50, total: 200, done: false });
        return new Promise((resolve) => {
          resolveDownload = () => resolve(null);
        });
      }
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });

    await screen.findByRole('radio', { name: 'tiny' });
    const tinyRow = rowFor('tiny');
    await fireEvent.click(within(tinyRow).getByRole('button', { name: /download tiny/i }));

    await waitFor(() => {
      const bar = within(tinyRow).getByRole('progressbar');
      expect(bar).toHaveAttribute('aria-valuenow', '25');
    });

    resolveDownload?.();

    await waitFor(() => expect(within(tinyRow).getByText(/downloaded/i)).toBeInTheDocument());
    expect(
      within(tinyRow).queryByRole('button', { name: /download tiny/i })
    ).not.toBeInTheDocument();
  });

  it('calls onPresenceChange after the initial probe, on selection change, and after a download completes', async () => {
    let channel: ProgressChannel | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') {
        const model = (args as { model: string }).model;
        return model === 'base';
      }
      if (cmd === 'download_whisper_model') {
        channel = (args as { on_progress: ProgressChannel }).on_progress;
        return null;
      }
    });

    const onPresenceChange = vi.fn();
    render(TranscriptionWhisperPane, { props: { onPresenceChange } });

    await waitFor(() => expect(onPresenceChange).toHaveBeenCalledWith('base', true));

    await screen.findByRole('radio', { name: 'small' });
    await fireEvent.click(screen.getByRole('radio', { name: 'small' }));
    await waitFor(() => expect(onPresenceChange).toHaveBeenCalledWith('small', false));

    await fireEvent.click(screen.getByRole('button', { name: /download small/i }));
    await waitFor(() => expect(channel).toBeDefined());
    channel!.onmessage({ received: 100, total: 100, done: true });

    await waitFor(() =>
      expect(onPresenceChange).toHaveBeenLastCalledWith('small', expect.any(Boolean))
    );
  });

  it('persists an already-downloaded selection as asr.whisper_model via the shared store', async () => {
    let saved: AppConfig | undefined;
    mount({
      whisperModel: 'base',
      downloaded: { small: true },
      setConfigSpy: (cfg) => {
        saved = cfg;
      }
    });

    // Wait for the presence probe to resolve — clicking before `downloadedMap`
    // settles would race the fail-closed guard with stale (undownloaded) data.
    await waitFor(() =>
      expect(within(rowFor('small')).getByText(/downloaded/i)).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('radio', { name: 'small' }));

    await waitFor(() => expect(saved?.asr.whisper_model).toBe('small'));
  });

  it('surfaces a Tauri LensError message when persisting a selected (already-downloaded) model fails', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return (args as { model: string }).model === 'small';
      if (cmd === 'set_config') throw { kind: 'Io', message: 'config write failed' };
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });
    await waitFor(() =>
      expect(within(rowFor('small')).getByText(/downloaded/i)).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('radio', { name: 'small' }));

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('config write failed'));
  });

  it('never persists an undownloaded selection (fail-closed: no degradation arm for LocalWhisper + missing model)', async () => {
    const setConfigSpy = vi.fn();
    mount({
      whisperModel: 'base',
      downloaded: {},
      setConfigSpy
    });

    await screen.findByRole('radio', { name: 'small' });
    await fireEvent.click(screen.getByRole('radio', { name: 'small' }));

    // Selection still updates locally (the row highlights, size stays browsable)
    // — only the persisted config must not point at an undownloaded model.
    await waitFor(() =>
      expect(screen.getByRole('radio', { name: 'small' })).toHaveAttribute('aria-checked', 'true')
    );
    expect(setConfigSpy).not.toHaveBeenCalled();
  });

  it("disables only the downloading row's own button, leaving siblings clickable", async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'download_whisper_model') return new Promise(() => {});
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });
    await screen.findByRole('radio', { name: 'tiny' });

    await fireEvent.click(within(rowFor('tiny')).getByRole('button', { name: /download tiny/i }));

    await waitFor(() =>
      expect(within(rowFor('tiny')).getByRole('button', { name: /download tiny/i })).toBeDisabled()
    );
    expect(
      within(rowFor('small')).getByRole('button', { name: /download small/i })
    ).not.toBeDisabled();
    expect(
      within(rowFor('base')).getByRole('button', { name: /download base/i })
    ).not.toBeDisabled();
  });

  it('guards against a remounted pane starting a second invoke of a model already downloading', async () => {
    let invokeCount = 0;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'download_whisper_model') {
        invokeCount += 1;
        return new Promise(() => {});
      }
    });

    const first = render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });
    await screen.findByRole('radio', { name: 'tiny' });
    await fireEvent.click(within(rowFor('tiny')).getByRole('button', { name: /download tiny/i }));
    await waitFor(() => expect(invokeCount).toBe(1));

    first.unmount();

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });
    await screen.findByRole('radio', { name: 'tiny' });
    const downloadBtn = within(rowFor('tiny')).getByRole('button', { name: /download tiny/i });

    expect(downloadBtn).toBeDisabled();
    // `disabled` alone would make this test pass even with the `id in activeDownloads`
    // guard deleted (a disabled button never dispatches click). Clear it so the click
    // actually reaches handleDownload and the guard itself is what stops a 2nd invoke.
    (downloadBtn as HTMLButtonElement).disabled = false;
    await fireEvent.click(downloadBtn);
    expect(invokeCount).toBe(1);
  });

  it('surfaces a Tauri LensError message for the model list instead of the generic fallback', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') {
        throw { kind: 'Internal', message: 'engine not started' };
      }
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('engine not started'));
  });

  it('surfaces a Tauri LensError message for a failed download instead of the generic fallback', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'download_whisper_model') {
        throw { kind: 'Io', message: 'disk unavailable' };
      }
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });
    await screen.findByRole('radio', { name: 'tiny' });
    await fireEvent.click(within(rowFor('tiny')).getByRole('button', { name: /download tiny/i }));

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('disk unavailable'));
  });

  it('marks the persisted+downloaded model Active even while another row is being previewed', async () => {
    mount({ whisperModel: 'base', downloaded: { base: true } });

    await waitFor(() => expect(within(rowFor('base')).getByText(/^active$/i)).toBeInTheDocument());

    await screen.findByRole('radio', { name: 'small' });
    await fireEvent.click(screen.getByRole('radio', { name: 'small' }));
    await waitFor(() =>
      expect(screen.getByRole('radio', { name: 'small' })).toHaveAttribute('aria-checked', 'true')
    );

    // small is now the highlighted/previewed row, but base — the persisted model — still
    // carries the Active badge, so "previewing" and "will run" stay visually distinct.
    expect(within(rowFor('base')).getByText(/^active$/i)).toBeInTheDocument();
    expect(within(rowFor('small')).queryByText(/^active$/i)).not.toBeInTheDocument();
  });

  it('allows two different models to download concurrently and completes both independently', async () => {
    let resolveTiny: (() => void) | undefined;
    let resolveSmall: (() => void) | undefined;
    const channels: Partial<Record<'tiny' | 'small', ProgressChannel>> = {};

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'download_whisper_model') {
        const model = (args as { model: 'tiny' | 'small' }).model;
        channels[model] = (args as { on_progress: ProgressChannel }).on_progress;
        return new Promise((resolve) => {
          if (model === 'tiny') resolveTiny = () => resolve(null);
          else resolveSmall = () => resolve(null);
        });
      }
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });
    await screen.findByRole('radio', { name: 'tiny' });

    await fireEvent.click(within(rowFor('tiny')).getByRole('button', { name: /download tiny/i }));
    await fireEvent.click(within(rowFor('small')).getByRole('button', { name: /download small/i }));

    await waitFor(() => {
      expect(within(rowFor('tiny')).getByRole('progressbar')).toBeInTheDocument();
      expect(within(rowFor('small')).getByRole('progressbar')).toBeInTheDocument();
    });

    channels.tiny?.onmessage({ received: 100, total: 100, done: true });
    resolveTiny?.();

    await waitFor(() =>
      expect(within(rowFor('tiny')).queryByRole('progressbar')).not.toBeInTheDocument()
    );
    expect(
      within(rowFor('tiny')).getByRole('button', { name: /download tiny/i })
    ).not.toBeDisabled();
    expect(within(rowFor('small')).getByRole('progressbar')).toBeInTheDocument();

    channels.small?.onmessage({ received: 100, total: 100, done: true });
    resolveSmall?.();

    await waitFor(() =>
      expect(within(rowFor('small')).queryByRole('progressbar')).not.toBeInTheDocument()
    );
    expect(
      within(rowFor('small')).getByRole('button', { name: /download small/i })
    ).not.toBeDisabled();
  });

  it('self-heals a wedged download once the watchdog window elapses, re-enabling the button', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'download_whisper_model') return new Promise(() => {});
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });
    await screen.findByRole('radio', { name: 'tiny' });

    // Fake timers must be installed before the click so the watchdog's own `setTimeout`
    // (armed synchronously inside handleDownload) is one vitest can fast-forward.
    vi.useFakeTimers();
    try {
      await fireEvent.click(within(rowFor('tiny')).getByRole('button', { name: /download tiny/i }));
      expect(within(rowFor('tiny')).getByRole('button', { name: /download tiny/i })).toBeDisabled();

      await vi.advanceTimersByTimeAsync(45_000);
    } finally {
      vi.useRealTimers();
    }

    await waitFor(() =>
      expect(
        within(rowFor('tiny')).getByRole('button', { name: /download tiny/i })
      ).not.toBeDisabled()
    );
    expect(screen.getByRole('alert')).toHaveTextContent(/stalled/i);
  });

  it('renders a Cancel control only for the row that is downloading', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'download_whisper_model') {
        const ch = (args as { on_progress: ProgressChannel }).on_progress;
        ch.onmessage({ received: 10, total: 100, done: false });
        // Never resolves — keeps the download "in flight" for the assertions below.
        return new Promise(() => {});
      }
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });

    await screen.findByRole('radio', { name: 'tiny' });
    expect(screen.queryByRole('button', { name: /cancel/i })).not.toBeInTheDocument();

    const tinyRow = rowFor('tiny');
    await fireEvent.click(within(tinyRow).getByRole('button', { name: /download tiny/i }));

    await waitFor(() =>
      expect(within(tinyRow).getByRole('button', { name: /cancel tiny download/i })).toBeEnabled()
    );
    expect(
      within(rowFor('small')).queryByRole('button', { name: /cancel/i })
    ).not.toBeInTheDocument();
  });

  it('cancels via the per-row key, targeting the clicked row and not the other in-flight download', async () => {
    const cancelArgs: unknown[] = [];
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'download_whisper_model') return new Promise(() => {});
      if (cmd === 'cancel_download') {
        cancelArgs.push(args);
        return true;
      }
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });
    await screen.findByRole('radio', { name: 'tiny' });

    await fireEvent.click(within(rowFor('tiny')).getByRole('button', { name: /download tiny/i }));
    await fireEvent.click(within(rowFor('small')).getByRole('button', { name: /download small/i }));
    await waitFor(() =>
      expect(
        within(rowFor('small')).getByRole('button', { name: /cancel small download/i })
      ).toBeInTheDocument()
    );

    await fireEvent.click(
      within(rowFor('small')).getByRole('button', { name: /cancel small download/i })
    );

    // Exact keys, not just "was called": mockIPC forwards whatever it is handed, so a
    // camelCase or mis-shaped payload would pass a looser assertion yet fail for real.
    await waitFor(() => expect(cancelArgs).toEqual([{ key: { kind: 'whisper', id: 'small' } }]));
  });

  it('keeps Download disabled after a cancel click until the invoke promise settles', async () => {
    let rejectTiny: ((e: unknown) => void) | undefined;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'download_whisper_model') {
        return new Promise((_resolve, reject) => {
          rejectTiny = reject;
        });
      }
      if (cmd === 'cancel_download') return true;
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });
    await screen.findByRole('radio', { name: 'tiny' });

    const tinyRow = rowFor('tiny');
    await fireEvent.click(within(tinyRow).getByRole('button', { name: /download tiny/i }));
    await fireEvent.click(
      await within(tinyRow).findByRole('button', { name: /cancel tiny download/i })
    );

    await waitFor(() =>
      expect(within(tinyRow).getByRole('button', { name: /cancel tiny download/i })).toBeDisabled()
    );
    expect(within(tinyRow).getByRole('button', { name: /download tiny/i })).toBeDisabled();

    rejectTiny?.({ kind: 'Cancelled', message: 'cancelled: download' });

    await waitFor(() =>
      expect(within(tinyRow).getByRole('button', { name: /download tiny/i })).toBeEnabled()
    );
  });

  it('shows no error banner when a download rejects as Cancelled', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'download_whisper_model') {
        throw { kind: 'Cancelled', message: 'cancelled: whisper small' };
      }
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });
    await screen.findByRole('radio', { name: 'tiny' });

    const tinyRow = rowFor('tiny');
    await fireEvent.click(within(tinyRow).getByRole('button', { name: /download tiny/i }));

    await waitFor(() =>
      expect(within(tinyRow).getByRole('button', { name: /download tiny/i })).toBeEnabled()
    );
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByText(/cancelled/i)).not.toBeInTheDocument();
  });

  it('renders the InsufficientSpace byte counts plus the Storage-settings pointer', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'download_whisper_model') {
        throw {
          kind: 'InsufficientSpace',
          message: 'Not enough disk space: 487 MB needed, 112 MB available.'
        };
      }
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });
    await screen.findByRole('radio', { name: 'tiny' });
    await fireEvent.click(within(rowFor('tiny')).getByRole('button', { name: /download tiny/i }));

    const alert = await waitFor(() => screen.getByRole('alert'));
    expect(alert).toHaveTextContent('487 MB needed, 112 MB available.');
    expect(alert).toHaveTextContent(/Settings → Storage/);
  });

  it('does not fire the wedge watchdog across a retry-length gap that keeps ticking', async () => {
    let channel: ProgressChannel | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'download_whisper_model') {
        channel = (args as { on_progress: ProgressChannel }).on_progress;
        return new Promise(() => {});
      }
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });
    await screen.findByRole('radio', { name: 'tiny' });

    vi.useFakeTimers();
    try {
      await fireEvent.click(within(rowFor('tiny')).getByRole('button', { name: /download tiny/i }));

      // Three attempts' worth of idle-read timeout plus backoff (≈96 s), each attempt
      // announcing itself with the engine's attempt-start tick — total elapsed far
      // exceeds WEDGE_TIMEOUT_MS, so only the re-arming keeps the watchdog quiet.
      for (const total of [null, 4_000, 4_000]) {
        await vi.advanceTimersByTimeAsync(32_000);
        channel?.onmessage({ received: 1_000, total, done: false });
      }
      await vi.advanceTimersByTimeAsync(6_000);
    } finally {
      vi.useRealTimers();
    }

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(within(rowFor('tiny')).getByRole('button', { name: /download tiny/i })).toBeDisabled();
  });
});
