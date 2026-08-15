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
      const ch = (args as { onProgress: ProgressChannel }).onProgress;
      opts.onDownloadChannel?.(ch, (args as { model: string }).model);
      return null;
    }
  });

  const onPresenceChange = vi.fn();
  const result = render(TranscriptionWhisperPane, { props: { onPresenceChange } });
  return { ...result, onPresenceChange };
}

/** The radio button's accessible name is the bare model id (`aria-label={m.id}`),
 *  so rows are looked up by role+name rather than by visible text. The Download
 *  button/"Downloaded" badge live in a sibling slot outside the radio (see the
 *  a11y test below) — `.parentElement` is the shared row that contains both. */
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
        const ch = (args as { onProgress: ProgressChannel }).onProgress;
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
        channel = (args as { onProgress: ProgressChannel }).onProgress;
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

  it('renders no Cancel control, including mid-download', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'download_whisper_model') {
        const ch = (args as { onProgress: ProgressChannel }).onProgress;
        ch.onmessage({ received: 10, total: 100, done: false });
        // Never resolves — keeps the download "in flight" for the assertion below.
        return new Promise(() => {});
      }
    });

    render(TranscriptionWhisperPane, { props: { onPresenceChange: vi.fn() } });

    await screen.findByRole('radio', { name: 'tiny' });
    const tinyRow = rowFor('tiny');
    await fireEvent.click(within(tinyRow).getByRole('button', { name: /download tiny/i }));

    await waitFor(() => expect(within(tinyRow).getByRole('progressbar')).toBeInTheDocument());
    expect(screen.queryByRole('button', { name: /cancel/i })).not.toBeInTheDocument();
  });
});
