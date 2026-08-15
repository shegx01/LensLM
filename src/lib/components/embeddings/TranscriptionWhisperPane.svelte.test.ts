import { render, screen, fireEvent, waitFor, within } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import { resetConfig } from '$lib/models/app-config.svelte.js';
import TranscriptionWhisperPane from './TranscriptionWhisperPane.svelte';

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

describe('TranscriptionWhisperPane', () => {
  it('lists tiny/base/small with size labels and marks the default model', async () => {
    mount({ downloaded: {} });

    const tinyRow = (await screen.findByText('tiny')).closest('[role="radio"]') as HTMLElement;
    const baseRow = screen.getByText('base').closest('[role="radio"]') as HTMLElement;
    const smallRow = screen.getByText('small').closest('[role="radio"]') as HTMLElement;

    expect(within(tinyRow).getByText(/74 ?mb/i)).toBeInTheDocument();
    expect(within(baseRow).getByText(/141 ?mb/i)).toBeInTheDocument();
    expect(within(smallRow).getByText(/465 ?mb/i)).toBeInTheDocument();

    expect(within(baseRow).getByText(/recommended/i)).toBeInTheDocument();
    expect(within(tinyRow).queryByText(/recommended/i)).not.toBeInTheDocument();
    expect(within(smallRow).queryByText(/recommended/i)).not.toBeInTheDocument();
  });

  it('reflects per-model downloaded presence', async () => {
    mount({ downloaded: { tiny: false, base: true, small: false } });

    const baseRow = (await screen.findByText('base')).closest('[role="radio"]') as HTMLElement;
    await waitFor(() => expect(within(baseRow).getByText(/downloaded/i)).toBeInTheDocument());

    const tinyRow = screen.getByText('tiny').closest('[role="radio"]') as HTMLElement;
    expect(within(tinyRow).getByRole('button', { name: /download tiny/i })).toBeInTheDocument();
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

    const tinyRow = (await screen.findByText('tiny')).closest('[role="radio"]') as HTMLElement;
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

    const smallRadio = (await screen.findByText('small')).closest('[role="radio"]') as HTMLElement;
    await fireEvent.click(smallRadio);
    await waitFor(() => expect(onPresenceChange).toHaveBeenCalledWith('small', false));

    await fireEvent.click(screen.getByRole('button', { name: /download small/i }));
    await waitFor(() => expect(channel).toBeDefined());
    channel!.onmessage({ received: 100, total: 100, done: true });

    await waitFor(() =>
      expect(onPresenceChange).toHaveBeenLastCalledWith('small', expect.any(Boolean))
    );
  });

  it('persists the selected model as asr.whisper_model via the shared store', async () => {
    let saved: AppConfig | undefined;
    mount({
      whisperModel: 'base',
      downloaded: {},
      setConfigSpy: (cfg) => {
        saved = cfg;
      }
    });

    const smallRadio = (await screen.findByText('small')).closest('[role="radio"]') as HTMLElement;
    await fireEvent.click(smallRadio);

    await waitFor(() => expect(saved?.asr.whisper_model).toBe('small'));
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

    const tinyRow = (await screen.findByText('tiny')).closest('[role="radio"]') as HTMLElement;
    await fireEvent.click(within(tinyRow).getByRole('button', { name: /download tiny/i }));

    await waitFor(() => expect(within(tinyRow).getByRole('progressbar')).toBeInTheDocument());
    expect(screen.queryByRole('button', { name: /cancel/i })).not.toBeInTheDocument();
  });
});
