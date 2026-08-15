import { render, screen, fireEvent, waitFor, within } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig, AsrConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import { resetConfig } from '$lib/models/app-config.svelte.js';
import TranscriptionSection from './TranscriptionSection.svelte';

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

function baseAsr(overrides?: Partial<AsrConfig>): AsrConfig {
  return {
    backend: '',
    whisper_model: 'base',
    language: null,
    translate: false,
    cloud_provider: null,
    cloud_base_url: '',
    cloud_model: '',
    cloud_api_key: '',
    apple_min_confidence: 0.5,
    ...overrides
  };
}

function mount(opts: {
  asr?: Partial<AsrConfig>;
  audioCloudConsent?: boolean;
  appleAvailable?: boolean;
  whisperDownloaded?: Record<string, boolean>;
  setConfigSpy?: (cfg: AppConfig) => void;
  onDownloadChannel?: (ch: ProgressChannel, model: string) => void;
}) {
  const asr = baseAsr(opts.asr);
  const downloaded = opts.whisperDownloaded ?? {};
  mockIPC((cmd, args) => {
    if (cmd === 'get_config') {
      return baseAppConfig({ asr, audio_cloud_consent: opts.audioCloudConsent ?? false });
    }
    if (cmd === 'set_config') {
      opts.setConfigSpy?.((args as { config: AppConfig }).config);
      return null;
    }
    if (cmd === 'asr_apple_native_available') return opts.appleAvailable ?? false;
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
  return render(TranscriptionSection);
}

function radiogroup(): HTMLElement {
  return screen.getByRole('radiogroup', { name: 'Transcription engine' });
}

function rowFor(name: RegExp | string): HTMLElement {
  return within(radiogroup()).getByText(name).closest('[role="radio"]') as HTMLElement;
}

describe('TranscriptionSection', () => {
  it('renders all four engine rows', async () => {
    mount({});
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const group = within(radiogroup());
    expect(group.getByText('Automatic')).toBeInTheDocument();
    expect(group.getByText('Apple (on-device)')).toBeInTheDocument();
    expect(group.getByText('Local Whisper')).toBeInTheDocument();
    expect(group.getByText('Cloud')).toBeInTheDocument();
  });

  it('selecting Automatic persists backend as an empty string', async () => {
    const setConfigSpy = vi.fn();
    mount({ asr: { backend: 'local_whisper' }, whisperDownloaded: { base: true }, setConfigSpy });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Automatic');
    await fireEvent.click(row);
    await waitFor(() => expect(setConfigSpy).toHaveBeenCalled());
    const cfg = setConfigSpy.mock.calls.at(-1)![0] as AppConfig;
    expect(cfg.asr.backend).toBe('');
  });

  it('selecting the unavailable Apple row never calls set_config', async () => {
    const setConfigSpy = vi.fn();
    mount({ appleAvailable: false, setConfigSpy });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Apple (on-device)');
    await fireEvent.click(row);
    await waitFor(() => expect(row).toHaveAttribute('aria-checked', 'true'));
    expect(setConfigSpy).not.toHaveBeenCalled();
  });

  it('stock install (no whisper model, apple unavailable) shows Automatic as Needs setup, not Active', async () => {
    mount({ appleAvailable: false, whisperDownloaded: {} });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Automatic');
    await waitFor(() => expect(within(row).queryByText('Active')).toBeNull());
    await waitFor(() => expect(within(row).getByText('Needs setup')).toBeInTheDocument());
  });

  it('a persisted cloud backend with a blank base URL renders selected + Needs setup', async () => {
    mount({ asr: { backend: 'cloud', cloud_base_url: '' } });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Cloud');
    await waitFor(() => expect(row).toHaveAttribute('aria-checked', 'true'));
    expect(within(row).getByText('Needs setup')).toBeInTheDocument();
  });

  it('whisper presence flipping to true activates Local Whisper', async () => {
    const setConfigSpy = vi.fn();
    const downloaded: Record<string, boolean> = { base: false };
    mount({
      asr: { backend: 'local_whisper', whisper_model: 'base' },
      whisperDownloaded: downloaded,
      setConfigSpy,
      onDownloadChannel: (ch) => {
        downloaded.base = true;
        ch.onmessage({ received: 1, total: 1, done: true });
      }
    });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const whisperRow = rowFor('Local Whisper');
    await waitFor(() => expect(within(whisperRow).queryByText('Active')).toBeNull());

    const downloadBtn = await screen.findByRole('button', { name: /download base/i });
    await fireEvent.click(downloadBtn);

    await waitFor(() => expect(within(whisperRow).getByText('Active')).toBeInTheDocument());
    const cfg = setConfigSpy.mock.calls.at(-1)![0] as AppConfig;
    expect(cfg.asr.backend).toBe('local_whisper');
  });
});
