import { render, screen, fireEvent, waitFor, within } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppleAsrAvailability, AsrBackend } from '$lib/asr/ipc.js';
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
  appleAvailability?: AppleAsrAvailability;
  whisperDownloaded?: Record<string, boolean>;
  /** `'reject'` simulates the command throwing. A function is re-evaluated per call, so
   *  a test can make the router's answer depend on the config the panel just wrote. */
  resolvedBackend?: AsrBackend | 'reject' | ((cfg: AppConfig) => AsrBackend);
  setConfigSpy?: (cfg: AppConfig) => void;
  onDownloadChannel?: (ch: ProgressChannel, model: string) => void;
}) {
  // Round-trips through one in-memory config so a write is visible to the next read,
  // as the real backend does — a fresh snapshot per read would hide every persist.
  let current = baseAppConfig({
    asr: baseAsr(opts.asr),
    audio_cloud_consent: opts.audioCloudConsent ?? false
  });
  const downloaded = opts.whisperDownloaded ?? {};
  mockIPC((cmd, args) => {
    if (cmd === 'get_config') return current;
    if (cmd === 'set_config') {
      current = (args as { config: AppConfig }).config;
      opts.setConfigSpy?.(current);
      return null;
    }
    if (cmd === 'asr_apple_native_available') return opts.appleAvailability ?? 'not_built';
    if (cmd === 'list_whisper_models') return MODELS;
    if (cmd === 'whisper_model_downloaded') {
      return downloaded[(args as { model: string }).model] ?? false;
    }
    if (cmd === 'download_whisper_model') {
      const ch = (args as { on_progress: ProgressChannel }).on_progress;
      opts.onDownloadChannel?.(ch, (args as { model: string }).model);
      return null;
    }
    if (cmd === 'resolve_asr_backend') {
      if (opts.resolvedBackend === 'reject') {
        throw { kind: 'Internal', message: 'router unavailable' };
      }
      if (typeof opts.resolvedBackend === 'function') return opts.resolvedBackend(current);
      return opts.resolvedBackend ?? 'local_whisper';
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
  it('the Apple row names an outdated macOS as the blocker', async () => {
    mount({ appleAvailability: { unsupported: { macos_too_old: { found: 15, required: 26 } } } });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Apple (on-device)');
    await waitFor(() =>
      expect(within(row).getByText(/needs macOS 26 or later/i)).toBeInTheDocument()
    );
  });

  it('the Apple row names a missing bridge differently from an unsupported device', async () => {
    mount({ appleAvailability: 'not_built' });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Apple (on-device)');
    await waitFor(() => expect(within(row).getByText(/bridge/i)).toBeInTheDocument());
    expect(within(row).queryByText(/needs macOS/i)).toBeNull();
  });

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
    mount({ appleAvailability: 'not_built', setConfigSpy });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Apple (on-device)');
    await fireEvent.click(row);
    await waitFor(() => expect(row).toHaveAttribute('aria-checked', 'true'));
    expect(setConfigSpy).not.toHaveBeenCalled();
  });

  it('stock install (no whisper model, apple unavailable) shows Automatic as Needs setup, not Active', async () => {
    mount({ appleAvailability: 'not_built', whisperDownloaded: {} });
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

  it('a persisted cloud backend with no provider set renders Needs setup, not Active', async () => {
    mount({
      asr: {
        backend: 'cloud',
        cloud_provider: null,
        cloud_base_url: 'https://api.openai.com',
        cloud_model: 'whisper-1',
        cloud_api_key: 'sk-test'
      },
      audioCloudConsent: true
    });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Cloud');
    await waitFor(() => expect(row).toHaveAttribute('aria-checked', 'true'));
    expect(within(row).getByText('Needs setup')).toBeInTheDocument();
  });

  it('a persisted cloud backend with whitespace-only fields renders Needs setup, not Active', async () => {
    mount({
      asr: {
        backend: 'cloud',
        cloud_provider: 'open_ai_compatible',
        cloud_base_url: '   ',
        cloud_model: '   ',
        cloud_api_key: '   '
      },
      audioCloudConsent: true
    });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Cloud');
    await waitFor(() => expect(row).toHaveAttribute('aria-checked', 'true'));
    expect(within(row).getByText('Needs setup')).toBeInTheDocument();
  });

  it('selecting Cloud with a complete, consented config activates it immediately', async () => {
    const setConfigSpy = vi.fn();
    mount({
      asr: {
        backend: 'local_whisper',
        cloud_provider: 'open_ai_compatible',
        cloud_base_url: 'https://api.openai.com',
        cloud_model: 'whisper-1',
        cloud_api_key: 'sk-test'
      },
      audioCloudConsent: true,
      setConfigSpy
    });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Cloud');
    await fireEvent.click(row);
    await waitFor(() => expect(setConfigSpy).toHaveBeenCalled());
    const cfg = setConfigSpy.mock.calls.at(-1)![0] as AppConfig;
    expect(cfg.asr.backend).toBe('cloud');
  });

  it('surfaces a failed persist as an alert instead of silently no-writing', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        return baseAppConfig({ asr: baseAsr({ backend: 'local_whisper' }) });
      }
      if (cmd === 'set_config') {
        throw { kind: 'Internal', message: 'disk write failed' };
      }
      if (cmd === 'asr_apple_native_available') return 'not_built';
      if (cmd === 'list_whisper_models') return MODELS;
      if (cmd === 'whisper_model_downloaded') return false;
      if (cmd === 'resolve_asr_backend') return 'local_whisper';
    });
    render(TranscriptionSection);
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Automatic');
    await fireEvent.click(row);
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/disk write failed/i));
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

  it('Automatic states which engine it resolves to', async () => {
    mount({ appleAvailability: 'available', resolvedBackend: 'apple_native' });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    await waitFor(() =>
      expect(screen.getByText(/currently resolves to Apple \(on-device\)/i)).toBeInTheDocument()
    );
  });

  it('a rejected resolve_asr_backend renders an explicit unknown state, never a guess', async () => {
    mount({ appleAvailability: 'available', resolvedBackend: 'reject' });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    await waitFor(() =>
      expect(screen.getByText(/couldn't determine which engine/i)).toBeInTheDocument()
    );
    expect(screen.queryByText(/currently resolves to/i)).toBeNull();
  });

  it('a backend token the catalog cannot name renders unknown, not the raw wire token', async () => {
    mount({
      appleAvailability: 'available',
      resolvedBackend: 'faster_whisper' as AsrBackend
    });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    await waitFor(() =>
      expect(screen.getByText(/couldn't determine which engine/i)).toBeInTheDocument()
    );
    expect(document.body.textContent).not.toContain('faster_whisper');
  });
});

// The capability notice is the only surface that reads `activeEngine`. `automaticStatusText`
// is a different derived, so asserting on it alone leaves a re-added client-side guess
// (`activeEngine = isUsable(persistedEngine) ? … : null`) passing.
describe('TranscriptionSection — the active engine is the router’s, not a client guess', () => {
  it('names the resolved engine in the capability notice even when the guess would differ', async () => {
    // Apple is available and the persisted engine is Automatic, so a client-side guess
    // would say Automatic; the command says Whisper, and the notice must follow it.
    mount({
      appleAvailability: 'available',
      whisperDownloaded: { base: true },
      resolvedBackend: 'local_whisper'
    });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    await waitFor(() =>
      expect(screen.getByText(/honours translate directly/i)).toBeInTheDocument()
    );
    expect(screen.queryByText(/Automatic prefers on-device/i)).toBeNull();
    expect(screen.queryByText(/reroutes Apple transcription/i)).toBeNull();
  });

  it('names Apple in the capability notice when the router resolves to Apple', async () => {
    mount({ appleAvailability: 'available', resolvedBackend: 'apple_native' });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    await waitFor(() =>
      expect(screen.getByText(/reroutes Apple transcription to Local Whisper/i)).toBeInTheDocument()
    );
    expect(screen.queryByText(/honours translate directly/i)).toBeNull();
  });

  it('warns that Local Whisper cannot run translate with no model on disk', async () => {
    mount({
      asr: { translate: true },
      appleAvailability: 'available',
      whisperDownloaded: {},
      resolvedBackend: 'local_whisper'
    });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const notice = await screen.findByRole('alert');
    await waitFor(() => expect(notice).toHaveTextContent(/no Whisper model is downloaded/i));
    expect(notice).toHaveTextContent(/transcription will fail/i);
  });
});

describe('TranscriptionSection — the resolved engine is re-read after router inputs change', () => {
  it('re-asks the router when Translate is toggled', async () => {
    mount({
      appleAvailability: 'available',
      whisperDownloaded: { base: true },
      resolvedBackend: (cfg) => (cfg.asr.translate ? 'local_whisper' : 'apple_native')
    });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    await waitFor(() =>
      expect(screen.getByText(/currently resolves to Apple \(on-device\)/i)).toBeInTheDocument()
    );

    await fireEvent.click(screen.getByRole('switch', { name: /translate to english/i }));

    await waitFor(() =>
      expect(screen.getByText(/currently resolves to Local Whisper/i)).toBeInTheDocument()
    );
    expect(screen.queryByText(/currently resolves to Apple/i)).toBeNull();
  });

  it('re-asks the router when the Cloud pane persists a change', async () => {
    mount({
      asr: {
        backend: 'cloud',
        cloud_provider: 'open_ai_compatible',
        cloud_base_url: 'https://api.openai.com',
        cloud_model: 'whisper-1',
        cloud_api_key: 'sk-test'
      },
      audioCloudConsent: true,
      appleAvailability: 'available',
      whisperDownloaded: { base: true },
      resolvedBackend: (cfg) => (cfg.asr.backend === 'cloud' ? 'cloud' : 'apple_native')
    });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    await waitFor(() => expect(screen.getByText(/ignores translate/i)).toBeInTheDocument());

    const baseUrlInput = screen.getByLabelText<HTMLInputElement>(/base url/i);
    await fireEvent.input(baseUrlInput, { target: { value: '' } });
    await fireEvent.blur(baseUrlInput);

    await waitFor(() =>
      expect(screen.getByText(/reroutes Apple transcription to Local Whisper/i)).toBeInTheDocument()
    );
    expect(screen.queryByText(/ignores translate/i)).toBeNull();
  });
});

describe('TranscriptionSection — the Cloud row honours the transport gate', () => {
  it('a cleartext http:// endpoint on a public host is Needs setup, not Active', async () => {
    mount({
      asr: {
        backend: 'cloud',
        cloud_provider: 'open_ai_compatible',
        cloud_base_url: 'http://api.openai.com',
        cloud_model: 'whisper-1',
        cloud_api_key: 'sk-test'
      },
      audioCloudConsent: true
    });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Cloud');
    await waitFor(() => expect(within(row).getByText('Needs setup')).toBeInTheDocument());
    expect(within(row).queryByText('Active')).toBeNull();
  });

  it('a private-network http:// endpoint stays Active', async () => {
    mount({
      asr: {
        backend: 'cloud',
        cloud_provider: 'open_ai_compatible',
        cloud_base_url: 'http://192.168.1.5:9000',
        cloud_model: 'whisper-1',
        cloud_api_key: 'sk-test'
      },
      audioCloudConsent: true
    });
    await screen.findByRole('radiogroup', { name: 'Transcription engine' });
    const row = rowFor('Cloud');
    await waitFor(() => expect(within(row).getByText('Active')).toBeInTheDocument());
  });
});
