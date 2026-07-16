import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import type { TtsEngineCatalogEntry } from '$lib/onboarding/system-check.js';
import TtsConfigPanel from './TtsConfigPanel.svelte';

const baseConfig = baseAppConfig;

/** A 3-engine catalog fixture mirroring lens-core's `tts_catalog_serialized` shape (#194). */
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
      preset_voices: [],
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
      preset_voices: [],
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

/** Drive the download progress channel to completion. */
function driveDownload(args: unknown): null {
  const ch = (args as { onProgress?: { onmessage?: (m: unknown) => void } }).onProgress;
  ch?.onmessage?.({ received: 100, total: 100, done: true });
  return null;
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('TtsConfigPanel — voices', () => {
  it('persists the picked host/guest voices to AppConfig.voices on save', async () => {
    let written: AppConfig | null = null;
    const oncheck = vi.fn().mockResolvedValue(undefined);
    const oncollapse = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'download_tts_model') return driveDownload(args);
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'list_tts_voices')
        return [
          { id: 'tara', name: 'Tara', gender: 'female' },
          { id: 'leo', name: 'Leo', gender: 'male' }
        ];
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck, oncollapse } });
    await fireEvent.click(screen.getByRole('button', { name: /download voice engine/i }));
    await waitFor(() => expect(screen.getByText(/voice engine ready/i)).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /save voice settings/i }));

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).voices).toEqual({
      host: 'leo',
      guest: 'tara'
    });
    expect(oncollapse).toHaveBeenCalledOnce();
  });

  it('shows an inline error and disables Save when the voice list is empty (no stub voices)', async () => {
    const setConfig = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'download_tts_model') return driveDownload(args);
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'list_tts_voices') return [];
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /download voice engine/i }));

    await waitFor(() =>
      expect(
        screen.getByText(/couldn't load voices — is the engine installed\?/i)
      ).toBeInTheDocument()
    );
    // No fake voice IDs rendered, and Save is disabled.
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /save voice settings/i })).toBeDisabled();
    expect(setConfig).not.toHaveBeenCalled();
  });
});

describe('TtsConfigPanel — cloud (ElevenLabs)', () => {
  it('persists the ElevenLabs provider + entered key to AppConfig.tts (RMW), then re-checks and collapses', async () => {
    let written: AppConfig | null = null;
    const oncheck = vi.fn().mockResolvedValue(undefined);
    const oncollapse = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck, oncollapse } });

    // Switch to the Cloud tab.
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    // Enter an ElevenLabs API key.
    const keyField = screen.getByLabelText(/api key/i);
    await fireEvent.input(keyField, { target: { value: 'sk-elevenlabs-1234' } });

    // Save.
    await fireEvent.click(screen.getByRole('button', { name: /^save$/i }));

    await waitFor(() => expect(written).not.toBeNull());
    // Standard client-side RMW: set_config receives the whole config with `tts`
    // populated and every other field round-tripped untouched.
    expect((written as unknown as AppConfig).tts).toEqual({
      version: 1,
      backend: { cloud: 'eleven_labs' },
      model: '',
      cloud: { kind: 'eleven_labs', api_key: 'sk-elevenlabs-1234', base_url: '' }
    });
    await waitFor(() => expect(oncheck).toHaveBeenCalledOnce());
    expect(oncollapse).toHaveBeenCalledOnce();
  });

  it('disables Save until a key is entered', async () => {
    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const save = screen.getByRole('button', { name: /^save$/i });
    expect(save).toBeDisabled();

    await fireEvent.input(screen.getByLabelText(/api key/i), { target: { value: 'sk-x' } });
    expect(save).not.toBeDisabled();
  });

  it('masks a previously-saved ElevenLabs key: Save disabled initially, enabled after editing', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        const cfg = baseConfig();
        return {
          ...cfg,
          tts: {
            version: 1,
            backend: { cloud: 'eleven_labs' },
            model: '',
            cloud: { kind: 'eleven_labs', api_key: 'sk-saved-eleven', base_url: '' }
          }
        };
      }
      if (cmd === 'set_config') return null;
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const keyField = screen.getByLabelText(/api key/i);
    const save = screen.getByRole('button', { name: /^save$/i });

    // Real key kept out of the DOM; masked placeholder shown.
    await waitFor(() => expect(keyField).toHaveValue(''));
    expect(keyField).not.toHaveValue('sk-saved-eleven');
    expect(keyField).toHaveAttribute('placeholder', expect.stringMatching(/saved/i));

    expect(save).toBeDisabled();

    await fireEvent.focus(keyField);
    await fireEvent.input(keyField, { target: { value: 'sk-new-eleven' } });
    expect(save).not.toBeDisabled();
  });

  it('surfaces an inline error and does NOT collapse when the save fails', async () => {
    const oncheck = vi.fn().mockResolvedValue(undefined);
    const oncollapse = vi.fn();
    mockIPC((cmd) => {
      if (cmd === 'get_config') throw new Error('disk full');
    });

    render(TtsConfigPanel, { props: { oncheck, oncollapse } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));
    await fireEvent.input(screen.getByLabelText(/api key/i), { target: { value: 'sk-x' } });
    await fireEvent.click(screen.getByRole('button', { name: /^save$/i }));

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(oncollapse).not.toHaveBeenCalled();
  });
});

describe('TtsConfigPanel — local engine detection', () => {
  it('skips the download step and shows voices when the engine is already on disk', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return { ...baseConfig(), voices: { host: 'leo', guest: 'tara' } };
      if (cmd === 'tts_model_downloaded') return true;
      if (cmd === 'list_tts_voices')
        return [
          { id: 'tara', name: 'Tara', gender: 'female' },
          { id: 'leo', name: 'Leo', gender: 'male' }
        ];
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });

    await waitFor(() => expect(screen.getByText(/voice engine ready/i)).toBeInTheDocument());
    expect(
      screen.queryByRole('button', { name: /download voice engine/i })
    ).not.toBeInTheDocument();
  });
});

describe('TtsConfigPanel — engine selector from the catalog (#194)', () => {
  it('renders engines from the catalog with capability gating (Qwen disabled off Apple Silicon)', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: false });
      if (cmd === 'tts_model_downloaded') return false;
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });

    const orpheusRadio = await screen.findByRole('radio', { name: /orpheus/i });
    expect(orpheusRadio).not.toBeDisabled();

    const qwenRadio = screen.getByRole('radio', { name: /qwen3-tts/i });
    expect(qwenRadio).toBeDisabled();
    expect(screen.getByText(/requires apple silicon/i)).toBeInTheDocument();

    // Cloud is its own tab, not a Local-selector entry.
    expect(screen.queryByRole('radio', { name: /^cloud$/i })).not.toBeInTheDocument();
  });

  it('shows model size + language-capability label next to Download, before first fetch', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'tts_model_downloaded') return false;
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });

    await waitFor(() => expect(screen.getByText('~2.3 GB')).toBeInTheDocument());
    expect(screen.getAllByText(/english only/i).length).toBeGreaterThan(0);
    // Still pre-fetch: Download is offered, no progress yet.
    expect(screen.getByRole('button', { name: /download voice engine/i })).toBeInTheDocument();
  });

  it('persists the selected engine into AppConfig.tts.backend alongside voices on Save', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: true });
      if (cmd === 'tts_model_downloaded') return true;
      if (cmd === 'list_tts_voices')
        return [
          { id: 'tara', name: 'Tara', gender: 'female' },
          { id: 'leo', name: 'Leo', gender: 'male' }
        ];
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, {
      props: { oncheck: vi.fn().mockResolvedValue(undefined), oncollapse: vi.fn() }
    });

    const qwenRadio = await screen.findByRole('radio', { name: /qwen3-tts/i });
    await fireEvent.click(qwenRadio);

    // Wait for the post-switch voice list (Qwen preset voices) to resolve and prefill the pickers.
    await waitFor(() => expect(screen.getByLabelText(/^host voice/i)).toHaveValue('leo'));
    await fireEvent.click(screen.getByRole('button', { name: /save voice settings/i }));

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).tts.backend).toBe('qwen3_local');
    expect((written as unknown as AppConfig).voices).toEqual({ host: 'leo', guest: 'tara' });
  });

  it('preserves a previously-saved Cloud API key when saving a local engine', async () => {
    const savedCloud = { kind: 'eleven_labs' as const, api_key: 'sk-keep-me', base_url: '' };
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config')
        return {
          ...baseConfig(),
          tts: { version: 1, backend: 'orpheus' as const, model: '', cloud: savedCloud }
        };
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: true });
      if (cmd === 'tts_model_downloaded') return true;
      if (cmd === 'list_tts_voices')
        return [
          { id: 'tara', name: 'Tara', gender: 'female' },
          { id: 'leo', name: 'Leo', gender: 'male' }
        ];
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, {
      props: { oncheck: vi.fn().mockResolvedValue(undefined), oncollapse: vi.fn() }
    });

    const qwenRadio = await screen.findByRole('radio', { name: /qwen3-tts/i });
    await fireEvent.click(qwenRadio);
    await waitFor(() => expect(screen.getByLabelText(/^host voice/i)).toHaveValue('leo'));
    await fireEvent.click(screen.getByRole('button', { name: /save voice settings/i }));

    await waitFor(() => expect(written).not.toBeNull());
    // Local engine is active, but the stored Cloud key survives (not wiped to null).
    expect((written as unknown as AppConfig).tts.backend).toBe('qwen3_local');
    expect((written as unknown as AppConfig).tts.cloud).toEqual(savedCloud);
  });
});
