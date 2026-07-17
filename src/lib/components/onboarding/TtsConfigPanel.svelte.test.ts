import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import type { TtsEngineCatalogEntry, TtsVoice } from '$lib/onboarding/system-check.js';
import TtsConfigPanel from './TtsConfigPanel.svelte';

const baseConfig = baseAppConfig;

const DEFAULT_PRESET_VOICES: TtsVoice[] = [
  { id: 'leo', name: 'Leo', gender: 'male' },
  { id: 'tara', name: 'Tara', gender: 'female' }
];

/** A sample of the curated OpenAI-compatible cloud voice set (#195), mirroring
 *  the real gender buckets from `lens-core/src/tts/cloud/mod.rs::OPENAI_VOICES`. */
const CLOUD_VOICES: TtsVoice[] = [
  { id: 'alloy', name: 'Alloy', gender: 'female' },
  { id: 'onyx', name: 'Onyx', gender: 'male' }
];

/** A 3-engine catalog fixture mirroring lens-core's `tts_catalog_serialized` shape (#194).
 *  Orpheus and Qwen3Local both carry preset voices by default — the voice picker (#194)
 *  reads them straight from this static catalog, never from `list_tts_voices`. Cloud's
 *  `available`/`preset_voices` are overridable (#195) so tests can exercise both the
 *  no-key/no-presets state and the post-save/curated-voices state. */
function catalogFixture(overrides?: {
  qwenAvailable?: boolean;
  orpheusVoices?: TtsVoice[];
  qwenVoices?: TtsVoice[];
  cloudAvailable?: boolean;
  cloudVoices?: TtsVoice[];
}): TtsEngineCatalogEntry[] {
  const qwenAvailable = overrides?.qwenAvailable ?? false;
  const orpheusVoices = overrides?.orpheusVoices ?? DEFAULT_PRESET_VOICES;
  const qwenVoices = overrides?.qwenVoices ?? DEFAULT_PRESET_VOICES;
  const cloudAvailable = overrides?.cloudAvailable ?? false;
  const cloudVoices = overrides?.cloudVoices ?? [];
  return [
    {
      id: 'orpheus',
      platform: 'cross_platform',
      needs_key: false,
      available: true,
      unavailable_reason: null,
      multilingual: false,
      supported_languages: ['english'],
      preset_voices: orpheusVoices,
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
      preset_voices: qwenVoices,
      model_size_bytes: 4_500_000_000,
      language_capability_label: '10 languages',
      required_model_ids: []
    },
    {
      id: 'cloud',
      platform: 'cross_platform',
      needs_key: true,
      available: cloudAvailable,
      unavailable_reason: cloudAvailable ? null : 'Requires an API key',
      multilingual: true,
      supported_languages: [],
      preset_voices: cloudVoices,
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

/** Drive the `prepare_qwen_model` progress channel through a partial update, then completion. */
function drivePrepare(args: unknown): null {
  const ch = (args as { onProgress?: { onmessage?: (m: unknown) => void } }).onProgress;
  ch?.onmessage?.({ received: 50, total: 100, done: false });
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
  it('persists the default host/guest voices reactively right after download completes (no Save button)', async () => {
    let written: AppConfig | null = null;
    const oncheck = vi.fn().mockResolvedValue(undefined);
    const oncollapse = vi.fn();
    const listVoicesSpy = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'download_tts_model') return driveDownload(args);
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'list_tts_voices') {
        listVoicesSpy();
        return [];
      }
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck, oncollapse } });
    // Wait for the catalog fetch to resolve so `selectedEntry` (and its preset_voices) is populated
    // before Download runs.
    await waitFor(() => expect(screen.getAllByText(/english only/i).length).toBeGreaterThan(0));
    await fireEvent.click(screen.getByRole('button', { name: /download voice engine/i }));
    await waitFor(() => expect(screen.getByText(/voice engine ready/i)).toBeInTheDocument());

    // No explicit Save button on the Local tab — a freshly-downloaded engine's
    // default voice selection persists on its own.
    expect(screen.queryByRole('button', { name: /save voice settings/i })).not.toBeInTheDocument();

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).voices).toEqual({
      host: 'leo',
      guest: 'tara'
    });
    // Reactive persist does not drive the onboarding re-check/collapse flow.
    expect(oncheck).not.toHaveBeenCalled();
    expect(oncollapse).not.toHaveBeenCalled();
    // Preset voices come from the catalog, not a runtime IPC round trip.
    expect(listVoicesSpy).not.toHaveBeenCalled();
  });

  it('opens the host-voice picker, lists options from preset_voices, and selecting one persists immediately', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'download_tts_model') return driveDownload(args);
      if (cmd === 'tts_engine_catalog')
        return catalogFixture({
          orpheusVoices: [
            { id: 'leo', name: 'Leo', gender: 'male' },
            { id: 'milo', name: 'Milo', gender: 'male' }
          ]
        });
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await waitFor(() => expect(screen.getAllByText(/english only/i).length).toBeGreaterThan(0));
    await fireEvent.click(screen.getByRole('button', { name: /download voice engine/i }));
    await waitFor(() => expect(screen.getByText(/voice engine ready/i)).toBeInTheDocument());

    // Defaults to the first preset voice until the user picks another.
    const trigger = screen.getByLabelText(/^host voice/i);
    expect(trigger).toHaveTextContent('Leo');
    // Reset the write captured by the download-time default persist so the
    // assertion below is scoped to the explicit voice-pick persist.
    written = null;

    // bits-ui Select opens on trigger keydown (Enter/Space/Arrow) — this avoids
    // relying on PointerEvent pointer-capture semantics happy-dom doesn't model.
    await fireEvent.keyDown(trigger, { key: 'Enter' });

    const leoOption = await screen.findByRole('option', { name: 'Leo' });
    const miloOption = screen.getByRole('option', { name: 'Milo' });
    expect(leoOption).toBeInTheDocument();
    expect(miloOption).toBeInTheDocument();

    // Selection fires on `pointerup` (bits-ui's item handler), not `click`.
    await fireEvent.pointerUp(miloOption);

    await waitFor(() => expect(trigger).toHaveTextContent('Milo'));

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).voices.host).toBe('milo');
  });

  it('shows an inline error and no voice picker when the voice list is empty (no stub voices, no persist)', async () => {
    const setConfig = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'download_tts_model') return driveDownload(args);
      if (cmd === 'tts_engine_catalog') return catalogFixture({ orpheusVoices: [] });
      if (cmd === 'get_config') return baseConfig();
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
    // No fake voice IDs rendered, and there is nothing to persist.
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /save voice settings/i })).not.toBeInTheDocument();
    expect(setConfig).not.toHaveBeenCalled();
  });
});

describe('TtsConfigPanel — cloud (OpenAI-compatible, reactive, #195)', () => {
  it('has no Save button — the Cloud tab persists reactively like the Local tab', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    expect(screen.queryByRole('button', { name: /^save$/i })).not.toBeInTheDocument();
  });

  it('names the endpoint as any OpenAI-speech-API-compatible provider (Groq/DeepInfra/self-hosted), not just OpenAI', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    expect(screen.getByText(/groq/i)).toBeInTheDocument();
    expect(screen.getByText(/deepinfra/i)).toBeInTheDocument();
    expect(screen.getByText(/localai/i)).toBeInTheDocument();
  });

  it('labels the curated preset pickers as OpenAI-specific and hints non-OpenAI providers to use free-text voice ids', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ cloudVoices: CLOUD_VOICES });
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    await screen.findByLabelText(/host speaker/i);
    expect(screen.getByRole('heading', { name: /openai voices/i })).toBeInTheDocument();
    expect(
      screen.getByText(/using another provider\? enter its own voice ids\./i)
    ).toBeInTheDocument();
  });

  it('entering a fresh key and blurring persists it and flips Cloud from unavailable to available (catalog re-fetch)', async () => {
    let written: AppConfig | null = null;
    let keySaved = false;
    const oncheck = vi.fn().mockResolvedValue(undefined);
    const oncollapse = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      // Mirrors the real backend: `tts_engine_catalog` re-derives `available` from
      // whatever key is currently persisted, every time it is invoked.
      if (cmd === 'tts_engine_catalog') return catalogFixture({ cloudAvailable: keySaved });
      if (cmd === 'set_config') {
        keySaved = true;
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck, oncollapse } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    await waitFor(() => expect(screen.getByText(/requires an api key/i)).toBeInTheDocument());

    const keyField = screen.getByLabelText(/api key/i);
    await fireEvent.input(keyField, { target: { value: 'sk-openai-1234' } });
    await fireEvent.blur(keyField);

    await waitFor(() => expect(written).not.toBeNull());
    // `open_ai_compatible` is the only kind the backend adapter dispatches (#195).
    expect((written as unknown as AppConfig).tts).toEqual({
      version: 1,
      backend: { cloud: 'open_ai_compatible' },
      model: '',
      cloud: {
        kind: 'open_ai_compatible',
        api_key: 'sk-openai-1234',
        base_url: 'https://api.openai.com'
      }
    });

    await waitFor(() => expect(screen.getByText(/cloud is available/i)).toBeInTheDocument());
    expect(screen.queryByText(/requires an api key/i)).not.toBeInTheDocument();

    // A reactive persist is not a Save-button submit: no re-check, no collapse.
    expect(oncheck).not.toHaveBeenCalled();
    expect(oncollapse).not.toHaveBeenCalled();
  });

  it('masks a previously-saved key and clears it for fresh entry on focus', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        const cfg = baseConfig();
        return {
          ...cfg,
          tts: {
            version: 1,
            backend: { cloud: 'open_ai_compatible' },
            model: '',
            cloud: { kind: 'open_ai_compatible', api_key: 'sk-saved-openai', base_url: '' }
          }
        };
      }
      if (cmd === 'set_config') return null;
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const keyField = screen.getByLabelText(/api key/i);

    // Real key kept out of the DOM; masked placeholder shown once onMount's
    // `get_config` fetch resolves.
    await waitFor(() =>
      expect(keyField).toHaveAttribute('placeholder', expect.stringMatching(/saved/i))
    );
    expect(keyField).toHaveValue('');
    expect(keyField).not.toHaveValue('sk-saved-openai');

    await fireEvent.focus(keyField);
    expect(keyField).toHaveValue('');
  });

  it('blurring the key field empty while editing does NOT wipe the saved key', async () => {
    let written: AppConfig | null = null;
    const setConfigSpy = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') {
        const cfg = baseConfig();
        return {
          ...cfg,
          tts: {
            version: 1,
            backend: { cloud: 'open_ai_compatible' },
            model: '',
            cloud: { kind: 'open_ai_compatible', api_key: 'sk-saved-openai', base_url: '' }
          }
        };
      }
      if (cmd === 'tts_engine_catalog') return catalogFixture({ cloudAvailable: true });
      if (cmd === 'set_config') {
        setConfigSpy();
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const keyField = screen.getByLabelText(/api key/i);
    // Wait for onMount's `get_config` fetch to resolve (hasSavedKey/masked
    // placeholder loaded) before racing focus/blur against it.
    await waitFor(() =>
      expect(keyField).toHaveAttribute('placeholder', expect.stringMatching(/saved/i))
    );

    // Focus enters "replace" mode (clears the masked field); blurring without
    // typing anything must re-mask, not persist a blank key over the real one.
    await fireEvent.focus(keyField);
    await fireEvent.blur(keyField);

    expect(setConfigSpy).not.toHaveBeenCalled();
    expect(written).toBeNull();
    // Re-masked: focusing again still shows the empty/masked state, not a wiped key.
    expect(keyField).toHaveAttribute('placeholder', expect.stringMatching(/saved/i));
  });

  it('surfaces an inline error when a reactive persist fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') throw new Error('disk full');
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const keyField = screen.getByLabelText(/api key/i);
    await fireEvent.input(keyField, { target: { value: 'sk-x' } });
    await fireEvent.blur(keyField);

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
  });

  it('shows the unavailable banner until a key exists', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    await waitFor(() => expect(screen.getByText(/requires an api key/i)).toBeInTheDocument());
  });

  it('editing the base URL and blurring persists it, resending the already-saved key', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config')
        return {
          ...baseConfig(),
          tts: {
            version: 1,
            backend: { cloud: 'open_ai_compatible' as const },
            model: '',
            cloud: {
              kind: 'open_ai_compatible' as const,
              api_key: 'sk-already-saved',
              base_url: 'https://api.openai.com'
            }
          }
        };
      if (cmd === 'tts_engine_catalog') return catalogFixture({ cloudAvailable: true });
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const baseUrlField = screen.getByLabelText(/base url/i);
    await waitFor(() => expect(baseUrlField).toHaveValue('https://api.openai.com'));

    await fireEvent.input(baseUrlField, { target: { value: 'https://my-gateway.example.com' } });
    await fireEvent.blur(baseUrlField);

    await waitFor(() => expect(written).not.toBeNull());
    // The real key is resent untouched — masking never risks wiping it.
    expect((written as unknown as AppConfig).tts.cloud?.api_key).toBe('sk-already-saved');
    expect((written as unknown as AppConfig).tts.cloud?.base_url).toBe(
      'https://my-gateway.example.com'
    );
  });

  it('accepts free-text voice ids for host/guest on blur when no curated cloud voices exist', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const hostField = screen.getByLabelText(/host speaker/i);
    // Let onMount's async config fetch settle first — it unconditionally (re)sets
    // the free-text voice fields from the fetched config, which would otherwise
    // race with and clobber a value typed immediately after render.
    await waitFor(() => expect(hostField).toHaveValue(''));
    await fireEvent.input(hostField, { target: { value: 'my-host-voice' } });
    await fireEvent.blur(hostField);

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).voices.host).toBe('my-host-voice');

    written = null;
    const guestField = screen.getByLabelText(/guest speaker/i);
    await fireEvent.input(guestField, { target: { value: 'my-guest-voice' } });
    await fireEvent.blur(guestField);

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).voices.guest).toBe('my-guest-voice');
  });

  it('changing a curated voice picker persists immediately (no blur needed)', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ cloudVoices: CLOUD_VOICES });
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const hostTrigger = await screen.findByLabelText(/host speaker/i);
    const guestTrigger = screen.getByLabelText(/guest speaker/i);
    // Male voice (Onyx) defaults into Host, female (Alloy) into Guest — same
    // gender-bucket convention as the Local engine pickers. Mount-time defaulting
    // doesn't itself persist (mirrors Local's onMount, which doesn't either) —
    // only an explicit pick does.
    await waitFor(() => expect(hostTrigger).toHaveTextContent('Onyx'));
    expect(guestTrigger).toHaveTextContent('Alloy');
    expect(written).toBeNull();

    // bits-ui Select opens on trigger keydown; selection fires on `pointerup`.
    await fireEvent.keyDown(guestTrigger, { key: 'Enter' });
    // Only one non-custom female voice in this fixture — pick the custom escape
    // hatch instead to prove a *different* selection re-persists immediately.
    const customOption = await screen.findByRole('option', { name: /custom voice id/i });
    await fireEvent.pointerUp(customOption);

    await waitFor(() => expect(written).not.toBeNull());
    // Selecting "Custom voice ID…" resolves to an empty voice id until the
    // escape-hatch text field is filled in — still persisted immediately.
    expect((written as unknown as AppConfig).voices.guest).toBe('');
  });

  it('lets the user override a curated pick with a free-text custom voice ID, persisted on blur', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ cloudVoices: CLOUD_VOICES });
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const hostTrigger = await screen.findByLabelText(/host speaker/i);
    await waitFor(() => expect(hostTrigger).toHaveTextContent('Onyx'));

    await fireEvent.keyDown(hostTrigger, { key: 'Enter' });
    const customOption = await screen.findByRole('option', { name: /custom voice id/i });
    await fireEvent.pointerUp(customOption);

    const customInput = await screen.findByPlaceholderText(/e\.g\. alloy/i);
    await fireEvent.input(customInput, { target: { value: 'my-self-hosted-voice' } });
    await fireEvent.blur(customInput);

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).voices.host).toBe('my-self-hosted-voice');
  });
});

describe('TtsConfigPanel — local engine detection', () => {
  it('skips the download step and shows voices when the engine is already on disk', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return { ...baseConfig(), voices: { host: 'leo', guest: 'tara' } };
      if (cmd === 'tts_model_downloaded') return true;
      if (cmd === 'tts_engine_catalog') return catalogFixture();
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

  it('persists the selected engine into AppConfig.tts.backend alongside voices reactively (no Save button)', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: true });
      if (cmd === 'tts_model_downloaded') return true;
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

    // Wait for the post-switch voice list (Qwen preset voices, straight from the catalog) to prefill the pickers.
    // The picker is a bits-ui Select now: the trigger's label-associated element is a button
    // (no native `.value`), so assert the displayed voice name instead of a form value.
    await waitFor(() => expect(screen.getByLabelText(/^host voice/i)).toHaveTextContent('Leo'));

    expect(screen.queryByRole('button', { name: /save voice settings/i })).not.toBeInTheDocument();
    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).tts.backend).toBe('qwen3_local');
    expect((written as unknown as AppConfig).voices).toEqual({ host: 'leo', guest: 'tara' });
  });

  it('preserves a previously-saved Cloud API key when switching the local engine', async () => {
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
    await waitFor(() => expect(screen.getByLabelText(/^host voice/i)).toHaveTextContent('Leo'));

    // Reactive persist (triggered by pickEngine, no Save click) still round-trips
    // through the cloud-preserving helper.
    await waitFor(() => expect(written).not.toBeNull());
    // Local engine is active, but the stored Cloud key survives (not wiped to null).
    expect((written as unknown as AppConfig).tts.backend).toBe('qwen3_local');
    expect((written as unknown as AppConfig).tts.cloud).toEqual(savedCloud);
  });

  it('selecting Qwen3Local populates voice dropdowns from catalog preset_voices, with no list_tts_voices call and no load error', async () => {
    const listVoicesSpy = vi.fn();
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: true });
      if (cmd === 'tts_model_downloaded') return true;
      if (cmd === 'list_tts_voices') {
        listVoicesSpy();
        return [];
      }
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });

    const qwenRadio = await screen.findByRole('radio', { name: /qwen3-tts/i });
    await fireEvent.click(qwenRadio);

    await waitFor(() => expect(screen.getByLabelText(/^host voice/i)).toHaveTextContent('Leo'));
    expect(screen.getByLabelText(/co-host voice/i)).toHaveTextContent('Tara');
    expect(
      screen.queryByText(/couldn't load voices — is the engine installed\?/i)
    ).not.toBeInTheDocument();
    expect(listVoicesSpy).not.toHaveBeenCalled();
  });
});

describe('TtsConfigPanel — Qwen3-TTS prepare/download (#194)', () => {
  it('shows the download step (size, language label, download button) for a not-yet-downloaded Qwen — no voice pickers', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: true });
      if (cmd === 'tts_model_downloaded') {
        // Qwen presence check ignores `model` — assert it's called with the
        // empty-string sentinel from the handoff contract.
        expect((args as { engine: string; model: string }).model).toBe('');
        return (args as { engine: string }).engine !== 'qwen3_local';
      }
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });

    const qwenRadio = await screen.findByRole('radio', { name: /qwen3-tts/i });
    await fireEvent.click(qwenRadio);

    await waitFor(() => expect(screen.getByText('~4.5 GB')).toBeInTheDocument());
    expect(screen.getAllByText(/10 languages/i).length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: /download voice engine/i })).toBeInTheDocument();
    expect(screen.queryByLabelText(/^host voice/i)).not.toBeInTheDocument();
  });

  it('clicking Download for Qwen invokes prepare_qwen_model, drives the progress bar, then reveals catalog voices and persists', async () => {
    let written: AppConfig | null = null;
    const prepareSpy = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: true });
      if (cmd === 'tts_model_downloaded')
        return (args as { engine: string }).engine !== 'qwen3_local';
      if (cmd === 'prepare_qwen_model') {
        prepareSpy();
        return drivePrepare(args);
      }
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });

    const qwenRadio = await screen.findByRole('radio', { name: /qwen3-tts/i });
    await fireEvent.click(qwenRadio);
    await waitFor(() => expect(screen.getByText('~4.5 GB')).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /download voice engine/i }));

    expect(prepareSpy).toHaveBeenCalledOnce();
    await waitFor(() => expect(screen.getByText(/voice engine ready/i)).toBeInTheDocument());

    await waitFor(() => expect(screen.getByLabelText(/^host voice/i)).toHaveTextContent('Leo'));
    expect(screen.getByLabelText(/co-host voice/i)).toHaveTextContent('Tara');

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).tts.backend).toBe('qwen3_local');
    expect((written as unknown as AppConfig).voices).toEqual({ host: 'leo', guest: 'tara' });
  });

  it('skips the download step and shows voices immediately when Qwen is already downloaded', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config')
        return {
          ...baseConfig(),
          tts: { version: 1, backend: 'qwen3_local' as const, model: '', cloud: null }
        };
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: true });
      if (cmd === 'tts_model_downloaded') return true;
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });

    await waitFor(() => expect(screen.getByText(/voice engine ready/i)).toBeInTheDocument());
    expect(
      screen.queryByRole('button', { name: /download voice engine/i })
    ).not.toBeInTheDocument();
  });

  it('surfaces a download error via downloadError when prepare_qwen_model rejects', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ qwenAvailable: true });
      if (cmd === 'tts_model_downloaded')
        return (args as { engine: string }).engine !== 'qwen3_local';
      if (cmd === 'prepare_qwen_model') throw new Error('download failed');
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });

    const qwenRadio = await screen.findByRole('radio', { name: /qwen3-tts/i });
    await fireEvent.click(qwenRadio);
    await waitFor(() => expect(screen.getByText('~4.5 GB')).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /download voice engine/i }));

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/download failed/i));
    expect(screen.getByRole('button', { name: /download voice engine/i })).toBeInTheDocument();
  });
});
