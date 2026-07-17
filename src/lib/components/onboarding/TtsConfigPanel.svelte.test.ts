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

describe('TtsConfigPanel — cloud (OpenAI-compatible, #195)', () => {
  it('persists the OpenAI-compatible provider + key + default base URL to AppConfig.tts (RMW), then re-checks and collapses', async () => {
    let written: AppConfig | null = null;
    const oncheck = vi.fn().mockResolvedValue(undefined);
    const oncollapse = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck, oncollapse } });

    // Switch to the Cloud tab.
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const keyField = screen.getByLabelText(/api key/i);
    await fireEvent.input(keyField, { target: { value: 'sk-openai-1234' } });

    await fireEvent.click(screen.getByRole('button', { name: /^save$/i }));

    await waitFor(() => expect(written).not.toBeNull());
    // Standard client-side RMW: set_config receives the whole config with `tts`
    // populated and every other field round-tripped untouched. `open_ai_compatible`
    // is the only kind the backend adapter dispatches (#195).
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
    // No curated cloud voices in this fixture and no free-text entry — persists empty,
    // never a stale voice id left over from a local engine.
    expect((written as unknown as AppConfig).voices).toEqual({ host: '', guest: '' });
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

  it('masks a previously-saved key: Save disabled initially, enabled after editing', async () => {
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
    const save = screen.getByRole('button', { name: /^save$/i });

    // Real key kept out of the DOM; masked placeholder shown.
    await waitFor(() => expect(keyField).toHaveValue(''));
    expect(keyField).not.toHaveValue('sk-saved-openai');
    expect(keyField).toHaveAttribute('placeholder', expect.stringMatching(/saved/i));

    await fireEvent.focus(keyField);
    await fireEvent.input(keyField, { target: { value: 'sk-new-openai' } });
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

  it('shows the unavailable banner and blocks Save until a key is entered', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    await waitFor(() => expect(screen.getByText(/requires an api key/i)).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /^save$/i })).toBeDisabled();
  });

  it('re-fetches the catalog after saving so Cloud flips from unavailable to available (Critic #3)', async () => {
    let keySaved = false;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      // Mirrors the real backend: `tts_engine_catalog` re-derives `available` from
      // whatever key is currently persisted, every time it is invoked.
      if (cmd === 'tts_engine_catalog') return catalogFixture({ cloudAvailable: keySaved });
      if (cmd === 'set_config') {
        keySaved = true;
        return null;
      }
    });

    render(TtsConfigPanel, {
      props: { oncheck: vi.fn().mockResolvedValue(undefined), oncollapse: vi.fn() }
    });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    await waitFor(() => expect(screen.getByText(/requires an api key/i)).toBeInTheDocument());

    await fireEvent.input(screen.getByLabelText(/api key/i), { target: { value: 'sk-refresh' } });
    await fireEvent.click(screen.getByRole('button', { name: /^save$/i }));

    await waitFor(() => expect(screen.getByText(/cloud is available/i)).toBeInTheDocument());
    expect(screen.queryByText(/requires an api key/i)).not.toBeInTheDocument();
  });

  it('accepts free-text voice ids for host/guest when no curated cloud voices exist, and persists a custom base URL', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, {
      props: { oncheck: vi.fn().mockResolvedValue(undefined), oncollapse: vi.fn() }
    });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    await fireEvent.input(screen.getByLabelText(/api key/i), { target: { value: 'sk-x' } });
    await fireEvent.input(screen.getByLabelText(/base url/i), {
      target: { value: 'https://my-tts-gateway.example.com' }
    });
    await fireEvent.input(screen.getByLabelText(/host speaker/i), {
      target: { value: 'my-host-voice' }
    });
    await fireEvent.input(screen.getByLabelText(/guest speaker/i), {
      target: { value: 'my-guest-voice' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /^save$/i }));

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).tts.cloud?.base_url).toBe(
      'https://my-tts-gateway.example.com'
    );
    expect((written as unknown as AppConfig).voices).toEqual({
      host: 'my-host-voice',
      guest: 'my-guest-voice'
    });
  });

  it('offers curated cloud voice pickers (gender-filtered) that default to the first option and persist the pick', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ cloudVoices: CLOUD_VOICES });
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, {
      props: { oncheck: vi.fn().mockResolvedValue(undefined), oncollapse: vi.fn() }
    });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const hostTrigger = await screen.findByLabelText(/host speaker/i);
    const guestTrigger = screen.getByLabelText(/guest speaker/i);
    // Male voice (Onyx) defaults into Host, female (Alloy) into Guest — same
    // gender-bucket convention as the Local engine pickers.
    await waitFor(() => expect(hostTrigger).toHaveTextContent('Onyx'));
    expect(guestTrigger).toHaveTextContent('Alloy');

    await fireEvent.input(screen.getByLabelText(/api key/i), { target: { value: 'sk-x' } });
    await fireEvent.click(screen.getByRole('button', { name: /^save$/i }));

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).voices).toEqual({ host: 'onyx', guest: 'alloy' });
  });

  it('lets the user override a curated pick with a free-text custom voice ID', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ cloudVoices: CLOUD_VOICES });
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, {
      props: { oncheck: vi.fn().mockResolvedValue(undefined), oncollapse: vi.fn() }
    });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const hostTrigger = await screen.findByLabelText(/host speaker/i);
    await waitFor(() => expect(hostTrigger).toHaveTextContent('Onyx'));

    // bits-ui Select opens on trigger keydown; selection fires on `pointerup`.
    await fireEvent.keyDown(hostTrigger, { key: 'Enter' });
    const customOption = await screen.findByRole('option', { name: /custom voice id/i });
    await fireEvent.pointerUp(customOption);

    const customInput = await screen.findByPlaceholderText(/e\.g\. alloy/i);
    await fireEvent.input(customInput, { target: { value: 'my-self-hosted-voice' } });

    await fireEvent.input(screen.getByLabelText(/api key/i), { target: { value: 'sk-x' } });
    await fireEvent.click(screen.getByRole('button', { name: /^save$/i }));

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).voices.host).toBe('my-self-hosted-voice');
  });

  it('allows editing the base URL without retyping an already-saved key (no key wipe)', async () => {
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

    render(TtsConfigPanel, {
      props: { oncheck: vi.fn().mockResolvedValue(undefined), oncollapse: vi.fn() }
    });
    await fireEvent.click(screen.getByRole('tab', { name: /cloud/i }));

    const keyField = screen.getByLabelText(/api key/i);
    // Wait for onMount's `get_config` fetch to resolve (masked-key state loaded)
    // before asserting Save's enabled state.
    await waitFor(() =>
      expect(keyField).toHaveAttribute('placeholder', expect.stringMatching(/saved/i))
    );

    const save = screen.getByRole('button', { name: /^save$/i });
    // Not editing the key — Save is enabled purely to persist the base-URL change.
    expect(save).not.toBeDisabled();

    await fireEvent.input(screen.getByLabelText(/base url/i), {
      target: { value: 'https://my-gateway.example.com' }
    });
    await fireEvent.click(save);

    await waitFor(() => expect(written).not.toBeNull());
    // The real key is resent untouched — masking never risks wiping it.
    expect((written as unknown as AppConfig).tts.cloud?.api_key).toBe('sk-already-saved');
    expect((written as unknown as AppConfig).tts.cloud?.base_url).toBe(
      'https://my-gateway.example.com'
    );
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
