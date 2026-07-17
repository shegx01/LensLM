import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import type { TtsEngineCatalogEntry, TtsVoice } from '$lib/onboarding/system-check.js';
import CloudTtsForm from './CloudTtsForm.svelte';

const CLOUD_VOICES: TtsVoice[] = [
  { id: 'alloy', name: 'Alloy', gender: 'female' },
  { id: 'onyx', name: 'Onyx', gender: 'male' }
];

function catalogFixture(overrides?: {
  cloudAvailable?: boolean;
  cloudVoices?: TtsVoice[];
}): TtsEngineCatalogEntry[] {
  const cloudAvailable = overrides?.cloudAvailable ?? false;
  const cloudVoices = overrides?.cloudVoices ?? [];
  return [
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

function cloudKeyedConfig(): AppConfig {
  return {
    ...baseAppConfig(),
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
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('CloudTtsForm — single-sourced host/guest voice picker snippet (AC-6/AC-7)', () => {
  it('renders both host and guest pickers with distinct labels and ids from one snippet', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return cloudKeyedConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ cloudVoices: CLOUD_VOICES });
    });

    // The parent hands down an already-populated, shared catalog.
    render(CloudTtsForm, {
      props: { catalog: catalogFixture({ cloudVoices: CLOUD_VOICES }), active: true }
    });

    const host = await screen.findByLabelText(/host speaker/i);
    const guest = screen.getByLabelText(/guest speaker/i);
    // Distinct ids (duplicate ids would break label association + are invalid HTML).
    expect(host.id).toBe('tts-cloud-host-voice');
    expect(guest.id).toBe('tts-cloud-guest-voice');
    // Default gender buckets: male → host, female → guest.
    await waitFor(() => expect(host).toHaveTextContent('Onyx'));
    expect(guest).toHaveTextContent('Alloy');
  });

  it('the SAME persist wiring drives both roles: host and guest free-text ids persist on blur', async () => {
    let written: AppConfig | null = null;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return cloudKeyedConfig();
      // No curated voices → both pickers fall back to free-text inputs.
      if (cmd === 'tts_engine_catalog') return catalogFixture();
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(CloudTtsForm, { props: { catalog: [], active: true } });

    const hostField = await screen.findByLabelText(/host speaker/i);
    await waitFor(() => expect(hostField).toHaveValue(''));
    await fireEvent.input(hostField, { target: { value: 'host-voice-x' } });
    await fireEvent.blur(hostField);
    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).voices.host).toBe('host-voice-x');

    written = null;
    const guestField = screen.getByLabelText(/guest speaker/i);
    await fireEvent.input(guestField, { target: { value: 'guest-voice-y' } });
    await fireEvent.blur(guestField);
    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).voices.guest).toBe('guest-voice-y');
  });

  it('per-role custom placeholders are parameterized (host "e.g. alloy", guest "e.g. onyx")', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return cloudKeyedConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ cloudVoices: CLOUD_VOICES });
    });

    render(CloudTtsForm, {
      props: { catalog: catalogFixture({ cloudVoices: CLOUD_VOICES }), active: true }
    });

    const hostTrigger = await screen.findByLabelText(/host speaker/i);
    await waitFor(() => expect(hostTrigger).toHaveTextContent('Onyx'));
    await fireEvent.keyDown(hostTrigger, { key: 'Enter' });
    const hostCustom = await screen.findByRole('option', { name: /custom voice id/i });
    await fireEvent.pointerUp(hostCustom);

    const hostCustomInput = await screen.findByPlaceholderText(/e\.g\. alloy/i);
    expect(hostCustomInput).toHaveAttribute('id', 'tts-cloud-host-voice-custom');
  });
});

describe('CloudTtsForm — key save refreshes availability', () => {
  it('entering a fresh key flips Cloud from unavailable to available via catalog re-fetch', async () => {
    let keySaved = false;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseAppConfig();
      if (cmd === 'tts_engine_catalog') return catalogFixture({ cloudAvailable: keySaved });
      if (cmd === 'set_config') {
        keySaved = true;
        return null;
      }
    });

    render(CloudTtsForm, { props: { catalog: [], active: true } });

    await waitFor(() => expect(screen.getByText(/requires an api key/i)).toBeInTheDocument());
    const keyField = screen.getByLabelText(/api key/i);
    await fireEvent.input(keyField, { target: { value: 'sk-openai-1234' } });
    await fireEvent.blur(keyField);

    await waitFor(() => expect(screen.getByText(/cloud is available/i)).toBeInTheDocument());
    expect(screen.queryByText(/requires an api key/i)).not.toBeInTheDocument();
  });
});
