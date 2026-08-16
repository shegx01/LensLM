import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import { appConfigStore, persist, resetConfig } from '$lib/models/app-config.svelte.js';
import TranscriptionCloudPane from './TranscriptionCloudPane.svelte';

/** A config with a complete, ready-to-activate Cloud setup except for what `opts` overrides. */
function cloudConfig(
  opts: Partial<AppConfig['asr']> & { audioCloudConsent?: boolean } = {}
): AppConfig {
  const { audioCloudConsent, ...asrOverrides } = opts;
  return baseAppConfig({
    audio_cloud_consent: audioCloudConsent ?? true,
    asr: {
      backend: 'local_whisper',
      whisper_model: 'base',
      language: null,
      translate: false,
      cloud_provider: 'open_ai_compatible',
      cloud_base_url: 'https://api.openai.com',
      cloud_model: 'whisper-1',
      cloud_api_key: 'sk-already-saved',
      apple_min_confidence: 0.5,
      ...asrOverrides
    }
  });
}

/** Round-trips get_config/set_config through one in-memory config, mirroring the real
 *  read-modify-write backend so `persist()`'s re-read sees each write. */
function mockBackend(initial: AppConfig): { savedConfigs: AppConfig[] } {
  let current = initial;
  const savedConfigs: AppConfig[] = [];
  mockIPC((cmd, args) => {
    if (cmd === 'get_config') return current;
    if (cmd === 'set_config') {
      current = (args as { config: AppConfig }).config;
      savedConfigs.push(current);
      return null;
    }
  });
  return { savedConfigs };
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
  resetConfig();
});

describe('TranscriptionCloudPane', () => {
  it("switching provider prefills base URL and model with that provider's preset", async () => {
    mockBackend(cloudConfig());
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    const providerTrigger = screen.getByLabelText(/^provider$/i);
    await fireEvent.keyDown(providerTrigger, { key: 'Enter' });
    const deepgramOption = await screen.findByRole('option', { name: /deepgram/i });
    await fireEvent.pointerUp(deepgramOption);

    const modelInput = screen.getByLabelText<HTMLInputElement>(/model/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.deepgram.com'));
    expect(modelInput.value).toBe('nova-3');
  });

  it('base URL and model stay editable after a provider switch', async () => {
    mockBackend(cloudConfig());
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    await fireEvent.input(baseUrlInput, { target: { value: 'http://localhost:8090' } });
    expect(baseUrlInput.value).toBe('http://localhost:8090');

    const modelInput = screen.getByLabelText<HTMLInputElement>(/model/i);
    await fireEvent.input(modelInput, { target: { value: 'custom-model' } });
    expect(modelInput.value).toBe('custom-model');
  });

  it('persists an edited base URL on blur', async () => {
    const { savedConfigs } = mockBackend(cloudConfig());
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    await fireEvent.input(baseUrlInput, { target: { value: 'http://localhost:8090' } });
    await fireEvent.blur(baseUrlInput);

    await waitFor(() => expect(savedConfigs.length).toBeGreaterThan(0));
    expect(savedConfigs.at(-1)?.asr.cloud_base_url).toBe('http://localhost:8090');
  });

  it('renders a saved API key masked', async () => {
    mockBackend(cloudConfig());
    render(TranscriptionCloudPane);

    const keyInput = await screen.findByLabelText<HTMLInputElement>(/api key/i);
    await waitFor(() => expect(keyInput.placeholder).toMatch(/saved/i));
    expect(keyInput.value).toBe('');
  });

  it('a save that only touches the base URL does not blank the stored key', async () => {
    const { savedConfigs } = mockBackend(cloudConfig());
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    await fireEvent.input(baseUrlInput, { target: { value: 'http://localhost:8090' } });
    await fireEvent.blur(baseUrlInput);

    await waitFor(() => expect(savedConfigs.length).toBeGreaterThan(0));
    expect(savedConfigs.at(-1)?.asr.cloud_api_key).toBe('sk-already-saved');
  });

  it('rejects a non-http(s) base URL and does not persist it', async () => {
    const { savedConfigs } = mockBackend(cloudConfig());
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    await fireEvent.input(baseUrlInput, { target: { value: 'not-a-url' } });
    await fireEvent.blur(baseUrlInput);

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/https:\/\//i));
    expect(savedConfigs.length).toBe(0);
  });

  it('rejects a cleartext http:// base URL on a non-loopback host — the API key is bearer-sent there', async () => {
    const { savedConfigs } = mockBackend(cloudConfig());
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    await fireEvent.input(baseUrlInput, { target: { value: 'http://api.example.com' } });
    await fireEvent.blur(baseUrlInput);

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/localhost/i));
    expect(savedConfigs.length).toBe(0);
  });

  it.each(['http://localhost:8090', 'http://127.0.0.1:8090', 'http://[::1]:8090'])(
    'accepts a loopback http:// base URL (%s) for self-hosted servers',
    async (url) => {
      const { savedConfigs } = mockBackend(cloudConfig());
      render(TranscriptionCloudPane);

      const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
      await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

      await fireEvent.input(baseUrlInput, { target: { value: url } });
      await fireEvent.blur(baseUrlInput);

      await waitFor(() => expect(savedConfigs.length).toBeGreaterThan(0));
      expect(savedConfigs.at(-1)?.asr.cloud_base_url).toBe(url);
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    }
  );

  it('shows why cloud is unavailable and does not activate while consent is off', async () => {
    const { savedConfigs } = mockBackend(cloudConfig({ audioCloudConsent: false }));
    render(TranscriptionCloudPane);

    await screen.findByText(/needs audio consent/i);

    const baseUrlInput = screen.getByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));
    await fireEvent.input(baseUrlInput, { target: { value: 'http://localhost:8090' } });
    await fireEvent.blur(baseUrlInput);

    await waitFor(() => expect(savedConfigs.length).toBeGreaterThan(0));
    expect(savedConfigs.at(-1)?.asr.backend).not.toBe('cloud');
  });

  it('granting consent through the shared store alone does not activate Cloud from this pane', async () => {
    const { savedConfigs } = mockBackend(cloudConfig({ audioCloudConsent: false }));
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    // Simulates PrivacySection granting consent through the same shared store, with no
    // further interaction with this pane. Activation now happens by selecting the Cloud
    // row in TranscriptionSection (see TranscriptionSection.svelte.test.ts), not here.
    await persist((cfg) => ({ ...cfg, audio_cloud_consent: true }));
    await new Promise((r) => setTimeout(r, 0));

    expect(savedConfigs.length).toBe(1);
    expect(savedConfigs.at(-1)?.asr.backend).not.toBe('cloud');
  });

  it('a typeahead keypress resolving to the already-selected provider is a no-op that preserves the saved key', async () => {
    const { savedConfigs } = mockBackend(cloudConfig());
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    // bits-ui's Select resolves a single "o" keypress on the (closed) trigger to
    // "OpenAI-compatible" via typeahead and fires onValueChange unconditionally —
    // even though it's already the selected provider. This must not clear the key.
    const providerTrigger = screen.getByLabelText(/^provider$/i);
    await fireEvent.keyDown(providerTrigger, { key: 'o' });
    await new Promise((r) => setTimeout(r, 0));

    expect(savedConfigs.length).toBe(0);
    const keyInput = await screen.findByLabelText<HTMLInputElement>(/api key/i);
    await waitFor(() => expect(keyInput.placeholder).toMatch(/saved/i));
  });

  it("switching provider clears the previous provider's API key instead of carrying it forward", async () => {
    const { savedConfigs } = mockBackend(cloudConfig());
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    const providerTrigger = screen.getByLabelText(/^provider$/i);
    await fireEvent.keyDown(providerTrigger, { key: 'Enter' });
    const deepgramOption = await screen.findByRole('option', { name: /deepgram/i });
    await fireEvent.pointerUp(deepgramOption);

    await waitFor(() => expect(savedConfigs.length).toBeGreaterThan(0));
    expect(savedConfigs.at(-1)?.asr.cloud_provider).toBe('deepgram');
    expect(savedConfigs.at(-1)?.asr.cloud_api_key).toBe('');
    expect(savedConfigs.at(-1)?.asr.backend).not.toBe('cloud');
  });

  it('clearing the Base URL while cloud is the active backend demotes it instead of leaving it pointed at cloud', async () => {
    const { savedConfigs } = mockBackend(cloudConfig({ backend: 'cloud' }));
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    await fireEvent.input(baseUrlInput, { target: { value: '' } });
    await fireEvent.blur(baseUrlInput);

    await waitFor(() => expect(savedConfigs.length).toBeGreaterThan(0));
    expect(savedConfigs.at(-1)?.asr.backend).toBe('');
  });

  // Every other test here asserts within a single mount. The pane's `hydrated` flag is
  // component-local and the engine rows sit in an {:else if} chain, so switching rows
  // and back genuinely remounts — that is where a key can end up under a wrong vendor.
  it('keeps a saved key bound to its own provider across a remount after a required field is cleared', async () => {
    const { savedConfigs } = mockBackend(
      cloudConfig({
        backend: 'cloud',
        cloud_provider: 'deepgram',
        cloud_base_url: 'https://api.deepgram.com',
        cloud_model: 'nova-3',
        cloud_api_key: 'dg-secret'
      })
    );
    const firstMount = render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.deepgram.com'));

    await fireEvent.input(baseUrlInput, { target: { value: '' } });
    await fireEvent.blur(baseUrlInput);

    // The store re-read lands after the write is recorded; the remount must hydrate from
    // the settled snapshot, or it reads pre-write state and proves nothing.
    await waitFor(() => expect(appConfigStore.asr?.cloud_base_url).toBe(''));
    expect(savedConfigs.at(-1)?.asr.backend).toBe('');
    expect(savedConfigs.at(-1)?.asr.cloud_provider).toBe('deepgram');

    firstMount.unmount();
    render(TranscriptionCloudPane);

    const remountedBaseUrl = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(remountedBaseUrl.value).toBe('https://api.deepgram.com'));

    await fireEvent.input(remountedBaseUrl, { target: { value: 'https://api.deepgram.com/v2' } });
    await fireEvent.blur(remountedBaseUrl);

    await waitFor(() => expect(savedConfigs.length).toBeGreaterThan(1));
    expect(savedConfigs.at(-1)?.asr.cloud_api_key).toBe('dg-secret');
    expect(savedConfigs.at(-1)?.asr.cloud_provider).toBe('deepgram');
  });

  it('re-syncs the displayed model from the stored config after a successful persist, not the value that was typed', async () => {
    let stored = cloudConfig();
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return stored;
      if (cmd === 'set_config') {
        const sent = (args as { config: AppConfig }).config;
        // Simulates engine-side normalization returning a different value than what
        // this pane sent — proves the resync reads the store, not the local echo.
        stored = { ...sent, asr: { ...sent.asr, cloud_model: 'engine-assigned-model' } };
        return null;
      }
    });
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    const modelInput = screen.getByLabelText<HTMLInputElement>(/model/i);
    await fireEvent.input(modelInput, { target: { value: 'typed-model' } });
    await fireEvent.blur(modelInput);

    await waitFor(() => expect(modelInput.value).toBe('engine-assigned-model'));
  });

  it('after clearing the Base URL, displays what the engine stored — not the typed blank, nor the provider preset', async () => {
    // The stored value is distinct from both the typed blank and the preset, so the
    // assertion can only pass if the resync read the store.
    const engineNormalized = 'https://engine-normalized.example';
    let stored = cloudConfig({ backend: 'cloud' });
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return stored;
      if (cmd === 'set_config') {
        const sent = (args as { config: AppConfig }).config;
        stored = { ...sent, asr: { ...sent.asr, cloud_base_url: engineNormalized } };
        return null;
      }
    });
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    await fireEvent.input(baseUrlInput, { target: { value: '' } });
    await fireEvent.blur(baseUrlInput);

    await waitFor(() => expect(baseUrlInput.value).toBe(engineNormalized));
  });

  it('surfaces a Tauri LensError message instead of the generic fallback', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return cloudConfig();
      if (cmd === 'set_config') {
        throw { kind: 'Validation', message: 'cloud ASR base URL rejected by backend' };
      }
    });
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    await fireEvent.input(baseUrlInput, { target: { value: 'http://localhost:9999' } });
    await fireEvent.blur(baseUrlInput);

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent('cloud ASR base URL rejected by backend')
    );
  });
});
