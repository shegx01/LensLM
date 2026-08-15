import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import { persist, resetConfig } from '$lib/models/app-config.svelte.js';
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

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/valid base url/i));
    expect(savedConfigs.length).toBe(0);
  });

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

  it('activates Cloud automatically once consent is granted through the shared store', async () => {
    const { savedConfigs } = mockBackend(cloudConfig({ audioCloudConsent: false }));
    render(TranscriptionCloudPane);

    const baseUrlInput = await screen.findByLabelText<HTMLInputElement>(/base url/i);
    await waitFor(() => expect(baseUrlInput.value).toBe('https://api.openai.com'));

    // Simulates PrivacySection granting consent through the same shared store — no
    // further interaction with this pane.
    await persist((cfg) => ({ ...cfg, audio_cloud_consent: true }));

    await waitFor(() => expect(savedConfigs.at(-1)?.asr.backend).toBe('cloud'));
    expect(savedConfigs.at(-1)?.asr.cloud_api_key).toBe('sk-already-saved');
  });
});
