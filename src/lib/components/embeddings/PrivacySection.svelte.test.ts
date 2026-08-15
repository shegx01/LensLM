import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { appConfigStore, resetConfig } from '$lib/models/app-config.svelte.js';
import PrivacySection from './PrivacySection.svelte';

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
  resetConfig();
});

/** A get_config payload carrying only the fields this section reads. */
function config(opts: { textConsent: boolean; audioConsent: boolean }): Partial<AppConfig> {
  return {
    enrichment: {
      enabled: true,
      coref_strategy: 'none',
      cloud_consent: opts.textConsent,
      chat_model: { provider: 'openai', model: 'gpt-4o' }
    },
    audio_cloud_consent: opts.audioConsent,
    models: [
      {
        provider: 'openai',
        base_url: '',
        model: 'gpt-4o',
        context: 128000,
        temperature: 0.7,
        api_key: 'x'
      }
    ],
    tts: { version: 1, backend: 'orpheus', model: '', clouds: {} },
    asr: {
      backend: '',
      whisper_model: 'base',
      translate: false,
      cloud_base_url: '',
      cloud_model: '',
      cloud_api_key: '',
      apple_min_confidence: 0.5
    }
  };
}

describe('PrivacySection', () => {
  it('reflects persisted enrichment.cloud_consent and audio_cloud_consent on mount', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config({ textConsent: true, audioConsent: false });
    });

    render(PrivacySection);

    const textToggle = await screen.findByRole('switch', { name: /allow cloud text models/i });
    const audioToggle = await screen.findByRole('switch', { name: /allow cloud audio/i });
    await waitFor(() => expect(textToggle).toHaveAttribute('aria-checked', 'true'));
    expect(audioToggle).toHaveAttribute('aria-checked', 'false');
  });

  it('hydrates audio_cloud_consent from the shared store when mounted alone, with no AI Model section co-mounted', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config({ textConsent: false, audioConsent: true });
    });

    render(PrivacySection);

    const audioToggle = await screen.findByRole('switch', { name: /allow cloud audio/i });
    await waitFor(() => expect(audioToggle).toHaveAttribute('aria-checked', 'true'));
  });

  it('makes zero set_config calls on mount', async () => {
    let setConfigCalls = 0;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config({ textConsent: false, audioConsent: true });
      if (cmd === 'set_config') setConfigCalls += 1;
    });

    render(PrivacySection);

    await screen.findByRole('switch', { name: /allow cloud audio/i });
    expect(setConfigCalls).toBe(0);
  });

  it('flipping the LLM/text toggle writes enrichment.cloud_consent with enrichment siblings intact', async () => {
    let saved: AppConfig | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return config({ textConsent: false, audioConsent: false });
      if (cmd === 'set_config') {
        saved = (args as { config: AppConfig }).config;
      }
    });

    render(PrivacySection);

    const textToggle = await screen.findByRole('switch', { name: /allow cloud text models/i });
    await waitFor(() => expect(textToggle).toHaveAttribute('aria-checked', 'false'));

    await fireEvent.click(textToggle);

    await waitFor(() => expect(textToggle).toHaveAttribute('aria-checked', 'true'));
    expect(saved?.enrichment.cloud_consent).toBe(true);
    expect(saved?.enrichment.enabled).toBe(true);
    expect(saved?.enrichment.chat_model).toEqual({ provider: 'openai', model: 'gpt-4o' });
  });

  it('flipping the audio toggle writes top-level audio_cloud_consent through the store, without mutating enrichment', async () => {
    let saved: AppConfig | undefined;
    let current = config({ textConsent: true, audioConsent: false }) as AppConfig;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return current;
      if (cmd === 'set_config') {
        saved = (args as { config: AppConfig }).config;
        current = saved;
      }
    });

    render(PrivacySection);

    const audioToggle = await screen.findByRole('switch', { name: /allow cloud audio/i });
    await waitFor(() => expect(audioToggle).toHaveAttribute('aria-checked', 'false'));

    await fireEvent.click(audioToggle);

    await waitFor(() => expect(audioToggle).toHaveAttribute('aria-checked', 'true'));
    expect(saved?.audio_cloud_consent).toBe(true);
    expect(saved?.enrichment.cloud_consent).toBe(true);
    expect(saved?.enrichment.enabled).toBe(true);
    expect(saved?.enrichment.chat_model).toEqual({ provider: 'openai', model: 'gpt-4o' });
    // The Switch renders its own optimistic aria-checked on click ahead of the store's
    // round trip, so the store assertion needs its own wait, not the DOM's.
    await waitFor(() => expect(appConfigStore.audioCloudConsent).toBe(true));
  });

  it('reverts the toggle when set_config fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config({ textConsent: false, audioConsent: false });
      if (cmd === 'set_config') throw new Error('write failed');
    });

    render(PrivacySection);

    const textToggle = await screen.findByRole('switch', { name: /allow cloud text models/i });
    await waitFor(() => expect(textToggle).toHaveAttribute('aria-checked', 'false'));

    await fireEvent.click(textToggle);

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/write failed/i));
    expect(textToggle).toHaveAttribute('aria-checked', 'false');
  });

  it('keeps the optimistically-mutated consent value and surfaces an error when the post-write re-read fails (R12)', async () => {
    let saved: AppConfig | undefined;
    let writeHappened = false;
    let rereadAttempted = false;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') {
        if (writeHappened && !rereadAttempted) {
          rereadAttempted = true;
          throw new Error('reread failed');
        }
        return config({ textConsent: false, audioConsent: false });
      }
      if (cmd === 'set_config') {
        writeHappened = true;
        saved = (args as { config: AppConfig }).config;
      }
    });

    render(PrivacySection);

    const audioToggle = await screen.findByRole('switch', { name: /allow cloud audio/i });
    await waitFor(() => expect(audioToggle).toHaveAttribute('aria-checked', 'false'));

    await fireEvent.click(audioToggle);

    await waitFor(() => expect(rereadAttempted).toBe(true));
    expect(saved?.audio_cloud_consent).toBe(true);
    expect(appConfigStore.audioCloudConsent).toBe(true);
    expect(appConfigStore.loadError).not.toBeNull();
  });

  it('shows an error state instead of a confident "off" consent toggle when the initial load fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') throw new Error('engine unreachable');
    });

    render(PrivacySection);

    await waitFor(() => expect(appConfigStore.loadError).not.toBeNull());
    expect(screen.queryByRole('switch', { name: /allow cloud audio/i })).not.toBeInTheDocument();
    expect(screen.getByText(/couldn't load cloud audio consent/i)).toBeInTheDocument();
  });

  it('shows "No data leaves this device" when everything is local', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config')
        return {
          enrichment: { enabled: false, coref_strategy: 'none', cloud_consent: false },
          audio_cloud_consent: false,
          models: [],
          tts: { version: 1, backend: 'orpheus', model: '', clouds: {} },
          asr: {
            backend: '',
            whisper_model: 'base',
            translate: false,
            cloud_base_url: '',
            cloud_model: '',
            cloud_api_key: '',
            apple_min_confidence: 0.5
          }
        };
    });

    render(PrivacySection);

    await waitFor(() =>
      expect(screen.getByText(/no data leaves this device/i)).toBeInTheDocument()
    );
  });

  it('shows the cloud LLM egress row when a cloud chat model is pinned', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config({ textConsent: true, audioConsent: false });
    });

    render(PrivacySection);

    await waitFor(() => expect(screen.getByText(/chat & notes model/i)).toBeInTheDocument());
    expect(screen.getAllByText(/cloud/i).length).toBeGreaterThan(0);
  });
});
