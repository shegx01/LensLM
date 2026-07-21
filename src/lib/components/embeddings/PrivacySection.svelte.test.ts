import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import PrivacySection from './PrivacySection.svelte';

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
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
    tts: { version: 1, backend: 'orpheus', model: '', cloud: null },
    asr: {
      backend: '',
      whisper_model: 'base',
      translate: false,
      cloud_base_url: '',
      cloud_model: '',
      cloud_api_key: ''
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

  it('flipping the audio toggle writes top-level audio_cloud_consent without mutating enrichment', async () => {
    let saved: AppConfig | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return config({ textConsent: true, audioConsent: false });
      if (cmd === 'set_config') {
        saved = (args as { config: AppConfig }).config;
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

  it('shows "No data leaves this device" when everything is local', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config')
        return {
          enrichment: { enabled: false, coref_strategy: 'none', cloud_consent: false },
          audio_cloud_consent: false,
          models: [],
          tts: { version: 1, backend: 'orpheus', model: '', cloud: null },
          asr: {
            backend: '',
            whisper_model: 'base',
            translate: false,
            cloud_base_url: '',
            cloud_model: '',
            cloud_api_key: ''
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
