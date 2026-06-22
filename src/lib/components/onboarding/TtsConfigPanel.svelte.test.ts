import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import TtsConfigPanel from './TtsConfigPanel.svelte';

function baseConfig(): AppConfig {
  return {
    theme: 'dark',
    accent: 'purple',
    models: [],
    endpoints: {},
    voices: { host: '', guest: '' },
    tts: { provider: '', api_key: '' },
    paths: { data_dir: '' },
    tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
    onboarding_complete: false,
    embedding_model: ''
  };
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
      if (cmd === 'download_tts_engine') return driveDownload(args);
      if (cmd === 'list_tts_voices')
        return [
          { id: 'am_michael', name: 'Michael', gender: 'male' },
          { id: 'af_heart', name: 'Heart', gender: 'female' }
        ];
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck, oncollapse } });
    await fireEvent.click(screen.getByRole('button', { name: /download kokoro/i }));
    await waitFor(() => expect(screen.getByText(/kokoro engine ready/i)).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /save voice settings/i }));

    await waitFor(() => expect(written).not.toBeNull());
    expect((written as unknown as AppConfig).voices).toEqual({
      host: 'am_michael',
      guest: 'af_heart'
    });
    expect(oncollapse).toHaveBeenCalledOnce();
  });

  it('shows an inline error and disables Save when the voice list is empty (no stub voices)', async () => {
    const setConfig = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'download_tts_engine') return driveDownload(args);
      if (cmd === 'list_tts_voices') return []; // engine not really available
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
    });

    render(TtsConfigPanel, { props: { oncheck: vi.fn(), oncollapse: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /download kokoro/i }));

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
      provider: 'elevenlabs',
      api_key: 'sk-elevenlabs-1234'
    });
    // Re-runs the system check and collapses on success (same as the LLM panel).
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
