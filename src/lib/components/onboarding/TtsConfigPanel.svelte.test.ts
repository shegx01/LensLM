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
