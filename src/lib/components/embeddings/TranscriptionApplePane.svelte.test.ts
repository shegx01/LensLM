import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import { resetConfig } from '$lib/models/app-config.svelte.js';
import TranscriptionApplePane from './TranscriptionApplePane.svelte';

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
  resetConfig();
});

function mount(opts: {
  appleMinConfidence?: number;
  available?: boolean;
  setConfigSpy?: (cfg: AppConfig) => void;
}) {
  mockIPC((cmd, args) => {
    if (cmd === 'get_config') {
      return baseAppConfig({
        asr: {
          backend: '',
          whisper_model: 'base',
          language: null,
          translate: false,
          cloud_provider: null,
          cloud_base_url: '',
          cloud_model: '',
          cloud_api_key: '',
          apple_min_confidence: opts.appleMinConfidence ?? 0.5
        }
      });
    }
    if (cmd === 'set_config') {
      opts.setConfigSpy?.((args as { config: AppConfig }).config);
      return null;
    }
  });
  return render(TranscriptionApplePane, { props: { available: opts.available ?? true } });
}

describe('TranscriptionApplePane', () => {
  it('renders Strict/Balanced/Lenient presets', async () => {
    mount({});
    expect(await screen.findByText('Strict')).toBeInTheDocument();
    expect(screen.getByText('Balanced')).toBeInTheDocument();
    expect(screen.getByText('Lenient')).toBeInTheDocument();
  });

  it('selects Balanced by default (0.5)', async () => {
    mount({ appleMinConfidence: 0.5 });
    const row = (await screen.findByText('Balanced')).closest('[role="radio"]') as HTMLElement;
    await waitFor(() => expect(row).toHaveAttribute('aria-checked', 'true'));
  });

  it('clicking Strict persists apple_min_confidence 0.7', async () => {
    const setConfigSpy = vi.fn();
    mount({ setConfigSpy });
    const row = (await screen.findByText('Strict')).closest('[role="radio"]') as HTMLElement;
    await fireEvent.click(row);
    await waitFor(() => expect(setConfigSpy).toHaveBeenCalled());
    const cfg = setConfigSpy.mock.calls[0][0] as AppConfig;
    expect(cfg.asr.apple_min_confidence).toBe(0.7);
  });

  it('clicking Lenient persists apple_min_confidence 0.3', async () => {
    const setConfigSpy = vi.fn();
    mount({ setConfigSpy });
    const row = (await screen.findByText('Lenient')).closest('[role="radio"]') as HTMLElement;
    await fireEvent.click(row);
    await waitFor(() => expect(setConfigSpy).toHaveBeenCalled());
    const cfg = setConfigSpy.mock.calls[0][0] as AppConfig;
    expect(cfg.asr.apple_min_confidence).toBe(0.3);
  });

  it('writes the preset even when unavailable (no-write invariant is backend-only)', async () => {
    const setConfigSpy = vi.fn();
    mount({ available: false, setConfigSpy });
    const row = (await screen.findByText('Strict')).closest('[role="radio"]') as HTMLElement;
    await fireEvent.click(row);
    await waitFor(() => expect(setConfigSpy).toHaveBeenCalled());
    const cfg = setConfigSpy.mock.calls[0][0] as AppConfig;
    expect(cfg.asr.apple_min_confidence).toBe(0.7);
  });

  it('a stored value matching no preset renders as the nearest without rewriting on load', async () => {
    const setConfigSpy = vi.fn();
    mount({ appleMinConfidence: 0.42, setConfigSpy });
    const row = (await screen.findByText('Balanced')).closest('[role="radio"]') as HTMLElement;
    await waitFor(() => expect(row).toHaveAttribute('aria-checked', 'true'));
    expect(setConfigSpy).not.toHaveBeenCalled();
  });

  it('clicking the nearest preset after a non-matching load writes the canonical value', async () => {
    const setConfigSpy = vi.fn();
    mount({ appleMinConfidence: 0.42, setConfigSpy });
    const row = (await screen.findByText('Balanced')).closest('[role="radio"]') as HTMLElement;
    await waitFor(() => expect(row).toHaveAttribute('aria-checked', 'true'));
    await fireEvent.click(row);
    await waitFor(() => expect(setConfigSpy).toHaveBeenCalled());
    const cfg = setConfigSpy.mock.calls[0][0] as AppConfig;
    expect(cfg.asr.apple_min_confidence).toBe(0.5);
  });

  it('shows a not-yet-available notice without disabling the presets', async () => {
    mount({ available: false });
    await screen.findByText('Strict');
    expect(screen.getByText(/available on this device/i)).toBeInTheDocument();
    const row = screen.getByText('Strict').closest('[role="radio"]') as HTMLElement;
    expect(row).not.toHaveAttribute('disabled');
  });

  it('describes whole-clip re-transcription, not word-level filtering, and the no-model caveat', async () => {
    mount({});
    await screen.findByText('Strict');
    expect(screen.getByText(/re-transcribed on local whisper/i)).toBeInTheDocument();
    expect(screen.getByText(/kept as-is/i)).toBeInTheDocument();
    expect(screen.queryByText(/most confident words/i)).not.toBeInTheDocument();
  });
});
