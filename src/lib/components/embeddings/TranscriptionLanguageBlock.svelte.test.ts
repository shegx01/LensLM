import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import { resetConfig } from '$lib/models/app-config.svelte.js';
import type { AsrBackend } from '$lib/asr/ipc.js';
import TranscriptionLanguageBlock from './TranscriptionLanguageBlock.svelte';

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
  resetConfig();
});

function config(overrides?: Partial<AppConfig['asr']>): AppConfig {
  return baseAppConfig({ asr: { ...baseAppConfig().asr, ...overrides } });
}

async function openLanguageSelect(): Promise<void> {
  const trigger = screen.getByLabelText(/spoken language/i);
  await fireEvent.keyDown(trigger, { key: 'Enter' });
}

describe('TranscriptionLanguageBlock — language persistence', () => {
  it('selecting Auto-detect persists the literal null', async () => {
    let written: AppConfig | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return config({ language: 'Es' });
      if (cmd === 'set_config') written = (args as { config: AppConfig }).config;
    });

    render(TranscriptionLanguageBlock, { props: { activeEngine: 'local_whisper' } });
    await screen.findByLabelText(/spoken language/i);

    await openLanguageSelect();
    const option = await screen.findByRole('option', { name: 'Auto-detect' });
    await fireEvent.pointerUp(option);

    await waitFor(() => expect(written).not.toBeUndefined());
    expect(written?.asr.language).toBeNull();
  });

  it('picking Spanish persists the exact Rust token "Es", never a lowercase form', async () => {
    let written: AppConfig | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'set_config') written = (args as { config: AppConfig }).config;
    });

    render(TranscriptionLanguageBlock, { props: { activeEngine: 'local_whisper' } });
    await screen.findByLabelText(/spoken language/i);

    await openLanguageSelect();
    const option = await screen.findByRole('option', { name: 'Spanish' });
    await fireEvent.pointerUp(option);

    await waitFor(() => expect(written).not.toBeUndefined());
    expect(written?.asr.language).toBe('Es');
  });

  it('the free-text hatch persists { Other: "ar" } on blur', async () => {
    let written: AppConfig | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'set_config') written = (args as { config: AppConfig }).config;
    });

    render(TranscriptionLanguageBlock, { props: { activeEngine: 'local_whisper' } });
    await screen.findByLabelText(/spoken language/i);

    await openLanguageSelect();
    const option = await screen.findByRole('option', { name: /other/i });
    await fireEvent.pointerUp(option);

    const codeField = await screen.findByLabelText(/language code/i);
    await fireEvent.input(codeField, { target: { value: 'ar' } });
    await fireEvent.blur(codeField);

    await waitFor(() => expect(written).not.toBeUndefined());
    expect(written?.asr.language).toEqual({ Other: 'ar' });
  });

  it('surfaces a rejected language save as a real LensError shape, not the generic fallback', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config({ language: 'Es' });
      if (cmd === 'set_config') {
        throw { kind: 'Io', message: 'failed to write config file' };
      }
    });

    render(TranscriptionLanguageBlock, { props: { activeEngine: 'local_whisper' } });
    await screen.findByLabelText(/spoken language/i);

    await openLanguageSelect();
    const option = await screen.findByRole('option', { name: 'Auto-detect' });
    await fireEvent.pointerUp(option);

    await waitFor(() =>
      expect(screen.getByText('failed to write config file')).toBeInTheDocument()
    );
  });

  it('leaves the persisted language untouched when the free-text field is blurred empty', async () => {
    let setConfigCalls = 0;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'set_config') setConfigCalls += 1;
    });

    render(TranscriptionLanguageBlock, { props: { activeEngine: 'local_whisper' } });
    await screen.findByLabelText(/spoken language/i);

    await openLanguageSelect();
    const option = await screen.findByRole('option', { name: /other/i });
    await fireEvent.pointerUp(option);

    const codeField = await screen.findByLabelText(/language code/i);
    await fireEvent.blur(codeField);

    expect(setConfigCalls).toBe(0);
  });
});

describe('TranscriptionLanguageBlock — translate persistence', () => {
  it('toggling the translate checkbox persists asr.translate', async () => {
    let written: AppConfig | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return config({ translate: false });
      if (cmd === 'set_config') written = (args as { config: AppConfig }).config;
    });

    render(TranscriptionLanguageBlock, { props: { activeEngine: 'local_whisper' } });
    const toggle = await screen.findByRole('switch', { name: /translate/i });
    await waitFor(() => expect(toggle).toHaveAttribute('aria-checked', 'false'));

    await fireEvent.click(toggle);

    await waitFor(() => expect(written).not.toBeUndefined());
    expect(written?.asr.translate).toBe(true);
  });

  it('surfaces a rejected save as a real LensError shape, not the generic fallback', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config({ translate: false });
      if (cmd === 'set_config') {
        throw { kind: 'Io', message: 'failed to write config file' };
      }
    });

    render(TranscriptionLanguageBlock, { props: { activeEngine: 'local_whisper' } });
    const toggle = await screen.findByRole('switch', { name: /translate/i });
    await waitFor(() => expect(toggle).toHaveAttribute('aria-checked', 'false'));

    await fireEvent.click(toggle);

    await waitFor(() =>
      expect(screen.getByText('failed to write config file')).toBeInTheDocument()
    );
  });
});

describe('TranscriptionLanguageBlock — tri-state capability notice', () => {
  async function renderWithEngine(
    engine: AsrBackend | null,
    asrOverrides?: Partial<AppConfig['asr']>
  ) {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config(asrOverrides);
      if (cmd === 'whisper_model_downloaded') return false;
    });
    return render(TranscriptionLanguageBlock, { props: { activeEngine: engine } });
  }

  it('renders nothing when no engine is active', async () => {
    await renderWithEngine(null);
    await screen.findByLabelText(/spoken language/i);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('Local Whisper: translate is honoured', async () => {
    await renderWithEngine('local_whisper');
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(/honours translate/i));
  });

  it('Cloud: translate is ignored', async () => {
    await renderWithEngine('cloud');
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(/ignores translate/i));
  });

  it('Apple with translate off: info notice names the reroute without alarming', async () => {
    await renderWithEngine('apple_native', { translate: false });
    await waitFor(() =>
      expect(screen.getByRole('status')).toHaveTextContent(/reroutes? .*local whisper/i)
    );
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('Apple with translate on and no Whisper model downloaded: warns of the reroute and the failure', async () => {
    await renderWithEngine('apple_native', { translate: true });
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(
        /reroutes apple transcription to local whisper/i
      )
    );
    expect(screen.getByRole('alert')).toHaveTextContent(/fail/i);
  });

  it('Apple with translate on and a Whisper model already downloaded: reroute notice without a failure warning', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config({ translate: true });
      if (cmd === 'whisper_model_downloaded') return true;
    });
    render(TranscriptionLanguageBlock, { props: { activeEngine: 'apple_native' } });

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(
        /reroutes apple transcription to local whisper/i
      )
    );
    expect(screen.getByRole('alert')).not.toHaveTextContent(/fail/i);
  });

  // "Honoured" alone is the dangerous half-truth: the run still needs the model on disk,
  // and whisper_engine_for has no fallback arm when it is missing.
  it('Local Whisper with translate on and no Whisper model downloaded: warns of the failure', async () => {
    await renderWithEngine('local_whisper', { translate: true });
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/no whisper model is downloaded/i)
    );
    expect(screen.getByRole('alert')).toHaveTextContent(/fail/i);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('Local Whisper with translate on and a Whisper model already downloaded: no failure warning', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config({ translate: true });
      if (cmd === 'whisper_model_downloaded') return true;
    });
    render(TranscriptionLanguageBlock, { props: { activeEngine: 'local_whisper' } });

    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(/honours translate/i));
    expect(screen.getByRole('status')).not.toHaveTextContent(/fail/i);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('updates the notice when the activeEngine prop changes', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config();
      if (cmd === 'whisper_model_downloaded') return false;
    });
    const { rerender } = render(TranscriptionLanguageBlock, {
      props: { activeEngine: 'local_whisper' }
    });
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(/honours translate/i));

    await rerender({ activeEngine: 'cloud' });
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(/ignores translate/i));
  });
});
