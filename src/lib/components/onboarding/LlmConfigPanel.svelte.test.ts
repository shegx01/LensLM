import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import SystemCheckRow from './SystemCheckRow.svelte';
import type { CheckResult } from '$lib/onboarding/system-check.js';

// Helpers ──────────────────────────────────────────────────────────────────

/** A CheckResult for the llm_runtime row with the configure action. */
function llmRow(over: Partial<CheckResult> = {}): CheckResult {
  return {
    id: 'llm_runtime',
    label: 'LLM runtime',
    status: 'pass',
    detail: 'Auto-detected: Ollama 0.3.2',
    action: 'configure',
    ...over
  };
}

/** A base AppConfig for IPC mocks. */
function baseConfig() {
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

beforeEach(() => {
  // Activate the Tauri path in all helpers (isTauri() reads globalThis.isTauri).
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

// ──────────────────────────────────────────────────────────────────────────
// Panel expand / collapse
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — expand/collapse', () => {
  it('panel is hidden by default; clicking Configure expands it', async () => {
    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });

    // Panel content is not visible before clicking
    expect(screen.queryByRole('tab', { name: /local/i })).not.toBeInTheDocument();

    const btn = screen.getByRole('button', { name: /configure/i });
    expect(btn).not.toBeDisabled();
    await fireEvent.click(btn);

    // After click: segmented tabs appear
    await waitFor(() => expect(screen.getByRole('tab', { name: /local/i })).toBeInTheDocument());
    expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument();
  });

  it('clicking Configure a second time collapses the panel', async () => {
    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });

    const btn = screen.getByRole('button', { name: /configure/i });
    await fireEvent.click(btn);
    await waitFor(() => expect(screen.getByRole('tab', { name: /local/i })).toBeInTheDocument());

    // Collapse
    await fireEvent.click(btn);
    await waitFor(() =>
      expect(screen.queryByRole('tab', { name: /local/i })).not.toBeInTheDocument()
    );
  });

  it('Configure button carries aria-expanded that flips with the panel', async () => {
    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    const btn = screen.getByRole('button', { name: /configure/i });

    expect(btn).toHaveAttribute('aria-expanded', 'false');
    await fireEvent.click(btn);
    expect(btn).toHaveAttribute('aria-expanded', 'true');
    await fireEvent.click(btn);
    expect(btn).toHaveAttribute('aria-expanded', 'false');
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Auto-detect
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — Auto-detect', () => {
  it('populates model select with detected models on reachable response', async () => {
    mockIPC((cmd) => {
      if (cmd === 'detect_llm')
        return { reachable: true, version: 'Ollama 0.3.2', models: ['llama3.2:3b', 'mistral:7b'] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });

    // Expand panel
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() => expect(screen.getByLabelText(/api endpoint/i)).toBeInTheDocument());

    // Click Auto-detect
    await fireEvent.click(screen.getByRole('button', { name: /auto-detect/i }));

    // Version confirmation text appears
    await waitFor(() => expect(screen.getByText(/ollama 0\.3\.2 detected/i)).toBeInTheDocument());

    // Model select appears with detected models
    const select = screen.getByRole('combobox');
    expect(select).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'llama3.2:3b' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'mistral:7b' })).toBeInTheDocument();
  });

  it('shows "Not detected" hint when reachable is false', async () => {
    mockIPC((cmd) => {
      if (cmd === 'detect_llm') return { reachable: false, version: null, models: [] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() => expect(screen.getByLabelText(/api endpoint/i)).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /auto-detect/i }));

    await waitFor(() => expect(screen.getByText(/not detected/i)).toBeInTheDocument());
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Local tab — "Test connection" saves config + probes endpoint
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — Test connection (local tab)', () => {
  it('calls set_config with the ollama provider entry and collapses when connection succeeds', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama 0.3.2', models: [] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });

    // Expand
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /test connection/i })).toBeInTheDocument()
    );

    // Click Test connection
    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    // set_config was called with the ollama entry
    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            models: expect.arrayContaining([
              expect.objectContaining({
                provider: 'ollama',
                base_url: 'http://localhost:11434',
                model: 'llama3.2:3b'
              })
            ])
          })
        })
      )
    );

    // oncheck was called (re-run system check)
    await waitFor(() => expect(oncheck).toHaveBeenCalledOnce());
  });

  it('persists the picked context window (default 8192) — not a stale hardcode', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama 0.3.2', models: [] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /test connection/i })).toBeInTheDocument()
    );

    // Pick the 32K context option.
    await fireEvent.click(screen.getByRole('button', { name: '32K' }));
    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            models: expect.arrayContaining([
              expect.objectContaining({ provider: 'ollama', context: 32768 })
            ])
          })
        })
      )
    );
  });

  it('prioritizes a custom context window typed into the input (out-of-preset size)', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama 0.3.2', models: [] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /test connection/i })).toBeInTheDocument()
    );

    // Type a context size that is NOT one of the presets (e.g. 64K).
    await fireEvent.input(screen.getByLabelText(/custom context window/i), {
      target: { value: '65536' }
    });
    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            models: expect.arrayContaining([
              expect.objectContaining({ provider: 'ollama', context: 65536 })
            ])
          })
        })
      )
    );
  });

  it('surfaces a connection error inline and does NOT collapse on IPC failure', async () => {
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd) => {
      if (cmd === 'get_config') throw new Error('disk full');
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /test connection/i })).toBeInTheDocument()
    );

    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());

    // Panel remains open (tabs still visible)
    expect(screen.getByRole('tab', { name: /local/i })).toBeInTheDocument();
  });

  it('shows "Could not reach" message when endpoint is not reachable', async () => {
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'detect_llm') return { reachable: false, version: null, models: [] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /test connection/i })).toBeInTheDocument()
    );

    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(screen.getByRole('alert').textContent).toMatch(/could not reach/i);

    // Panel stays open when not reachable
    expect(screen.getByRole('tab', { name: /local/i })).toBeInTheDocument();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Cloud API tab — "Save" saves config + collapses
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — Save (cloud tab)', () => {
  it('calls set_config with openai-compatible provider when Cloud API tab is active', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });

    // Expand and switch to Cloud API tab
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    // A key must be entered for Save to enable (no key saved for this provider).
    await fireEvent.input(screen.getByLabelText(/api key/i), {
      target: { value: 'sk-test-key' }
    });

    // Save
    await fireEvent.click(screen.getByRole('button', { name: /save/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            models: expect.arrayContaining([
              expect.objectContaining({
                provider: 'openai-compatible',
                // Real OpenAI API model id (default provider), not a derived slug.
                model: 'gpt-4o',
                context: 128000
              })
            ])
          })
        })
      )
    );
  });

  it('forwards the entered Cloud API key into set_config (locks the contract)', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });

    // Expand and switch to Cloud API tab.
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    // Type an API key into the Cloud API key field.
    const keyField = screen.getByLabelText(/api key/i);
    await fireEvent.input(keyField, { target: { value: 'sk-test-1234' } });

    // Save.
    await fireEvent.click(screen.getByRole('button', { name: /save/i }));

    // The entered api_key must reach set_config on the openai-compatible entry.
    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            models: expect.arrayContaining([
              expect.objectContaining({
                provider: 'openai-compatible',
                api_key: 'sk-test-1234'
              })
            ])
          })
        })
      )
    );
  });

  it('masks a previously-saved cloud key: Save disabled initially, enabled after editing the key', async () => {
    const oncheck = vi.fn().mockResolvedValue(undefined);

    // get_config returns a saved openai-compatible cloud config WITH a key.
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        const cfg = baseConfig();
        return {
          ...cfg,
          models: [
            {
              provider: 'openai-compatible',
              base_url: 'https://api.openai.com/v1',
              model: 'gpt-4o',
              context: 128000,
              temperature: 0.7,
              api_key: 'sk-saved-secret'
            }
          ]
        };
      }
      if (cmd === 'set_config') return null;
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });

    // Expand and switch to the Cloud API tab.
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    const keyField = screen.getByLabelText(/api key/i);
    const save = screen.getByRole('button', { name: /save/i });

    // The real key is NOT in the DOM (masked placeholder, empty value).
    await waitFor(() => expect(keyField).toHaveValue(''));
    expect(keyField).not.toHaveValue('sk-saved-secret');
    expect(keyField).toHaveAttribute('placeholder', expect.stringMatching(/saved/i));

    // Save is disabled while the saved key is untouched.
    expect(save).toBeDisabled();

    // Editing the key enables Save.
    await fireEvent.focus(keyField);
    await fireEvent.input(keyField, { target: { value: 'sk-new-key' } });
    expect(save).not.toBeDisabled();
  });

  it('does not bleed the masked-disabled state across providers: switching to an unsaved card enables normal entry', async () => {
    const oncheck = vi.fn().mockResolvedValue(undefined);

    // A saved OpenAI cloud key — only the OpenAI card should be masked.
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        const cfg = baseConfig();
        return {
          ...cfg,
          models: [
            {
              provider: 'openai-compatible',
              base_url: 'https://api.openai.com/v1',
              model: 'gpt-4o',
              context: 128000,
              temperature: 0.7,
              api_key: 'sk-saved-secret'
            }
          ]
        };
      }
      if (cmd === 'set_config') return null;
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });

    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    const save = screen.getByRole('button', { name: /save/i });
    const keyField = screen.getByLabelText(/api key/i);

    // On the SAVED (OpenAI) card: masked + Save disabled until re-entry.
    await waitFor(() => expect(keyField).toHaveValue(''));
    expect(keyField).toHaveAttribute('placeholder', expect.stringMatching(/saved/i));
    expect(save).toBeDisabled();

    // Switch to an UNSAVED provider card (Anthropic): no masking, normal entry.
    await fireEvent.click(screen.getByRole('radio', { name: /anthropic/i }));

    // Save is disabled with an empty field…
    expect(save).toBeDisabled();
    expect(keyField).toHaveAttribute('placeholder', expect.stringMatching(/paste api key/i));

    // …and enabled once a non-empty key is typed (no bleed-through disable).
    await fireEvent.input(keyField, { target: { value: 'sk-anthropic-key' } });
    expect(save).not.toBeDisabled();
  });

  it('forwards a custom model id entered in the Cloud Model field', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });

    // Expand and switch to the Cloud API tab.
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    // Override the seeded default with a model the app shipped without. (Both the
    // Local and Cloud panels label a field "Model", so target the cloud input.)
    const modelField = screen.getByLabelText('Model', { selector: '#llm-cloud-model' });
    await fireEvent.input(modelField, { target: { value: 'gpt-5-turbo' } });

    // A key must be entered for Save to enable.
    await fireEvent.input(screen.getByLabelText(/api key/i), {
      target: { value: 'sk-test-key' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /save/i }));

    // The user's model id must reach set_config, not the hardcoded default.
    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            models: expect.arrayContaining([
              expect.objectContaining({
                provider: 'openai-compatible',
                model: 'gpt-5-turbo'
              })
            ])
          })
        })
      )
    );
  });
});
