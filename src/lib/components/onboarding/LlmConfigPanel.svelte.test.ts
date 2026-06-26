import { render, screen, fireEvent, waitFor, within } from '@testing-library/svelte';
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
    detail: 'Local LLM reachable',
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
    enrichment: { enabled: false, coref_strategy: 'llm_inline' as const, cloud_consent: false },
    paths: { data_dir: '' },
    tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
    onboarding_complete: false,
    embedding_model: ''
  };
}

/** A models.dev-style catalog map for `list_provider_models`. Includes a
 * reasoning model (claude, with context_limit + cost.input) and a non-reasoning
 * one (gpt-4o), so capability-keyed UI can be exercised both ways. */
function cloudCatalog(provider: string): Record<string, unknown> {
  if (provider === 'anthropic') {
    return {
      'claude-3-5-sonnet-latest': {
        id: 'claude-3-5-sonnet-latest',
        name: 'Claude 3.5 Sonnet',
        reasoning: true,
        reasoning_options: [{ type: 'toggle' }],
        tool_call: true,
        temperature: true,
        modalities: { input: ['text'], output: ['text'] },
        context_limit: 200000,
        output_limit: 8192,
        open_weights: false,
        cost: { input: 3, output: 15 }
      }
    };
  }
  // Default: openai. Includes 'gpt-4o' (reasoning=false) so the legacy cloud Save
  // test still resolves to model 'gpt-4o'.
  return {
    'gpt-4o': {
      id: 'gpt-4o',
      name: 'GPT-4o',
      reasoning: false,
      reasoning_options: [],
      tool_call: true,
      temperature: true,
      modalities: { input: ['text'], output: ['text'] },
      context_limit: 128000,
      output_limit: 16384,
      open_weights: false,
      cost: { input: 2.5, output: 10 }
    }
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

    // Provider-neutral reachability confirmation text appears (no vendor/version)
    await waitFor(() => expect(screen.getByText(/local server reachable/i)).toBeInTheDocument());

    // Model select appears with detected models. Target the model picker by its
    // id (the panel also renders an enrichment "Pronoun resolution" combobox).
    const select = screen.getByLabelText('Model', { selector: '#llm-model-local' });
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
  it('calls set_config with the real cloud provider id when Cloud API tab is active', async () => {
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
                // The REAL provider id of the selected card (default OpenAI), not a
                // blanket 'openai-compatible' (fix #1).
                provider: 'openai',
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

    // The entered api_key must reach set_config on the real-provider entry.
    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            models: expect.arrayContaining([
              expect.objectContaining({
                provider: 'openai',
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
      // Catalog with TWO openai models so the picker can select a non-default one.
      if (cmd === 'list_provider_models')
        return {
          'gpt-4o': cloudCatalog('openai')['gpt-4o'],
          'gpt-5-turbo': {
            id: 'gpt-5-turbo',
            name: 'GPT-5 Turbo',
            reasoning: false,
            reasoning_options: [],
            tool_call: true,
            temperature: true,
            modalities: { input: ['text'], output: ['text'] },
            context_limit: 256000,
            output_limit: 16384,
            open_weights: false,
            cost: { input: 5, output: 15 }
          }
        };
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });

    // Expand and switch to the Cloud API tab.
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    // Pick a non-default catalog model from the picker. (Both the Local and Cloud
    // panels label a field "Model", so target the cloud select by id.)
    const modelField = screen.getByLabelText('Model', {
      selector: '#llm-cloud-model'
    }) as HTMLSelectElement;
    await waitFor(() =>
      expect(within(modelField).getByRole('option', { name: 'GPT-5 Turbo' })).toBeInTheDocument()
    );
    await fireEvent.change(modelField, { target: { value: 'gpt-5-turbo' } });

    // A key must be entered for Save to enable.
    await fireEvent.input(screen.getByLabelText(/api key/i), {
      target: { value: 'sk-test-key' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /save/i }));

    // The user's chosen model id must reach set_config, not the seeded default.
    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            models: expect.arrayContaining([
              expect.objectContaining({
                provider: 'openai',
                model: 'gpt-5-turbo'
              })
            ])
          })
        })
      )
    );
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Enrichment (M4 Phase 3) — toggle + coref select + cloud consent
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — Enrichment prefs', () => {
  it('renders the enable toggle + coref select; cloud consent is HIDDEN on the local tab', async () => {
    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));

    await waitFor(() =>
      expect(screen.getByRole('switch', { name: /enable enrichment/i })).toBeInTheDocument()
    );
    // Coref select present.
    expect(
      screen.getByLabelText(/pronoun resolution/i, { selector: '#enrichment-coref' })
    ).toBeInTheDocument();
    // The cloud consent checkbox only appears on the Cloud API tab.
    expect(screen.queryByRole('checkbox', { name: /send document text/i })).not.toBeInTheDocument();
  });

  it('shows the cloud privacy/cost consent note ONLY on the Cloud API tab', async () => {
    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );

    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    const consent = screen.getByRole('checkbox', { name: /send document text/i });
    expect(consent).toBeInTheDocument();
    // Defaults OFF (explicit-enable, not a dark pattern).
    expect(consent).not.toBeChecked();
    // The disclosure names the cost + that text leaves the machine.
    expect(screen.getByText(/leaves your machine/i)).toBeInTheDocument();
    expect(screen.getByText(/api costs/i)).toBeInTheDocument();
  });

  it('persists enrichment prefs on the local Test connection (consent forced false)', async () => {
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

    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    // The enrichment section was persisted (local defaults: enabled + llm_inline,
    // consent forced false because local text never leaves the machine).
    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            enrichment: expect.objectContaining({
              enabled: true,
              coref_strategy: 'llm_inline',
              cloud_consent: false
            })
          })
        })
      )
    );
  });

  it('cloud Save persists cloud_consent and gates enabled on it', async () => {
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
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    // Grant consent + provide a key.
    await fireEvent.click(screen.getByRole('checkbox', { name: /send document text/i }));
    await fireEvent.input(screen.getByLabelText(/api key/i), {
      target: { value: 'sk-test-key' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /save/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            enrichment: expect.objectContaining({
              enabled: true,
              cloud_consent: true
            })
          })
        })
      )
    );
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Capability-aware model pickers (M4 Phase 3, Stage 3)
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — capability-aware pickers', () => {
  it('renders catalog models in the cloud model select for the selected provider', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'list_provider_models')
        return cloudCatalog((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    // The openai catalog model (GPT-4o) renders as an option in the cloud picker.
    // Scope to the cloud model <select> (the coref-override picker also lists ids).
    const cloudSelect = (await waitFor(() =>
      screen.getByLabelText('Model', { selector: '#llm-cloud-model' })
    )) as HTMLSelectElement;
    await waitFor(() =>
      expect(within(cloudSelect).getByRole('option', { name: 'GPT-4o' })).toBeInTheDocument()
    );

    // Switching to Anthropic loads its catalog model (Claude 3.5 Sonnet).
    await fireEvent.click(screen.getByRole('radio', { name: /anthropic/i }));
    await waitFor(() =>
      expect(
        within(cloudSelect).getByRole('option', { name: 'Claude 3.5 Sonnet' })
      ).toBeInTheDocument()
    );
  });

  it('on open: renders the loaded list immediately, then calls refresh_models and re-reads list_provider_models to converge to the fresh catalog', async () => {
    // First `list_provider_models` read returns the LOADED (stale) catalog: just
    // GPT-4o. After `refresh_models` runs, the SAME command returns a fresh
    // catalog that adds a new model and drops the old one — proving the picker
    // re-reads after the refresh and converges to the live list.
    let refreshed = false;
    const refreshCalled = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'refresh_models') {
        refreshCalled();
        refreshed = true;
        return true;
      }
      if (cmd === 'list_provider_models') {
        if ((args as { provider: string }).provider !== 'openai') return {};
        // Before refresh: only the loaded GPT-4o. After refresh: a NEW model
        // appears and GPT-4o is gone (catalog rotated live).
        return refreshed
          ? {
              'gpt-6-omega': {
                id: 'gpt-6-omega',
                name: 'GPT-6 Omega',
                reasoning: true,
                reasoning_options: [],
                tool_call: true,
                temperature: true,
                modalities: { input: ['text'], output: ['text'] },
                context_limit: 512000,
                output_limit: 32768,
                open_weights: false,
                cost: { input: 8, output: 24 }
              }
            }
          : (cloudCatalog('openai') as Record<string, unknown>);
      }
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    const cloudSelect = (await waitFor(() =>
      screen.getByLabelText('Model', { selector: '#llm-cloud-model' })
    )) as HTMLSelectElement;

    // The loaded list is rendered immediately (GPT-4o present right away).
    await waitFor(() =>
      expect(within(cloudSelect).getByRole('option', { name: 'GPT-4o' })).toBeInTheDocument()
    );

    // refresh_models is invoked on open…
    await waitFor(() => expect(refreshCalled).toHaveBeenCalled());

    // …and after the refresh the picker re-reads and converges to the FRESH
    // catalog: the new model appears, the removed one disappears.
    await waitFor(() =>
      expect(within(cloudSelect).getByRole('option', { name: 'GPT-6 Omega' })).toBeInTheDocument()
    );
    expect(within(cloudSelect).queryByRole('option', { name: 'GPT-4o' })).not.toBeInTheDocument();
  });

  it('on open: keeps the loaded list and does NOT error when the live refresh fails (offline)', async () => {
    const refreshCalled = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'list_ollama_models') return [];
      // refresh_models REJECTS (offline / HTTP error) — the picker must swallow it.
      if (cmd === 'refresh_models') {
        refreshCalled();
        throw new Error('network unreachable');
      }
      if (cmd === 'list_provider_models')
        return cloudCatalog((args as { provider: string }).provider);
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    const cloudSelect = (await waitFor(() =>
      screen.getByLabelText('Model', { selector: '#llm-cloud-model' })
    )) as HTMLSelectElement;

    // The loaded list renders and SURVIVES the failed refresh (offline floor).
    await waitFor(() =>
      expect(within(cloudSelect).getByRole('option', { name: 'GPT-4o' })).toBeInTheDocument()
    );
    await waitFor(() => expect(refreshCalled).toHaveBeenCalled());
    // No error surfaced (onboarding stays non-blocking) and the option remains.
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(within(cloudSelect).getByRole('option', { name: 'GPT-4o' })).toBeInTheDocument();
  });

  it('shows the compact capability/cost helper (context + input + output cost)', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'list_provider_models')
        return cloudCatalog((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    // gpt-4o: context_limit 128000, cost { input: 2.5, output: 10 } →
    // "128K Context · ~$2.5/1M input · ~$10/1M output".
    await waitFor(() =>
      expect(screen.getByText('128K Context · ~$2.5/1M input · ~$10/1M output')).toBeInTheDocument()
    );
  });

  it('omits the cost clauses for a model with no cost (Context clause only)', async () => {
    // A single-model openai catalog with a context_limit but NO cost.
    const noCost = (provider: string) =>
      provider === 'openai'
        ? {
            'gpt-4o': {
              id: 'gpt-4o',
              name: 'GPT-4o',
              reasoning: false,
              reasoning_options: [],
              tool_call: true,
              temperature: true,
              modalities: { input: ['text'], output: ['text'] },
              context_limit: 1_050_000,
              output_limit: 16384,
              open_weights: false
            }
          }
        : {};
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'list_provider_models') return noCost((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    // 1,050,000 → "1.05M"; no cost → only the Context clause renders.
    await waitFor(() => expect(screen.getByText('1.05M Context')).toBeInTheDocument());
    expect(screen.queryByText(/1M input/)).not.toBeInTheDocument();
    expect(screen.queryByText(/1M output/)).not.toBeInTheDocument();
  });

  it('omits the Context clause for a model with no context_limit', async () => {
    // A single-model openai catalog with cost but NO context_limit.
    const noLimit = (provider: string) =>
      provider === 'openai'
        ? {
            'gpt-4o': {
              id: 'gpt-4o',
              name: 'GPT-4o',
              reasoning: false,
              reasoning_options: [],
              tool_call: true,
              temperature: true,
              modalities: { input: ['text'], output: ['text'] },
              context_limit: null,
              output_limit: null,
              open_weights: false,
              cost: { input: 5, output: 25 }
            }
          }
        : {};
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'list_provider_models') return noLimit((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    // No context_limit → no Context clause; both cost clauses still render.
    await waitFor(() =>
      expect(screen.getByText('~$5/1M input · ~$25/1M output')).toBeInTheDocument()
    );
    // The compact "<n>K/M/B Context" clause is absent (the unrelated "Context
    // Window" label on the Local tab is intentionally not matched).
    expect(screen.queryByText(/[\d.]+[KMB] Context/)).not.toBeInTheDocument();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Smart default — newest text-capable model (M4 Phase 3, enrichment)
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — smart default model', () => {
  /** A two-model openai catalog: an OLDER model and a NEWER one, so the smart
   * default (newest text-capable) must resolve to the newer id, NOT the seeded
   * 'gpt-4o'. The newer one carries a distinct id to prove the seed was beaten. */
  function datedOpenaiCatalog(): Record<string, unknown> {
    return {
      'gpt-4o': {
        id: 'gpt-4o',
        name: 'GPT-4o',
        reasoning: false,
        reasoning_options: [],
        tool_call: true,
        temperature: true,
        modalities: { input: ['text'], output: ['text'] },
        context_limit: 128000,
        output_limit: 16384,
        open_weights: false,
        last_updated: '2024-08-01',
        release_date: '2024-08-01',
        cost: { input: 2.5, output: 10 }
      },
      'gpt-5-pro': {
        id: 'gpt-5-pro',
        name: 'GPT-5 Pro',
        reasoning: true,
        reasoning_options: [],
        tool_call: true,
        temperature: true,
        modalities: { input: ['text'], output: ['text'] },
        context_limit: 400000,
        output_limit: 32768,
        open_weights: false,
        last_updated: '2026-01-15',
        release_date: '2026-01-10',
        cost: { input: 5, output: 20 }
      }
    };
  }

  it('defaults a no-saved-model provider to the NEWEST text-capable model (not the seed)', async () => {
    const setConfig = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'list_provider_models') {
        return (args as { provider: string }).provider === 'openai' ? datedOpenaiCatalog() : {};
      }
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    const cloudSelect = (await waitFor(() =>
      screen.getByLabelText('Model', { selector: '#llm-cloud-model' })
    )) as HTMLSelectElement;

    // The select value resolves to the NEWEST text model, not the 'gpt-4o' seed.
    await waitFor(() => expect(cloudSelect.value).toBe('gpt-5-pro'));

    // And saving forwards that smart default (no explicit pick needed).
    await fireEvent.input(screen.getByLabelText(/api key/i), { target: { value: 'sk-test-key' } });
    await fireEvent.click(screen.getByRole('button', { name: /save/i }));
    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            models: expect.arrayContaining([
              expect.objectContaining({ provider: 'openai', model: 'gpt-5-pro' })
            ])
          })
        })
      )
    );
  });

  it('preserves a previously-saved model instead of overriding with the smart default', async () => {
    // A saved openai config pins the OLDER 'gpt-4o'; the catalog also offers the
    // newer 'gpt-5-pro'. The restored choice must be kept (not re-defaulted).
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') {
        const cfg = baseConfig();
        return {
          ...cfg,
          models: [
            {
              provider: 'openai',
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
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'list_provider_models') {
        return (args as { provider: string }).provider === 'openai' ? datedOpenaiCatalog() : {};
      }
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    const cloudSelect = (await waitFor(() =>
      screen.getByLabelText('Model', { selector: '#llm-cloud-model' })
    )) as HTMLSelectElement;

    // The saved 'gpt-4o' is kept even though 'gpt-5-pro' is newer.
    await waitFor(() => expect(cloudSelect.value).toBe('gpt-4o'));
  });

  it('falls back to the per-provider seed when the catalog is empty/offline', async () => {
    // list_provider_models returns an empty map (filtered/offline) → the picker
    // must seed the field with the provider default so it is never blank.
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'list_provider_models') return {};
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument()
    );
    await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));

    const cloudSelect = (await waitFor(() =>
      screen.getByLabelText('Model', { selector: '#llm-cloud-model' })
    )) as HTMLSelectElement;

    // The seeded openai default renders as the single fallback option.
    await waitFor(() => expect(cloudSelect.value).toBe('gpt-4o'));
    expect(within(cloudSelect).getByRole('option', { name: 'gpt-4o' })).toBeInTheDocument();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Routing + coref-model override persistence (M4 Phase 3, Stage 3)
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — routing + coref override', () => {
  it('renders the routing selector and persists local_first on the local Test connection', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama 0.3.2', models: [] };
      if (cmd === 'list_provider_models')
        return cloudCatalog((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
    await waitFor(() =>
      expect(
        screen.getByLabelText(/routing/i, { selector: '#enrichment-routing' })
      ).toBeInTheDocument()
    );

    // Switch routing to Local-first.
    await fireEvent.change(screen.getByLabelText(/routing/i, { selector: '#enrichment-routing' }), {
      target: { value: 'local_first' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            enrichment: expect.objectContaining({
              routing: { kind: 'local_first' }
            })
          })
        })
      )
    );
  });

  it('persists a picked coref model as a TaskModel and clears it to null on default', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama 0.3.2', models: [] };
      if (cmd === 'list_provider_models')
        return cloudCatalog((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return ['llama3.2:3b', 'qwen2.5:7b'];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await fireEvent.click(screen.getByRole('button', { name: /configure/i }));

    // Coref-model picker offers the live Ollama models on the local tab. Scope to
    // the coref-override <select> (the local model picker also lists pulled ids).
    const corefSelect = (await waitFor(() =>
      screen.getByLabelText(/coreference model/i, { selector: '#enrichment-coref-model' })
    )) as HTMLSelectElement;
    await waitFor(() =>
      expect(within(corefSelect).getByRole('option', { name: 'qwen2.5:7b' })).toBeInTheDocument()
    );

    // Pick a non-default coref model → forwards a TaskModel {provider, model}.
    await fireEvent.change(corefSelect, { target: { value: 'qwen2.5:7b' } });
    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            enrichment: expect.objectContaining({
              coref_model: { provider: 'ollama', model: 'qwen2.5:7b' }
            })
          })
        })
      )
    );

    // Reset to the default option ("") → forwards coref_model: null.
    await fireEvent.change(corefSelect, { target: { value: '' } });
    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenLastCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            enrichment: expect.objectContaining({ coref_model: null })
          })
        })
      )
    );
  });
});
