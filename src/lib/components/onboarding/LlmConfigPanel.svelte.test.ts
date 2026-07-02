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

/** A models.dev-style catalog map for `list_provider_models`. */
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

/**
 * Drive the Cloud provider combobox (bits-ui `Combobox`). Opening requires a
 * `pointerDown` on the trigger; SELECTING an option requires a `pointerUp`.
 */
async function selectCloudProvider(name: RegExp | string): Promise<void> {
  const trigger = screen.getByRole('button', { name: /show providers/i });
  await fireEvent.pointerDown(trigger);
  const option = await screen.findByRole('option', { name });
  await fireEvent.pointerUp(option);
}

/** Expand the LlmConfigPanel by clicking the Configure button. */
async function expandPanel(): Promise<void> {
  await fireEvent.click(screen.getByRole('button', { name: /configure/i }));
  await waitFor(() => expect(screen.getByRole('tab', { name: /local/i })).toBeInTheDocument());
}

/** Switch to the Cloud API tab (panel already expanded). */
async function switchToCloud(): Promise<void> {
  await fireEvent.click(screen.getByRole('tab', { name: /cloud api/i }));
}

beforeEach(() => {
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

    expect(screen.queryByRole('tab', { name: /local/i })).not.toBeInTheDocument();

    const btn = screen.getByRole('button', { name: /configure/i });
    expect(btn).not.toBeDisabled();
    await fireEvent.click(btn);

    await waitFor(() => expect(screen.getByRole('tab', { name: /local/i })).toBeInTheDocument());
    expect(screen.getByRole('tab', { name: /cloud api/i })).toBeInTheDocument();
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
// Two model roles — labels present on both tabs
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — two model roles', () => {
  it('renders BOTH "Enrichment model" and "Studio & Chat model" selectors on the local tab', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'list_ollama_models') return ['llama3.2:3b', 'mistral:7b'];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();

    await waitFor(() =>
      expect(
        screen.getByLabelText('Enrichment model', { selector: '#llm-model-local' })
      ).toBeInTheDocument()
    );
    expect(
      screen.getByLabelText('Studio & Chat model', { selector: '#studio-chat-model-local' })
    ).toBeInTheDocument();
    // Helper text for each role (appears on both the local + cloud panels).
    expect(
      screen.getAllByText(/enrich sources \(coreference \+ structural mapping\)/i).length
    ).toBeGreaterThan(0);
    expect(screen.getAllByText(/used when chat\/studio ships/i).length).toBeGreaterThan(0);
  });

  it('renders BOTH role selectors on the cloud tab', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'list_provider_models')
        return cloudCatalog((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();
    await switchToCloud();

    await waitFor(() =>
      expect(
        screen.getByLabelText('Enrichment model', { selector: '#llm-cloud-model' })
      ).toBeInTheDocument()
    );
    expect(
      screen.getByLabelText('Studio & Chat model', { selector: '#studio-chat-model-cloud' })
    ).toBeInTheDocument();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Local model picker vs. free-text pull prompt (Rev 2)
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — local model picker / pull prompt', () => {
  it('renders a picker of detected models (no hardcoded default; enrichment auto-selects first)', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'list_ollama_models') return ['llama3.2:3b', 'mistral:7b'];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();

    const enrichment = (await waitFor(() =>
      screen.getByLabelText('Enrichment model', { selector: '#llm-model-local' })
    )) as HTMLSelectElement;
    expect(enrichment.tagName).toBe('SELECT');
    expect(within(enrichment).getByRole('option', { name: 'llama3.2:3b' })).toBeInTheDocument();
    expect(within(enrichment).getByRole('option', { name: 'mistral:7b' })).toBeInTheDocument();
    // ENRICHMENT auto-selects the FIRST detected model (not a hardcoded llama3.2:3b).
    await waitFor(() => expect(enrichment.value).toBe('llama3.2:3b'));
  });

  it('renders a free-text field with the placeholder + pull command + Re-check when no models', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();

    const enrichment = (await waitFor(() =>
      screen.getByLabelText('Enrichment model', { selector: '#llm-model-local' })
    )) as HTMLInputElement;
    expect(enrichment.tagName).toBe('INPUT');
    expect(enrichment).toHaveValue('');
    expect(enrichment).toHaveAttribute('placeholder', 'e.g. llama3.2:3b');
    // Copyable pull command as PLAIN text in <code> (defaults to the suggestion).
    expect(screen.getAllByText(/ollama pull llama3\.2:3b/i).length).toBeGreaterThan(0);
    // Re-check button present for the enrichment role.
    expect(
      screen.getByRole('button', { name: /re-check ollama models for enrichment model/i })
    ).toBeInTheDocument();
  });

  it('Re-check re-loads Ollama models and switches the field to a picker', async () => {
    let pulled = false;
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'list_ollama_models') return pulled ? ['llama3.2:3b'] : [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();

    // Initially: free-text (no models).
    await waitFor(() =>
      expect(
        (screen.getByLabelText('Enrichment model', { selector: '#llm-model-local' }) as HTMLElement)
          .tagName
      ).toBe('INPUT')
    );

    // A model is now pulled; clicking Re-check populates the picker.
    pulled = true;
    await fireEvent.click(
      screen.getByRole('button', { name: /re-check ollama models for enrichment model/i })
    );

    const enrichment = (await waitFor(() =>
      screen.getByLabelText('Enrichment model', { selector: '#llm-model-local' })
    )) as HTMLSelectElement;
    await waitFor(() => expect(enrichment.tagName).toBe('SELECT'));
    await waitFor(() => expect(enrichment.value).toBe('llama3.2:3b'));
  });

  it('the Studio & Chat picker includes a "Not set" option mapping to ""', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'list_ollama_models') return ['llama3.2:3b'];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();

    const studio = (await waitFor(() =>
      screen.getByLabelText('Studio & Chat model', { selector: '#studio-chat-model-local' })
    )) as HTMLSelectElement;
    expect(within(studio).getByRole('option', { name: /not set/i })).toBeInTheDocument();
    // Studio/chat stays empty until explicitly picked (does NOT auto-select).
    expect(studio.value).toBe('');
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Auto-detect
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — Auto-detect', () => {
  it('populates the enrichment model select with detected models on reachable response', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'detect_llm')
        return { reachable: true, version: 'Ollama 0.3.2', models: ['llama3.2:3b', 'mistral:7b'] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();
    await waitFor(() => expect(screen.getByLabelText(/api endpoint/i)).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /auto-detect/i }));

    await waitFor(() =>
      expect(screen.getByText(/configure your preferred llm/i)).toBeInTheDocument()
    );

    const select = screen.getByLabelText('Enrichment model', {
      selector: '#llm-model-local'
    }) as HTMLSelectElement;
    expect(select).toBeInTheDocument();
    expect(within(select).getByRole('option', { name: 'llama3.2:3b' })).toBeInTheDocument();
    expect(within(select).getByRole('option', { name: 'mistral:7b' })).toBeInTheDocument();
  });

  it('shows "Not detected" hint when reachable is false', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'detect_llm') return { reachable: false, version: null, models: [] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();
    await waitFor(() => expect(screen.getByLabelText(/api endpoint/i)).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: /auto-detect/i }));

    await waitFor(() => expect(screen.getByText(/not detected/i)).toBeInTheDocument());
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Empty-model guard — disables Test Connection / Save
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — empty enrichment-model guard', () => {
  it('disables local Test Connection while the enrichment model is empty', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'list_ollama_models') return []; // free-text, starts empty
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();

    const test = await waitFor(() => screen.getByRole('button', { name: /test connection/i }));
    expect(test).toBeDisabled();

    // Typing an enrichment model enables it.
    await fireEvent.input(
      screen.getByLabelText('Enrichment model', { selector: '#llm-model-local' }),
      { target: { value: 'llama3.2:3b' } }
    );
    expect(test).not.toBeDisabled();
  });

  it('does NOT disable Test Connection when only the studio/chat model is empty', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'list_ollama_models') return ['llama3.2:3b']; // enrichment auto-selects
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();

    // Enrichment auto-selected → Test enabled even though studio/chat is "Not set".
    const test = await waitFor(() => screen.getByRole('button', { name: /test connection/i }));
    await waitFor(() => expect(test).not.toBeDisabled());
    const studio = screen.getByLabelText('Studio & Chat model', {
      selector: '#studio-chat-model-local'
    }) as HTMLSelectElement;
    expect(studio.value).toBe('');
  });

  it('disables the cloud Save while the enrichment model is empty (catalog-less provider)', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'list_provider_models') return {};
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();
    await switchToCloud();

    // Switch to the custom (catalog-less) provider so the enrichment model is
    // free-text and can be emptied.
    await selectCloudProvider(/custom \(openai-compatible\)/i);

    const enrichment = screen.getByLabelText('Enrichment model', {
      selector: '#llm-cloud-model'
    }) as HTMLInputElement;
    await fireEvent.input(enrichment, { target: { value: '' } });
    await fireEvent.input(screen.getByLabelText(/api key/i), { target: { value: 'sk-key' } });

    const save = screen.getByRole('button', { name: /save/i });
    expect(save).toBeDisabled();

    await fireEvent.input(enrichment, { target: { value: 'my-model' } });
    expect(save).not.toBeDisabled();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Enrichment validation gate — blocking (local)
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — enrichment blocking (local Test connection)', () => {
  it('blocks save + does NOT call oncheck when the enrichment model is invalid', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'list_ollama_models') return ['llama3.2:3b'];
      if (cmd === 'validate_model_interactive')
        return { status: 'invalid', reason: "Model 'llama3.2:3b' is not installed." };
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama', models: [] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await expandPanel();
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /test connection/i })).not.toBeDisabled()
    );

    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    // The invalid reason is shown inline and the config was NOT persisted.
    await waitFor(() =>
      expect(
        screen.getAllByRole('alert').some((el) => /not installed/i.test(el.textContent ?? ''))
      ).toBe(true)
    );
    expect(setConfig).not.toHaveBeenCalled();
    expect(oncheck).not.toHaveBeenCalled();
  });

  it('persists + calls oncheck when the enrichment model is valid', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'list_ollama_models') return ['llama3.2:3b'];
      if (cmd === 'validate_model_interactive') return { status: 'valid' };
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama', models: [] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await expandPanel();
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /test connection/i })).not.toBeDisabled()
    );

    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            models: expect.arrayContaining([
              expect.objectContaining({ provider: 'ollama', model: 'llama3.2:3b' })
            ])
          })
        })
      )
    );
    await waitFor(() => expect(oncheck).toHaveBeenCalledOnce());
  });

  it('persists the picked context window (default 8192) on a valid enrichment model', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'list_ollama_models') return ['llama3.2:3b'];
      if (cmd === 'validate_model_interactive') return { status: 'valid' };
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama', models: [] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await expandPanel();
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /test connection/i })).not.toBeDisabled()
    );

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
});

// ──────────────────────────────────────────────────────────────────────────
// Studio & Chat — non-blocking + chat_model persistence
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — studio/chat non-blocking + chat_model persist', () => {
  it('shows the studio/chat invalid status BUT still saves + calls oncheck (non-blocking)', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'list_ollama_models') return ['llama3.2:3b', 'qwen2.5:7b'];
      if (cmd === 'validate_model_interactive') {
        const model = (args as { model: string }).model;
        // Enrichment (llama3.2:3b) valid; studio/chat (qwen2.5:7b) invalid.
        return model === 'qwen2.5:7b'
          ? { status: 'invalid', reason: "Model 'qwen2.5:7b' is not installed." }
          : { status: 'valid' };
      }
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama', models: [] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await expandPanel();

    const studio = (await waitFor(() =>
      screen.getByLabelText('Studio & Chat model', { selector: '#studio-chat-model-local' })
    )) as HTMLSelectElement;
    await fireEvent.change(studio, { target: { value: 'qwen2.5:7b' } });

    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    // Studio/chat invalid status is shown…
    await waitFor(() =>
      expect(screen.getAllByText(/'qwen2\.5:7b' is not installed/i).length).toBeGreaterThan(0)
    );
    // …but the save proceeded and oncheck ran (non-blocking), with chat_model set.
    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            enrichment: expect.objectContaining({
              chat_model: { provider: 'ollama', model: 'qwen2.5:7b' }
            })
          })
        })
      )
    );
    await waitFor(() => expect(oncheck).toHaveBeenCalledOnce());
  });

  it('persists chat_model = null when the studio/chat model is unpicked', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'list_ollama_models') return ['llama3.2:3b'];
      if (cmd === 'validate_model_interactive') return { status: 'valid' };
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama', models: [] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await expandPanel();
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /test connection/i })).not.toBeDisabled()
    );

    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            enrichment: expect.objectContaining({ chat_model: null })
          })
        })
      )
    );
  });
});

// ──────────────────────────────────────────────────────────────────────────
// chat_model round-trip — restore + preserve
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — chat_model round-trip', () => {
  it('restores a saved ollama chat_model into the local Studio & Chat picker', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') {
        const cfg = baseConfig();
        return {
          ...cfg,
          enrichment: { ...cfg.enrichment, chat_model: { provider: 'ollama', model: 'qwen2.5:7b' } }
        };
      }
      if (cmd === 'set_config') return null;
      if (cmd === 'list_ollama_models') return ['llama3.2:3b', 'qwen2.5:7b'];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();

    const studio = (await waitFor(() =>
      screen.getByLabelText('Studio & Chat model', { selector: '#studio-chat-model-local' })
    )) as HTMLSelectElement;
    await waitFor(() => expect(studio.value).toBe('qwen2.5:7b'));

    // The ENRICHMENT model is NOT clobbered by chat_model (auto-selects first).
    const enrichment = screen.getByLabelText('Enrichment model', {
      selector: '#llm-model-local'
    }) as HTMLSelectElement;
    await waitFor(() => expect(enrichment.value).toBe('llama3.2:3b'));
  });

  it('restores a saved cloud chat_model into the cloud Studio & Chat picker', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') {
        const cfg = baseConfig();
        return {
          ...cfg,
          models: [
            {
              provider: 'openai',
              base_url: '',
              model: 'gpt-4o',
              context: 128000,
              temperature: 0.7,
              api_key: 'sk-saved-secret'
            }
          ],
          enrichment: { ...cfg.enrichment, chat_model: { provider: 'openai', model: 'gpt-4o' } }
        };
      }
      if (cmd === 'set_config') return null;
      if (cmd === 'list_provider_models')
        return cloudCatalog((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();
    await switchToCloud();

    const studio = (await waitFor(() =>
      screen.getByLabelText('Studio & Chat model', { selector: '#studio-chat-model-cloud' })
    )) as HTMLSelectElement;
    await waitFor(() => expect(studio.value).toBe('gpt-4o'));
  });

  it('preserves chat_model across a subsequent save (round-trip, not cleared)', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') {
        const cfg = baseConfig();
        return {
          ...cfg,
          models: [
            {
              provider: 'openai',
              base_url: '',
              model: 'gpt-4o',
              context: 128000,
              temperature: 0.7,
              api_key: 'sk-saved-secret'
            }
          ],
          enrichment: {
            ...cfg.enrichment,
            enabled: true,
            cloud_consent: true,
            chat_model: { provider: 'openai', model: 'gpt-4o' }
          }
        };
      }
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'list_provider_models')
        return cloudCatalog((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'validate_model_interactive') return { status: 'valid' };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await expandPanel();
    await switchToCloud();

    const studio = (await waitFor(() =>
      screen.getByLabelText('Studio & Chat model', { selector: '#studio-chat-model-cloud' })
    )) as HTMLSelectElement;
    await waitFor(() => expect(studio.value).toBe('gpt-4o'));

    // Grant consent so the enrichment gate runs (valid), then re-enter the key + save.
    await fireEvent.click(screen.getByRole('checkbox', { name: /send document text/i }));
    const keyField = screen.getByLabelText(/api key/i);
    await fireEvent.focus(keyField);
    await fireEvent.input(keyField, { target: { value: 'sk-new-key' } });
    await fireEvent.click(screen.getByRole('button', { name: /save/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            enrichment: expect.objectContaining({
              chat_model: { provider: 'openai', model: 'gpt-4o' }
            })
          })
        })
      )
    );
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Cloud Save — real provider id + key forwarding (enrichment gate off w/o consent)
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — Save (cloud tab)', () => {
  it('persists the real cloud provider id + entered key (no consent → enrichment gate skipped)', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'list_provider_models')
        return cloudCatalog((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await expandPanel();
    await switchToCloud();

    await fireEvent.input(screen.getByLabelText(/api key/i), { target: { value: 'sk-test-1234' } });
    await fireEvent.click(screen.getByRole('button', { name: /save/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            models: expect.arrayContaining([
              expect.objectContaining({
                provider: 'openai',
                model: 'gpt-4o',
                api_key: 'sk-test-1234',
                context: 128000
              })
            ])
          })
        })
      )
    );
    await waitFor(() => expect(oncheck).toHaveBeenCalledOnce());
  });

  it('cloud enrichment gate BLOCKS save when consent granted + enrichment model invalid', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'list_provider_models')
        return cloudCatalog((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'validate_model_interactive')
        return { status: 'invalid', reason: 'API key invalid or model not accessible' };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await expandPanel();
    await switchToCloud();

    await fireEvent.click(screen.getByRole('checkbox', { name: /send document text/i }));
    await fireEvent.input(screen.getByLabelText(/api key/i), { target: { value: 'bad-key' } });
    await fireEvent.click(screen.getByRole('button', { name: /save/i }));

    await waitFor(() =>
      expect(
        screen.getAllByRole('alert').some((el) => /api key invalid/i.test(el.textContent ?? ''))
      ).toBe(true)
    );
    expect(setConfig).not.toHaveBeenCalled();
    expect(oncheck).not.toHaveBeenCalled();
  });

  it('masks a previously-saved cloud key: Save disabled initially, enabled after editing', async () => {
    const oncheck = vi.fn().mockResolvedValue(undefined);

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
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await expandPanel();
    await switchToCloud();

    const keyField = screen.getByLabelText(/api key/i);
    const save = screen.getByRole('button', { name: /save/i });

    await waitFor(() => expect(keyField).toHaveValue(''));
    expect(keyField).toHaveAttribute('placeholder', expect.stringMatching(/saved/i));
    expect(save).toBeDisabled();

    await fireEvent.focus(keyField);
    await fireEvent.input(keyField, { target: { value: 'sk-new-key' } });
    expect(save).not.toBeDisabled();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Cloud provider combobox — grouped, searchable
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — cloud provider combobox', () => {
  async function openCloudCombobox() {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'list_provider_models')
        return cloudCatalog((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();
    await switchToCloud();
    await fireEvent.pointerDown(screen.getByRole('button', { name: /show providers/i }));
  }

  it('renders grouped Popular + All headings and lists the new providers when open', async () => {
    await openCloudCombobox();

    await waitFor(() => expect(screen.getByText('Popular')).toBeInTheDocument());
    expect(screen.getByText('All')).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /groq/i })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /deepseek/i })).toBeInTheDocument();
    expect(
      screen.getByRole('option', { name: /custom \(openai-compatible\)/i })
    ).toBeInTheDocument();
  });

  it('type-to-filter narrows the list to DeepSeek and hides OpenAI', async () => {
    await openCloudCombobox();

    const input = screen.getByRole('combobox', { name: /cloud provider/i });
    await fireEvent.input(input, { target: { value: 'deep' } });

    await waitFor(() =>
      expect(screen.getByRole('option', { name: /deepseek/i })).toBeInTheDocument()
    );
    expect(screen.queryByRole('option', { name: /^openai$/i })).not.toBeInTheDocument();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Cloud enrichment model picker — catalog-driven
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — cloud enrichment picker', () => {
  it('renders catalog models in the cloud enrichment select for the selected provider', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') return null;
      if (cmd === 'list_provider_models')
        return cloudCatalog((args as { provider: string }).provider);
      if (cmd === 'list_ollama_models') return [];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();
    await switchToCloud();

    const cloudSelect = (await waitFor(() =>
      screen.getByLabelText('Enrichment model', { selector: '#llm-cloud-model' })
    )) as HTMLSelectElement;
    await waitFor(() =>
      expect(within(cloudSelect).getByRole('option', { name: 'GPT-4o' })).toBeInTheDocument()
    );

    await selectCloudProvider(/anthropic/i);
    await waitFor(() =>
      expect(
        within(cloudSelect).getByRole('option', { name: 'Claude 3.5 Sonnet' })
      ).toBeInTheDocument()
    );
  });

  it('forwards a custom enrichment model id picked from the catalog', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
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
    await expandPanel();
    await switchToCloud();

    const modelField = screen.getByLabelText('Enrichment model', {
      selector: '#llm-cloud-model'
    }) as HTMLSelectElement;
    await waitFor(() =>
      expect(within(modelField).getByRole('option', { name: 'GPT-5 Turbo' })).toBeInTheDocument()
    );
    await fireEvent.change(modelField, { target: { value: 'gpt-5-turbo' } });

    await fireEvent.input(screen.getByLabelText(/api key/i), { target: { value: 'sk-test-key' } });
    await fireEvent.click(screen.getByRole('button', { name: /save/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            models: expect.arrayContaining([
              expect.objectContaining({ provider: 'openai', model: 'gpt-5-turbo' })
            ])
          })
        })
      )
    );
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Enrichment prefs + opt-out
// ──────────────────────────────────────────────────────────────────────────

describe('LlmConfigPanel — Enrichment prefs', () => {
  it('renders the enable toggle + coref select; cloud consent HIDDEN on the local tab', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'list_ollama_models') return ['llama3.2:3b'];
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck: vi.fn() } });
    await expandPanel();

    await waitFor(() =>
      expect(screen.getByRole('switch', { name: /enable enrichment/i })).toBeInTheDocument()
    );
    expect(
      screen.getByLabelText(/pronoun resolution/i, { selector: '#enrichment-coref' })
    ).toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: /send document text/i })).not.toBeInTheDocument();
  });

  it('opt-out (enrichment OFF) shows the tradeoff text AND allows save without validation', async () => {
    const setConfig = vi.fn();
    const oncheck = vi.fn().mockResolvedValue(undefined);
    const validate = vi.fn();

    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
      if (cmd === 'list_ollama_models') return []; // enrichment field empty → disabled unless opted out
      if (cmd === 'validate_model_interactive') {
        validate();
        return { status: 'valid' };
      }
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama', models: [] };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await expandPanel();

    // Turn enrichment OFF.
    await fireEvent.click(screen.getByRole('switch', { name: /enable enrichment/i }));

    // Tradeoff text appears.
    await waitFor(() =>
      expect(screen.getByText(/without enrichment quality boosts/i)).toBeInTheDocument()
    );

    // Give the enrichment model a value so the (empty-model) guard doesn't block.
    await fireEvent.input(
      screen.getByLabelText('Enrichment model', { selector: '#llm-model-local' }),
      { target: { value: 'llama3.2:3b' } }
    );
    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    // Enrichment validation was SKIPPED (opt-out), but the save proceeded + oncheck ran.
    await waitFor(() => expect(oncheck).toHaveBeenCalledOnce());
    expect(validate).not.toHaveBeenCalled();
    expect(setConfig).toHaveBeenCalled();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Routing + coref-model override persistence
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
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama', models: [] };
      if (cmd === 'list_ollama_models') return ['llama3.2:3b'];
      if (cmd === 'validate_model_interactive') return { status: 'valid' };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await expandPanel();
    await waitFor(() =>
      expect(
        screen.getByLabelText(/routing/i, { selector: '#enrichment-routing' })
      ).toBeInTheDocument()
    );

    await fireEvent.change(screen.getByLabelText(/routing/i, { selector: '#enrichment-routing' }), {
      target: { value: 'local_first' }
    });
    await fireEvent.click(screen.getByRole('button', { name: /test connection/i }));

    await waitFor(() =>
      expect(setConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({
            enrichment: expect.objectContaining({ routing: { kind: 'local_first' } })
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
      if (cmd === 'detect_llm') return { reachable: true, version: 'Ollama', models: [] };
      if (cmd === 'list_ollama_models') return ['llama3.2:3b', 'qwen2.5:7b'];
      if (cmd === 'validate_model_interactive') return { status: 'valid' };
    });

    render(SystemCheckRow, { props: { result: llmRow(), oncheck } });
    await expandPanel();

    const corefSelect = (await waitFor(() =>
      screen.getByLabelText(/coreference model/i, { selector: '#enrichment-coref-model' })
    )) as HTMLSelectElement;
    await waitFor(() =>
      expect(within(corefSelect).getByRole('option', { name: 'qwen2.5:7b' })).toBeInTheDocument()
    );

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
