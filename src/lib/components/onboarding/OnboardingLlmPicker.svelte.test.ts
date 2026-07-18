import { render, screen, fireEvent, waitFor, within } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import OnboardingLlmPicker from './OnboardingLlmPicker.svelte';
import { resetChatProvider } from '$lib/models/chat-provider.svelte.js';
import type { SaveApi } from '$lib/onboarding/system-check.js';

/** A base AppConfig with non-default enrichment prefs, to prove Save preserves them. */
function baseConfig() {
  return {
    theme: 'dark',
    accent: 'purple',
    models: [],
    endpoints: {},
    voices: { host: '', guest: '' },
    tts: { provider: '', api_key: '' },
    enrichment: { enabled: true, coref_strategy: 'none' as const, cloud_consent: false },
    paths: { data_dir: '' },
    tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
    onboarding_complete: false,
    embedding_model: ''
  };
}

/** Renders the picker and captures the latest onready api (fires reactively). */
function renderPicker() {
  let api: SaveApi | null = null;
  render(OnboardingLlmPicker, { props: { onready: (a: SaveApi) => (api = a) } });
  return {
    get api(): SaveApi {
      if (!api) throw new Error('onready never fired');
      return api;
    }
  };
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  resetChatProvider();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('OnboardingLlmPicker — local-only render', () => {
  it('renders the endpoint (default localhost:11434), a model field, and Auto-detect', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_ollama_models') return ['llama3.2:3b'];
    });

    renderPicker();

    const endpoint = screen.getByLabelText('Endpoint') as HTMLInputElement;
    expect(endpoint.value).toBe('http://localhost:11434');
    expect(
      screen.getByLabelText('Model', { selector: '#onboarding-llm-model' })
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /auto-detect/i })).toBeInTheDocument();
  });

  it('shows the Settings › AI Model pointer for cloud/other providers', () => {
    mockIPC((cmd) => {
      if (cmd === 'list_ollama_models') return [];
    });
    renderPicker();
    expect(screen.getByText(/settings › ai model/i)).toBeInTheDocument();
  });

  it('renders NO cloud combobox / routing / coref / context / consent controls', () => {
    mockIPC((cmd) => {
      if (cmd === 'list_ollama_models') return [];
    });
    renderPicker();

    expect(screen.queryByRole('button', { name: /show providers/i })).not.toBeInTheDocument();
    expect(document.querySelector('#enrichment-routing')).toBeNull();
    expect(document.querySelector('#enrichment-coref')).toBeNull();
    expect(document.querySelector('#llm-context-window')).toBeNull();
    expect(screen.queryByRole('checkbox', { name: /send document text/i })).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/temperature/i)).not.toBeInTheDocument();
  });
});

describe('OnboardingLlmPicker — local model field (salvaged)', () => {
  it('renders a picker of listed Ollama models', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_ollama_models') return ['llama3.2:3b', 'mistral:7b'];
    });
    renderPicker();

    const select = (await waitFor(() => {
      const el = screen.getByLabelText('Model', { selector: '#onboarding-llm-model' });
      expect(el.tagName).toBe('SELECT');
      return el;
    })) as HTMLSelectElement;
    expect(within(select).getByRole('option', { name: 'llama3.2:3b' })).toBeInTheDocument();
    expect(within(select).getByRole('option', { name: 'mistral:7b' })).toBeInTheDocument();
  });

  it('Auto-detect populates the model select from a reachable endpoint', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_ollama_models') return [];
      if (cmd === 'detect_llm')
        return { reachable: true, version: 'Ollama 0.3.2', models: ['llama3.2:3b', 'mistral:7b'] };
    });
    renderPicker();

    await fireEvent.click(screen.getByRole('button', { name: /auto-detect/i }));

    const select = (await waitFor(() => {
      const el = screen.getByLabelText('Model', { selector: '#onboarding-llm-model' });
      expect(el.tagName).toBe('SELECT');
      return el;
    })) as HTMLSelectElement;
    expect(within(select).getByRole('option', { name: 'llama3.2:3b' })).toBeInTheDocument();
  });
});

describe('OnboardingLlmPicker — Variant-B persist', () => {
  it('Save upserts the ollama models[] entry and pins enrichment.chat_model', async () => {
    const setConfigs: Array<Record<string, unknown>> = [];
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfigs.push((args as { config: Record<string, unknown> }).config);
        return null;
      }
      if (cmd === 'list_ollama_models') return ['llama3.2:3b'];
      if (cmd === 'has_chat_provider') return true;
    });

    const picker = renderPicker();
    const select = (await waitFor(() => {
      const el = screen.getByLabelText('Model', { selector: '#onboarding-llm-model' });
      expect(el.tagName).toBe('SELECT');
      return el;
    })) as HTMLSelectElement;
    await fireEvent.change(select, { target: { value: 'llama3.2:3b' } });
    await picker.api.save();

    const modelWrite = setConfigs.find((c) => Array.isArray(c.models) && c.models.length > 0);
    expect(modelWrite).toBeDefined();
    expect(modelWrite!.models).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          provider: 'ollama',
          model: 'llama3.2:3b',
          base_url: 'http://localhost:11434'
        })
      ])
    );

    const enrichWrite = setConfigs.find(
      (c) => (c.enrichment as { chat_model?: unknown }).chat_model != null
    );
    expect(enrichWrite).toBeDefined();
    expect(enrichWrite!.enrichment).toEqual(
      expect.objectContaining({ chat_model: { provider: 'ollama', model: 'llama3.2:3b' } })
    );
  });

  it('Save preserves prior enrichment.enabled and OMITS routing/coref_model/map_model', async () => {
    const setConfigs: Array<Record<string, unknown>> = [];
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfigs.push((args as { config: Record<string, unknown> }).config);
        return null;
      }
      if (cmd === 'list_ollama_models') return ['llama3.2:3b'];
      if (cmd === 'has_chat_provider') return true;
    });

    const picker = renderPicker();
    const select = (await waitFor(() => {
      const el = screen.getByLabelText('Model', { selector: '#onboarding-llm-model' });
      expect(el.tagName).toBe('SELECT');
      return el;
    })) as HTMLSelectElement;
    await fireEvent.change(select, { target: { value: 'llama3.2:3b' } });
    await picker.api.save();

    const enrichWrite = setConfigs.find(
      (c) => (c.enrichment as { chat_model?: unknown }).chat_model != null
    );
    const enrichment = enrichWrite!.enrichment as Record<string, unknown>;
    // enabled/coref_strategy are preserved from prior (baseConfig's enrichment).
    expect(enrichment.enabled).toBe(true);
    expect(enrichment.coref_strategy).toBe('none');
    expect(enrichment).not.toHaveProperty('routing');
    expect(enrichment).not.toHaveProperty('coref_model');
    expect(enrichment).not.toHaveProperty('map_model');
  });

  it('Save is a no-op while no model is chosen (never pins an empty chat_model)', async () => {
    const setConfigs: unknown[] = [];
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return baseConfig();
      if (cmd === 'set_config') {
        setConfigs.push(args);
        return null;
      }
      if (cmd === 'list_ollama_models') return []; // free-text, starts empty
    });

    const picker = renderPicker();
    await picker.api.save();
    expect(setConfigs).toHaveLength(0);
  });
});
