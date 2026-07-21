import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { AppConfig, ModelConfig } from '$lib/theme/types.js';
import type { ModelInfo } from '$lib/models/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import { resetChatProvider } from '$lib/models/chat-provider.svelte.js';
import { resetActiveModel } from '$lib/models/active-model.svelte.js';
import { resetConfig } from '$lib/models/app-config.svelte.js';
import ActiveModelSection from './ActiveModelSection.svelte';

function textModel(id: string, ctx: number | null = 128000): ModelInfo {
  return {
    id,
    name: id,
    family: null,
    reasoning: false,
    reasoning_options: [],
    tool_call: true,
    temperature: true,
    modalities: { input: ['text'], output: ['text'] },
    context_limit: ctx,
    output_limit: null,
    open_weights: false,
    cost: null,
    last_updated: '2025-01-01',
    release_date: '2025-01-01'
  };
}

function keyedEntry(provider: string, model = '', api_key = 'secret'): ModelConfig {
  return { provider, base_url: '', model, context: 128000, temperature: 0.7, api_key };
}

interface SetupOpts {
  /** Per-provider (catalogKey) cloud model maps returned by `list_provider_models`. */
  models?: Record<string, Record<string, ModelInfo>>;
  ollama?: string[];
}

interface SetupHandle {
  writes: AppConfig[];
  counts: Record<string, number>;
  /** Mutate the current config out-of-band (simulates a concurrent Providers edit). */
  patch: (mutate: (cfg: AppConfig) => AppConfig) => void;
}

function setup(cfg: AppConfig, opts: SetupOpts = {}): SetupHandle {
  const writes: AppConfig[] = [];
  const counts: Record<string, number> = {};
  let current = cfg;
  mockIPC((cmd, args) => {
    counts[cmd] = (counts[cmd] ?? 0) + 1;
    if (cmd === 'get_config') return current;
    if (cmd === 'set_config') {
      current = (args as { config: AppConfig }).config;
      writes.push(current);
      return null;
    }
    if (cmd === 'list_provider_models') {
      const provider = (args as { provider: string }).provider;
      return opts.models?.[provider] ?? {};
    }
    if (cmd === 'list_ollama_models') return opts.ollama ?? [];
    if (cmd === 'has_chat_provider') return true;
    if (cmd === 'list_active_model_candidates') return { active: null, candidates: [] };
    return undefined;
  });
  return {
    writes,
    counts,
    patch: (mutate) => {
      current = mutate(current);
    }
  };
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  resetChatProvider();
  resetActiveModel();
  resetConfig();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('ActiveModelSection', () => {
  it('renders the section without a Save button', async () => {
    setup(baseAppConfig());
    render(ActiveModelSection);
    expect(await screen.findByRole('heading', { name: 'Active model' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^save$/i })).not.toBeInTheDocument();
  });

  // D-T1: the provider dropdown lists ONLY usable providers (keyless cloud absent), and
  // selecting a provider reveals only its own models (no mixed provider·model soup).
  it('lists only usable providers and reveals only the selected provider models (D-T1)', async () => {
    const cfg = baseAppConfig({
      models: [
        keyedEntry('openai', 'gpt-4o'),
        keyedEntry('anthropic', '', 'k2'),
        // google has an empty key → not usable.
        {
          provider: 'google',
          base_url: '',
          model: '',
          context: 128000,
          temperature: 0.7,
          api_key: ''
        }
      ],
      enrichment: {
        enabled: true,
        coref_strategy: 'llm_inline',
        cloud_consent: true,
        chat_model: { provider: 'openai', model: 'gpt-4o' }
      }
    });
    setup(cfg, {
      models: {
        openai: { 'gpt-4o': textModel('gpt-4o') },
        anthropic: { 'claude-sonnet-4-5': textModel('claude-sonnet-4-5') }
      }
    });

    render(ActiveModelSection);
    await screen.findByRole('heading', { name: 'Active model' });

    // The provider trigger shows the pinned provider's label; opening the listbox
    // (bits-ui Select) surfaces only usable providers — no keyless google.
    const providerTrigger = await screen.findByLabelText('Provider');
    await waitFor(() => expect(providerTrigger).toHaveTextContent('OpenAI'));
    await fireEvent.keyDown(providerTrigger, { key: 'Enter' });
    const optionLabels = (await screen.findAllByRole('option')).map((o) => o.textContent?.trim());
    expect(optionLabels).toEqual(['OpenAI', 'Anthropic']);
    expect(optionLabels).not.toContain('Google (Gemini)');

    // Initial provider (openai, from the pin) shows only openai models.
    await waitFor(() => expect(screen.getByRole('radio', { name: /gpt-4o/ })).toBeInTheDocument());
    expect(screen.queryByRole('radio', { name: /claude-sonnet-4-5/ })).not.toBeInTheDocument();

    // Switch to anthropic → only anthropic models remain.
    const anthropicOption = await screen.findByRole('option', { name: 'Anthropic' });
    await fireEvent.pointerUp(anthropicOption);
    await waitFor(() =>
      expect(screen.getByRole('radio', { name: /claude-sonnet-4-5/ })).toBeInTheDocument()
    );
    expect(screen.queryByRole('radio', { name: /gpt-4o/ })).not.toBeInTheDocument();
  });

  // D-T2: choosing a cloud model pins via saveActiveModel (preserving credentials) +
  // saveEnrichmentPrefs (chat_model set, cloud_consent flipped true), then refreshes stores.
  it('pins a cloud model preserving credentials and flipping cloud_consent (D-T2)', async () => {
    const cfg = baseAppConfig({
      models: [
        {
          provider: 'openai',
          base_url: 'https://custom/v1',
          model: '',
          context: 8192,
          temperature: 0.7,
          api_key: 'secret'
        }
      ],
      enrichment: { enabled: false, coref_strategy: 'llm_inline', cloud_consent: false }
    });
    const h = setup(cfg, { models: { openai: { 'gpt-4o': textModel('gpt-4o') } } });

    render(ActiveModelSection);
    await screen.findByRole('heading', { name: 'Active model' });

    const refreshBefore = h.counts['list_active_model_candidates'] ?? 0;
    await fireEvent.click(await screen.findByRole('radio', { name: /gpt-4o/ }));

    await waitFor(() => expect(h.writes.some((w) => w.enrichment.chat_model)).toBe(true));

    // saveActiveModel preserved base_url + api_key while pinning the model and its context_limit.
    const modelWrite = h.writes.find((w) =>
      w.models.some((m) => m.provider === 'openai' && m.model === 'gpt-4o')
    );
    expect(modelWrite).toBeDefined();
    const entry = modelWrite!.models.find((m) => m.provider === 'openai')!;
    expect(entry.api_key).toBe('secret');
    expect(entry.base_url).toBe('https://custom/v1');
    expect(entry.context).toBe(128000);

    // saveEnrichmentPrefs pinned chat_model + flipped cloud_consent, without flipping enabled.
    const enrWrite = h.writes.find((w) => w.enrichment.chat_model);
    expect(enrWrite!.enrichment.chat_model).toMatchObject({ provider: 'openai', model: 'gpt-4o' });
    expect(enrWrite!.enrichment.cloud_consent).toBe(true);
    expect(enrWrite!.enrichment.enabled).toBe(false);

    // Both stores refreshed after the pin.
    expect(h.counts['has_chat_provider']).toBeGreaterThan(0);
    await waitFor(() =>
      expect(h.counts['list_active_model_candidates'] ?? 0).toBeGreaterThan(refreshBefore)
    );
  });

  // D-T3a: a cloud model shows NO context control (only the ctx chip) and persists context_limit.
  it('shows no context control for a cloud model, only the ctx chip (D-T3)', async () => {
    const cfg = baseAppConfig({
      models: [keyedEntry('openai', 'gpt-4o')],
      enrichment: {
        enabled: true,
        coref_strategy: 'llm_inline',
        cloud_consent: true,
        chat_model: { provider: 'openai', model: 'gpt-4o' }
      }
    });
    setup(cfg, { models: { openai: { 'gpt-4o': textModel('gpt-4o', 128000) } } });

    render(ActiveModelSection);
    await screen.findByRole('heading', { name: 'Active model' });
    await waitFor(() => expect(screen.getByRole('radio', { name: /gpt-4o/ })).toBeInTheDocument());

    // No manual ContextWindowField for cloud.
    expect(screen.queryByRole('group', { name: 'Context window size' })).not.toBeInTheDocument();
    // Context is surfaced only as the model-row chip.
    expect(screen.getByText('128K ctx')).toBeInTheDocument();
  });

  // D-T3b: a local/Ollama model shows the manual ContextWindowField.
  it('shows the manual ContextWindowField for a local model (D-T3)', async () => {
    const cfg = baseAppConfig({
      models: [
        {
          provider: 'ollama',
          base_url: 'http://localhost:11434',
          model: 'llama3.2:3b',
          context: 8192,
          temperature: 0.7,
          api_key: ''
        }
      ],
      enrichment: {
        enabled: true,
        coref_strategy: 'llm_inline',
        cloud_consent: false,
        chat_model: { provider: 'ollama', model: 'llama3.2:3b' }
      }
    });
    setup(cfg, { ollama: ['llama3.2:3b', 'qwen2.5:7b'] });

    render(ActiveModelSection);
    await screen.findByRole('heading', { name: 'Active model' });

    await waitFor(() =>
      expect(screen.getByRole('group', { name: 'Context window size' })).toBeInTheDocument()
    );
  });

  // D-T4 (regression): a key edited in Providers between snapshots is NOT reverted by the
  // model pin — saveActiveModel merges base_url/api_key from the CURRENT config in-mutator.
  it('never reverts a concurrently-edited API key when pinning a model (D-T4)', async () => {
    const cfg = baseAppConfig({
      models: [
        {
          provider: 'openai',
          base_url: '',
          model: '',
          context: 8192,
          temperature: 0.7,
          api_key: 'old-key'
        }
      ],
      enrichment: { enabled: false, coref_strategy: 'llm_inline', cloud_consent: false }
    });
    const h = setup(cfg, { models: { openai: { 'gpt-4o': textModel('gpt-4o') } } });

    render(ActiveModelSection);
    await screen.findByRole('heading', { name: 'Active model' });
    await screen.findByRole('radio', { name: /gpt-4o/ });

    // Simulate the user editing the key in the Providers section after this section loaded.
    h.patch((c) => ({
      ...c,
      models: c.models.map((m) => (m.provider === 'openai' ? { ...m, api_key: 'new-key' } : m))
    }));

    await fireEvent.click(screen.getByRole('radio', { name: /gpt-4o/ }));

    await waitFor(() =>
      expect(
        h.writes.some((w) => w.models.some((m) => m.provider === 'openai' && m.model === 'gpt-4o'))
      ).toBe(true)
    );
    const entry = h.writes
      .find((w) => w.models.some((m) => m.provider === 'openai' && m.model === 'gpt-4o'))!
      .models.find((m) => m.provider === 'openai')!;
    expect(entry.api_key).toBe('new-key');
    expect(
      h.writes.some((w) => w.models.some((m) => m.provider === 'openai' && m.api_key === 'old-key'))
    ).toBe(false);
  });

  // FIX 3: a keyed custom (OpenAI-compatible) provider has no catalog, so it exposes a
  // free-text Model input (not a "No models found" dead-end) and can pin the typed model.
  it('pins a free-text model for a custom provider (AC6/AC8)', async () => {
    const cfg = baseAppConfig({
      models: [
        {
          provider: 'openai-compatible',
          base_url: 'https://self.host/v1',
          model: '',
          context: 8192,
          temperature: 0.7,
          api_key: 'ck'
        }
      ],
      enrichment: { enabled: false, coref_strategy: 'llm_inline', cloud_consent: false }
    });
    const h = setup(cfg);

    render(ActiveModelSection);
    await screen.findByRole('heading', { name: 'Active model' });

    // Custom provider → free-text Model input, no radio list, no dead-end message.
    const modelInput = await screen.findByLabelText('Model');
    expect(screen.queryByText('No models found for this provider.')).not.toBeInTheDocument();
    expect(screen.queryByRole('radiogroup', { name: 'Model' })).not.toBeInTheDocument();

    // Type a model id and commit on blur → pins via saveActiveModel + saveEnrichmentPrefs.
    (modelInput as HTMLInputElement).value = 'my-local-model';
    modelInput.dispatchEvent(new InputEvent('input', { bubbles: true }));
    await fireEvent.blur(modelInput);

    await waitFor(() => expect(h.writes.some((w) => w.enrichment.chat_model)).toBe(true));
    const enrWrite = h.writes.find((w) => w.enrichment.chat_model)!;
    expect(enrWrite.enrichment.chat_model).toMatchObject({
      provider: 'openai-compatible',
      model: 'my-local-model'
    });
    // Cloud consent flips true for a custom (non-local) provider.
    expect(enrWrite.enrichment.cloud_consent).toBe(true);

    // saveActiveModel preserved the credentials while pinning the typed model.
    const modelWrite = h.writes.find((w) =>
      w.models.some((m) => m.provider === 'openai-compatible' && m.model === 'my-local-model')
    )!;
    const entry = modelWrite.models.find((m) => m.provider === 'openai-compatible')!;
    expect(entry.api_key).toBe('ck');
    expect(entry.base_url).toBe('https://self.host/v1');
  });

  // FIX 5a: temperature is model-level — hidden until a model is pinned (no dead control).
  it('hides the temperature control until a model is pinned', async () => {
    const cfg = baseAppConfig({
      models: [keyedEntry('openai', '')],
      enrichment: { enabled: false, coref_strategy: 'llm_inline', cloud_consent: false }
    });
    setup(cfg, { models: { openai: { 'gpt-4o': textModel('gpt-4o') } } });

    render(ActiveModelSection);
    await screen.findByRole('heading', { name: 'Active model' });
    await screen.findByRole('radio', { name: /gpt-4o/ });

    // No pin yet → no temperature slider.
    expect(screen.queryByLabelText('Temperature')).not.toBeInTheDocument();

    // Pin a model → the temperature control appears.
    await fireEvent.click(screen.getByRole('radio', { name: /gpt-4o/ }));
    await waitFor(() => expect(screen.getByLabelText('Temperature')).toBeInTheDocument());
  });
});
