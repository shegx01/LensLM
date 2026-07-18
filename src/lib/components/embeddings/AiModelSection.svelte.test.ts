import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import type { ModelInfo } from '$lib/models/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import { resetChatProvider } from '$lib/models/chat-provider.svelte.js';
import AiModelSection from './AiModelSection.svelte';

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

interface SetupOpts {
  provider?: Record<string, ModelInfo>;
  ollama?: string[];
  hasProvider?: boolean;
}

function setup(cfg: AppConfig, opts: SetupOpts = {}): AppConfig[] {
  const writes: AppConfig[] = [];
  mockIPC((cmd, args) => {
    if (cmd === 'get_config') return cfg;
    if (cmd === 'set_config') {
      writes.push((args as { config: AppConfig }).config);
      return null;
    }
    if (cmd === 'list_provider_models') return opts.provider ?? {};
    if (cmd === 'list_ollama_models') return opts.ollama ?? [];
    if (cmd === 'has_chat_provider') return opts.hasProvider ?? true;
    if (cmd === 'validate_model_interactive') return { status: 'valid' };
    if (cmd === 'detect_llm') return { reachable: true, version: 'v1', models: opts.ollama ?? [] };
    if (cmd === 'refresh_models') return false;
    return undefined;
  });
  return writes;
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  resetChatProvider();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('AiModelSection', () => {
  it('renders the panel and has no Save button (AC-1, AC-4)', async () => {
    setup(baseAppConfig());
    render(AiModelSection);
    expect(await screen.findByRole('heading', { name: 'AI Model' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^save$/i })).not.toBeInTheDocument();
  });

  it('initializes from the resolved provider and does NOT repin on open (deterministic init)', async () => {
    const cfg = baseAppConfig({
      models: [
        {
          provider: 'ollama',
          base_url: 'http://localhost:11434',
          model: 'llama3.2:3b',
          context: 8192,
          temperature: 0.7,
          api_key: ''
        },
        {
          provider: 'openai',
          base_url: '',
          model: 'gpt-4o',
          context: 128000,
          temperature: 0.7,
          api_key: 'secret'
        }
      ],
      enrichment: { enabled: true, coref_strategy: 'llm_inline', cloud_consent: true }
    });
    const writes = setup(cfg, { provider: { 'gpt-4o': textModel('gpt-4o') } });

    render(AiModelSection);
    await screen.findByRole('heading', { name: 'AI Model' });

    // CloudFirst + consent → resolves to the cloud entry, not blind models[0].
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: 'Cloud API' })).toHaveAttribute(
        'aria-selected',
        'true'
      )
    );
    // Open-without-edit must not write.
    await new Promise((r) => setTimeout(r, 30));
    expect(writes).toHaveLength(0);
  });

  it('preserves a masked API key when another field changes (KEY-WIPE)', async () => {
    const cfg = baseAppConfig({
      models: [
        {
          provider: 'openai',
          base_url: '',
          model: 'gpt-4o',
          context: 128000,
          temperature: 0.7,
          api_key: 'secret-key'
        }
      ],
      enrichment: {
        enabled: true,
        coref_strategy: 'llm_inline',
        cloud_consent: true,
        routing: { kind: 'explicit', provider: 'openai', model: 'gpt-4o' }
      }
    });
    const writes = setup(cfg, { provider: { 'gpt-4o': textModel('gpt-4o') } });

    render(AiModelSection);
    await screen.findByRole('heading', { name: 'AI Model' });

    await fireEvent.click(await screen.findByRole('button', { name: '32K' }));

    await waitFor(() => expect(writes.length).toBeGreaterThan(0));
    const modelWrite = writes.find((w) =>
      w.models.some((m) => m.provider === 'openai' && m.context === 32768)
    );
    expect(modelWrite).toBeDefined();
    const entry = modelWrite!.models.find((m) => m.provider === 'openai')!;
    expect(entry.api_key).toBe('secret-key');
  });

  it('preserves a masked API key on focus-then-blur without typing (KEY-WIPE)', async () => {
    const cfg = baseAppConfig({
      models: [
        {
          provider: 'openai',
          base_url: '',
          model: 'gpt-4o',
          context: 128000,
          temperature: 0.7,
          api_key: 'secret-key'
        }
      ],
      enrichment: {
        enabled: true,
        coref_strategy: 'llm_inline',
        cloud_consent: true,
        routing: { kind: 'explicit', provider: 'openai', model: 'gpt-4o' }
      }
    });
    const writes = setup(cfg, { provider: { 'gpt-4o': textModel('gpt-4o') } });

    render(AiModelSection);
    await screen.findByRole('heading', { name: 'AI Model' });

    // Focusing the masked field flips it into edit mode with an empty buffer;
    // blurring without a keystroke must not persist the empty string as the key.
    const keyInput = await screen.findByLabelText('API Key');
    await fireEvent.focus(keyInput);
    await fireEvent.blur(keyInput);

    const wipe = writes.find((w) =>
      w.models.some((m) => m.provider === 'openai' && m.api_key === '')
    );
    expect(wipe).toBeUndefined();
  });

  it('never rewrites prior enrichment flags for a local save (CLOBBER-GUARD)', async () => {
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
        cloud_consent: true,
        routing: { kind: 'explicit', provider: 'ollama', model: 'llama3.2:3b' }
      }
    });
    const writes = setup(cfg, { ollama: ['llama3.2:3b', 'qwen2.5:7b'] });

    render(AiModelSection);
    await screen.findByRole('heading', { name: 'AI Model' });

    await fireEvent.click(await screen.findByRole('button', { name: '16K' }));

    await waitFor(() => expect(writes.length).toBeGreaterThan(0));
    const enrWrite = writes.find((w) => w.enrichment.routing?.kind === 'explicit');
    expect(enrWrite).toBeDefined();
    expect(enrWrite!.enrichment.enabled).toBe(true);
    expect(enrWrite!.enrichment.coref_strategy).toBe('llm_inline');
    // cloud_consent untouched (preserved, not forced false) for a local provider.
    expect(enrWrite!.enrichment.cloud_consent).toBe(true);
  });

  it('writes routing=Explicit and cloud_consent when a cloud provider is selected (AC-2)', async () => {
    const cfg = baseAppConfig({
      models: [],
      enrichment: { enabled: false, coref_strategy: 'llm_inline', cloud_consent: false }
    });
    const writes = setup(cfg, { provider: { 'gpt-4o': textModel('gpt-4o') } });

    render(AiModelSection);
    await screen.findByRole('heading', { name: 'AI Model' });

    await fireEvent.click(screen.getByRole('tab', { name: 'Cloud API' }));

    await waitFor(() => expect(writes.some((w) => w.enrichment.cloud_consent === true)).toBe(true));
    const enrWrite = writes.find((w) => w.enrichment.routing?.kind === 'explicit');
    expect(enrWrite).toBeDefined();
    expect(enrWrite!.enrichment.cloud_consent).toBe(true);
    // enabled must NOT be flipped on by this panel.
    expect(enrWrite!.enrichment.enabled).toBe(false);
  });

  it('persists the temperature slider into the chat entry (AC-2)', async () => {
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
        routing: { kind: 'explicit', provider: 'ollama', model: 'llama3.2:3b' }
      }
    });
    const writes = setup(cfg, { ollama: ['llama3.2:3b'] });

    render(AiModelSection);
    await screen.findByRole('heading', { name: 'AI Model' });

    const slider = screen.getByLabelText('Temperature');
    // Range bind:value updates on input; persist fires on change (drag then release).
    await fireEvent.input(slider, { target: { value: '1.2' } });
    await fireEvent.change(slider, { target: { value: '1.2' } });

    await waitFor(() => expect(writes.length).toBeGreaterThan(0));
    const modelWrite = writes.find((w) =>
      w.models.some((m) => m.provider === 'ollama' && Math.abs(m.temperature - 1.2) < 1e-6)
    );
    expect(modelWrite).toBeDefined();
  });

  it('Advanced override pins BOTH coref_model and map_model (AC-3)', async () => {
    const cfg = baseAppConfig({
      models: [
        {
          provider: 'openai',
          base_url: '',
          model: 'gpt-4o',
          context: 128000,
          temperature: 0.7,
          api_key: 'secret'
        }
      ],
      enrichment: {
        enabled: true,
        coref_strategy: 'llm_inline',
        cloud_consent: true,
        routing: { kind: 'explicit', provider: 'openai', model: 'gpt-4o' }
      }
    });
    const writes = setup(cfg, {
      provider: { 'gpt-4o': textModel('gpt-4o'), 'gpt-4o-mini': textModel('gpt-4o-mini') }
    });

    render(AiModelSection);
    await screen.findByRole('heading', { name: 'AI Model' });

    await fireEvent.click(
      screen.getByRole('button', { name: /advanced: separate enrichment model/i })
    );
    const overrideSelect = await screen.findByLabelText('Enrichment model');
    await fireEvent.change(overrideSelect, { target: { value: 'gpt-4o-mini' } });

    await waitFor(() => expect(writes.some((w) => w.enrichment.coref_model)).toBe(true));
    const w = writes.find((x) => x.enrichment.coref_model)!;
    expect(w.enrichment.coref_model).toEqual({ provider: 'openai', model: 'gpt-4o-mini' });
    expect(w.enrichment.map_model).toEqual({ provider: 'openai', model: 'gpt-4o-mini' });
  });

  it('Auto-detect surfaces a reachable local server (AC-6)', async () => {
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
        routing: { kind: 'explicit', provider: 'ollama', model: 'llama3.2:3b' }
      }
    });
    setup(cfg, { ollama: ['llama3.2:3b'] });

    render(AiModelSection);
    await screen.findByRole('heading', { name: 'AI Model' });

    await fireEvent.click(await screen.findByRole('button', { name: /auto-detect local models/i }));
    expect(await screen.findByText(/connected — v1/i)).toBeInTheDocument();
  });
});
