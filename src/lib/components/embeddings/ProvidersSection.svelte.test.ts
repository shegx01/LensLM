import { render, screen, fireEvent, waitFor, within } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { tick } from 'svelte';
import { openUrl } from '@tauri-apps/plugin-opener';
import type { AppConfig } from '$lib/theme/types.js';
import type { ModelInfo, ProviderEntry } from '$lib/models/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';
import { resetChatProvider } from '$lib/models/chat-provider.svelte.js';
import { resetActiveModel } from '$lib/models/active-model.svelte.js';
import { resetConfig } from '$lib/models/app-config.svelte.js';
import ProvidersSection from './ProvidersSection.svelte';
import ActiveModelSection from './ActiveModelSection.svelte';

vi.mock('@tauri-apps/plugin-opener', () => ({ openUrl: vi.fn() }));

function textModel(id: string): ModelInfo {
  return {
    id,
    name: id,
    family: null,
    reasoning: false,
    reasoning_options: [],
    tool_call: true,
    temperature: true,
    modalities: { input: ['text'], output: ['text'] },
    context_limit: 128000,
    output_limit: null,
    open_weights: false,
    cost: null,
    last_updated: '2025-01-01',
    release_date: '2025-01-01'
  };
}

function providerEntry(
  id: string,
  name: string,
  modelIds: string[],
  opts: { doc?: string; env?: string[] } = {}
): ProviderEntry {
  return {
    id,
    name,
    env: opts.env ?? [],
    doc: opts.doc ?? null,
    models: Object.fromEntries(modelIds.map((m) => [m, textModel(m)]))
  };
}

interface SetupOpts {
  /** Value returned by `list_models` (counts + doc/env). Throw a marker to reject. */
  catalog?: Record<string, ProviderEntry> | 'reject';
  ollama?: string[];
}

function setup(cfg: AppConfig, opts: SetupOpts = {}): AppConfig[] {
  const writes: AppConfig[] = [];
  let current = cfg;
  mockIPC((cmd, args) => {
    if (cmd === 'get_config') return current;
    if (cmd === 'set_config') {
      current = (args as { config: AppConfig }).config;
      writes.push(current);
      return null;
    }
    if (cmd === 'list_models') {
      if (opts.catalog === 'reject') throw new Error('catalog unavailable');
      return opts.catalog ?? {};
    }
    if (cmd === 'list_ollama_models') return opts.ollama ?? [];
    if (cmd === 'has_chat_provider') return true;
    if (cmd === 'list_active_model_candidates') return { active: null, candidates: [] };
    return undefined;
  });
  return writes;
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  vi.mocked(openUrl).mockClear();
  resetChatProvider();
  resetActiveModel();
  resetConfig();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

function listbox() {
  return within(screen.getByRole('listbox', { name: 'Providers' }));
}

// Svelte 5 `bind:value` captures a real InputEvent — @testing-library's `fireEvent.input`
// dispatches a plain Event that the binding ignores under happy-dom.
async function typeInto(el: HTMLElement, value: string): Promise<void> {
  (el as HTMLInputElement).value = value;
  el.dispatchEvent(new InputEvent('input', { bubbles: true }));
  await tick();
}

describe('ProvidersSection', () => {
  it('renders the section and every provider row, with no Save button', async () => {
    setup(baseAppConfig());
    render(ProvidersSection);

    expect(await screen.findByRole('heading', { name: 'Providers' })).toBeInTheDocument();
    // Ollama + 10 cloud providers = 11 rows.
    expect(screen.getAllByRole('option')).toHaveLength(11);
    expect(screen.queryByRole('button', { name: /^save$/i })).not.toBeInTheDocument();
  });

  // C-T1: editing an API key (commit) persists via saveProviderCredential and PRESERVES model.
  it('editing an API key preserves the existing model (C-T1)', async () => {
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
      ]
    });
    const writes = setup(cfg, {
      catalog: { openai: providerEntry('openai', 'OpenAI', ['gpt-4o']) }
    });
    render(ProvidersSection);
    await screen.findByRole('heading', { name: 'Providers' });

    await fireEvent.click(listbox().getByText('OpenAI').closest('button')!);

    const keyInput = await screen.findByLabelText('API Key');
    await fireEvent.focus(keyInput);
    await typeInto(keyInput, 'new-key');
    await fireEvent.blur(keyInput);

    await waitFor(() => expect(writes.length).toBeGreaterThan(0));
    const entry = writes.at(-1)!.models.find((m) => m.provider === 'openai')!;
    expect(entry.api_key).toBe('new-key');
    // Model / context / temperature are preserved by the credential-only merge.
    expect(entry.model).toBe('gpt-4o');
    expect(entry.context).toBe(128000);
  });

  // C-T1: editing a custom Base URL (blur) preserves the existing model + api_key.
  it('editing a Base URL preserves the existing model and key (C-T1)', async () => {
    const cfg = baseAppConfig({
      models: [
        {
          provider: 'openai-compatible',
          base_url: 'https://old.example/v1',
          model: 'my-model',
          context: 32768,
          temperature: 0.5,
          api_key: 'ck'
        }
      ]
    });
    const writes = setup(cfg);
    render(ProvidersSection);
    await screen.findByRole('heading', { name: 'Providers' });

    await fireEvent.click(listbox().getByText('Custom (OpenAI-compatible)').closest('button')!);

    const baseInput = await screen.findByLabelText('Base URL');
    await typeInto(baseInput, 'https://new.example/v1');
    await fireEvent.blur(baseInput);

    await waitFor(() => expect(writes.length).toBeGreaterThan(0));
    const entry = writes.at(-1)!.models.find((m) => m.provider === 'openai-compatible')!;
    expect(entry.base_url).toBe('https://new.example/v1');
    expect(entry.model).toBe('my-model');
    // Untouched masked key stays saved.
    expect(entry.api_key).toBe('ck');
  });

  // C-T2a: a decoupled credential save for a brand-new provider writes model:''.
  it('saves a credential-only entry (model:"") for a new provider (C-T2)', async () => {
    const writes = setup(baseAppConfig());
    render(ProvidersSection);
    await screen.findByRole('heading', { name: 'Providers' });

    await fireEvent.click(listbox().getByText('Anthropic').closest('button')!);

    const keyInput = await screen.findByLabelText('API Key');
    await fireEvent.focus(keyInput);
    await typeInto(keyInput, 'ant-key');
    await fireEvent.blur(keyInput);

    await waitFor(() => expect(writes.length).toBeGreaterThan(0));
    const entry = writes.at(-1)!.models.find((m) => m.provider === 'anthropic')!;
    expect(entry.api_key).toBe('ant-key');
    // Credential-only: no model chosen yet.
    expect(entry.model).toBe('');
  });

  // C-T2b: key-wipe — focus+blur without typing must NOT persist an empty key.
  it('does not wipe a saved key on focus-then-blur without typing (C-T2)', async () => {
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
      ]
    });
    const writes = setup(cfg, {
      catalog: { openai: providerEntry('openai', 'OpenAI', ['gpt-4o']) }
    });
    render(ProvidersSection);
    await screen.findByRole('heading', { name: 'Providers' });

    await fireEvent.click(listbox().getByText('OpenAI').closest('button')!);

    const keyInput = await screen.findByLabelText('API Key');
    await fireEvent.focus(keyInput);
    await fireEvent.blur(keyInput);

    // Allow any (mis)write to settle, then assert the key was never emptied.
    await new Promise((r) => setTimeout(r, 20));
    expect(
      writes.some((w) => w.models.some((m) => m.provider === 'openai' && m.api_key === ''))
    ).toBe(false);
  });

  // C-T3: counts render as "N models" and degrade to "—".
  it('renders text-capable model counts and degrades to "—" (C-T3)', async () => {
    setup(baseAppConfig(), {
      catalog: { openai: providerEntry('openai', 'OpenAI', ['a', 'b', 'c']) }
    });
    render(ProvidersSection);
    await screen.findByRole('heading', { name: 'Providers' });

    const openaiRow = () => listbox().getByText('OpenAI').closest('button')!;
    await waitFor(() => expect(within(openaiRow()).getByText('3 models')).toBeInTheDocument());

    // A provider absent from the catalog degrades to "—".
    const cohereRow = listbox().getByText('Cohere').closest('button')!;
    expect(within(cohereRow).getByText('—')).toBeInTheDocument();
    // The catalog-less custom endpoint is always "—".
    const customRow = listbox().getByText('Custom (OpenAI-compatible)').closest('button')!;
    expect(within(customRow).getByText('—')).toBeInTheDocument();
  });

  it('degrades all cloud counts to "—" when the catalog rejects (C-T3)', async () => {
    setup(baseAppConfig(), { catalog: 'reject' });
    render(ProvidersSection);
    await screen.findByRole('heading', { name: 'Providers' });

    const openaiRow = listbox().getByText('OpenAI').closest('button')!;
    await waitFor(() => expect(within(openaiRow).getByText('—')).toBeInTheDocument());
  });

  // FIX 1: a key saved in Providers surfaces as a usable provider in Active model within
  // the same pane open — both sections derive from one reactive config store, no remount.
  it('propagates a newly-saved key to Active model in-session', async () => {
    const writes = setup(baseAppConfig(), {
      catalog: { anthropic: providerEntry('anthropic', 'Anthropic', ['claude-sonnet-4-5']) }
    });
    render(ProvidersSection);
    render(ActiveModelSection);

    await screen.findByRole('heading', { name: 'Providers' });
    await screen.findByRole('heading', { name: 'Active model' });

    // Active model starts with no usable providers → the setup prompt, no Provider control.
    await screen.findByText(/Set up a provider under Providers first/);
    expect(screen.queryByLabelText('Provider')).not.toBeInTheDocument();

    // Save an Anthropic key under Providers.
    await fireEvent.click(listbox().getByText('Anthropic').closest('button')!);
    const keyInput = await screen.findByLabelText('API Key');
    await fireEvent.focus(keyInput);
    await typeInto(keyInput, 'ant-key');
    await fireEvent.blur(keyInput);

    await waitFor(() => expect(writes.length).toBeGreaterThan(0));

    // Active model now offers a Provider dropdown without any remount.
    await waitFor(() => expect(screen.getByLabelText('Provider')).toBeInTheDocument());
  });

  // The "Provider docs" control opens the doc URL in the system browser via the
  // opener plugin (a webview `<a href>` won't) — it is a button, not a link.
  it('opens the provider docs URL in the system browser', async () => {
    const docHref = 'https://platform.openai.com/docs';
    setup(baseAppConfig(), {
      catalog: { openai: providerEntry('openai', 'OpenAI', ['gpt-4o'], { doc: docHref }) }
    });
    render(ProvidersSection);
    await screen.findByRole('heading', { name: 'Providers' });

    await fireEvent.click(listbox().getByText('OpenAI').closest('button')!);

    const docsButton = await screen.findByRole('button', { name: 'Provider docs' });
    await fireEvent.click(docsButton);

    await waitFor(() => expect(openUrl).toHaveBeenCalledWith(docHref));
  });
});
