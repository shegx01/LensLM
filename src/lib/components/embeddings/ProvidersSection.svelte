<!--
  ProvidersSection — the "Providers" half of the AI Model panel. A master-detail
  surface: the left rail lists every provider (local Ollama + the cloud catalog) with a
  text-capable model count and a key-set/reachable indicator; the right panel edits the
  SELECTED provider's credentials only (base URL / API key) — never a model, temperature,
  or context. Credentials persist reactively via saveProviderCredential (no Save button),
  decoupled from the model pin so the two halves never clobber each other's fields.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { invoke, isTauri } from '@tauri-apps/api/core';
  import { openUrl } from '@tauri-apps/plugin-opener';
  import type { ModelConfig } from '$lib/theme/types.js';
  import type { ProviderEntry } from '$lib/models/types.js';
  import { providerModelCounts } from '$lib/models/catalog.js';
  import { saveProviderCredential } from '$lib/onboarding/llm-config.js';
  import { refreshChatProvider } from '$lib/models/chat-provider.svelte.js';
  import { refreshActiveModel } from '$lib/models/active-model.svelte.js';
  import { appConfigStore, refreshConfig } from '$lib/models/app-config.svelte.js';
  import {
    providerDescriptors,
    isReachable,
    isUsable,
    LOCAL_DEFAULT_ENDPOINT,
    type ProviderDescriptor
  } from '$lib/models/providers.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import ProviderLogo from './ProviderLogo.svelte';
  import ApiKeyField from '$lib/components/llm/ApiKeyField.svelte';
  import ExternalLink from '@lucide/svelte/icons/external-link';

  const PROVIDERS = providerDescriptors();

  let selectedId = $state('ollama');
  let counts = $state<Record<string, number>>({});
  let ollamaCount = $state<number | null>(null);
  let catalog = $state<Record<string, ProviderEntry>>({});
  let saveError = $state<string | null>(null);

  // Detail-panel editing buffers for the selected provider.
  let baseUrl = $state(LOCAL_DEFAULT_ENDPOINT);
  let apiKeyValue = $state('');
  let editingKey = $state(false);

  const selected = $derived(PROVIDERS.find((p) => p.id === selectedId) ?? PROVIDERS[0]);
  const selectedEntry = $derived(entryFor(selectedId));
  const hasSavedKey = $derived((selectedEntry?.api_key.trim() ?? '') !== '');
  const selectedCatalog = $derived(catalog[selected.catalogKey ?? selected.id]);
  const docUrl = $derived(selectedCatalog?.doc ?? null);
  const envVar = $derived(selectedCatalog?.env?.[0] ?? null);

  function entryFor(id: string): ModelConfig | undefined {
    return appConfigStore.models.find((m) => m.provider === id);
  }

  // Open the provider's docs in the system browser; a webview `<a href>` won't.
  // Best-effort — a no-op outside Tauri and swallow errors so it never throws into the UI.
  async function openProviderDocs(): Promise<void> {
    if (!isTauri() || docUrl == null) return;
    try {
      await openUrl(docUrl);
    } catch {
      // Opening the external browser is non-critical; leave the UI untouched.
    }
  }

  /** Count label for a row: text-capable model count, or `—` when unknown. */
  function countLabel(row: ProviderDescriptor): string {
    if (row.id === 'ollama') return ollamaCount == null ? '—' : `${ollamaCount} models`;
    if (row.catalogKey == null) return '—';
    const c = counts[row.catalogKey];
    return c == null ? '—' : `${c} models`;
  }

  function statusTag(row: ProviderDescriptor): string | null {
    if (row.kind === 'local') return isReachable(ollamaCount) ? 'Reachable' : null;
    return (entryFor(row.id)?.api_key.trim() ?? '') !== '' ? 'Key set' : null;
  }

  function syncBuffers(): void {
    apiKeyValue = '';
    editingKey = false;
    baseUrl = entryFor(selectedId)?.base_url || selected.baseUrl;
  }

  function selectProvider(id: string): void {
    selectedId = id;
    syncBuffers();
  }

  async function loadOllama(): Promise<void> {
    const base = entryFor('ollama')?.base_url || LOCAL_DEFAULT_ENDPOINT;
    try {
      const ids = await invoke<string[]>('list_ollama_models', { base_url: base });
      ollamaCount = ids.length;
    } catch {
      ollamaCount = null;
    }
  }

  async function loadCatalog(): Promise<void> {
    try {
      catalog = await invoke<Record<string, ProviderEntry>>('list_models');
    } catch {
      catalog = {};
    }
  }

  onMount(async () => {
    if (!isTauri()) return;
    await refreshConfig();
    // Seed the detail buffers from config before the slower catalog/Ollama fetches, so a
    // late resolve can't clobber a credential the user is already editing.
    syncBuffers();
    counts = await providerModelCounts();
    await Promise.all([loadCatalog(), loadOllama()]);
  });

  async function persistCredential(): Promise<void> {
    if (!isTauri()) return;
    saveError = null;
    const row = selected;
    const existing = entryFor(row.id);
    // Focusing a masked key flips editingKey before any keystroke; an empty buffer still
    // means "keep the saved key", so guard on the buffer too (never wipe an existing key).
    const keyMasked =
      row.kind !== 'local' && hasSavedKey && (!editingKey || apiKeyValue.trim() === '');
    const api_key = row.kind === 'local' ? '' : keyMasked ? (existing?.api_key ?? '') : apiKeyValue;
    const base_url =
      row.kind === 'local' || row.kind === 'custom' ? baseUrl : (existing?.base_url ?? '');

    try {
      await saveProviderCredential({ provider: row.id, base_url, api_key });
      await refreshConfig();
      if (row.kind !== 'local') {
        editingKey = false;
        apiKeyValue = '';
      }
      await refreshChatProvider();
      await refreshActiveModel();
    } catch (err) {
      saveError = err instanceof Error ? err.message : 'Could not save provider credentials.';
    }
  }
</script>

<section class="flex flex-col" aria-label="Providers settings">
  <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Providers</h2>
  <p class="mt-1 text-[0.8rem] text-muted-foreground">
    Connect the model providers you want to use. Set a key here, then pick a model under Active
    model.
  </p>

  <div class="mt-6 grid grid-cols-1 items-start gap-3.5 md:grid-cols-[minmax(200px,0.85fr)_1.15fr]">
    <div
      class="no-scrollbar flex max-h-[360px] flex-col gap-1.5 overflow-y-auto"
      role="listbox"
      aria-label="Providers"
    >
      {#each PROVIDERS as row (row.id)}
        {@const isSel = row.id === selectedId}
        {@const tag = statusTag(row)}
        <button
          type="button"
          role="option"
          aria-selected={isSel}
          onclick={() => selectProvider(row.id)}
          class="flex w-full items-center gap-2.5 rounded-[10px] border px-3 py-2.5 text-left transition-[background-color,border-color,transform] duration-150 active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring {isSel
            ? 'border-primary/40 bg-primary/10'
            : 'border-transparent hover:bg-muted'}"
        >
          <ProviderLogo id={row.id} name={row.name} size={26} />
          <span class="min-w-0 flex-1">
            <span class="flex items-center gap-2 text-[0.8rem] font-bold text-foreground">
              <span
                class="size-[7px] shrink-0 rounded-full {isUsable(
                  row,
                  entryFor(row.id),
                  ollamaCount
                )
                  ? 'bg-primary'
                  : 'bg-muted-foreground/50'}"
                aria-hidden="true"
              ></span>
              <span class="truncate">{row.name}</span>
              {#if tag}
                <span
                  class="shrink-0 rounded-full bg-primary/15 px-1.5 py-px text-[0.58rem] font-bold uppercase tracking-[0.05em] text-primary"
                  >{tag}</span
                >
              {/if}
            </span>
            <span class="mt-px block text-[0.68rem] tabular-nums text-muted-foreground"
              >{countLabel(row)}</span
            >
          </span>
        </button>
      {/each}
    </div>

    <div class="rounded-xl border border-border bg-card p-[18px]">
      <div class="flex items-center gap-3">
        <ProviderLogo id={selected.id} name={selected.name} size={34} />
        <div class="min-w-0">
          <div class="truncate text-[0.95rem] font-extrabold text-foreground">{selected.name}</div>
          <div class="text-[0.7rem] tabular-nums text-muted-foreground">{countLabel(selected)}</div>
        </div>
      </div>

      {#if docUrl}
        <button
          type="button"
          onclick={() => void openProviderDocs()}
          class="mt-2 inline-flex items-center gap-1.5 text-[0.66rem] font-bold text-primary hover:underline"
        >
          Provider docs
          <ExternalLink class="size-3" aria-hidden="true" />
        </button>
      {/if}

      {#if selected.kind === 'local' || selected.kind === 'custom'}
        <div class="mt-4 flex flex-col gap-1.5">
          <label for="provider-base-url" class="text-[0.68rem] font-medium text-muted-foreground">
            Base URL
          </label>
          <Input
            id="provider-base-url"
            type="url"
            bind:value={baseUrl}
            onblur={() => void persistCredential()}
            placeholder={selected.kind === 'local'
              ? LOCAL_DEFAULT_ENDPOINT
              : 'https://api.openai.com/v1'}
            autocomplete="off"
            spellcheck={false}
          />
        </div>
      {/if}

      {#if selected.kind === 'cloud' || selected.kind === 'custom'}
        <div class="mt-4 flex flex-col gap-1.5">
          <ApiKeyField
            id="provider-api-key"
            bind:value={apiKeyValue}
            bind:editing={editingKey}
            {hasSavedKey}
            oncommit={() => void persistCredential()}
          />
          {#if envVar}
            <p class="text-[0.64rem] text-muted-foreground">
              Falls back to the <code class="rounded bg-muted px-1 py-px font-mono text-[0.62rem]"
                >{envVar}</code
              > environment variable if left blank.
            </p>
          {/if}
        </div>
      {/if}

      {#if selected.kind === 'local'}
        <p class="mt-3.5 rounded-[9px] bg-muted px-3 py-2.5 text-[0.72rem] text-muted-foreground">
          Ollama runs on your machine — no API key needed. Models you pull locally appear under
          Active model.
        </p>
      {/if}

      {#if saveError}
        <p class="mt-3 text-[0.75rem] text-destructive" role="alert">{saveError}</p>
      {/if}
    </div>
  </div>
</section>
