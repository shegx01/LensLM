<!--
  ApiKeyField — masked, key-wipe-safe API-key input shared by the Settings AI Model
  panel and (issue #217) onboarding's curated 3-preset picker. A saved key is shown
  masked; the field only becomes editable on focus/input, and the caller must NOT
  persist the (empty) editing value as the key while `editing` is false — it should
  reconstruct the prior key from config instead. Reused as-is inside each preset card.
-->
<script lang="ts">
  import { Input } from '$lib/components/ui/input/index.js';

  let {
    value = $bindable(''),
    editing = $bindable(false),
    hasSavedKey = false,
    id,
    label = 'API Key',
    oncommit
  }: {
    /** The key the user is typing. Empty while a saved key stays masked. */
    value?: string;
    /** True once the user starts replacing a saved key. */
    editing?: boolean;
    /** A key is already persisted for the selected provider. */
    hasSavedKey?: boolean;
    id: string;
    label?: string;
    /** Fires on blur so the caller can persist the (masked-safe) key. */
    oncommit?: () => void;
  } = $props();

  const masked = $derived(hasSavedKey && !editing);

  // First focus/keystroke on a masked field clears it and switches to edit mode so
  // the placeholder never leaks into the persisted value.
  function startEditing(): void {
    if (masked) {
      editing = true;
      value = '';
    }
  }
</script>

<div class="flex flex-col gap-1.5">
  <label for={id} class="text-[0.68rem] font-medium text-muted-foreground">{label}</label>
  <Input
    {id}
    type="password"
    bind:value
    placeholder={masked ? '•••••••••• saved — click to replace' : 'Paste API key…'}
    autocomplete="new-password"
    onfocus={startEditing}
    oninput={startEditing}
    onblur={() => oncommit?.()}
  />
  {#if masked}
    <p class="text-[0.72rem] leading-relaxed text-muted-foreground">
      A key is already saved. Click the field to replace it.
    </p>
  {/if}
</div>
