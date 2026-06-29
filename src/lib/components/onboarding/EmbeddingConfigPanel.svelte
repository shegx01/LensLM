<script lang="ts">
  // Onboarding embedding-config panel (plan Step 9). The inline expansion under
  // the system-check "Embedding model" row.
  //
  // This is now a THIN onboarding SHELL around the shared
  // `<EmbeddingsSection mode="global">` component — it owns ONLY the
  // onboarding-specific chrome (the re-embed warning banner + the expand/collapse
  // + oncheck/oncollapse callbacks) and delegates the provider selector, model
  // list, fastembed cache detection / Install, and Ollama detect-only flow to the
  // shared section. This kills the ~200-line fork that previously drifted from
  // `EmbeddingsSection` (e.g. the re-embed warning was hardcoded inline instead of
  // using the shared `REEMBED_WARNING` constant).
  //
  // Tokens only — light + dark + every accent ([[theming-light-dark-accent]]).

  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import EmbeddingsSection from '$lib/components/embeddings/EmbeddingsSection.svelte';
  import { REEMBED_WARNING } from '$lib/embeddings/models.js';

  let {
    oncheck,
    oncollapse
  }: {
    oncheck: () => Promise<void>;
    oncollapse: () => void;
  } = $props();

  // After the shared section persists the global default, re-run the onboarding
  // system-check (so the "Embedding model" row flips to Pass) and collapse.
  async function onSet(): Promise<void> {
    await oncheck();
    oncollapse();
  }
</script>

<div class="flex flex-col gap-3 pt-3">
  <!-- Re-embed warning banner (shared REEMBED_WARNING — single source of truth) -->
  <div
    class="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/15 px-3 py-2.5"
  >
    <TriangleAlert class="mt-0.5 size-4 shrink-0 text-amber-500" />
    <p class="text-[0.78rem] leading-relaxed text-amber-500">
      {REEMBED_WARNING}
    </p>
  </div>

  <!-- Delegated provider selector + model list + install / detect logic -->
  <EmbeddingsSection mode="global" compact showHeader={false} onchange={onSet} />
</div>
