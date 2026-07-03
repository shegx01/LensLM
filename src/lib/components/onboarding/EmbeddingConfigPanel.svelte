<script lang="ts">
  // Thin onboarding shell around EmbeddingsSection. Owns only the re-embed
  // warning banner and the oncheck/oncollapse callbacks; delegates everything
  // else to the shared component to avoid drift.

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
  <div
    class="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/15 px-3 py-2.5"
  >
    <TriangleAlert class="mt-0.5 size-4 shrink-0 text-amber-500" />
    <p class="text-[0.78rem] leading-relaxed text-amber-500">
      {REEMBED_WARNING}
    </p>
  </div>

  <EmbeddingsSection mode="global" compact showHeader={false} onchange={onSet} />
</div>
