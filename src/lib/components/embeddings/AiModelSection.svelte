<!--
  AiModelSection — the "AI Model" preferences pane. A thin composition of the two
  purpose-scoped sections: Providers (credentials) → Active model (the model pin).
  All fields live in the children; this shell only stacks them.
-->
<script lang="ts">
  import ProvidersSection from './ProvidersSection.svelte';
  import ActiveModelSection from './ActiveModelSection.svelte';
</script>

<div class="flex flex-col gap-10" aria-label="AI Model settings">
  <ProvidersSection />
  <ActiveModelSection />
</div>
