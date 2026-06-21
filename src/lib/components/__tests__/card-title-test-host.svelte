<script lang="ts">
  // Test-only host: renders <CardTitle> so vitest can inspect its classes.
  import { CardTitle } from '$lib/components/ui/card/index.js';

  let { label = 'Card Title' }: { label?: string } = $props();
</script>

<CardTitle>{label}</CardTitle>
