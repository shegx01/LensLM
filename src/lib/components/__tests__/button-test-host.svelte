<script lang="ts">
  // Test-only host: lets vitest render <Button>text</Button> with real snippet
  // children (passing `children` as a plain function does not render text).
  import { Button } from '$lib/components/ui/button/index.js';

  let { label = 'Button', onclick }: { label?: string; onclick?: () => void } = $props();
</script>

<Button {onclick}>{label}</Button>
