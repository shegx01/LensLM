<script lang="ts">
  // Test-only host: renders <DialogTitle> inside a Dialog.Root context so
  // vitest can inspect its classes without a "Context not found" error.
  import { Dialog, DialogTitle } from '$lib/components/ui/dialog/index.js';

  let { label = 'Dialog Title' }: { label?: string } = $props();
</script>

<Dialog>
  <DialogTitle>{label}</DialogTitle>
</Dialog>
