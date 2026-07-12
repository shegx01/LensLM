<!-- Right-aligned neutral bubble for the user's question (AC5). -->
<script lang="ts">
  interface Props {
    content: string;
  }

  let { content }: Props = $props();
</script>

<div class="flex justify-end px-4 pt-4">
  <div
    class="max-w-[75%] rounded-2xl rounded-br-md bg-muted px-4 py-2.5 text-sm text-foreground whitespace-pre-wrap break-words"
  >
    {content}
  </div>
</div>
