<!-- Right-aligned accent bubble for the user's question (AC5). -->
<script lang="ts">
  interface Props {
    content: string;
  }

  let { content }: Props = $props();
</script>

<div class="flex justify-end px-4 pt-4">
  <div
    class="max-w-[75%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm text-primary-foreground whitespace-pre-wrap break-words"
  >
    {content}
  </div>
</div>
