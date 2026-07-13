<!-- Clamps rendered note HTML to `lineClamp` lines, with a Show more/less toggle
     that removes the clamp. The clamp is pure CSS (-webkit-line-clamp), so the
     full sanitized HTML is always in the DOM — expand/collapse never re-renders
     model content. The toggle only appears when the content actually overflows,
     measured after mount and on resize. -->
<script lang="ts">
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ChevronUp from '@lucide/svelte/icons/chevron-up';

  interface Props {
    /** Pre-sanitized HTML (DOMPurify output) — rendered via `{@html}`. */
    html: string;
    lineClamp?: number;
    class?: string;
    /** Post-render hook over the body element (e.g. mermaid hydration). Runs after
     *  each render; must be idempotent/restore-safe. */
    onrender?: (body: HTMLElement) => void;
  }

  let { html, lineClamp = 4, class: className = '', onrender }: Props = $props();

  let expanded = $state(false);
  let overflows = $state(false);
  let bodyRef = $state<HTMLElement | null>(null);

  function measure(): void {
    const el = bodyRef;
    if (!el) return;
    // scrollHeight exceeds clientHeight only while the clamp is truncating.
    overflows = el.scrollHeight - el.clientHeight > 1;
  }

  // Run the post-render hook (e.g. mermaid hydration) after each {@html} render.
  $effect(() => {
    void html;
    if (bodyRef && onrender) onrender(bodyRef);
  });

  // Re-measure when content or the (collapsed) layout changes. While expanded we
  // skip measuring — the toggle stays visible so the user can collapse again.
  $effect(() => {
    void html;
    if (expanded) return;
    measure();
    if (typeof ResizeObserver === 'undefined' || !bodyRef) return;
    const ro = new ResizeObserver(measure);
    ro.observe(bodyRef);
    return () => ro.disconnect();
  });
</script>

<div
  bind:this={bodyRef}
  class="note-markdown text-sm leading-relaxed text-foreground {className}"
  class:clamped={!expanded}
  style:--note-line-clamp={lineClamp}
>
  <!-- eslint-disable-next-line svelte/no-at-html-tags -->
  {@html html}
</div>

{#if overflows || expanded}
  <button
    type="button"
    class="mt-1.5 inline-flex items-center gap-1 text-xs font-medium text-primary transition-opacity hover:opacity-80"
    aria-expanded={expanded}
    onclick={() => (expanded = !expanded)}
  >
    {#if expanded}
      Show less
      <ChevronUp class="size-3.5" strokeWidth={2} />
    {:else}
      Show more
      <ChevronDown class="size-3.5" strokeWidth={2} />
    {/if}
  </button>
{/if}

<style>
  .clamped {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: var(--note-line-clamp);
    line-clamp: var(--note-line-clamp);
    overflow: hidden;
  }
  :global(.note-markdown p) {
    margin: 0 0 0.5em;
  }
  :global(.note-markdown p:last-child) {
    margin-bottom: 0;
  }
  :global(.note-markdown ul),
  :global(.note-markdown ol) {
    margin: 0.25em 0 0.5em;
    padding-left: 1.25em;
  }
  :global(.note-markdown li) {
    margin: 0.15em 0;
  }
  :global(.note-markdown code) {
    background: var(--muted);
    border-radius: 0.25rem;
    padding: 0.1em 0.35em;
    font-size: 0.85em;
  }
  :global(.note-markdown pre) {
    background: var(--muted);
    border-radius: 0.5rem;
    padding: 0.6em 0.8em;
    overflow-x: auto;
    margin: 0.5em 0;
  }
  :global(.note-markdown pre code) {
    background: none;
    padding: 0;
  }
  :global(.note-markdown a) {
    color: var(--primary);
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  :global(.note-markdown strong) {
    font-weight: 600;
  }
</style>
