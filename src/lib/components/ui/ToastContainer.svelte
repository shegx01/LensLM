<script lang="ts">
  import { toastStore, dismissToast } from '$lib/sources/toast.svelte.js';

  let toasts = $derived(toastStore.toasts);
</script>

<!-- Fixed overlay, bottom-right. Mount outside the booting/onboarding conditional
     so toasts are visible on every surface. z-[100] clears the modal overlay (z-50). -->
<div
  class="pointer-events-none fixed right-4 bottom-4 z-[100] flex flex-col items-end gap-2"
  role="status"
  aria-live="polite"
  aria-label="Notifications"
>
  {#each toasts as toast (toast.id)}
    <div
      class="pointer-events-auto flex max-w-sm items-start gap-3 rounded-lg border border-border bg-popover px-4 py-3 text-sm text-popover-foreground shadow-lg"
    >
      <span class="flex-1 leading-snug">{toast.message}</span>
      <button
        type="button"
        class="mt-0.5 shrink-0 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
        aria-label="Dismiss notification"
        onclick={() => dismissToast(toast.id)}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>
    </div>
  {/each}
</div>
