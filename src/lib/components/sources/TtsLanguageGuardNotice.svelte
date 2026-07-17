<!-- Reusable inline-reason notice for the TTS engine-aware language guard (#194).
     Display-only: renders a GuardVerdict's block reason + offending sources/languages.
     NOT mounted on the synthesis button here — #28/#161 wires that. -->
<script lang="ts">
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';

  /** Mirrors lens-core `tts::catalog::OffendingSource` (serde field names). */
  export interface OffendingSource {
    source_id: string;
    language: string;
  }

  /** Mirrors lens-core `tts::catalog::GuardVerdict` (serde field names). */
  export interface GuardVerdict {
    allow: boolean;
    reason: string | null;
    offending: OffendingSource[];
  }

  let { verdict }: { verdict: GuardVerdict } = $props();

  /** `"german"` -> `"German"` for display; catalog languages serialize lowercase. */
  function displayLang(lang: string): string {
    return lang.length > 0 ? lang.charAt(0).toUpperCase() + lang.slice(1) : lang;
  }
</script>

{#if !verdict.allow}
  <div
    class="flex items-start gap-2 rounded-lg border border-destructive/25 bg-destructive/5 px-3 py-2.5"
    role="alert"
  >
    <TriangleAlert class="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden="true" />
    <div class="min-w-0">
      {#if verdict.reason}
        <p class="text-xs font-semibold text-destructive">{verdict.reason}</p>
      {/if}
      {#if verdict.offending.length > 0}
        <ul class="mt-1 flex flex-col gap-0.5 text-xs text-muted-foreground">
          {#each verdict.offending as offending (offending.source_id)}
            <li>{offending.source_id} — {displayLang(offending.language)}</li>
          {/each}
        </ul>
      {/if}
    </div>
  </div>
{/if}
