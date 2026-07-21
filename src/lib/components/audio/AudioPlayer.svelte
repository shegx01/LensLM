<!-- Minimal Audio Overview player (#29): visible play/pause + draggable seek + time.
     Speed and ±15s skip are keyboard-only, scoped to this player's focus so they never
     collide with the app-wide shortcut map (#35) — a visible hint + aria-live
     announcements make the keyboard-only functions discoverable and accessible.

     Plays from a Blob objectURL rather than the asset: src directly: WKWebView's
     AVFoundation media loader deadlocks the whole app on pause/seek over the custom
     asset scheme (tauri-apps/tauri#10426). Fetching the (small) WAV into memory and
     playing a blob: URL bypasses that broken media path. -->
<script lang="ts">
  import { convertFileSrc } from '@tauri-apps/api/core';
  import Play from '@lucide/svelte/icons/play';
  import Pause from '@lucide/svelte/icons/pause';

  let { path }: { path: string } = $props();

  const RATES = [0.75, 1, 1.25, 1.5, 1.75, 2] as const;
  const DEFAULT_RATE_INDEX = 1;
  const SEEK_STEP_S = 5;
  const SKIP_STEP_S = 15;

  let audioEl = $state<HTMLAudioElement | undefined>(undefined);
  let objectUrl = $state<string | null>(null);
  let loadError = $state(false);
  let playing = $state(false);
  let currentTime = $state(0);
  let duration = $state(0);
  let rateIndex = $state(DEFAULT_RATE_INDEX);
  /** Visually-hidden live-region text — the only feedback for keyboard-only skip/speed. */
  let announcement = $state('');

  const rate = $derived(RATES[rateIndex]);
  const maxDuration = $derived(Number.isFinite(duration) ? duration : 0);

  $effect(() => {
    const src = convertFileSrc(path);
    let cancelled = false;
    let created: string | null = null;
    loadError = false;
    playing = false;
    currentTime = 0;
    (async () => {
      try {
        const res = await fetch(src);
        if (!res.ok) throw new Error(String(res.status));
        const blob = await res.blob();
        if (cancelled) return;
        created = URL.createObjectURL(blob);
        objectUrl = created;
      } catch {
        if (!cancelled) loadError = true;
      }
    })();
    return () => {
      cancelled = true;
      if (created) URL.revokeObjectURL(created);
      objectUrl = null;
    };
  });

  function clamp(v: number, lo: number, hi: number): number {
    return Math.min(hi, Math.max(lo, v));
  }

  function formatTime(seconds: number): string {
    if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  function togglePlay(): void {
    if (!audioEl) return;
    if (playing) audioEl.pause();
    else void audioEl.play().catch(() => {});
  }

  function seekTo(value: number): void {
    if (!audioEl) return;
    const clamped = clamp(value, 0, maxDuration);
    audioEl.currentTime = clamped;
    currentTime = clamped;
  }

  function skip(delta: number): void {
    seekTo(currentTime + delta);
    announcement = `${delta > 0 ? 'Skipped forward' : 'Skipped back'} ${Math.abs(delta)} seconds`;
  }

  function cycleRate(dir: 1 | -1): void {
    if (!audioEl) return;
    rateIndex = clamp(rateIndex + dir, 0, RATES.length - 1);
    audioEl.playbackRate = rate;
    announcement = `Speed ${rate}×`;
  }

  /** Scoped to this element's focus (tabindex on the wrapper) — never a document-level listener. */
  function handleKeydown(e: KeyboardEvent): void {
    if (!objectUrl) return;
    const key = e.key.toLowerCase();
    if (key === ' ' || key === 'spacebar') {
      e.preventDefault();
      togglePlay();
    } else if (key === 'arrowright') {
      e.preventDefault();
      seekTo(currentTime + SEEK_STEP_S);
    } else if (key === 'arrowleft') {
      e.preventDefault();
      seekTo(currentTime - SEEK_STEP_S);
    } else if (key === '[') {
      e.preventDefault();
      cycleRate(-1);
    } else if (key === ']') {
      e.preventDefault();
      cycleRate(1);
    } else if (key === 'j') {
      e.preventDefault();
      skip(-SKIP_STEP_S);
    } else if (key === 'l') {
      e.preventDefault();
      skip(SKIP_STEP_S);
    }
  }
</script>

<!-- Composite media-player widget (ARIA APG pattern): the group itself is the
     focus target for the keyboard-only transport shortcuts (space/arrows/J/L/[/]),
     so tabindex + keydown on a non-interactive role are intentional here. -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<div
  class="audio-player flex flex-col gap-2 rounded-lg border border-border/70 bg-card p-2.5"
  role="group"
  aria-label="Audio overview player. Space plays or pauses. Left and right arrows seek 5 seconds. J and L skip 15 seconds. Bracket keys change playback speed."
  tabindex="0"
  onkeydown={handleKeydown}
>
  {#if loadError}
    <p class="py-2 text-center text-xs text-destructive" role="alert">
      Couldn&rsquo;t load the audio overview.
    </p>
  {:else if !objectUrl}
    <p class="py-2 text-center text-xs text-muted-foreground" aria-live="polite">Loading&hellip;</p>
  {:else}
    <!-- svelte-ignore a11y_media_has_caption -->
    <audio
      bind:this={audioEl}
      src={objectUrl}
      preload="auto"
      ontimeupdate={() => (currentTime = audioEl?.currentTime ?? 0)}
      onloadedmetadata={() => {
        duration = audioEl?.duration ?? 0;
        // A freshly-mounted <audio> defaults playbackRate to 1.0; re-apply the current
        // rate so the "N× speed" badge doesn't lie after a source swap.
        if (audioEl) audioEl.playbackRate = rate;
      }}
      onplay={() => (playing = true)}
      onpause={() => (playing = false)}
      onended={() => (playing = false)}
    ></audio>

    <div class="flex items-center gap-2.5">
      <button
        type="button"
        class="press grid size-8 shrink-0 place-items-center rounded-full bg-primary text-primary-foreground transition-[transform,opacity] hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-label={playing ? 'Pause' : 'Play'}
        onclick={togglePlay}
      >
        {#if playing}
          <Pause class="size-3.5" fill="currentColor" strokeWidth={0} />
        {:else}
          <Play class="ml-0.5 size-3.5" fill="currentColor" strokeWidth={0} />
        {/if}
      </button>

      <input
        type="range"
        class="seek-range flex-1"
        min="0"
        max={maxDuration}
        step="0.1"
        value={currentTime}
        aria-label="Seek"
        oninput={(e) => seekTo(+e.currentTarget.value)}
      />

      <span class="shrink-0 text-[0.7rem] tabular-nums text-muted-foreground">
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>
    </div>

    {#if rate !== 1}
      <span
        class="speed-badge self-start rounded-full bg-muted px-1.5 py-px text-[0.625rem] font-semibold text-muted-foreground"
      >
        {rate}&times; speed
      </span>
    {/if}

    <p class="hint text-[0.65rem] leading-relaxed text-muted-foreground/60">
      <kbd>Space</kbd> play/pause &middot; <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> seek &middot;
      <kbd>J</kbd>/<kbd>L</kbd> skip 15s &middot; <kbd>[</kbd>/<kbd>]</kbd> speed
    </p>

    <span class="sr-only" aria-live="polite">{announcement}</span>
  {/if}
</div>

<style>
  .press:active {
    transform: scale(calc(1 - 0.04 * var(--rail-motion, 1)));
  }
  .audio-player:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--ring);
  }
  .seek-range {
    height: 4px;
    accent-color: var(--primary);
  }
  kbd {
    display: inline-block;
    padding: 0 4px;
    border-radius: 4px;
    background: var(--muted);
    font-family: inherit;
    font-size: 0.95em;
  }
</style>
