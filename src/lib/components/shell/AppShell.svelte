<script lang="ts">
  import Aperture from '@lucide/svelte/icons/aperture';
</script>

<!-- Full-viewport app shell on a canvas ("container wall", bg-background). The
     LEFT rail is a floating panel inset from the window edges (subtle border +
     tiny shadow for a crisp elevation); the macOS native traffic lights
     (titleBarStyle "Overlay") sit on its top row. Each region has a top drag bar
     (data-tauri-drag-region) so the window can be moved by its top edge. -->
<div class="grid h-svh w-full grid-cols-[256px_1fr_320px] bg-background">
  <!-- LEFT: floating sidebar panel — equal gutter on all sides, rounded, border +
       tiny shadow; native traffic lights overlay the top drag row. M3 list below. -->
  <aside
    class="m-2 flex flex-col overflow-hidden rounded-xl border border-sidebar-border bg-sidebar text-sidebar-foreground shadow-sm"
  >
    <!-- Top drag row reserved for the native macOS traffic lights (positioned via
         trafficLightPosition in tauri.conf). Generous height so there is clear
         space above the lights and the brand row never feels crowded. -->
    <div data-tauri-drag-region class="h-14 shrink-0"></div>
    <!-- App brand: the "Lens" mark + name, on its own row below the traffic lights
         (matches the design + the onboarding Aperture brand mark). -->
    <div class="flex items-center gap-2 px-4 pb-3">
      <div
        class="flex size-7 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground"
      >
        <Aperture class="size-4" />
      </div>
      <span class="text-base font-semibold">Lens</span>
    </div>
    <p class="px-4 text-xs font-medium tracking-wide text-muted-foreground uppercase">Notebooks</p>
  </aside>

  <!-- CENTER: workspace on the canvas — top drag bar, then skeletal placeholder -->
  <main class="flex flex-col overflow-hidden">
    <div data-tauri-drag-region class="h-[var(--titlebar-h)] shrink-0"></div>
    <div class="flex flex-1 flex-col items-center justify-center gap-2">
      <Aperture class="size-8 text-muted-foreground/40" />
      <p class="text-sm text-muted-foreground">Your workspace</p>
      <p class="text-xs text-muted-foreground/60">Select or create a notebook to begin</p>
    </div>
  </main>

  <!-- RIGHT: sources & studio rail — flush panel with a hairline divider; M4 fills -->
  <aside class="flex flex-col overflow-hidden border-l border-border bg-card text-card-foreground">
    <div data-tauri-drag-region class="flex h-[var(--titlebar-h)] items-center px-4">
      <span class="text-xs font-medium tracking-wide text-muted-foreground uppercase"
        >Sources &amp; Studio</span
      >
    </div>
  </aside>
</div>
