<script lang="ts">
  import { onMount } from 'svelte';
  import LoaderCircle from '@lucide/svelte/icons/loader-circle';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import Sun from '@lucide/svelte/icons/sun';
  import Moon from '@lucide/svelte/icons/moon';
  import Monitor from '@lucide/svelte/icons/monitor';
  import {
    Card,
    CardHeader,
    CardTitle,
    CardDescription,
    CardContent,
    CardFooter
  } from '$lib/components/ui/card/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import SystemCheckRow from '$lib/components/onboarding/SystemCheckRow.svelte';
  import {
    runSystemCheck,
    type CheckResult,
    type CheckAction
  } from '$lib/onboarding/system-check.js';
  import { completeOnboarding } from '$lib/onboarding/completeOnboarding.js';
  import { setMode, userPrefersMode } from 'mode-watcher';
  import { persistTheme, type Mode } from '$lib/theme/index.js';

  // The first-run system-check screen. State-based: the layout renders this
  // (instead of the app) while onboarding is incomplete, and flips to the app
  // when `oncomplete` fires AFTER the flag is durably persisted. No navigation.
  let { oncomplete }: { oncomplete: () => void } = $props();

  let results = $state<CheckResult[]>([]);
  let loading = $state(true);
  let finishing = $state(false);
  // Inline error surfaces (no silent failure, no empty card with a live
  // Continue): `checkError` when the probes themselves failed to run;
  // `continueError` when persisting onboarding_complete failed.
  let checkError = $state<string | null>(null);
  let continueError = $state<string | null>(null);

  // Continue is blocked ONLY when LocalBackend or DiskPermissions FAIL (plan
  // change #12 / ADR decision #6). LLM `fail` warns-but-allows; any `pending`
  // (embedding/vector) never blocks. A failed probe RUN also blocks Continue.
  const BLOCKING_IDS = ['local_backend', 'disk_permissions'] as const;
  const blocked = $derived(
    checkError !== null ||
      results.some((r) => (BLOCKING_IDS as readonly string[]).includes(r.id) && r.status === 'fail')
  );

  // Footer summary, verbatim per the design: "{readyCount} of {totalCount} checks passed".
  const readyCount = $derived(results.filter((r) => r.status === 'pass').length);
  const totalCount = $derived(results.length);

  async function check(): Promise<void> {
    loading = true;
    checkError = null;
    try {
      results = await runSystemCheck();
    } catch (err) {
      // The probes failed to run at all: surface it inline and keep Continue
      // blocked rather than presenting an empty card with a live button.
      console.error('SystemCheck: runSystemCheck failed', err);
      results = [];
      checkError = 'Could not run the system check. Please retry.';
    } finally {
      loading = false;
    }
  }

  async function handleContinue(): Promise<void> {
    finishing = true;
    continueError = null;
    try {
      // Persist the flag FIRST (RMW). Only on success do we hand control back to
      // the layout, which flips to the app. On failure we stay on this screen.
      await completeOnboarding();
      oncomplete();
    } catch (err) {
      console.error('SystemCheck: completeOnboarding failed', err);
      continueError = 'Could not save your setup. Please try again.';
    } finally {
      finishing = false;
    }
  }

  // SystemCheckRow action affordances. `retry` re-runs the whole check. The
  // `configure`/`choose` actions target Settings, which is not built until a
  // later milestone — they are rendered disabled (see SystemCheckRow) so we
  // never ship a dead, silently-no-op button.
  function handleAction(action: CheckAction): void {
    if (action === 'retry') void check();
  }

  // Compact onboarding-local theme toggle: cycles light → dark → system.
  // Keeps the shared ThemeSwitcher on /showcase untouched.
  const CYCLE: Mode[] = ['light', 'dark', 'system'];
  const CYCLE_ICON = { light: Sun, dark: Moon, system: Monitor } as const;
  const CYCLE_LABEL = { light: 'Light', dark: 'Dark', system: 'System' } as const;

  const currentMode = $derived(userPrefersMode.current ?? 'system');
  const NextIcon = $derived(CYCLE_ICON[currentMode]);
  const nextLabel = $derived(CYCLE_LABEL[currentMode]);

  function cycleTheme(): void {
    const idx = CYCLE.indexOf(currentMode);
    const next = CYCLE[(idx + 1) % CYCLE.length];
    setMode(next);
    persistTheme(next);
  }

  onMount(() => {
    void check();
  });
</script>

<div class="absolute top-4 right-4 z-10">
  <Button
    variant="outline"
    size="icon"
    aria-label={`Theme: ${nextLabel}`}
    onclick={cycleTheme}
    class="size-9 rounded-lg"
  >
    <NextIcon class="size-4" />
  </Button>
</div>

<main class="flex min-h-svh items-center justify-center p-4">
  <Card class="w-full max-w-lg">
    <CardHeader class="items-center text-center">
      <div
        class="bg-primary mx-auto mb-2 flex size-12 items-center justify-center rounded-2xl text-primary-foreground"
        aria-hidden="true"
      >
        <ScanSearch class="size-6" />
      </div>
      <CardTitle class="text-2xl">System check</CardTitle>
      <CardDescription>Verifying your local intelligence engine before launch</CardDescription>
    </CardHeader>

    <CardContent class="flex flex-col gap-3">
      {#if loading}
        <div
          class="text-muted-foreground flex items-center justify-center gap-2 py-12 text-sm"
          aria-live="polite"
        >
          <LoaderCircle class="size-4 animate-spin" />
          Checking your system…
        </div>
      {:else if checkError}
        <div
          class="text-destructive flex items-center justify-center gap-2 py-12 text-center text-sm"
          role="alert"
        >
          <TriangleAlert class="size-4 shrink-0" />
          {checkError}
        </div>
      {:else}
        {#each results as result (result.id)}
          <SystemCheckRow {result} onaction={handleAction} oncheck={check} />
        {/each}
      {/if}
    </CardContent>

    <CardFooter class="flex flex-col gap-3 pt-2">
      {#if continueError}
        <p class="text-destructive w-full text-center text-sm" role="alert">{continueError}</p>
      {/if}
      {#if !loading && !checkError}
        <p class="text-muted-foreground w-full text-center text-[11px]">
          {readyCount} of {totalCount} checks passed
        </p>
      {/if}
      <Button
        class="h-11 w-full"
        onclick={handleContinue}
        disabled={loading || finishing || blocked}
      >
        Continue to setup
        <ArrowRight />
      </Button>
    </CardFooter>
  </Card>
</main>
