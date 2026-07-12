<!--
  RetrievalSection — per-notebook graph-retrieval toggle + advisory benchmark (#158b).
  The toggle is always usable; the eval verdict is advisory only and never gates it.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { Switch } from '$lib/components/ui/switch/index.js';
  import Network from '@lucide/svelte/icons/network';
  import {
    getNotebookGraphRetrievalEnabled,
    setNotebookGraphRetrievalEnabled,
    latestNotebookEval,
    runNotebookGraphEval,
    type EvalReportDto,
    type EvalPhaseDto
  } from '$lib/embeddings/ipc.js';

  let { notebookId }: { notebookId: string } = $props();

  let enabled = $state(false);
  let toggleSaving = $state(false);
  let toggleError = $state<string | null>(null);

  let report = $state<EvalReportDto | null>(null);
  let running = $state(false);
  let phase = $state<EvalPhaseDto | null>(null);
  let runError = $state<string | null>(null);
  let noProvider = $state(false);
  let skippedReason = $state<string | null>(null);

  onMount(async () => {
    try {
      enabled = await getNotebookGraphRetrievalEnabled(notebookId);
    } catch {
      // Non-fatal: default OFF.
    }
    try {
      report = await latestNotebookEval(notebookId);
    } catch {
      // Non-fatal: treat as no prior benchmark.
    }
  });

  async function handleToggle(next: boolean): Promise<void> {
    const prev = enabled;
    enabled = next;
    toggleSaving = true;
    toggleError = null;
    try {
      await setNotebookGraphRetrievalEnabled(notebookId, next);
    } catch (err) {
      toggleError = err instanceof Error ? err.message : 'Could not save setting.';
      enabled = prev;
    } finally {
      toggleSaving = false;
    }
  }

  // AC-16 pre-gate deviation: no reliable cheap client-side "chat provider configured"
  // signal exists — the engine derives a provider from enrichment.routing/models even
  // when enrichment.chat_model is null, so no-provider is surfaced only on the run reject.
  const NO_PROVIDER_MESSAGE = 'no chat model configured';

  async function runBenchmark(): Promise<void> {
    if (running) return;
    running = true;
    phase = null;
    runError = null;
    noProvider = false;
    skippedReason = null;
    try {
      const outcome = await runNotebookGraphEval(notebookId, (p) => {
        phase = p;
      });
      if (outcome.status === 'ran') {
        report = await latestNotebookEval(notebookId);
      } else {
        report = null;
        skippedReason = outcome.reason;
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'The benchmark could not run.';
      const kind = err instanceof Error ? (err as Error & { kind?: string }).kind : undefined;
      if (kind === 'Model' && message === NO_PROVIDER_MESSAGE) {
        noProvider = true;
      } else {
        runError = message;
      }
    } finally {
      running = false;
      phase = null;
    }
  }

  const pct = (v: number): string => `${Math.round(v * 100)}%`;

  const friendlyDate = (iso: string): string => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime())
      ? iso
      : d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  };

  const progressLabel = $derived(
    phase === 'generating_qa' ? 'Generating test questions…' : running ? 'Finishing…' : ''
  );
</script>

<section class="flex flex-col" aria-label="Retrieval settings">
  <h2 class="text-xl font-extrabold tracking-[-0.4px] text-foreground">Retrieval</h2>
  <p class="mt-1 text-[0.8rem] text-muted-foreground">
    Graph retrieval expands answers with entities and relationships mined from this notebook's
    sources, on top of the usual hybrid search.
  </p>

  <div class="mt-6">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Graph retrieval
    </p>

    <label
      class="mt-3 flex cursor-pointer items-center justify-between gap-4 rounded-[10px] border border-border bg-card px-4 py-3.5 transition-colors hover:border-border/80"
    >
      <span class="min-w-0 flex-1">
        <span class="block text-[0.78rem] font-bold text-foreground">Use graph retrieval</span>
        <span class="mt-0.5 block text-[0.68rem] text-muted-foreground">
          When enabled, answers in this notebook can draw on the entity graph in addition to hybrid
          vector search.
        </span>
      </span>
      <Switch
        checked={enabled}
        disabled={toggleSaving}
        aria-label="Use graph retrieval"
        onCheckedChange={handleToggle}
      />
    </label>

    {#if toggleError}
      <p class="mt-3 text-[0.75rem] text-destructive" role="alert">{toggleError}</p>
    {/if}
  </div>

  <div class="mt-8">
    <p class="text-[0.65rem] font-bold uppercase tracking-[0.08em] text-muted-foreground/70">
      Benchmark
    </p>

    {#if report}
      <div class="mt-3 rounded-[10px] border border-border bg-card px-4 py-3.5">
        {#if report.passed}
          <p class="flex items-start gap-2 text-[0.8rem] font-bold text-foreground">
            <Network class="mt-0.5 size-4 shrink-0 text-primary" aria-hidden="true" />
            <span
              >Graph retrieval improves this notebook: +{report.delta_pp.toFixed(1)}pp recall@5</span
            >
          </p>
        {:else}
          <p class="flex items-start gap-2 text-[0.8rem] font-bold text-foreground">
            <Network class="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
            <span>No measurable benefit — keeping it off is safe</span>
          </p>
        {/if}

        <dl class="mt-3 grid grid-cols-2 gap-x-6 gap-y-1.5 text-[0.7rem]">
          <div class="flex justify-between gap-3">
            <dt class="text-muted-foreground">Graph recall@5</dt>
            <dd class="font-semibold text-foreground">{pct(report.graph_recall)}</dd>
          </div>
          <div class="flex justify-between gap-3">
            <dt class="text-muted-foreground">Hybrid recall@5</dt>
            <dd class="font-semibold text-foreground">{pct(report.hybrid_recall)}</dd>
          </div>
          <div class="flex justify-between gap-3">
            <dt class="text-muted-foreground">Delta</dt>
            <dd class="font-semibold text-foreground">{report.delta_pp.toFixed(1)}pp</dd>
          </div>
          <div class="flex justify-between gap-3">
            <dt class="text-muted-foreground">p95 latency</dt>
            <dd class="font-semibold text-foreground">{Math.round(report.p95_ms)}ms</dd>
          </div>
          <div class="flex justify-between gap-3">
            <dt class="text-muted-foreground">Sample size</dt>
            <dd class="font-semibold text-foreground">{report.sample_n}</dd>
          </div>
        </dl>
        <p class="mt-2 text-[0.65rem] text-muted-foreground">ran {friendlyDate(report.ran_at)}</p>
      </div>
    {:else}
      <p class="mt-3 text-[0.75rem] text-muted-foreground">
        No benchmark has run yet. Run one to measure whether graph retrieval improves recall for
        this notebook's sources.
      </p>
    {/if}

    <button
      type="button"
      onclick={runBenchmark}
      disabled={running}
      class="mt-4 inline-flex h-9 items-center justify-center rounded-lg bg-primary px-4 text-[0.78rem] font-semibold text-primary-foreground transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60"
    >
      {#if running}
        {progressLabel || 'Running…'}
      {:else if report}
        Re-run benchmark
      {:else}
        Run benchmark
      {/if}
    </button>

    {#if skippedReason}
      <p class="mt-3 text-[0.75rem] text-muted-foreground" role="status">{skippedReason}</p>
    {/if}

    {#if noProvider}
      <p class="mt-3 text-[0.75rem] text-muted-foreground" role="alert">
        Set up a chat model to run the benchmark.
      </p>
    {/if}

    {#if runError}
      <p class="mt-3 text-[0.75rem] text-destructive" role="alert">{runError}</p>
    {/if}
  </div>
</section>
