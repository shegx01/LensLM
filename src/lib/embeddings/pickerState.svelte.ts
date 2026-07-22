// Shared embedding-picker controller. Owns the state + IPC orchestration behind
// BOTH the Settings `EmbeddingsSection` and the onboarding `OnboardingEmbeddingPicker`
// so detection/install/persist logic lives in one place — the two components only
// differ in presentation. `mode: 'global'` persists the app-wide default;
// `mode: 'notebook'` re-embeds one notebook's coordinate.

import { invoke, isTauri } from '@tauri-apps/api/core';
import type { AppConfig } from '$lib/theme/types.js';
import { updateConfig } from '$lib/config.js';
import { prefersReducedMotion } from '$lib/motion/index.js';
import {
  EMBEDDING_MODELS,
  DEFAULT_EMBEDDING_MODEL,
  DEFAULT_EMBEDDING_BACKEND,
  resolveModel,
  resolveBackend,
  ollamaMatches,
  type EmbeddingBackend,
  type EmbeddingModelId
} from './models.js';
import {
  fastembedModelsCached,
  listOllamaModels,
  warmFastembedModel,
  getNotebookEmbeddingModel,
  setNotebookEmbeddingModel,
  gpuAcceleratedModels
} from './ipc.js';

export const DEFAULT_OLLAMA_ENDPOINT = 'http://localhost:11434';

// fastembed reports no byte progress, so the label advances on a fixed cadence —
// purely decorative; it does not reflect real download state.
const INSTALL_PHASES = ['Downloading…', 'Extracting…', 'Configuring…', 'Almost ready…'];
const INSTALL_PHASE_INTERVAL_MS = 1200;

const FASTEMBED_MODELS = EMBEDDING_MODELS.filter((m) => m.backends.includes('fastembed'));
const OLLAMA_MODELS = EMBEDDING_MODELS.filter((m) => m.backends.includes('ollama'));
const FIRST_OLLAMA: EmbeddingModelId = OLLAMA_MODELS[0]?.id ?? DEFAULT_EMBEDDING_MODEL;

/** Metadata-chip class shared by both picker views (dims / size / speed pills). */
export const CHIP_CLASS =
  'rounded-full bg-muted px-1.5 py-0.5 text-[0.62rem] font-semibold text-muted-foreground';

export interface EmbeddingPickerConfig {
  mode: 'global' | 'notebook';
  /** Required in notebook mode — the notebook whose coordinate this edits. */
  notebookId?: string | null;
  /** Fired after a successful persist / re-embed. */
  onchange?: () => void | Promise<void>;
}

export class EmbeddingPickerState {
  backend = $state<EmbeddingBackend>(DEFAULT_EMBEDDING_BACKEND);
  // Per-backend selection so switching provider keeps each side's pick.
  selectedByBackend = $state<Record<EmbeddingBackend, EmbeddingModelId>>({
    fastembed: DEFAULT_EMBEDDING_MODEL,
    ollama: FIRST_OLLAMA
  });
  // The persisted default (global) / active coordinate (notebook) — the value the
  // gate reads, kept distinct from the previewed selection above.
  activeModel = $state<EmbeddingModelId>(DEFAULT_EMBEDDING_MODEL);
  activeBackend = $state<EmbeddingBackend>(DEFAULT_EMBEDDING_BACKEND);
  coordinateIndexed = $state(false);

  fastembedCached = $state<Set<EmbeddingModelId>>(new Set());
  ollamaInstalled = $state<Set<EmbeddingModelId>>(new Set());
  ollamaEndpoint = $state(DEFAULT_OLLAMA_ENDPOINT);
  // GPU acceleration is per-model (candle+Metal for nomic; CPU otherwise), so the
  // badge is keyed on this set, not the provider (issue #91).
  gpuModels = $state<Set<string>>(new Set());

  refreshing = $state(false);
  installing = $state(false);
  installPhase = $state('');
  actionError = $state<string | null>(null);

  confirmOpen = $state(false);
  reembedDone = $state(0);
  reembedTotal = $state(0);
  reembedding = $state(false);

  readonly #mode: 'global' | 'notebook';
  readonly #notebookId: string | null;
  readonly #onchange?: () => void | Promise<void>;
  #phaseTimer: ReturnType<typeof setInterval> | null = null;

  constructor(config: EmbeddingPickerConfig) {
    this.#mode = config.mode;
    this.#notebookId = config.notebookId ?? null;
    this.#onchange = config.onchange;
  }

  readonly models = $derived(this.backend === 'fastembed' ? FASTEMBED_MODELS : OLLAMA_MODELS);
  readonly installed = $derived(
    this.backend === 'fastembed' ? this.fastembedCached : this.ollamaInstalled
  );
  readonly selectedModel = $derived(this.selectedByBackend[this.backend]);
  readonly focusedModel = $derived(
    this.models.find((m) => m.id === this.selectedModel) ?? this.models[0]
  );
  readonly selectedReady = $derived(this.installed.has(this.selectedModel));
  // Blocks selection changes while a write is in flight (install or re-embed).
  readonly busy = $derived(this.reembedding || this.installing);
  readonly isDirty = $derived(
    this.selectedModel !== this.activeModel || this.backend !== this.activeBackend
  );
  readonly needsInstall = $derived(this.backend === 'fastembed' && !this.selectedReady);
  // Whether the PERSISTED default is usable — drives Settings' Active markers.
  readonly activeReady = $derived(
    this.activeBackend === 'fastembed'
      ? this.fastembedCached.has(this.activeModel)
      : this.ollamaInstalled.has(this.activeModel)
  );
  readonly reembedPct = $derived(
    this.reembedTotal > 0
      ? Math.min(100, Math.round((this.reembedDone / this.reembedTotal) * 100))
      : 0
  );

  // On-device is always usable (bundled); Ollama only once ≥1 model is detected.
  providerReady(id: EmbeddingBackend): boolean {
    return id === 'fastembed' ? true : this.ollamaInstalled.size > 0;
  }

  async init(): Promise<void> {
    void gpuAcceleratedModels().then((ids) => (this.gpuModels = new Set(ids)));
    if (this.#mode === 'global') {
      if (isTauri()) {
        try {
          const cfg = await invoke<AppConfig>('get_config');
          this.activeBackend = resolveBackend(cfg.embedding_backend);
          this.activeModel = resolveModel(cfg.embedding_model).id;
          const ep = cfg.endpoints?.ollama;
          if (ep) this.ollamaEndpoint = ep;
        } catch {
          // fall back to defaults
        }
      }
    } else if (this.#notebookId) {
      const info = await getNotebookEmbeddingModel(this.#notebookId);
      this.activeBackend = resolveBackend(info.backend);
      this.activeModel = resolveModel(info.model_id).id;
      this.coordinateIndexed = info.status === 'active';
    }
    this.backend = this.activeBackend;
    this.selectedByBackend[this.activeBackend] = this.activeModel;
    await Promise.all([this.refreshFastembed(), this.refreshOllama()]);
  }

  async refreshFastembed(): Promise<void> {
    try {
      const ids = await fastembedModelsCached();
      this.fastembedCached = new Set(ids.map((id) => resolveModel(id).id));
    } catch {
      this.fastembedCached = new Set();
    }
  }

  async refreshOllama(): Promise<void> {
    this.refreshing = true;
    try {
      const names = await listOllamaModels(this.ollamaEndpoint);
      const found = new Set<EmbeddingModelId>();
      for (const m of OLLAMA_MODELS) {
        if (names.some((d) => ollamaMatches(d, m))) found.add(m.id);
      }
      this.ollamaInstalled = found;
    } catch {
      this.ollamaInstalled = new Set();
    } finally {
      this.refreshing = false;
    }
  }

  pickBackend(b: EmbeddingBackend): void {
    if (this.busy) return;
    this.backend = b;
    this.actionError = null;
  }

  pickModel(id: EmbeddingModelId): void {
    if (this.busy) return;
    this.selectedByBackend[this.backend] = id;
    this.actionError = null;
  }

  async install(): Promise<void> {
    if (this.installing) return; // re-entrancy guard (mirrors pickBackend/pickModel)
    const model = this.selectedModel;
    this.actionError = null;
    this.installing = true;
    this.installPhase = INSTALL_PHASES[0];
    let i = 0;
    // Reduced motion renders a static "Installing…", so the ticker is dead work.
    if (!prefersReducedMotion()) {
      this.#phaseTimer = setInterval(() => {
        i = Math.min(i + 1, INSTALL_PHASES.length - 1);
        this.installPhase = INSTALL_PHASES[i];
      }, INSTALL_PHASE_INTERVAL_MS);
    }
    try {
      await warmFastembedModel(model);
      await this.refreshFastembed();
      // Commit only once the weights are actually on disk — a warm that reports
      // success without populating the cache must not persist an unusable default.
      if (this.fastembedCached.has(model)) await this.commit();
      else this.actionError = 'The model did not finish installing. Try again.';
    } catch (err) {
      this.actionError = err instanceof Error ? err.message : 'Installation failed.';
    } finally {
      this.#clearPhaseTimer();
      this.installing = false;
    }
  }

  async commit(): Promise<void> {
    if (this.#mode === 'global') await this.#persistGlobal();
    else await this.#maybeReembed();
  }

  async #persistGlobal(): Promise<void> {
    this.actionError = null;
    const model = this.selectedModel;
    const backend = this.backend;
    try {
      await updateConfig((cfg) => ({
        ...cfg,
        embedding_model: model,
        embedding_backend: backend
      }));
      this.activeModel = model;
      this.activeBackend = backend;
      await this.#onchange?.();
    } catch (err) {
      this.actionError = err instanceof Error ? err.message : 'Could not save the default.';
    }
  }

  // An indexed coordinate change opens the confirm dialog; otherwise apply now.
  async #maybeReembed(): Promise<void> {
    if (this.coordinateIndexed && this.isDirty) {
      this.confirmOpen = true;
      return;
    }
    await this.runReembed();
  }

  async runReembed(): Promise<void> {
    if (!this.#notebookId) return;
    const model = this.selectedModel;
    const backend = this.backend;
    this.confirmOpen = false;
    this.actionError = null;
    this.reembedding = true;
    this.reembedDone = 0;
    this.reembedTotal = 0;
    try {
      await setNotebookEmbeddingModel(this.#notebookId, model, backend, (done, total) => {
        this.reembedDone = done;
        this.reembedTotal = total;
      });
      this.activeModel = model;
      this.activeBackend = backend;
      this.coordinateIndexed = true;
      await this.#onchange?.();
    } catch (err) {
      this.actionError = err instanceof Error ? err.message : 'Re-embedding failed.';
    } finally {
      this.reembedding = false;
    }
  }

  cancelReembed(): void {
    this.confirmOpen = false;
  }

  #clearPhaseTimer(): void {
    if (this.#phaseTimer) clearInterval(this.#phaseTimer);
    this.#phaseTimer = null;
  }

  dispose(): void {
    this.#clearPhaseTimer();
  }
}
