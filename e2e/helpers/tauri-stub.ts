import type { Page } from '@playwright/test';

// Shared fake Tauri runtime for the onboarding e2e suite.
//
// E2E runs against the plain SvelteKit dev server (NO native Tauri backend), so
// we inject a fake runtime BEFORE app boot via page.addInitScript(). Two pieces
// make the gate take the real (non-fail-open) path:
//   1. `window.isTauri = true` — @tauri-apps/api's isTauri() returns
//      `!!(globalThis || window).isTauri`.
//   2. `window.__TAURI_INTERNALS__.invoke(cmd, args)` — invoke() in
//      @tauri-apps/api/core delegates straight to this. The stub dispatches on
//      `cmd`, returns a Promise, and records set_config calls on the window so a
//      test can read them back via page.evaluate().

// SYNC-CHECK: must match src/lib/theme/types.ts AppConfig
export type ModelConfig = {
  provider: string;
  base_url: string;
  model: string;
  context: number;
  temperature: number;
  api_key: string;
};
export type VoiceConfig = { host: string; guest: string };
export type TtsConfig = { provider: string; api_key: string };
export type PathConfig = { data_dir: string };
export type TierThresholds = { tier1_token_cap: number; tier2_token_cap: number };
export type AppConfig = {
  theme: string;
  accent: string;
  user_name: string;
  models: ModelConfig[];
  endpoints: Record<string, string>;
  voices: VoiceConfig;
  tts: TtsConfig;
  paths: PathConfig;
  tier_thresholds: TierThresholds;
  onboarding_complete: boolean;
  embedding_model: string;
  embedding_backend: string;
};

// SYNC-CHECK: must match src/lib/onboarding/system-check.ts CheckResult
export type CheckResult = {
  id: string;
  label: string;
  status: 'pass' | 'fail';
  detail: string;
  action: 'configure' | 'choose' | null;
};

/** A full, correctly-shaped AppConfig with the given onboarding flag. */
export function makeConfig(onboardingComplete: boolean): AppConfig {
  return {
    theme: 'dark',
    accent: 'purple',
    user_name: '',
    models: [],
    endpoints: {},
    voices: { host: '', guest: '' },
    tts: { provider: '', api_key: '' },
    paths: { data_dir: '' },
    tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
    onboarding_complete: onboardingComplete,
    embedding_model: '',
    embedding_backend: ''
  };
}

/**
 * The three readiness gates mirroring the frozen CheckResult contract
 * (snake_case ids, lowercase statuses): llm_runtime, embedding_model,
 * text_to_speech. All `pass` so the Continue gate (every row must pass) is NOT
 * blocked — returning-user / Continue flows stay green. A test that needs a
 * failing gate passes its own `checks` override.
 */
export const DEFAULT_CHECKS: CheckResult[] = [
  {
    id: 'llm_runtime',
    label: 'LLM runtime',
    status: 'pass',
    detail: 'Ollama 0.3.2 detected',
    action: 'configure'
  },
  {
    id: 'embedding_model',
    label: 'Embedding model',
    status: 'pass',
    detail: 'Embedding model installed',
    action: 'choose'
  },
  {
    id: 'text_to_speech',
    label: 'Text-to-speech',
    status: 'pass',
    detail: 'Kokoro audio engine ready',
    action: 'choose'
  }
];

export type InstallTauriStubOptions = {
  /** What get_config reports for `onboarding_complete`. */
  onboardingComplete: boolean;
  /** Rows returned by run_system_check (defaults to DEFAULT_CHECKS). */
  checks?: CheckResult[];
  /** Model ids `fastembed_models_cached` reports as cached (defaults to none). */
  fastembedCached?: string[];
  /** Model names `list_ollama_models` reports (defaults to none — Ollama down). */
  ollamaModels?: string[];
  /** A notebook embedding coordinate returned by `get_notebook_embedding_model`. */
  notebookEmbedding?: { model_id: string; dim: number; backend: string; status: string };
  /** Notebook summaries returned by `list_notebooks` (defaults to none). */
  notebooks?: { id: string; title: string }[];
};

/**
 * Inject the fake Tauri runtime. set_config calls are recorded on
 * `window.__SET_CONFIG_CALLS__` for the test to assert against.
 */
export async function installTauriStub(
  page: Page,
  {
    onboardingComplete,
    checks = DEFAULT_CHECKS,
    fastembedCached = [],
    ollamaModels = [],
    notebookEmbedding,
    notebooks = []
  }: InstallTauriStubOptions
): Promise<void> {
  await page.addInitScript(
    ({ cfg, checks, fastembedCached, ollamaModels, notebookEmbedding, notebooks }) => {
      const w = window as unknown as {
        isTauri?: boolean;
        __TAURI_INTERNALS__?: Record<string, unknown>;
        __SET_CONFIG_CALLS__?: unknown[];
        __REEMBED_CALLS__?: unknown[];
      };
      w.isTauri = true;
      (globalThis as unknown as { isTauri?: boolean }).isTauri = true;
      w.__SET_CONFIG_CALLS__ = [];
      w.__REEMBED_CALLS__ = [];

      let currentCfg = cfg;

      w.__TAURI_INTERNALS__ = {
        invoke(cmd: string, args?: Record<string, unknown>): Promise<unknown> {
          switch (cmd) {
            case 'get_config':
              return Promise.resolve(currentCfg);
            case 'run_system_check':
              return Promise.resolve(checks);
            case 'set_config': {
              const next = args?.config as typeof cfg | undefined;
              w.__SET_CONFIG_CALLS__!.push(next);
              if (next) currentCfg = next; // reflect the write for any later read
              return Promise.resolve(null);
            }
            case 'detect_llm':
              // Default stub: not reachable (safe — no local server in CI).
              // Override via page.addInitScript if a test needs a reachable stub.
              return Promise.resolve({ reachable: false, version: null, models: [] });
            case 'install_embedding_model': {
              // Real command streams InstallProgress { status, completed, total }
              // over a Channel passed as `onProgress`. In the stub the arg is the
              // live Channel instance (not yet IPC-serialized), so we drive its
              // onmessage directly to exercise the progress path, then resolve.
              const ch = args?.onProgress as { onmessage?: (m: unknown) => void } | undefined;
              ch?.onmessage?.({ status: 'pulling manifest', completed: null, total: null });
              ch?.onmessage?.({ status: 'downloading', completed: 5000, total: 10000 });
              ch?.onmessage?.({ status: 'success', completed: 10000, total: 10000 });
              return Promise.resolve(null);
            }
            case 'download_tts_engine': {
              // Real command streams DownloadProgress { received, total, done }.
              const ch = args?.onProgress as { onmessage?: (m: unknown) => void } | undefined;
              ch?.onmessage?.({ received: 0, total: 90000000, done: false });
              ch?.onmessage?.({ received: 45000000, total: 90000000, done: false });
              ch?.onmessage?.({ received: 90000000, total: 90000000, done: true });
              return Promise.resolve(null);
            }
            case 'list_tts_voices':
              // Mirror the real Kokoro catalog shape (TtsVoice { id, name, gender }).
              return Promise.resolve([
                { id: 'af_heart', name: 'Heart', gender: 'female' },
                { id: 'am_michael', name: 'Michael', gender: 'male' }
              ]);
            case 'create_notebook':
              // Notebook-shaped; the Create notebook screen reads `.id` into the
              // draft. description/focus_mode are write-only in M1.
              return Promise.resolve({
                id: 'nb-e2e-1',
                title: (args?.title as string) ?? '',
                description: (args?.description as string | null) ?? null,
                focus_mode: (args?.focusMode as string | null) ?? null,
                created_at: '2026-01-01T00:00:00Z'
              });
            case 'add_source':
              // Records-only source row (no ingestion).
              return Promise.resolve({
                id: 'src-e2e-1',
                notebook_id: (args?.notebookId as string) ?? '',
                kind: 'file',
                title: (args?.title as string) ?? '',
                status: 'pending',
                locator: (args?.locator as string) ?? '',
                selected: 1
              });
            case 'list_sources':
              return Promise.resolve([]);
            case 'list_recent_documents':
              // Empty → the "Suggested from your library" section stays hidden.
              return Promise.resolve([]);
            case 'list_notebooks':
              // NotebookSummary-shaped rows so the sidebar can render + select.
              return Promise.resolve(
                notebooks.map((n) => ({
                  id: n.id,
                  title: n.title,
                  description: null,
                  focus_mode: null,
                  embedding_model: null,
                  embedding_backend: null,
                  created_at: '2026-01-01T00:00:00Z',
                  updated_at: '2026-01-01T00:00:00Z',
                  trashed_at: null,
                  source_count: 2
                }))
              );
            case 'list_trashed':
              return Promise.resolve([]);
            case 'list_trashed_sources':
              return Promise.resolve([]);
            case 'fastembed_models_cached':
              // Per-model fastembed on-disk cache probe (M4 4b-B).
              return Promise.resolve(fastembedCached);
            case 'list_ollama_models':
              // Detect-only Ollama /api/tags probe (graceful empty when down).
              return Promise.resolve(ollamaModels);
            case 'warm_fastembed_model':
              // fastembed Install = warm (download) the weights. No-op stub: by
              // resolving we also pretend the model is now cached for any later
              // fastembed_models_cached read in the same test (handled per-test).
              return Promise.resolve(null);
            case 'get_notebook_embedding_model':
              return Promise.resolve(
                notebookEmbedding ?? {
                  model_id: 'nomic-embed-text-v1.5',
                  dim: 768,
                  backend: 'fastembed',
                  status: 'none'
                }
              );
            case 'set_notebook_embedding_model': {
              // Records the change + streams a StreamEvent<ReembedProgress> run.
              w.__REEMBED_CALLS__!.push({
                notebookId: args?.notebookId,
                modelId: args?.modelId,
                backend: args?.backend
              });
              const ch = args?.onProgress as { onmessage?: (m: unknown) => void } | undefined;
              ch?.onmessage?.({ type: 'started' });
              ch?.onmessage?.({ type: 'chunk', data: { done: 1, total: 2 } });
              ch?.onmessage?.({ type: 'chunk', data: { done: 2, total: 2 } });
              ch?.onmessage?.({ type: 'done' });
              return Promise.resolve(null);
            }
            default:
              return Promise.resolve(null);
          }
        },
        transformCallback(callback: unknown) {
          return callback;
        },
        unregisterCallback() {},
        convertFileSrc(path: string) {
          return path;
        }
      };
    },
    {
      cfg: makeConfig(onboardingComplete),
      checks,
      fastembedCached,
      ollamaModels,
      notebookEmbedding,
      notebooks
    }
  );
}

/** Reads back the recorded set_config payloads from the page. */
export function readSetConfigCalls(
  page: Page
): Promise<
  { onboarding_complete?: boolean; embedding_model?: string; embedding_backend?: string }[]
> {
  return page.evaluate(
    () =>
      (
        window as unknown as {
          __SET_CONFIG_CALLS__?: {
            onboarding_complete?: boolean;
            embedding_model?: string;
            embedding_backend?: string;
          }[];
        }
      ).__SET_CONFIG_CALLS__ ?? []
  );
}

/** Reads back the recorded set_notebook_embedding_model calls from the page. */
export function readReembedCalls(
  page: Page
): Promise<{ notebookId?: string; modelId?: string; backend?: string }[]> {
  return page.evaluate(
    () =>
      (
        window as unknown as {
          __REEMBED_CALLS__?: { notebookId?: string; modelId?: string; backend?: string }[];
        }
      ).__REEMBED_CALLS__ ?? []
  );
}
