import type { Page } from '@playwright/test';

// Shared fake Tauri runtime for e2e (M1, plan §6).
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
export type PathConfig = { data_dir: string };
export type TierThresholds = { tier1_token_cap: number; tier2_token_cap: number };
export type AppConfig = {
  theme: string;
  accent: string;
  models: ModelConfig[];
  endpoints: Record<string, string>;
  voices: VoiceConfig;
  paths: PathConfig;
  tier_thresholds: TierThresholds;
  onboarding_complete: boolean;
  embedding_model: string;
};

// SYNC-CHECK: must match src/lib/onboarding/system-check.ts CheckResult
export type CheckResult = {
  id: string;
  label: string;
  status: 'pass' | 'fail' | 'pending';
  detail: string;
  action: 'configure' | 'choose' | 'retry' | null;
};

/** A full, correctly-shaped AppConfig with the given onboarding flag. */
export function makeConfig(onboardingComplete: boolean): AppConfig {
  return {
    theme: 'dark',
    accent: 'purple',
    models: [],
    endpoints: {},
    voices: { host: '', guest: '' },
    paths: { data_dir: '' },
    tier_thresholds: { tier1_token_cap: 4000, tier2_token_cap: 16000 },
    onboarding_complete: onboardingComplete,
    embedding_model: ''
  };
}

/**
 * Six rows mirroring the frozen CheckResult contract (snake_case ids, lowercase
 * statuses) — the backend now returns text_to_speech too. A deliberate mix of
 * pass / fail / pending exercises icon + action rendering. local_backend +
 * disk_permissions stay `pass` so Continue is NOT blocked (gating predicate,
 * plan change #12).
 */
export const DEFAULT_CHECKS: CheckResult[] = [
  {
    id: 'local_backend',
    label: 'Local backend',
    status: 'pass',
    detail: 'In-process engine ready',
    action: null
  },
  {
    id: 'llm_runtime',
    label: 'LLM runtime',
    status: 'fail',
    detail: 'No local LLM runtime detected',
    action: 'configure'
  },
  {
    id: 'embedding_model',
    label: 'Embedding model',
    status: 'pending',
    detail: 'Set up when you add your first source',
    action: 'choose'
  },
  {
    id: 'vector_database',
    label: 'Vector database',
    status: 'pending',
    detail: 'Built-in storage, set up automatically',
    action: null
  },
  {
    id: 'disk_permissions',
    label: 'Disk permissions',
    status: 'pass',
    detail: '/tmp/lens',
    action: null
  },
  {
    id: 'text_to_speech',
    label: 'Text-to-speech',
    status: 'pending',
    detail: 'Kokoro audio engine — download required',
    action: 'choose'
  }
];

export type InstallTauriStubOptions = {
  /** What get_config reports for `onboarding_complete`. */
  onboardingComplete: boolean;
  /** Rows returned by run_system_check (defaults to DEFAULT_CHECKS). */
  checks?: CheckResult[];
};

/**
 * Inject the fake Tauri runtime. set_config calls are recorded on
 * `window.__SET_CONFIG_CALLS__` for the test to assert against.
 */
export async function installTauriStub(
  page: Page,
  { onboardingComplete, checks = DEFAULT_CHECKS }: InstallTauriStubOptions
): Promise<void> {
  await page.addInitScript(
    ({ cfg, checks }) => {
      const w = window as unknown as {
        isTauri?: boolean;
        __TAURI_INTERNALS__?: Record<string, unknown>;
        __SET_CONFIG_CALLS__?: unknown[];
      };
      w.isTauri = true;
      (globalThis as unknown as { isTauri?: boolean }).isTauri = true;
      w.__SET_CONFIG_CALLS__ = [];

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
    { cfg: makeConfig(onboardingComplete), checks }
  );
}

/** Reads back the recorded set_config payloads from the page. */
export function readSetConfigCalls(page: Page): Promise<{ onboarding_complete?: boolean }[]> {
  return page.evaluate(
    () =>
      (window as unknown as { __SET_CONFIG_CALLS__?: { onboarding_complete?: boolean }[] })
        .__SET_CONFIG_CALLS__ ?? []
  );
}
