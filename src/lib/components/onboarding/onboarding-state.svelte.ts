// Onboarding draft store (Svelte 5 runes, module singleton).
//
// Holds the in-progress onboarding input so Back/Forward across the step
// machine preserves what the user typed without prop-drilling through the
// layout. This is intentionally a module-level `$state` singleton: the
// onboarding flow is a single first-run session and there is never more than
// one in flight. `resetDraft()` MUST be called after completion so a future
// re-arm (settings/showcase reset) starts from clean initial values rather than
// stale globals.
//
// PERSISTENCE NOTE: this store is the DRAFT only. Durable writes (user_name +
// accent via one RMW set_config at Make-it-yours; the notebook at Create;
// onboarding_complete at the end) happen in the screen components — never here.

/** A selected source file, captured from the dialog/recent-docs pickers.
 *  Same shape as the backend `RecentDocument` IPC payload (re-exported below). */
export interface DraftSource {
  path: string;
  name: string;
  ext: string;
  size: number;
  mtime: number;
}

/** A recent-document suggestion from the `list_recent_documents` command —
 *  identical shape to a {@link DraftSource}. Canonical type lives here. */
export type RecentDocument = DraftSource;

const INITIAL = {
  userName: '',
  accent: 'purple',
  nbName: '',
  nbDesc: '',
  focusMode: 'research',
  selectedSources: [] as DraftSource[],
  notebookId: null as string | null
};

let userName = $state(INITIAL.userName);
let accent = $state(INITIAL.accent);
let nbName = $state(INITIAL.nbName);
let nbDesc = $state(INITIAL.nbDesc);
let focusMode = $state(INITIAL.focusMode);
let selectedSources = $state<DraftSource[]>([...INITIAL.selectedSources]);
let notebookId = $state<string | null>(INITIAL.notebookId);

/**
 * The onboarding draft. Read the reactive getters in markup/`$derived`; call the
 * setters to mutate. Backed by module-level `$state` so every screen sees the
 * same live values.
 */
export const draft = {
  get userName() {
    return userName;
  },
  set userName(v: string) {
    userName = v;
  },
  get accent() {
    return accent;
  },
  set accent(v: string) {
    accent = v;
  },
  get nbName() {
    return nbName;
  },
  set nbName(v: string) {
    nbName = v;
  },
  get nbDesc() {
    return nbDesc;
  },
  set nbDesc(v: string) {
    nbDesc = v;
  },
  get focusMode() {
    return focusMode;
  },
  set focusMode(v: string) {
    focusMode = v;
  },
  get selectedSources() {
    return selectedSources;
  },
  set selectedSources(v: DraftSource[]) {
    selectedSources = v;
  },
  get notebookId() {
    return notebookId;
  },
  set notebookId(v: string | null) {
    notebookId = v;
  }
};

/**
 * Restore every draft field to its initial value. Call AFTER onboarding
 * completes so a subsequent re-arm starts clean (avoids stale module globals).
 */
export function resetDraft(): void {
  userName = INITIAL.userName;
  accent = INITIAL.accent;
  nbName = INITIAL.nbName;
  nbDesc = INITIAL.nbDesc;
  focusMode = INITIAL.focusMode;
  selectedSources = [...INITIAL.selectedSources];
  notebookId = INITIAL.notebookId;
}
