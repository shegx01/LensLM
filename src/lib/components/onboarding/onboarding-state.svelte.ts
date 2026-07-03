// Module-level $state singleton for the onboarding draft. `resetDraft()` MUST
// be called after completion so a re-arm starts from clean values. Durable
// writes happen in the screen components — never here.

/** Selected source file; same shape as the backend `RecentDocument` IPC payload. */
export interface DraftSource {
  path: string;
  name: string;
  ext: string;
  size: number;
  mtime: number;
}

/** Recent-document suggestion from `list_recent_documents`; identical shape to {@link DraftSource}. */
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

/** Onboarding draft — reactive getters/setters backed by module-level `$state`. */
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

/** Reset all draft fields to initial values. Call after onboarding completes. */
export function resetDraft(): void {
  userName = INITIAL.userName;
  accent = INITIAL.accent;
  nbName = INITIAL.nbName;
  nbDesc = INITIAL.nbDesc;
  focusMode = INITIAL.focusMode;
  selectedSources = [...INITIAL.selectedSources];
  notebookId = INITIAL.notebookId;
}
