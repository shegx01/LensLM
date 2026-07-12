// Test-only reactive backing for the notebookStore mock used by
// SourcesRail.reveal.svelte.test.ts. `rightRailCollapsed` must be real Svelte
// `$state` so the component's `{#if collapsed}` swap reacts when `focusSource`
// flips it — a plain `let` would never re-render the expanded rail, leaving the
// collapsed-path scroll unobservable (the AC6 gap this backs).
let rightRailCollapsed = $state(false);

export const railState = {
  get rightRailCollapsed(): boolean {
    return rightRailCollapsed;
  },
  set rightRailCollapsed(v: boolean) {
    rightRailCollapsed = v;
  }
};
