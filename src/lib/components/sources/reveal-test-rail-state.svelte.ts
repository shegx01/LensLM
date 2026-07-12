// Test-only reactive backing for the notebookStore mock — see
// SourcesRail.reveal.svelte.test.ts for why this must be real `$state`.
let rightRailCollapsed = $state(false);

export const railState = {
  get rightRailCollapsed(): boolean {
    return rightRailCollapsed;
  },
  set rightRailCollapsed(v: boolean) {
    rightRailCollapsed = v;
  }
};
