// Barrel re-export for the sources module.
//
// Consumers import from `$lib/sources` instead of reaching into individual
// files. Pattern matches `$lib/notebooks/index.ts`.

export * from './types.js';
export * from './ipc.js';
export * from './status.js';
export {
  sourcesStore,
  resetSourcesStore,
  loadSources,
  addSourceLocal,
  ingest,
  toggleSelected,
  removeSource,
  undoRemove,
  disposeTrashTimers,
  drainTrashQueueEntry,
  disposeAutoRefresh
} from './sources-state.svelte.js';
