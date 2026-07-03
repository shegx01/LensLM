// Barrel re-export for the sources module.

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
