// Barrel re-export for the notebooks module.
//
// Consumers import from `$lib/notebooks` instead of reaching into individual
// files. Pattern matches `$lib/theme/index.ts`.

export * from './types.js';
export * from './ipc.js';
export * from './format-time.js';
export * from './format.js';
export * from './notebook-color.js';
export {
  notebookStore,
  resetNotebookStore,
  loadNotebooks,
  loadTrashed,
  loadTrashedSources,
  refreshTrashed,
  refreshTrashedSources,
  createNotebookAction,
  renameNotebookAction,
  trashNotebookAction,
  restoreNotebookAction,
  purgeNotebookAction,
  restoreSourceFromTrash,
  purgeSourceAction,
  selectNotebook,
  openTrash,
  notebookColorClass
} from './notebooks-state.svelte.js';
