// IPC-boundary contract tests. These pin the exact argument keys sent to
// `invoke`, which module-level mocks in higher-level suites cannot catch.

import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => true,
  invoke: vi.fn()
}));

import { invoke } from '@tauri-apps/api/core';
import { touchNotebookActivity } from './ipc.js';

beforeEach(() => {
  vi.mocked(invoke).mockReset();
  vi.mocked(invoke).mockResolvedValue(undefined);
});

describe('touchNotebookActivity', () => {
  // Regression: `touch_notebook_activity` is a plain `#[tauri::command]`, so Tauri
  // v2 requires the camelCase key `notebookId`. A snake_case `notebook_id` key
  // silently fails the command, which broke reopen-last-notebook (MRU never bumped).
  it('invokes with the camelCase notebookId key', async () => {
    await touchNotebookActivity('nb-001');

    expect(invoke).toHaveBeenCalledWith('touch_notebook_activity', { notebookId: 'nb-001' });
  });
});

describe('touchNotebookActivity outside Tauri', () => {
  it('is a no-op that never calls invoke', async () => {
    vi.resetModules();
    vi.doMock('@tauri-apps/api/core', () => ({
      isTauri: () => false,
      invoke: vi.fn()
    }));
    const core = await import('@tauri-apps/api/core');
    const { touchNotebookActivity: guarded } = await import('./ipc.js');

    await guarded('nb-001');

    expect(core.invoke).not.toHaveBeenCalled();
    vi.doUnmock('@tauri-apps/api/core');
  });
});
