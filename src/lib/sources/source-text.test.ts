// IPC-boundary contract tests (issue #237): pin the exact camelCase argument keys
// sent to `invoke` (Tauri v2 requires this — a snake_case key silently fails the
// command) and the snippet memoization behavior.

import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => true,
  invoke: vi.fn()
}));

import { invoke } from '@tauri-apps/api/core';
import {
  resolveCitationSnippet,
  loadSourceView,
  resetCitationSnippetCache
} from './source-text.js';
import type { SnippetSegments, SourceView } from '$lib/chat/types.js';

const segments: SnippetSegments = {
  before: 'before ',
  marked: 'cited span',
  after: ' after',
  truncated_before: false,
  truncated_after: false
};

const view: SourceView = {
  before: 'whole text',
  marked: '',
  after: '',
  title: 'Doc',
  kind: 'text',
  truncated: false
};

beforeEach(() => {
  vi.mocked(invoke).mockReset();
  resetCitationSnippetCache();
});

describe('resolveCitationSnippet', () => {
  it('invokes resolve_citation_snippet with camelCase keys', async () => {
    vi.mocked(invoke).mockResolvedValue(segments);

    const result = await resolveCitationSnippet('src-1', 10, 20);

    expect(invoke).toHaveBeenCalledWith('resolve_citation_snippet', {
      sourceId: 'src-1',
      charStart: 10,
      charEnd: 20
    });
    expect(result).toEqual(segments);
  });

  it('memoizes by (sourceId, charStart, charEnd) — a repeated hover does not refetch', async () => {
    vi.mocked(invoke).mockResolvedValue(segments);

    await resolveCitationSnippet('src-1', 10, 20);
    await resolveCitationSnippet('src-1', 10, 20);

    expect(invoke).toHaveBeenCalledTimes(1);
  });

  it('does not share the memo cache across different spans or sources', async () => {
    vi.mocked(invoke).mockResolvedValue(segments);

    await resolveCitationSnippet('src-1', 10, 20);
    await resolveCitationSnippet('src-1', 10, 21);
    await resolveCitationSnippet('src-2', 10, 20);

    expect(invoke).toHaveBeenCalledTimes(3);
  });

  it('evicts a rejected fetch so a later hover can retry', async () => {
    vi.mocked(invoke).mockRejectedValueOnce(new Error('boom'));
    await expect(resolveCitationSnippet('src-1', 10, 20)).rejects.toThrow('boom');

    vi.mocked(invoke).mockResolvedValueOnce(segments);
    await expect(resolveCitationSnippet('src-1', 10, 20)).resolves.toEqual(segments);

    expect(invoke).toHaveBeenCalledTimes(2);
  });
});

describe('loadSourceView', () => {
  it('invokes load_source_view with camelCase keys and a span', async () => {
    vi.mocked(invoke).mockResolvedValue(view);

    await loadSourceView('src-1', 5, 15);

    expect(invoke).toHaveBeenCalledWith('load_source_view', {
      sourceId: 'src-1',
      charStart: 5,
      charEnd: 15
    });
  });

  it('defaults offsets to null when omitted (R4 — null-offset degradation)', async () => {
    vi.mocked(invoke).mockResolvedValue(view);

    await loadSourceView('src-1');

    expect(invoke).toHaveBeenCalledWith('load_source_view', {
      sourceId: 'src-1',
      charStart: null,
      charEnd: null
    });
  });

  it('is not memoized — two calls invoke twice', async () => {
    vi.mocked(invoke).mockResolvedValue(view);

    await loadSourceView('src-1', 5, 15);
    await loadSourceView('src-1', 5, 15);

    expect(invoke).toHaveBeenCalledTimes(2);
  });
});

describe('outside Tauri', () => {
  it('resolveCitationSnippet rejects without calling invoke', async () => {
    vi.resetModules();
    vi.doMock('@tauri-apps/api/core', () => ({
      isTauri: () => false,
      invoke: vi.fn()
    }));
    const core = await import('@tauri-apps/api/core');
    const { resolveCitationSnippet: guarded } = await import('./source-text.js');

    await expect(guarded('src-1', 0, 1)).rejects.toThrow();
    expect(core.invoke).not.toHaveBeenCalled();
    vi.doUnmock('@tauri-apps/api/core');
  });

  it('loadSourceView rejects without calling invoke', async () => {
    vi.resetModules();
    vi.doMock('@tauri-apps/api/core', () => ({
      isTauri: () => false,
      invoke: vi.fn()
    }));
    const core = await import('@tauri-apps/api/core');
    const { loadSourceView: guarded } = await import('./source-text.js');

    await expect(guarded('src-1')).rejects.toThrow();
    expect(core.invoke).not.toHaveBeenCalled();
    vi.doUnmock('@tauri-apps/api/core');
  });
});
