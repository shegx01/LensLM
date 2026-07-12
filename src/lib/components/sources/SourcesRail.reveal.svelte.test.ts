// SourcesRail.reveal.svelte.test.ts — reveal-in-rail integration tests (#23b AC6).
//
// Unlike SourcesRail.svelte.test.ts (which uses a closure mock of the sources
// store), this file drives the REAL $state-backed sources store — `addSourceLocal`
// + `focusSource` + `resetSourcesStore` — so the SourcesRail focus $effect is
// exercised through genuine Svelte reactivity, not a static pre-render snapshot.
// Only IPC, Tauri, and the notebooks store are mocked.

import { render } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { tick } from 'svelte';
import type { Source } from '$lib/sources/types.js';

// isTauri()=false makes listSources return [] and the store's auto-refresh a no-op.
vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => false,
  invoke: vi.fn(),
  Channel: vi.fn()
}));

vi.mock('@tauri-apps/plugin-dialog', () => ({
  open: vi.fn().mockResolvedValue(null)
}));

// `rightRailCollapsed` is backed by real Svelte `$state` (in the .svelte.ts helper)
// so the component's `{#if collapsed}` swap reacts when `focusSource` flips it.
// A plain `let` would never re-render the expanded rail, so the collapsed-path
// scroll would be unobservable (AC6 gap). The mock is built in vi.hoisted (so the
// vi.mock factory can see it) but delegates to `railState`, which is late-bound
// after the .svelte.ts import initialises the `$state`.
import { railState } from './reveal-test-rail-state.svelte.js';

const { mockNotebookStore, bindRailState } = vi.hoisted(() => {
  let bound: { rightRailCollapsed: boolean } | null = null;
  const mockNotebookStore = {
    get activeNotebookId() {
      return 'nb-001';
    },
    get activeNotebook() {
      return { id: 'nb-001', title: 'iGaming Market Analysis' };
    },
    get rightRailCollapsed() {
      return bound?.rightRailCollapsed ?? false;
    },
    set rightRailCollapsed(v: boolean) {
      if (bound) bound.rightRailCollapsed = v;
    },
    _setRightRailCollapsed(v: boolean) {
      if (bound) bound.rightRailCollapsed = v;
    }
  };
  const bindRailState = (rs: { rightRailCollapsed: boolean }) => {
    bound = rs;
  };
  return { mockNotebookStore, bindRailState };
});

bindRailState(railState);

vi.mock('$lib/notebooks/notebooks-state.svelte.js', () => ({
  notebookStore: mockNotebookStore,
  refreshTrashedSources: vi.fn().mockResolvedValue(undefined)
}));

// The REAL sources store — genuine $state, not a mock.
import {
  sourcesStore,
  addSourceLocal,
  focusSource,
  resetSourcesStore
} from '$lib/sources/sources-state.svelte.js';
import SourcesRail, { PULSE_MS } from './SourcesRail.svelte';

function makeSource(overrides?: Partial<Source>): Source {
  return {
    id: 'src-001',
    notebook_id: 'nb-001',
    kind: 'file',
    title: 'Market Analysis Report.md',
    status: 'indexed',
    locator: '/docs/Market Analysis Report.md',
    selected: 1,
    created_at: new Date().toISOString(),
    token_count: 2048,
    content_hash: 'abc123',
    raw_content_hash: null,
    trashed_at: null,
    enrichment_status: null,
    enrichment_meta: null,
    force_js_render: 0,
    error_meta: null,
    ...overrides
  };
}

beforeEach(() => {
  resetSourcesStore();
  mockNotebookStore._setRightRailCollapsed(false);
});

afterEach(() => {
  resetSourcesStore();
  mockNotebookStore._setRightRailCollapsed(false);
});

describe('SourcesRail — reveal-in-rail re-fire (real store, AC6)', () => {
  it('does not scroll on mount, fires once on focus, and RE-FIRES for the same id (focusNonce)', async () => {
    const scrollSpy = vi
      .spyOn(HTMLElement.prototype, 'scrollIntoView')
      .mockImplementation(() => {});

    // Populate the real store BEFORE mount; focus stays null so the effect's
    // mount-time run must be a no-op (proving mount alone doesn't scroll).
    addSourceLocal(makeSource({ id: 'src-002' }));
    addSourceLocal(makeSource({ id: 'src-001' }));

    render(SourcesRail);
    await tick();
    expect(scrollSpy).not.toHaveBeenCalled();

    // First focus post-mount: reactive focusedSourceId change drives one scroll.
    focusSource('src-002');
    await tick();
    await tick();
    expect(scrollSpy).toHaveBeenCalledTimes(1);
    expect(sourcesStore.focusedSourceId).toBe('src-002');

    // Re-focus the SAME id: focusedSourceId is unchanged, so only the focusNonce
    // bump can re-trigger the effect. A second scroll proves the nonce drives re-fire.
    focusSource('src-002');
    await tick();
    await tick();
    expect(scrollSpy).toHaveBeenCalledTimes(2);

    scrollSpy.mockRestore();
  });

  it('clears the pulse ring after PULSE_MS and does not leak the timer', async () => {
    vi.useFakeTimers();
    try {
      const scrollSpy = vi
        .spyOn(HTMLElement.prototype, 'scrollIntoView')
        .mockImplementation(() => {});

      addSourceLocal(makeSource({ id: 'src-001' }));

      const { container } = render(SourcesRail);
      await tick();

      focusSource('src-001');
      await tick();
      await tick();

      const li = () => container.querySelector('[data-source-id="src-001"]') as HTMLElement;
      expect(li().getAttribute('data-pulsing')).toBe('true');

      vi.advanceTimersByTime(PULSE_MS);
      await tick();

      expect(li().getAttribute('data-pulsing')).toBe('false');

      scrollSpy.mockRestore();
    } finally {
      vi.useRealTimers();
    }
  });

  it('collapsed rail: focus expands it first, then scrolls+pulses the row exactly once (AC6)', async () => {
    const scrollSpy = vi
      .spyOn(HTMLElement.prototype, 'scrollIntoView')
      .mockImplementation(() => {});

    // Start collapsed: the reactive $state-backed mock means the component mounts
    // the collapsed rail (no source <li>s), so nothing is scrollable until expand.
    mockNotebookStore._setRightRailCollapsed(true);
    addSourceLocal(makeSource({ id: 'src-001' }));

    const { container } = render(SourcesRail);
    await tick();
    // Collapsed rail has no source list to scroll into.
    expect(container.querySelector('[data-source-id="src-001"]')).toBeNull();
    expect(scrollSpy).not.toHaveBeenCalled();

    focusSource('src-001');
    await tick();
    await tick();

    // (a) The rail expanded.
    expect(mockNotebookStore.rightRailCollapsed).toBe(false);

    const li = container.querySelector('[data-source-id="src-001"]') as HTMLElement;
    expect(li).not.toBeNull();

    // (b) scrollIntoView fired on the target <li> — exactly once (no double-scroll
    // from the untracked collapsed read re-triggering the effect on expand).
    expect(scrollSpy).toHaveBeenCalledTimes(1);
    expect(scrollSpy.mock.instances[0]).toBe(li);

    // (c) The target gained the pulse state.
    expect(li.getAttribute('data-pulsing')).toBe('true');

    scrollSpy.mockRestore();
  });
});
