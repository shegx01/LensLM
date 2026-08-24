import { render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AppShell from './AppShell.svelte';
import { notebookStore, resetNotebookStore } from '$lib/notebooks/notebooks-state.svelte.js';
import { listTrashed } from '$lib/notebooks/ipc.js';
import { listTrashedSources } from '$lib/sources/ipc.js';
import { invoke } from '@tauri-apps/api/core';
import { setPlatform } from '$lib/shortcuts/platform.js';
import type { ActionId } from '$lib/shortcuts/registry.js';

// AppShell mounts NotebooksSidebar (loads notebooks via the store) + CommandPalette
// + NotebookCreateDialog. Mock the IPC layer so the store's loadNotebooks() resolves
// without a Tauri host, and stub @tauri-apps/api/core (isTauri/invoke).
vi.mock('$lib/notebooks/ipc.js', () => ({
  listNotebooks: vi.fn().mockResolvedValue([]),
  createNotebook: vi.fn(),
  renameNotebook: vi.fn(),
  trashNotebook: vi.fn(),
  restoreNotebook: vi.fn(),
  listTrashed: vi.fn().mockResolvedValue([]),
  purgeNotebook: vi.fn(),
  touchNotebookActivity: vi.fn().mockResolvedValue(undefined)
}));

vi.mock('$lib/sources/ipc.js', () => ({
  listSources: vi.fn().mockResolvedValue([]),
  addTextSource: vi
    .fn()
    .mockResolvedValue({ source: { id: 'src-new', status: 'pending' }, wasExisting: false }),
  addFileSource: vi.fn(),
  ingestSource: vi.fn(),
  setSourceSelected: vi.fn(),
  listTrashedSources: vi.fn().mockResolvedValue([]),
  purgeSource: vi.fn(),
  restoreSource: vi.fn()
}));

const mockIsTauri = vi.fn(() => false);

vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => mockIsTauri(),
  invoke: vi.fn()
}));

// The `isTauri` mock above defaults to false, so the real store's load() early-returns and
// `mockIPC` can never feed it a keymap — inject one through the store module instead.
let mockKeymap: Partial<Record<ActionId, string>> = {};
const ensureLoaded = vi.fn(async () => {});

vi.mock('$lib/models/app-config.svelte.js', () => ({
  appConfigStore: {
    get models() {
      return [];
    },
    get enrichment() {
      return { enabled: false, coref_strategy: 'llm_inline', cloud_consent: false };
    },
    get asr() {
      return null;
    },
    get audioCloudConsent() {
      return false;
    },
    get ttsCloudConsent() {
      return false;
    },
    get keymap() {
      return mockKeymap;
    },
    get loadError() {
      return null;
    },
    get staleError() {
      return null;
    },
    get persistError() {
      return null;
    }
  },
  ensureLoaded: () => ensureLoaded(),
  refreshConfig: vi.fn(async () => {}),
  persist: vi.fn(async () => {}),
  resetConfig: vi.fn()
}));

// Left in place across tests it would keep document.activeElement typing-shaped and
// silently arm the open-guard in every later case — afterEach removes it.
function focusTypingTarget(): HTMLInputElement {
  const input = document.createElement('input');
  input.dataset.typingFixture = '';
  document.body.appendChild(input);
  input.focus();
  return input;
}

function pressKey(init: KeyboardEventInit): KeyboardEvent {
  // cancelable+bubbles are load-bearing: `defaultPrevented` is permanently false on a
  // non-cancelable event, which would make the preventDefault assertions vacuous.
  const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
  window.dispatchEvent(event);
  return event;
}

// Minimal NotebookSummary fixture.
function makeNotebook(id: string): import('$lib/notebooks/types.js').NotebookSummary {
  return {
    id,
    title: `Notebook ${id}`,
    description: null,
    focus_mode: null,
    embedding_model: null,
    embedding_backend: null,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    trashed_at: null,
    last_activity_at: '2026-01-01T00:00:00Z',
    graph_retrieval_enabled: null,
    source_count: 0
  };
}

// Minimal AppConfig fixture — only the fields the auto-select path reads.
function makeConfig(
  reopenLastNotebook: boolean | undefined = true
): Partial<import('$lib/theme/types.js').AppConfig> {
  return {
    user_name: 'Test User',
    reopen_last_notebook: reopenLastNotebook
  };
}

beforeEach(async () => {
  vi.clearAllMocks();
  resetNotebookStore();
  mockKeymap = {};
  // happy-dom builds its userAgent from process.platform, so ambient detection resolves
  // to linux even on macOS — every modifier assertion here needs it pinned.
  setPlatform('darwin');
  mockIsTauri.mockReturnValue(false);
  const { listNotebooks } = await import('$lib/notebooks/ipc.js');
  vi.mocked(listNotebooks).mockResolvedValue([]);
  // AppShell auto-selecting a notebook mounts ChatPane, which hydrates via
  // `list_chat_messages`/`list_notes` through this same mocked `invoke` — keep
  // them answering `[]` regardless of each test's `get_config` override below.
  vi.mocked(invoke).mockImplementation((cmd: string) => {
    if (cmd === 'list_chat_messages' || cmd === 'list_notes') return Promise.resolve([]);
    return Promise.resolve(makeConfig(true));
  });
});

afterEach(() => {
  resetNotebookStore();
  setPlatform(null);
  for (const el of document.querySelectorAll('input[data-typing-fixture]')) el.remove();
});

describe('AppShell.svelte', () => {
  it('renders the sidebar, the centre empty state, and the right rail', async () => {
    render(AppShell);
    expect(screen.getByText('Notebooks')).toBeInTheDocument();
    expect(screen.getByLabelText(/search notebooks/i)).toBeInTheDocument();
    expect(screen.getByText('Sources')).toBeInTheDocument();
    await vi.waitFor(() => {
      expect(screen.getByText('Your workspace')).toBeInTheDocument();
      expect(screen.getByText(/select or create a notebook/i)).toBeInTheDocument();
    });
  });

  it('uses semantic landmarks for the regions', () => {
    const { container } = render(AppShell);
    expect(container.querySelectorAll('aside')).toHaveLength(2);
    expect(container.querySelector('main')).not.toBeNull();
  });

  it('grid uses the expanded left column when sidebar is not collapsed', () => {
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    expect(grid.className).toContain('grid-cols-[256px_1fr_320px]');
  });

  it('grid uses the 104px collapsed left column when sidebarCollapsed is true', async () => {
    notebookStore.sidebarCollapsed = true;
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    expect(grid.className).toContain('grid-cols-[104px_1fr_320px]');
  });

  it('collapsed rail shows the icon-only layout (no hover behaviour)', () => {
    notebookStore.sidebarCollapsed = true;
    render(AppShell);
    expect(screen.getByRole('button', { name: 'Expand sidebar' })).toBeInTheDocument();
    expect(screen.queryByText('Notebooks')).not.toBeInTheDocument();
  });

  it('the rail is always a normal-flow aside (no overlay / no hover flyout)', () => {
    notebookStore.sidebarCollapsed = true;
    const { container } = render(AppShell);
    const rail = container.querySelector('[data-sidebar-rail]') as HTMLElement;
    expect(rail.tagName).toBe('ASIDE');
    expect(rail.className).not.toContain('absolute');
    expect(container.querySelector('[data-sidebar-flyout]')).toBeNull();
  });

  it('expanded sidebar renders the full layout in normal flow', () => {
    notebookStore.sidebarCollapsed = false;
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    expect(grid.className).toContain('grid-cols-[256px_1fr_320px]');
    expect(screen.getByText('Notebooks')).toBeInTheDocument();
  });

  it('grid uses the 104px collapsed RIGHT column (matches left) when rightRailCollapsed is true', () => {
    notebookStore.rightRailCollapsed = true;
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    expect(grid.className).toContain('grid-cols-[256px_1fr_104px]');
  });

  it('grid uses the 320px expanded RIGHT column by default', () => {
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    expect(grid.className).toContain('grid-cols-[256px_1fr_320px]');
  });

  it('both rails collapsed yields the symmetric 104px/104px grid', () => {
    notebookStore.sidebarCollapsed = true;
    notebookStore.rightRailCollapsed = true;
    const { container } = render(AppShell);
    const grid = container.querySelector('div.grid') as HTMLElement;
    expect(grid.className).toContain('grid-cols-[104px_1fr_104px]');
  });

  it('onMount triggers listTrashed and listTrashedSources so badge counts load at startup', async () => {
    render(AppShell);
    await vi.waitFor(() => {
      expect(vi.mocked(listTrashed)).toHaveBeenCalled();
      expect(vi.mocked(listTrashedSources)).toHaveBeenCalled();
    });
  });

  it('auto-selects the first (MRU) notebook when reopen_last_notebook is true and notebooks exist', async () => {
    mockIsTauri.mockReturnValue(true);
    const { listNotebooks } = await import('$lib/notebooks/ipc.js');
    vi.mocked(listNotebooks).mockResolvedValue([makeNotebook('nb-1'), makeNotebook('nb-2')]);
    vi.mocked(invoke).mockImplementation((cmd: string) =>
      cmd === 'list_chat_messages' || cmd === 'list_notes'
        ? Promise.resolve([])
        : Promise.resolve(makeConfig(true))
    );

    render(AppShell);

    await vi.waitFor(() => {
      expect(screen.queryByText(/select or create a notebook/i)).not.toBeInTheDocument();
    });
  });

  it('does NOT auto-select and shows empty state when reopen_last_notebook is false', async () => {
    mockIsTauri.mockReturnValue(true);
    const { listNotebooks } = await import('$lib/notebooks/ipc.js');
    vi.mocked(listNotebooks).mockResolvedValue([makeNotebook('nb-1')]);
    vi.mocked(invoke).mockImplementation((cmd: string) =>
      cmd === 'list_chat_messages' || cmd === 'list_notes'
        ? Promise.resolve([])
        : Promise.resolve(makeConfig(false))
    );

    render(AppShell);

    await vi.waitFor(() => {
      expect(screen.getByText(/select or create a notebook/i)).toBeInTheDocument();
    });
  });

  it('does NOT auto-select when notebook list is empty, and empty state is shown', async () => {
    mockIsTauri.mockReturnValue(true);
    vi.mocked(invoke).mockImplementation((cmd: string) =>
      cmd === 'list_chat_messages' || cmd === 'list_notes'
        ? Promise.resolve([])
        : Promise.resolve(makeConfig(true))
    );

    render(AppShell);

    await vi.waitFor(() => {
      expect(screen.getByText(/select or create a notebook/i)).toBeInTheDocument();
    });
  });

  it('auto-selects despite get_config rejection — falls back to default-on behavior', async () => {
    mockIsTauri.mockReturnValue(true);
    const { listNotebooks } = await import('$lib/notebooks/ipc.js');
    vi.mocked(listNotebooks).mockResolvedValue([makeNotebook('nb-1'), makeNotebook('nb-2')]);
    vi.mocked(invoke).mockImplementation((cmd: string) =>
      cmd === 'list_chat_messages' || cmd === 'list_notes'
        ? Promise.resolve([])
        : Promise.reject(new Error('config read failed'))
    );

    render(AppShell);

    await vi.waitFor(() => {
      expect(notebookStore.activeNotebookId).toBe('nb-1');
    });
  });
});

describe('AppShell keyboard dispatch', () => {
  it('Meta+k on darwin toggles the palette open, then closed', () => {
    render(AppShell);

    const open = pressKey({ key: 'k', metaKey: true });
    expect(notebookStore.paletteOpen).toBe(true);
    expect(open.defaultPrevented).toBe(true);

    const close = pressKey({ key: 'k', metaKey: true });
    expect(notebookStore.paletteOpen).toBe(false);
    expect(close.defaultPrevented).toBe(true);
  });

  it('Ctrl+k on linux toggles the palette open, then closed', () => {
    setPlatform('linux');
    render(AppShell);

    pressKey({ key: 'k', ctrlKey: true });
    expect(notebookStore.paletteOpen).toBe(true);

    pressKey({ key: 'k', ctrlKey: true });
    expect(notebookStore.paletteOpen).toBe(false);
  });

  it('Meta+k on linux does nothing', () => {
    setPlatform('linux');
    render(AppShell);

    const event = pressKey({ key: 'k', metaKey: true });

    expect(notebookStore.paletteOpen).toBe(false);
    expect(event.defaultPrevented).toBe(false);
  });

  it('the typing guard blocks the open but never the close', () => {
    render(AppShell);
    focusTypingTarget();

    pressKey({ key: 'k', metaKey: true });
    expect(notebookStore.paletteOpen).toBe(false);

    notebookStore.paletteOpen = true;
    pressKey({ key: 'k', metaKey: true });
    expect(notebookStore.paletteOpen).toBe(false);
  });

  it('leaves defaultPrevented false when the typing guard suppresses the open', () => {
    render(AppShell);
    focusTypingTarget();

    const event = pressKey({ key: 'k', metaKey: true });

    expect(notebookStore.paletteOpen).toBe(false);
    expect(event.defaultPrevented).toBe(false);
  });

  it('a bare-letter override still types into an input instead of being swallowed', () => {
    mockKeymap = { 'palette.toggle': 'P' };
    render(AppShell);
    focusTypingTarget();

    const event = pressKey({ key: 'p' });

    expect(notebookStore.paletteOpen).toBe(false);
    expect(event.defaultPrevented).toBe(false);
  });

  it('a keymap override fires instead of the default binding', () => {
    mockKeymap = { 'palette.toggle': 'P' };
    render(AppShell);

    const overridden = pressKey({ key: 'p' });
    expect(notebookStore.paletteOpen).toBe(true);
    expect(overridden.defaultPrevented).toBe(true);

    notebookStore.paletteOpen = false;
    const stale = pressKey({ key: 'k', metaKey: true });
    expect(notebookStore.paletteOpen).toBe(false);
    expect(stale.defaultPrevented).toBe(false);
  });

  it('loads the keymap on mount so window-scope bindings work without the Settings panel', () => {
    render(AppShell);
    expect(ensureLoaded).toHaveBeenCalled();
  });
});
