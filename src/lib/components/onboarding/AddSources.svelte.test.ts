import { render, screen, waitFor, fireEvent } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AddSources from './AddSources.svelte';
import {
  draft,
  resetDraft,
  type RecentDocument
} from '$lib/components/onboarding/onboarding-state.svelte.js';
import type { AppConfig } from '$lib/theme/types.js';
import { baseAppConfig } from '$lib/test-fixtures.js';

// ── Fixtures ─────────────────────────────────────────────────────────────────

const SAMPLE_DOCS: RecentDocument[] = [
  { path: '/docs/report.pdf', name: 'report.pdf', ext: 'pdf', size: 512000, mtime: 0 },
  { path: '/docs/notes.md', name: 'notes.md', ext: 'md', size: 4096, mtime: 0 }
];

const MANY_DOCS: RecentDocument[] = [
  { path: '/docs/a.pdf', name: 'a.pdf', ext: 'pdf', size: 1000, mtime: 0 },
  { path: '/docs/b.pdf', name: 'b.pdf', ext: 'pdf', size: 1000, mtime: 0 },
  { path: '/docs/c.pdf', name: 'c.pdf', ext: 'pdf', size: 1000, mtime: 0 },
  { path: '/docs/d.pdf', name: 'd.pdf', ext: 'pdf', size: 1000, mtime: 0 },
  { path: '/docs/e.pdf', name: 'e.pdf', ext: 'pdf', size: 1000, mtime: 0 },
  { path: '/docs/f.pdf', name: 'f.pdf', ext: 'pdf', size: 1000, mtime: 0 }
];

function baseConfig(onboarding_complete = false): AppConfig {
  return baseAppConfig({ onboarding_complete });
}

// ── Mock @tauri-apps/plugin-dialog ────────────────────────────────────────────
// The dialog plugin is NOT proxied through invoke/mockIPC — it is a direct module
// import. We vi.mock it at the module level so the component's `browse()` handler
// receives a controllable stub.

vi.mock('@tauri-apps/plugin-dialog', () => ({
  open: vi.fn().mockResolvedValue(null)
}));

// ── Setup / teardown ─────────────────────────────────────────────────────────

beforeEach(() => {
  // Activate the Tauri path (isTauri() reads globalThis.isTauri).
  (globalThis as { isTauri?: boolean }).isTauri = true;
  // Always start with a clean draft so tests don't bleed state.
  resetDraft();
  draft.notebookId = 'nb-test-01';
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
  resetDraft();
  vi.clearAllMocks();
});

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('AddSources', () => {
  // ── Rendering ──────────────────────────────────────────────────────────────

  it('renders the "Add sources" title and subtitle', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return [];
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    expect(screen.getByText('Add sources')).toBeInTheDocument();
    expect(screen.getByText(/Attach documents, PDFs, or notes/i)).toBeInTheDocument();
  });

  it('renders the drop zone with "Drop files here" and browse link', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return [];
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    expect(screen.getByText('Drop files here')).toBeInTheDocument();
    expect(screen.getByText('browse')).toBeInTheDocument();
  });

  it('renders footer buttons: "Skip for now" and "Launch Lens"', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return [];
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    expect(screen.getByRole('button', { name: /skip for now/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /launch lens/i })).toBeInTheDocument();
  });

  // ── Suggestions ────────────────────────────────────────────────────────────

  it('hides the suggestions section when list_recent_documents returns []', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return [];
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    // Give mount a tick to resolve.
    await waitFor(() => {
      expect(screen.queryByText(/suggested from your library/i)).not.toBeInTheDocument();
    });
  });

  it('shows suggestion rows when list_recent_documents returns documents', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return SAMPLE_DOCS;
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    await waitFor(() => {
      expect(screen.getByText(/suggested from your library/i)).toBeInTheDocument();
    });
    expect(screen.getByText('report.pdf')).toBeInTheDocument();
    expect(screen.getByText('notes.md')).toBeInTheDocument();
  });

  it('hides the suggestions section when the IPC call throws', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') throw new Error('failed');
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    await waitFor(() => {
      expect(screen.queryByText(/suggested from your library/i)).not.toBeInTheDocument();
    });
  });

  it('caps suggestions at 4 rows even when more are returned', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return MANY_DOCS;
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    await waitFor(() => {
      expect(screen.getByText(/suggested from your library/i)).toBeInTheDocument();
    });
    // First 4 visible, 5th and 6th not rendered.
    expect(screen.getByText('a.pdf')).toBeInTheDocument();
    expect(screen.getByText('b.pdf')).toBeInTheDocument();
    expect(screen.getByText('c.pdf')).toBeInTheDocument();
    expect(screen.getByText('d.pdf')).toBeInTheDocument();
    expect(screen.queryByText('e.pdf')).not.toBeInTheDocument();
    expect(screen.queryByText('f.pdf')).not.toBeInTheDocument();
  });

  // ── Toggling suggestions ──────────────────────────────────────────────────

  it('toggling a suggestion adds it to selectedSources and shows ADDED FILES', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return SAMPLE_DOCS;
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    await waitFor(() => screen.getByText('report.pdf'));

    // Before any selection the count label is visible and ADDED FILES is hidden.
    expect(screen.getByText('No sources selected')).toBeInTheDocument();
    expect(screen.queryByText(/added files/i)).not.toBeInTheDocument();

    const reportRow = screen.getByRole('button', { name: /select report\.pdf/i });
    await fireEvent.click(reportRow);

    // After selection: ADDED FILES section appears; count label is suppressed.
    await waitFor(() => expect(screen.getByText(/added files/i)).toBeInTheDocument());
    expect(draft.selectedSources).toHaveLength(1);
    expect(draft.selectedSources[0].path).toBe('/docs/report.pdf');
  });

  it('toggling a selected suggestion removes it from selectedSources', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return SAMPLE_DOCS;
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    await waitFor(() => screen.getByText('report.pdf'));

    const reportRow = screen.getByRole('button', { name: /select report\.pdf/i });
    await fireEvent.click(reportRow);
    expect(draft.selectedSources).toHaveLength(1);

    // Row label flips to "Deselect" after selection.
    const deselect = screen.getByRole('button', { name: /deselect report\.pdf/i });
    await fireEvent.click(deselect);
    expect(draft.selectedSources).toHaveLength(0);
    expect(screen.getByText('No sources selected')).toBeInTheDocument();
  });

  it('selecting two suggestions shows both in ADDED FILES and hides count label', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return SAMPLE_DOCS;
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    await waitFor(() => screen.getByText('report.pdf'));

    await fireEvent.click(screen.getByRole('button', { name: /select report\.pdf/i }));
    await fireEvent.click(screen.getByRole('button', { name: /select notes\.md/i }));

    // Both files appear in ADDED FILES; the count label text is suppressed.
    await waitFor(() => expect(screen.getByText(/added files/i)).toBeInTheDocument());
    expect(draft.selectedSources).toHaveLength(2);
    // count label is gone while ADDED FILES is visible
    expect(screen.queryByText('2 sources selected')).not.toBeInTheDocument();
  });

  // ── ADDED FILES section ───────────────────────────────────────────────────

  it('hides "ADDED FILES" section when selectedSources is empty', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return [];
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    await waitFor(() => {
      expect(screen.queryByText(/added files/i)).not.toBeInTheDocument();
    });
  });

  it('shows "ADDED FILES" section after toggling a suggestion on', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return SAMPLE_DOCS;
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    await waitFor(() => screen.getByText('report.pdf'));

    await fireEvent.click(screen.getByRole('button', { name: /select report\.pdf/i }));

    await waitFor(() => {
      expect(screen.getByText(/added files/i)).toBeInTheDocument();
    });
    // The file name appears in ADDED FILES (multiple elements expected since it
    // also appears in Suggested — getAllByText is appropriate here).
    const instances = screen.getAllByText('report.pdf');
    expect(instances.length).toBeGreaterThanOrEqual(2);
  });

  it('delete button removes the file from selectedSources and hides section when empty', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return SAMPLE_DOCS;
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    await waitFor(() => screen.getByText('report.pdf'));

    // Add via toggle.
    await fireEvent.click(screen.getByRole('button', { name: /select report\.pdf/i }));
    await waitFor(() => expect(screen.getByText(/added files/i)).toBeInTheDocument());

    // Remove via delete button in ADDED FILES.
    const removeBtn = screen.getByRole('button', { name: /remove report\.pdf/i });
    await fireEvent.click(removeBtn);

    await waitFor(() => {
      expect(screen.queryByText(/added files/i)).not.toBeInTheDocument();
    });
    expect(draft.selectedSources).toHaveLength(0);
  });

  it('removing a file via delete button reverts suggestion row to unselected', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return SAMPLE_DOCS;
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });
    await waitFor(() => screen.getByText('report.pdf'));

    // Select suggestion.
    await fireEvent.click(screen.getByRole('button', { name: /select report\.pdf/i }));
    // Suggestion is now "Deselect" aria-label.
    expect(screen.getByRole('button', { name: /deselect report\.pdf/i })).toBeInTheDocument();

    // Remove via ADDED FILES delete.
    await fireEvent.click(screen.getByRole('button', { name: /remove report\.pdf/i }));

    // Suggestion row reverts to "Select".
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /select report\.pdf/i })).toBeInTheDocument();
    });
  });

  it('shows "+N more" when more than 4 files are added', async () => {
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return [];
    });

    // Pre-populate draft with 5 sources before rendering.
    draft.selectedSources = [
      { path: '/f/a.pdf', name: 'a.pdf', ext: 'pdf', size: 0, mtime: 0 },
      { path: '/f/b.pdf', name: 'b.pdf', ext: 'pdf', size: 0, mtime: 0 },
      { path: '/f/c.pdf', name: 'c.pdf', ext: 'pdf', size: 0, mtime: 0 },
      { path: '/f/d.pdf', name: 'd.pdf', ext: 'pdf', size: 0, mtime: 0 },
      { path: '/f/e.pdf', name: 'e.pdf', ext: 'pdf', size: 0, mtime: 0 }
    ];

    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });

    await waitFor(() => {
      expect(screen.getByText(/added files/i)).toBeInTheDocument();
    });
    // First 4 visible.
    expect(screen.getByText('a.pdf')).toBeInTheDocument();
    expect(screen.getByText('d.pdf')).toBeInTheDocument();
    // 5th is hidden, overflow label shown.
    expect(screen.queryByText('e.pdf')).not.toBeInTheDocument();
    expect(screen.getByText('+1 more')).toBeInTheDocument();
  });

  it('shows "ADDED FILES" section after browse() adds a file', async () => {
    const { open } = await import('@tauri-apps/plugin-dialog');
    vi.mocked(open).mockResolvedValueOnce(['/home/user/thesis.pdf']);

    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return [];
    });

    render(AddSources, { props: { oncomplete: vi.fn(), onback: vi.fn() } });

    const browseBtn = screen.getByRole('button', { name: /drop files here or click to browse/i });
    await fireEvent.click(browseBtn);

    await waitFor(() => {
      expect(screen.getByText(/added files/i)).toBeInTheDocument();
    });
    expect(screen.getByText('thesis.pdf')).toBeInTheDocument();
  });

  // ── Skip for now ───────────────────────────────────────────────────────────

  it('"Skip for now" calls completeOnboarding (set_config) and oncomplete without add_source', async () => {
    const oncomplete = vi.fn();
    let written: AppConfig | null = null;
    let addSourceCalled = false;

    mockIPC((cmd, args) => {
      if (cmd === 'list_recent_documents') return [];
      if (cmd === 'get_config') return baseConfig(false);
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return undefined;
      }
      if (cmd === 'add_source') {
        addSourceCalled = true;
        return undefined;
      }
    });

    render(AddSources, { props: { oncomplete, onback: vi.fn() } });
    const skipBtn = screen.getByRole('button', { name: /skip for now/i });
    await fireEvent.click(skipBtn);

    await waitFor(() => expect(oncomplete).toHaveBeenCalledOnce());
    expect(addSourceCalled).toBe(false);
    expect(written).not.toBeNull();
    expect((written as unknown as AppConfig).onboarding_complete).toBe(true);
  });

  it('"Skip for now" does not add any sources to draft', async () => {
    const oncomplete = vi.fn();
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return [];
      if (cmd === 'get_config') return baseConfig(false);
      if (cmd === 'set_config') return undefined;
    });

    render(AddSources, { props: { oncomplete, onback: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /skip for now/i }));
    await waitFor(() => expect(oncomplete).toHaveBeenCalledOnce());
    expect(draft.selectedSources).toHaveLength(0);
  });

  // ── Launch Lens ────────────────────────────────────────────────────────────

  it('"Launch Lens" with selected sources calls add_source then completes', async () => {
    const oncomplete = vi.fn();
    const addSourceCalls: unknown[] = [];
    let written: AppConfig | null = null;

    mockIPC((cmd, args) => {
      if (cmd === 'list_recent_documents') return SAMPLE_DOCS;
      if (cmd === 'get_config') return baseConfig(false);
      if (cmd === 'set_config') {
        written = (args as { config: AppConfig }).config;
        return undefined;
      }
      if (cmd === 'add_source') {
        addSourceCalls.push(args);
        return undefined;
      }
    });

    render(AddSources, { props: { oncomplete, onback: vi.fn() } });
    await waitFor(() => screen.getByText('report.pdf'));

    // Select one suggestion.
    await fireEvent.click(screen.getByRole('button', { name: /select report\.pdf/i }));

    await fireEvent.click(screen.getByRole('button', { name: /launch lens/i }));

    await waitFor(() => expect(oncomplete).toHaveBeenCalledOnce());
    expect(addSourceCalls).toHaveLength(1);
    expect(addSourceCalls[0]).toMatchObject({
      notebookId: 'nb-test-01',
      title: 'report.pdf',
      locator: '/docs/report.pdf'
    });
    expect((written as unknown as AppConfig).onboarding_complete).toBe(true);
  });

  it('retrying "Launch Lens" after a partial failure does NOT re-add already-added sources', async () => {
    const oncomplete = vi.fn();
    const addSourceLocators: string[] = [];
    // Fail the FIRST add_source for notes.md only; succeed on every later call.
    let notesAttempts = 0;

    mockIPC((cmd, args) => {
      if (cmd === 'list_recent_documents') return SAMPLE_DOCS;
      if (cmd === 'get_config') return baseConfig(false);
      if (cmd === 'set_config') return undefined;
      if (cmd === 'add_source') {
        const locator = (args as { locator: string }).locator;
        if (locator === '/docs/notes.md') {
          notesAttempts++;
          if (notesAttempts === 1) throw new Error('add_source failed');
        }
        addSourceLocators.push(locator);
        return undefined;
      }
    });

    render(AddSources, { props: { oncomplete, onback: vi.fn() } });
    await waitFor(() => screen.getByText('report.pdf'));

    // Select BOTH suggestions; report.pdf is inserted first, notes.md second.
    await fireEvent.click(screen.getByRole('button', { name: /select report\.pdf/i }));
    await fireEvent.click(screen.getByRole('button', { name: /select notes\.md/i }));

    // First attempt: report.pdf lands, notes.md throws → inline error, no oncomplete.
    await fireEvent.click(screen.getByRole('button', { name: /launch lens/i }));
    await waitFor(() => expect(screen.getByText(/could not save your setup/i)).toBeInTheDocument());
    expect(oncomplete).not.toHaveBeenCalled();
    expect(addSourceLocators).toEqual(['/docs/report.pdf']);

    // Retry: report.pdf is already added → skipped; only notes.md is inserted.
    await fireEvent.click(screen.getByRole('button', { name: /launch lens/i }));
    await waitFor(() => expect(oncomplete).toHaveBeenCalledOnce());
    expect(addSourceLocators).toEqual(['/docs/report.pdf', '/docs/notes.md']);
  });

  it('"Launch Lens" with no sources calls completeOnboarding without add_source', async () => {
    const oncomplete = vi.fn();
    let addSourceCalled = false;

    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return [];
      if (cmd === 'get_config') return baseConfig(false);
      if (cmd === 'set_config') return undefined;
      if (cmd === 'add_source') {
        addSourceCalled = true;
        return undefined;
      }
    });

    render(AddSources, { props: { oncomplete, onback: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /launch lens/i }));

    await waitFor(() => expect(oncomplete).toHaveBeenCalledOnce());
    expect(addSourceCalled).toBe(false);
  });

  it('shows inline error and does not advance when set_config throws on Launch', async () => {
    const oncomplete = vi.fn();

    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return [];
      if (cmd === 'get_config') return baseConfig(false);
      if (cmd === 'set_config') throw new Error('disk full');
    });

    render(AddSources, { props: { oncomplete, onback: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /launch lens/i }));

    await waitFor(() => expect(screen.getByText(/could not save your setup/i)).toBeInTheDocument());
    expect(oncomplete).not.toHaveBeenCalled();
  });

  it('shows inline error and does not advance when set_config throws on Skip', async () => {
    const oncomplete = vi.fn();

    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return [];
      if (cmd === 'get_config') return baseConfig(false);
      if (cmd === 'set_config') throw new Error('network error');
    });

    render(AddSources, { props: { oncomplete, onback: vi.fn() } });
    await fireEvent.click(screen.getByRole('button', { name: /skip for now/i }));

    await waitFor(() => expect(screen.getByText(/could not save your setup/i)).toBeInTheDocument());
    expect(oncomplete).not.toHaveBeenCalled();
  });

  // ── Back button ────────────────────────────────────────────────────────────

  it('Back button fires onback', async () => {
    const onback = vi.fn();
    mockIPC((cmd) => {
      if (cmd === 'list_recent_documents') return [];
    });
    render(AddSources, { props: { oncomplete: vi.fn(), onback } });
    await fireEvent.click(screen.getByRole('button', { name: /back/i }));
    expect(onback).toHaveBeenCalledOnce();
  });

  // ── Non-Tauri guard ────────────────────────────────────────────────────────

  it('does not call list_recent_documents and skips add_source when not in Tauri', async () => {
    delete (globalThis as { isTauri?: boolean }).isTauri;
    const oncomplete = vi.fn();
    let ipcCalled = false;

    mockIPC(() => {
      ipcCalled = true;
      return undefined;
    });

    render(AddSources, { props: { oncomplete, onback: vi.fn() } });
    // No suggestions visible — list_recent_documents not called.
    await waitFor(() => {
      expect(screen.queryByText(/suggested from your library/i)).not.toBeInTheDocument();
    });

    // Skip still resolves (no-op updateConfig).
    await fireEvent.click(screen.getByRole('button', { name: /skip for now/i }));
    await waitFor(() => expect(oncomplete).toHaveBeenCalledOnce());
    expect(ipcCalled).toBe(false);
  });
});
