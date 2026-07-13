// Component tests for NotesExportMenu: renders the trigger + menu items and wires
// them to the export actions. Hidden outside Tauri or when there are no notes.

import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const isTauriMock = vi.fn(() => true);
vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => isTauriMock()
}));

const copyAllNotesMock = vi.fn().mockResolvedValue(undefined);
const exportNotesToFileMock = vi.fn().mockResolvedValue(undefined);
vi.mock('$lib/notes/export.js', () => ({
  copyAllNotes: (...args: unknown[]) => copyAllNotesMock(...args),
  exportNotesToFile: (...args: unknown[]) => exportNotesToFileMock(...args)
}));

import NotesExportMenu from './NotesExportMenu.svelte';

beforeEach(() => {
  vi.clearAllMocks();
  isTauriMock.mockReturnValue(true);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('NotesExportMenu', () => {
  it('renders the export trigger when there are notes and running under Tauri', () => {
    render(NotesExportMenu, { props: { notebookId: 'nb-1', hasNotes: true } });
    expect(screen.getByRole('button', { name: 'Export notes' })).toBeInTheDocument();
  });

  it('hides the trigger when there are no notes', () => {
    render(NotesExportMenu, { props: { notebookId: 'nb-1', hasNotes: false } });
    expect(screen.queryByRole('button', { name: 'Export notes' })).not.toBeInTheDocument();
  });

  it('hides the trigger outside Tauri', () => {
    isTauriMock.mockReturnValue(false);
    render(NotesExportMenu, { props: { notebookId: 'nb-1', hasNotes: true } });
    expect(screen.queryByRole('button', { name: 'Export notes' })).not.toBeInTheDocument();
  });

  it('wires "Copy all notes" to copyAllNotes', async () => {
    render(NotesExportMenu, { props: { notebookId: 'nb-1', hasNotes: true } });
    await fireEvent.click(screen.getByRole('button', { name: 'Export notes' }));
    await fireEvent.click(await screen.findByText('Copy all notes'));
    expect(copyAllNotesMock).toHaveBeenCalledWith('nb-1');
  });

  it('wires "Export as .md" to exportNotesToFile with ext "md"', async () => {
    render(NotesExportMenu, { props: { notebookId: 'nb-1', hasNotes: true } });
    await fireEvent.click(screen.getByRole('button', { name: 'Export notes' }));
    await fireEvent.click(await screen.findByText('Export as .md'));
    expect(exportNotesToFileMock).toHaveBeenCalledWith('nb-1', 'md');
  });

  it('wires "Export as .txt" to exportNotesToFile with ext "txt"', async () => {
    render(NotesExportMenu, { props: { notebookId: 'nb-1', hasNotes: true } });
    await fireEvent.click(screen.getByRole('button', { name: 'Export notes' }));
    await fireEvent.click(await screen.findByText('Export as .txt'));
    expect(exportNotesToFileMock).toHaveBeenCalledWith('nb-1', 'txt');
  });
});
