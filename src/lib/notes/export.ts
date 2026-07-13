// Notes export (#25 C4): copy-all-to-clipboard + save-to-file (.md/.txt). Produces
// RAW markdown source (mermaid fences + `$$` math emitted as-is, not rendered) so
// the file is a faithful copy of `notes.content`.

import { isTauri } from '@tauri-apps/api/core';
import { writeText } from '@tauri-apps/plugin-clipboard-manager';
import { save } from '@tauri-apps/plugin-dialog';
import { writeTextFile } from '@tauri-apps/plugin-fs';
import { notebookStore } from '$lib/notebooks/notebooks-state.svelte.js';
import { notesStore } from './notes-state.svelte.js';
import type { Note } from './types.js';

/** One line per note: its content, followed by an italic source-title attribution if present. */
function formatNote(note: Note): string {
  const body = note.content.trim();
  if (note.source_title) return `${body}\n\n*Source: ${note.source_title}*`;
  return body;
}

/**
 * Concatenates a notebook's notes into a single markdown document: a title
 * heading, then `## KEY INSIGHTS` (chat-origin notes) and `## PERSONAL NOTES`
 * (manual notes), each preserving the store's pinned-first/newest-first order.
 * Raw markdown source — not rendered output.
 */
export function formatNotesMarkdown(notebookId: string): string {
  const title = notebookStore.notebooks.find((n) => n.id === notebookId)?.title ?? 'Notes';
  const sections: string[] = [`# ${title}`];

  const insightNotes = notesStore.groupedBySource(notebookId).flatMap((g) => g.notes);
  if (insightNotes.length > 0) {
    sections.push(['## KEY INSIGHTS', ...insightNotes.map(formatNote)].join('\n\n'));
  }

  const manualNotes = notesStore.manualNotes(notebookId);
  if (manualNotes.length > 0) {
    sections.push(['## PERSONAL NOTES', ...manualNotes.map(formatNote)].join('\n\n'));
  }

  return sections.join('\n\n');
}

/** Copies the notebook's formatted notes markdown to the clipboard. No-op outside Tauri. */
export async function copyAllNotes(notebookId: string): Promise<void> {
  if (!isTauri()) return;
  await writeText(formatNotesMarkdown(notebookId));
}

const EXPORT_FILTERS: Record<'md' | 'txt', { name: string; extensions: string[] }> = {
  md: { name: 'Markdown', extensions: ['md'] },
  txt: { name: 'Text', extensions: ['txt'] }
};

/**
 * Opens a save dialog and writes the notebook's formatted notes markdown to the
 * chosen `.md`/`.txt` path. No-op outside Tauri or if the user cancels the dialog.
 */
export async function exportNotesToFile(notebookId: string, ext: 'md' | 'txt'): Promise<void> {
  if (!isTauri()) return;
  const title = notebookStore.notebooks.find((n) => n.id === notebookId)?.title ?? 'Notes';
  const path = await save({
    defaultPath: `${title}.${ext}`,
    filters: [EXPORT_FILTERS[ext]]
  });
  if (!path) return;
  await writeTextFile(path, formatNotesMarkdown(notebookId));
}
