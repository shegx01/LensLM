import { render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it } from 'vitest';
import { notebookStore } from '$lib/notebooks/index.js';
import { resetNotebookStore } from '$lib/notebooks/notebooks-state.svelte.js';
import PreferencesShell from './PreferencesShell.svelte';

afterEach(() => {
  resetNotebookStore();
});

describe('PreferencesShell deep-link (AC-7)', () => {
  it('honors settingsSection = "ai" and renders the AI Model panel', async () => {
    notebookStore.openSettings('ai');
    render(PreferencesShell);
    // The AI Model panel composes the Providers + Active model sections.
    expect(await screen.findByRole('heading', { name: 'Providers' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Active model' })).toBeInTheDocument();
  });

  it('defaults to Embeddings when no section is requested', async () => {
    notebookStore.openSettings();
    render(PreferencesShell);
    expect(await screen.findByRole('heading', { name: 'Embeddings' })).toBeInTheDocument();
  });

  it.each([
    ['storage', 'Storage'],
    ['privacy', 'Privacy'],
    ['shortcuts', 'Shortcuts']
  ])('renders the %s panel when deep-linked (issue #32)', async (section, heading) => {
    notebookStore.openSettings(section);
    render(PreferencesShell);
    expect(await screen.findByRole('heading', { name: heading })).toBeInTheDocument();
  });
});
