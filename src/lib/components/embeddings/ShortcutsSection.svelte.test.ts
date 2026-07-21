import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import { SHORTCUTS } from '$lib/shortcuts/registry.js';
import ShortcutsSection from './ShortcutsSection.svelte';

describe('ShortcutsSection', () => {
  it('renders a heading for every group present in the registry', () => {
    render(ShortcutsSection);

    for (const group of ['Global', 'Chat', 'Audio player']) {
      expect(screen.getByText(group)).toBeInTheDocument();
    }
  });

  it('renders representative keys from each group', () => {
    render(ShortcutsSection);

    expect(screen.getByText('⌘K')).toBeInTheDocument();
    expect(screen.getByText('Space')).toBeInTheDocument();
    expect(screen.getByText('J')).toBeInTheDocument();
    expect(screen.getByText('L')).toBeInTheDocument();
  });

  it('renders exactly one row per registry entry (drift guard)', () => {
    const { container } = render(ShortcutsSection);

    const rows = container.querySelectorAll('[data-shortcut-row]');
    expect(rows.length).toBe(SHORTCUTS.length);
  });
});
