import { render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { setPlatform } from '$lib/shortcuts/platform.js';
import { ROWS } from '$lib/shortcuts/registry.js';
import ShortcutsSection from './ShortcutsSection.svelte';

describe('ShortcutsSection', () => {
  // Chips are platform-rendered; the frontend CI job runs on ubuntu, so the
  // expected glyphs are only stable with the platform pinned.
  beforeEach(() => setPlatform('darwin'));
  afterEach(() => setPlatform(null));

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

  it('renders exactly one row per display row (drift guard)', () => {
    const { container } = render(ShortcutsSection);

    const rows = container.querySelectorAll('[data-shortcut-row]');
    expect(rows.length).toBe(ROWS.length);
  });
});
