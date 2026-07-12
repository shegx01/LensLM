import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Scrubber from './Scrubber.svelte';

const items = [
  { id: 'a', label: 'First item' },
  { id: 'b', label: 'Second item' }
];

describe('Scrubber', () => {
  it('renders one jump target per item with its label', () => {
    render(Scrubber, { items, activeId: 'a', onjump: vi.fn() });
    expect(screen.getByRole('button', { name: 'Jump to: First item' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Jump to: Second item' })).toBeInTheDocument();
  });

  it('marks the active item with aria-current', () => {
    render(Scrubber, { items, activeId: 'b', onjump: vi.fn() });
    expect(screen.getByRole('button', { name: 'Jump to: Second item' })).toHaveAttribute(
      'aria-current',
      'true'
    );
    expect(screen.getByRole('button', { name: 'Jump to: First item' })).not.toHaveAttribute(
      'aria-current'
    );
  });

  it('fires onjump with the item id when clicked', async () => {
    const onjump = vi.fn();
    render(Scrubber, { items, activeId: 'a', onjump });
    await fireEvent.click(screen.getByRole('button', { name: 'Jump to: Second item' }));
    expect(onjump).toHaveBeenCalledWith('b');
  });

  it('uses the provided ariaLabel on the nav landmark', () => {
    render(Scrubber, { items, activeId: 'a', onjump: vi.fn(), ariaLabel: 'Notes timeline' });
    expect(screen.getByRole('navigation', { name: 'Notes timeline' })).toBeInTheDocument();
  });

  it('renders nothing for a single item (nothing to navigate)', () => {
    render(Scrubber, { items: [{ id: 'a', label: 'only one' }], activeId: 'a', onjump: vi.fn() });
    expect(screen.queryByRole('navigation')).toBeNull();
    expect(screen.queryByRole('button', { name: /^Jump to:/ })).toBeNull();
  });
});
