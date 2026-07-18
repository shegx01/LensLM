import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import Check from '@lucide/svelte/icons/check';
import SystemCheckTile from './SystemCheckTile.svelte';

function badge(container: HTMLElement): HTMLElement {
  const el = container.querySelector('span[aria-hidden="true"]');
  if (!el) throw new Error('icon badge not found');
  return el as HTMLElement;
}

describe('SystemCheckTile', () => {
  it('renders the two-line header (title + subtitle)', () => {
    render(SystemCheckTile, {
      props: { icon: Check, title: 'Local AI', subtitle: 'Runs privately on your machine' }
    });
    expect(screen.getByText('Local AI')).toBeInTheDocument();
    expect(screen.getByText('Runs privately on your machine')).toBeInTheDocument();
  });

  it('applies the default primary badge tint', () => {
    const { container } = render(SystemCheckTile, {
      props: { icon: Check, title: 'Local AI', subtitle: 'sub' }
    });
    const b = badge(container);
    expect(b.className).toContain('bg-primary/15');
    expect(b.className).toContain('text-primary');
  });

  it('honors a custom badge tint', () => {
    const { container } = render(SystemCheckTile, {
      props: {
        icon: Check,
        title: 'Embedding model',
        subtitle: 'sub',
        badgeClass: 'bg-destructive/15 text-destructive'
      }
    });
    const b = badge(container);
    expect(b.className).toContain('bg-destructive/15');
    expect(b.className).toContain('text-destructive');
  });

  it('uses token-only elevated chrome (shadow-tile + concentric radius)', () => {
    const { container } = render(SystemCheckTile, {
      props: { icon: Check, title: 'Local AI', subtitle: 'sub' }
    });
    const tile = container.querySelector('[class*="shadow-"]') as HTMLElement;
    expect(tile.className).toContain('shadow-[var(--shadow-tile)]');
    expect(tile.className).toContain('rounded-[13px]');
  });
});
