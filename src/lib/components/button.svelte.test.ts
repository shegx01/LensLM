import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Host from './button-test-host.svelte';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('Button', () => {
  it('default variant carries the bg-primary token class', () => {
    render(Host, { props: { label: 'Save' } });
    const btn = screen.getByRole('button', { name: 'Save' });
    // Token-wired: the default variant must resolve to the semantic primary token.
    expect(btn.className).toContain('bg-primary');
    expect(btn.className).toContain('text-primary-foreground');
  });

  it('fires onclick', async () => {
    const onclick = vi.fn();
    render(Host, { props: { label: 'Click', onclick } });
    await fireEvent.click(screen.getByRole('button', { name: 'Click' }));
    expect(onclick).toHaveBeenCalledOnce();
  });

  it('does NOT use a faux font-medium weight', () => {
    render(Host, { props: { label: 'Weight' } });
    const btn = screen.getByRole('button', { name: 'Weight' });
    expect(btn.className).not.toContain('font-medium');
    expect(btn.className).not.toContain('font-semibold');
  });
});
