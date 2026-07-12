// CollapsibleText: clamps rendered HTML, then a toggle expands/collapses it.
// happy-dom reports zero layout heights, so overflow is simulated by stubbing
// scrollHeight/clientHeight — the component keys the toggle's visibility on
// scrollHeight > clientHeight.

import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import CollapsibleText from './CollapsibleText.svelte';

let scrollHeightSpy: ReturnType<typeof vi.spyOn> | null = null;
let clientHeightSpy: ReturnType<typeof vi.spyOn> | null = null;

function simulateOverflow(scroll: number, client: number): void {
  scrollHeightSpy = vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockReturnValue(scroll);
  clientHeightSpy = vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(client);
}

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', undefined);
});

afterEach(() => {
  scrollHeightSpy?.mockRestore();
  clientHeightSpy?.mockRestore();
  scrollHeightSpy = null;
  clientHeightSpy = null;
  vi.unstubAllGlobals();
});

describe('CollapsibleText', () => {
  it('renders the provided HTML', () => {
    const { container } = render(CollapsibleText, {
      props: { html: '<p>hello world</p>' }
    });
    expect(container.querySelector('p')?.textContent).toBe('hello world');
  });

  it('clamps by default and shows no toggle when content fits', () => {
    simulateOverflow(40, 40);
    const { container } = render(CollapsibleText, { props: { html: '<p>short</p>' } });
    expect(container.querySelector('.clamped')).not.toBeNull();
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('shows a Show more toggle when content overflows, and expands/collapses it', async () => {
    simulateOverflow(200, 80);
    const { container } = render(CollapsibleText, {
      props: { html: '<p>a very long note that overflows the clamp</p>' }
    });

    const toggle = screen.getByRole('button', { name: /show more/i });
    expect(container.querySelector('.clamped')).not.toBeNull();

    await fireEvent.click(toggle);
    expect(container.querySelector('.clamped')).toBeNull();
    expect(screen.getByRole('button', { name: /show less/i })).toBeInTheDocument();

    await fireEvent.click(screen.getByRole('button', { name: /show less/i }));
    expect(container.querySelector('.clamped')).not.toBeNull();
    expect(screen.getByRole('button', { name: /show more/i })).toBeInTheDocument();
  });
});
