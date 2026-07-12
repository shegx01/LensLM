import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import { Switch } from './index.js';

// The track/thumb visibility styling keys off `data-[state=checked|unchecked]`.
// bits-ui emits `data-state`, NOT bare `data-checked`/`data-unchecked` — if a
// bits-ui bump ever changes that, this fails and the switch.svelte variants must
// be updated in lockstep (otherwise the switch renders invisible).
describe('bits-ui Switch data-state contract', () => {
  it('track exposes data-state and no bare data-checked/data-unchecked', () => {
    render(Switch, { props: { checked: true, 'aria-label': 'probe' } });
    const track = screen.getByRole('switch', { name: 'probe' });

    expect(track.getAttribute('data-state')).toBe('checked');
    expect(track.hasAttribute('data-checked')).toBe(false);
    expect(track.hasAttribute('data-unchecked')).toBe(false);
  });

  it('thumb exposes data-state so its translate variant matches', () => {
    const { container } = render(Switch, { props: { checked: true, 'aria-label': 'probe2' } });
    const thumb = container.querySelector('[data-slot="switch-thumb"]');

    expect(thumb).not.toBeNull();
    expect(thumb?.getAttribute('data-state')).toBe('checked');
  });
});
