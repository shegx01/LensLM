// Regression guards: card-title and dialog-title must NOT use font-medium or
// font-semibold. These were changed to font-bold (no-faux-600 rule). This test
// guards against `shadcn-svelte add --overwrite` silently reverting them.
import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import CardTitleHost from './__tests__/card-title-test-host.svelte';
import DialogTitleHost from './__tests__/dialog-title-test-host.svelte';

describe('CardTitle', () => {
  it('does NOT use a faux font-medium or font-semibold weight', () => {
    render(CardTitleHost, { props: { label: 'Test Card Title' } });
    const el = screen.getByText('Test Card Title');
    expect(el.className).not.toContain('font-medium');
    expect(el.className).not.toContain('font-semibold');
  });
});

describe('DialogTitle', () => {
  it('does NOT use a faux font-medium or font-semibold weight', () => {
    render(DialogTitleHost, { props: { label: 'Test Dialog Title' } });
    const el = screen.getByText('Test Dialog Title');
    expect(el.className).not.toContain('font-medium');
    expect(el.className).not.toContain('font-semibold');
  });
});
