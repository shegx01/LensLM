import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import Page from './+page.svelte';

describe('+page.svelte', () => {
  it('renders the app shell (not the old Hello World placeholder)', () => {
    render(Page);
    // The shell replaced the Hello World landing.
    expect(screen.queryByRole('heading', { name: /hello world/i })).not.toBeInTheDocument();
    // Three structural regions are present as labelled placeholders.
    expect(screen.getByText('Notebooks')).toBeInTheDocument();
    expect(screen.getByText(/sources & studio/i)).toBeInTheDocument();
    expect(screen.getByText('Your workspace')).toBeInTheDocument();
  });
});
