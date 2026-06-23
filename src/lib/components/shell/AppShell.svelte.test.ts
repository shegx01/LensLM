import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import AppShell from './AppShell.svelte';

describe('AppShell.svelte', () => {
  it('renders the three structural regions as skeletal placeholders', () => {
    render(AppShell);
    // Left rail (M3 seam), right rail (M4 seam), and centre workspace placeholder.
    expect(screen.getByText('Notebooks')).toBeInTheDocument();
    expect(screen.getByText(/sources & studio/i)).toBeInTheDocument();
    expect(screen.getByText('Your workspace')).toBeInTheDocument();
    expect(screen.getByText(/select or create a notebook/i)).toBeInTheDocument();
  });

  it('uses semantic landmarks for the regions', () => {
    const { container } = render(AppShell);
    // Two <aside> rails + one <main> workspace.
    expect(container.querySelectorAll('aside')).toHaveLength(2);
    expect(container.querySelector('main')).not.toBeNull();
  });
});
