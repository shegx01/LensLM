import { render, screen, fireEvent } from '@testing-library/svelte';
import { createRawSnippet } from 'svelte';
import { describe, expect, it, vi } from 'vitest';
import Settings2 from '@lucide/svelte/icons/settings-2';
import SettingsShell, { type NavItem } from './SettingsShell.svelte';

const NAV: NavItem[] = [
  { id: 'one', label: 'One', icon: Settings2, stub: false },
  { id: 'two', label: 'Two', icon: Settings2, stub: false },
  { id: 'later', label: 'Later', icon: Settings2, stub: true }
];

const content = createRawSnippet((active: () => string) => ({
  render: () => `<div data-testid="content">active:${active()}</div>`
}));

describe('SettingsShell', () => {
  it('renders the nav items and marks the active one with aria-current', () => {
    render(SettingsShell, { nav: NAV, active: 'one', content });
    expect(screen.getByRole('button', { name: /one/i })).toHaveAttribute('aria-current', 'page');
    expect(screen.getByRole('button', { name: /two/i })).not.toHaveAttribute('aria-current');
    expect(screen.getByTestId('content')).toHaveTextContent('active:one');
  });

  it('renders a stub item with "Soon" that does not activate on click', async () => {
    render(SettingsShell, { nav: NAV, active: 'one', content });
    const later = screen.getByRole('button', { name: /later/i });
    expect(later).toHaveAttribute('aria-disabled', 'true');
    expect(later).toHaveTextContent('Soon');
    await fireEvent.click(later);
    expect(later).not.toHaveAttribute('aria-current', 'page');
    expect(screen.getByTestId('content')).toHaveTextContent('active:one');
  });

  it('activates a non-stub item and fires onSelect on click', async () => {
    const onSelect = vi.fn();
    render(SettingsShell, { nav: NAV, active: 'one', onSelect, content });
    await fireEvent.click(screen.getByRole('button', { name: /two/i }));
    expect(onSelect).toHaveBeenCalledWith('two');
    expect(screen.getByRole('button', { name: /two/i })).toHaveAttribute('aria-current', 'page');
  });

  it('omits the Back button when onBack is absent and renders it when provided', async () => {
    const { unmount } = render(SettingsShell, { nav: NAV, active: 'one', content });
    expect(screen.queryByRole('button', { name: /back/i })).not.toBeInTheDocument();
    unmount();

    const onBack = vi.fn();
    render(SettingsShell, { nav: NAV, active: 'one', onBack, content });
    const back = screen.getByRole('button', { name: /back/i });
    await fireEvent.click(back);
    expect(onBack).toHaveBeenCalledOnce();
  });

  it('omits the section-group heading when label is absent', () => {
    const { unmount } = render(SettingsShell, { nav: NAV, active: 'one', content });
    expect(screen.queryByText('Preferences')).not.toBeInTheDocument();
    unmount();

    render(SettingsShell, { nav: NAV, active: 'one', label: 'Preferences', content });
    expect(screen.getByText('Preferences')).toBeInTheDocument();
  });
});
