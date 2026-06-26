// AccountFooter component tests.
//
// Covers: renders initials from userName, footer menu opens on click,
// closes on outside click / focusout, "Settings" is disabled, "Sign out" is absent.

import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

// Stub ThemeCycleButton (avoids mode-watcher interaction in tests)
vi.mock('$lib/components/ThemeCycleButton.svelte', () => ({
  default: function ThemeCycleButtonStub() {}
}));

// Stub mode-watcher to prevent missing ESM module error in happy-dom
vi.mock('mode-watcher', () => ({
  userPrefersMode: { current: 'system' },
  setMode: vi.fn()
}));

vi.mock('$lib/theme/index.js', () => ({
  persistTheme: vi.fn()
}));

import AccountFooter from './AccountFooter.svelte';

afterEach(() => {
  vi.clearAllMocks();
});

describe('AccountFooter', () => {
  it('renders initials from a full name', () => {
    render(AccountFooter, { props: { userName: 'Jamie Doe' } });
    expect(screen.getByText('JD')).toBeInTheDocument();
  });

  it('renders initials from a single-word name', () => {
    render(AccountFooter, { props: { userName: 'Jamie' } });
    expect(screen.getByText('J')).toBeInTheDocument();
  });

  it('renders "?" for empty userName', () => {
    render(AccountFooter, { props: { userName: '' } });
    expect(screen.getByText('?')).toBeInTheDocument();
  });

  it('displays the userName in the trigger row', () => {
    render(AccountFooter, { props: { userName: 'Jamie Doe' } });
    expect(screen.getByText('Jamie Doe')).toBeInTheDocument();
  });

  it('popover menu is not visible initially', () => {
    render(AccountFooter, { props: { userName: 'Jamie' } });
    expect(screen.queryByRole('menu')).not.toBeInTheDocument();
  });

  it('clicking the trigger opens the popover menu', async () => {
    render(AccountFooter, { props: { userName: 'Jamie' } });
    const trigger = screen.getByRole('button', { name: /account menu/i });
    await fireEvent.click(trigger);
    await waitFor(() => expect(screen.getByRole('menu')).toBeInTheDocument());
  });

  it('Settings menu item is disabled/aria-disabled', async () => {
    render(AccountFooter, { props: { userName: 'Jamie' } });
    await fireEvent.click(screen.getByRole('button', { name: /account menu/i }));
    await waitFor(() => screen.getByRole('menu'));
    const settingsItem = screen.queryByText(/settings/i);
    expect(settingsItem).toBeInTheDocument();
    // The settings item wraps in a TooltipTrigger with disabled; check aria-disabled
    const disabledEl =
      screen.getByText(/settings/i).closest('[aria-disabled="true"]') ??
      screen.getByText(/settings/i).closest('[disabled]');
    expect(disabledEl).toBeTruthy();
  });

  it('does NOT render a "Sign out" menu item', async () => {
    render(AccountFooter, { props: { userName: 'Jamie' } });
    await fireEvent.click(screen.getByRole('button', { name: /account menu/i }));
    await waitFor(() => screen.getByRole('menu'));
    expect(screen.queryByText(/sign out/i)).not.toBeInTheDocument();
  });

  it('pressing Esc closes the menu', async () => {
    render(AccountFooter, { props: { userName: 'Jamie' } });
    const trigger = screen.getByRole('button', { name: /account menu/i });
    await fireEvent.click(trigger);
    await waitFor(() => screen.getByRole('menu'));
    await fireEvent.keyDown(document.body, { key: 'Escape' });
    // Menu close is driven by keydown on the container, not document.body;
    // fire it on the container element
    const container = screen.getByRole('menu').parentElement!;
    await fireEvent.keyDown(container, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('menu')).not.toBeInTheDocument());
  });

  it('closes the menu on an outside pointerdown', async () => {
    render(AccountFooter, { props: { userName: 'Jamie' } });
    await fireEvent.click(screen.getByRole('button', { name: /account menu/i }));
    await waitFor(() => screen.getByRole('menu'));
    // A pointerdown anywhere outside the menu container must dismiss it.
    await fireEvent.pointerDown(document.body);
    await waitFor(() => expect(screen.queryByRole('menu')).not.toBeInTheDocument());
  });

  it('keeps the menu open on a pointerdown inside it', async () => {
    render(AccountFooter, { props: { userName: 'Jamie' } });
    await fireEvent.click(screen.getByRole('button', { name: /account menu/i }));
    const menu = await waitFor(() => screen.getByRole('menu'));
    await fireEvent.pointerDown(menu);
    expect(screen.getByRole('menu')).toBeInTheDocument();
  });

  it('contains a Switch theme entry in the menu', async () => {
    render(AccountFooter, { props: { userName: 'Jamie' } });
    await fireEvent.click(screen.getByRole('button', { name: /account menu/i }));
    await waitFor(() => screen.getByRole('menu'));
    expect(screen.getByText(/switch theme/i)).toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // Embeddings Inspector (DEV-only) menu item — consensus fix #3.
  // The real notebooks store starts with activeNotebookId = null, so the item
  // must render DISABLED.
  // -------------------------------------------------------------------------

  it('renders the Embeddings Inspector item in DEV, disabled when no active notebook', async () => {
    render(AccountFooter, { props: { userName: 'Jamie' } });
    await fireEvent.click(screen.getByRole('button', { name: /account menu/i }));
    await waitFor(() => screen.getByRole('menu'));
    const item = screen.getByText(/embeddings inspector/i);
    expect(item).toBeInTheDocument();
    const disabledEl = item.closest('[aria-disabled="true"]') ?? item.closest('[disabled]');
    expect(disabledEl).toBeTruthy();
  });
});
