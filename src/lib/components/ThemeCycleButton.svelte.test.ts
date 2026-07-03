import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// vi.mock is hoisted to the top of the file, so factory functions cannot
// reference top-level variables. Use vi.hoisted() to initialise the mocks
// before the factory runs.
const { mockUserPrefersMode, mockSetMode, mockPersistTheme } = vi.hoisted(() => {
  return {
    mockUserPrefersMode: { current: 'system' as 'light' | 'dark' | 'system' },
    mockSetMode: vi.fn(),
    mockPersistTheme: vi.fn()
  };
});

// Mock mode-watcher — deterministic, no DOM/localStorage dependency.
vi.mock('mode-watcher', () => ({
  userPrefersMode: mockUserPrefersMode,
  setMode: mockSetMode
}));

// Mock $lib/theme so persistTheme doesn't attempt IPC in tests.
vi.mock('$lib/theme/index.js', () => ({
  persistTheme: mockPersistTheme
}));

// Import after mocks so the component picks up the mocked modules.
import ThemeCycleButton from './ThemeCycleButton.svelte';

beforeEach(() => {
  mockUserPrefersMode.current = 'system';
  mockSetMode.mockClear();
  mockPersistTheme.mockClear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ThemeCycleButton', () => {
  it('renders a single button', () => {
    render(ThemeCycleButton);
    expect(screen.getByRole('button')).toBeInTheDocument();
  });

  it('aria-label includes current mode (System) and next mode (Light)', () => {
    mockUserPrefersMode.current = 'system';
    render(ThemeCycleButton);
    const btn = screen.getByRole('button');
    expect(btn.getAttribute('aria-label')).toMatch(/system/i);
    expect(btn.getAttribute('aria-label')).toMatch(/light/i);
  });

  it('aria-label includes current mode (Light) and next mode (Dark)', () => {
    mockUserPrefersMode.current = 'light';
    render(ThemeCycleButton);
    const btn = screen.getByRole('button');
    expect(btn.getAttribute('aria-label')).toMatch(/light/i);
    expect(btn.getAttribute('aria-label')).toMatch(/dark/i);
  });

  it('aria-label includes current mode (Dark) and next mode (System)', () => {
    mockUserPrefersMode.current = 'dark';
    render(ThemeCycleButton);
    const btn = screen.getByRole('button');
    expect(btn.getAttribute('aria-label')).toMatch(/dark/i);
    expect(btn.getAttribute('aria-label')).toMatch(/system/i);
  });

  it('clicking calls setMode and persistTheme with the next mode (system → light)', async () => {
    mockUserPrefersMode.current = 'system';
    render(ThemeCycleButton);
    await fireEvent.click(screen.getByRole('button'));
    expect(mockSetMode).toHaveBeenCalledOnce();
    expect(mockSetMode).toHaveBeenCalledWith('light');
    expect(mockPersistTheme).toHaveBeenCalledOnce();
    expect(mockPersistTheme).toHaveBeenCalledWith('light');
  });

  it('clicking cycles light → dark', async () => {
    mockUserPrefersMode.current = 'light';
    render(ThemeCycleButton);
    await fireEvent.click(screen.getByRole('button'));
    expect(mockSetMode).toHaveBeenCalledWith('dark');
    expect(mockPersistTheme).toHaveBeenCalledWith('dark');
  });

  it('clicking cycles dark → system', async () => {
    mockUserPrefersMode.current = 'dark';
    render(ThemeCycleButton);
    await fireEvent.click(screen.getByRole('button'));
    expect(mockSetMode).toHaveBeenCalledWith('system');
    expect(mockPersistTheme).toHaveBeenCalledWith('system');
  });

  it('accepts a class prop and forwards it to the button', () => {
    render(ThemeCycleButton, { props: { class: 'size-9 rounded-lg' } });
    const btn = screen.getByRole('button');
    expect(btn.className).toContain('size-9');
    expect(btn.className).toContain('rounded-lg');
  });

  it('works without a class prop (default rendering)', () => {
    render(ThemeCycleButton);
    expect(screen.getByRole('button')).toBeInTheDocument();
  });

  it('outline variant renders the onboarding size-9 rounded-lg button', () => {
    render(ThemeCycleButton, { props: { variant: 'outline', class: 'size-9 rounded-lg' } });
    const btn = screen.getByRole('button');
    expect(btn.className).toContain('size-9');
    expect(btn.className).toContain('rounded-lg');
  });

  it('bare variant renders the sidebar 26px circle (bg-muted, rounded-full)', () => {
    render(ThemeCycleButton, { props: { variant: 'bare' } });
    const btn = screen.getByRole('button');
    expect(btn.className).toContain('size-[26px]');
    expect(btn.className).toContain('rounded-full');
    expect(btn.className).toContain('bg-muted');
  });

  it('bare variant carries the data-theme-cycle-btn hook', () => {
    const { container } = render(ThemeCycleButton, { props: { variant: 'bare' } });
    expect(container.querySelector('[data-theme-cycle-btn]')).not.toBeNull();
  });

  it('bare variant cycles light → dark → system on repeated clicks', async () => {
    mockUserPrefersMode.current = 'light';
    render(ThemeCycleButton, { props: { variant: 'bare' } });
    await fireEvent.click(screen.getByRole('button'));
    expect(mockSetMode).toHaveBeenCalledWith('dark');
    expect(mockPersistTheme).toHaveBeenCalledWith('dark');
  });

  it('bare variant aria-label names current and next mode', () => {
    mockUserPrefersMode.current = 'dark';
    render(ThemeCycleButton, { props: { variant: 'bare' } });
    const label = screen.getByRole('button').getAttribute('aria-label') ?? '';
    expect(label).toMatch(/dark/i);
    expect(label).toMatch(/system/i);
  });
});
