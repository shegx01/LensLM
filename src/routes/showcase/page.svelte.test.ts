import { render, screen } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, describe, expect, it } from 'vitest';
import Showcase from './+page.svelte';

afterEach(() => {
  clearMocks();
  document.documentElement.classList.remove('dark');
});

describe('/showcase', () => {
  it('renders the design-system surface in light mode', () => {
    mockIPC(() => {});
    render(Showcase);
    expect(screen.getByRole('heading', { name: /lens design system/i })).toBeInTheDocument();
    // A primitive from the page is present (Button group).
    expect(screen.getByRole('button', { name: 'Primary' })).toBeInTheDocument();
    expect(screen.getByText(/Display — Extrabold 800/)).toBeInTheDocument();
    expect(screen.getByText('primary')).toBeInTheDocument();
  });

  it('renders under the .dark theme class', () => {
    mockIPC(() => {});
    document.documentElement.classList.add('dark');
    render(Showcase);
    expect(document.documentElement.classList.contains('dark')).toBe(true);
    expect(screen.getByRole('heading', { name: /lens design system/i })).toBeInTheDocument();
  });
});
