// GeneralSection tests — Animations preference control.
//
// Covers: selecting an option persists via updateConfig and mirrors the choice
// to `data-motion` on <html>; a failed write reverts both the state and the attr.
// Mocks $lib/config.js and @tauri-apps/api/core so no IPC occurs.

import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';

const { mockUpdateConfig, mockInvoke, mockIsTauri } = vi.hoisted(() => ({
  mockUpdateConfig: vi.fn().mockResolvedValue(undefined),
  mockInvoke: vi.fn(),
  mockIsTauri: vi.fn(() => false)
}));

vi.mock('$lib/config.js', () => ({ updateConfig: mockUpdateConfig }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: mockInvoke, isTauri: mockIsTauri }));

import GeneralSection from './GeneralSection.svelte';

beforeEach(() => {
  delete document.documentElement.dataset.motion;
  mockUpdateConfig.mockReset().mockResolvedValue(undefined);
  mockInvoke.mockReset();
  mockIsTauri.mockReset().mockReturnValue(false);
});

afterEach(() => {
  delete document.documentElement.dataset.motion;
  vi.clearAllMocks();
});

describe('GeneralSection — Animations', () => {
  it('selecting an option persists {animations: value} and sets data-motion', async () => {
    render(GeneralSection);
    await fireEvent.click(screen.getByRole('button', { name: /Animations: On/i }));

    await waitFor(() => expect(mockUpdateConfig).toHaveBeenCalledTimes(1));
    const mutate = mockUpdateConfig.mock.calls[0][0] as (cfg: AppConfig) => AppConfig;
    expect(mutate({ animations: 'system' } as AppConfig)).toMatchObject({ animations: 'on' });
    expect(document.documentElement.dataset.motion).toBe('on');
    expect(screen.getByRole('button', { name: /Animations: On/i })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
  });

  it('reverts state and data-motion when the write fails', async () => {
    mockUpdateConfig.mockRejectedValueOnce(new Error('disk full'));
    render(GeneralSection);

    // Default selection is "System"; switching to "Off" then failing must roll back.
    await fireEvent.click(screen.getByRole('button', { name: /Animations: Off/i }));

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Animations: System/i })).toHaveAttribute(
        'aria-pressed',
        'true'
      )
    );
    expect(document.documentElement.dataset.motion).toBe('system');
    expect(screen.getByRole('button', { name: /Animations: Off/i })).toHaveAttribute(
      'aria-pressed',
      'false'
    );
    expect(screen.getByRole('alert')).toBeInTheDocument();
  });
});
