import { render, screen, fireEvent } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Page from './+page.svelte';

afterEach(() => {
  clearMocks(); // reset IPC intercepts between tests
});

describe('+page.svelte', () => {
  it('renders the Hello World heading', () => {
    mockIPC(() => {});
    render(Page);
    expect(screen.getByRole('heading', { name: /hello world/i })).toBeInTheDocument();
  });

  it('invokes invoke_core_action with an empty payload on button click', async () => {
    const handler = vi.fn().mockResolvedValue('');
    mockIPC((cmd, args) => {
      if (cmd === 'invoke_core_action') return handler(cmd, args);
    });

    render(Page);
    await fireEvent.click(screen.getByRole('button', { name: /invoke core action/i }));

    expect(handler).toHaveBeenCalledOnce();
    expect(handler).toHaveBeenCalledWith('invoke_core_action', { payload: '' });
  });
});
