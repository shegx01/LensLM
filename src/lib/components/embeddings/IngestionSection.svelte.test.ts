import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { AppConfig } from '$lib/theme/types.js';
import IngestionSection from './IngestionSection.svelte';

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

/** A get_config payload carrying only the field this section reads. */
function config(jsRender: boolean): Partial<AppConfig> {
  return { js_render_enabled: jsRender };
}

describe('IngestionSection', () => {
  it('reflects the persisted js_render_enabled on mount', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config(false);
    });

    render(IngestionSection);

    const toggle = await screen.findByRole('switch', { name: /enable js rendering/i });
    await waitFor(() => expect(toggle).toHaveAttribute('aria-checked', 'false'));
  });

  it('toggles on and persists js_render_enabled: true via set_config', async () => {
    let saved: AppConfig | undefined;
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') return config(false);
      if (cmd === 'set_config') {
        saved = (args as { config: AppConfig }).config;
      }
    });

    render(IngestionSection);

    const toggle = await screen.findByRole('switch', { name: /enable js rendering/i });
    await waitFor(() => expect(toggle).toHaveAttribute('aria-checked', 'false'));

    await fireEvent.click(toggle);

    await waitFor(() => expect(toggle).toHaveAttribute('aria-checked', 'true'));
    expect(saved?.js_render_enabled).toBe(true);
  });

  it('optimistically toggles then reverts when set_config fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_config') return config(true);
      if (cmd === 'set_config') throw new Error('write failed');
    });

    render(IngestionSection);

    const toggle = await screen.findByRole('switch', { name: /enable js rendering/i });
    await waitFor(() => expect(toggle).toHaveAttribute('aria-checked', 'true'));

    await fireEvent.click(toggle);

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/write failed/i));
    expect(toggle).toHaveAttribute('aria-checked', 'true');
  });
});
