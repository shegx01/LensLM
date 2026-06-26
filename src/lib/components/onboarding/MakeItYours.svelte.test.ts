import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import MakeItYours from './MakeItYours.svelte';
import { draft, resetDraft } from '$lib/components/onboarding/onboarding-state.svelte.js';

// Stub document.documentElement.dataset so happy-dom doesn't throw
// when the component writes `document.documentElement.dataset.accent`.
function getAccentAttr(): string | undefined {
  return document.documentElement.dataset.accent;
}

beforeEach(() => {
  // Reset draft between tests so module-singleton state doesn't leak.
  resetDraft();
  // Clear the data-accent attribute.
  delete document.documentElement.dataset.accent;
});

afterEach(() => {
  clearMocks();
  delete document.documentElement.dataset.accent;
});

describe('MakeItYours', () => {
  it('renders the "Make it yours" heading', () => {
    render(MakeItYours, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    expect(screen.getByRole('heading', { name: /make it yours/i })).toBeInTheDocument();
  });

  it('Continue is disabled when name is empty', () => {
    render(MakeItYours, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    const btn = screen.getByRole('button', { name: /continue/i });
    expect(btn).toBeDisabled();
  });

  it('Continue becomes enabled after typing a name', async () => {
    render(MakeItYours, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    const input = screen.getByPlaceholderText(/e\.g\. Jamie or jdoe/i);
    const btn = screen.getByRole('button', { name: /continue/i });

    expect(btn).toBeDisabled();
    await fireEvent.input(input, { target: { value: 'Jamie' } });
    await waitFor(() => expect(btn).not.toBeDisabled());
  });

  it('Continue stays disabled when name is only whitespace', async () => {
    render(MakeItYours, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    const input = screen.getByPlaceholderText(/e\.g\. Jamie or jdoe/i);
    await fireEvent.input(input, { target: { value: '   ' } });
    const btn = screen.getByRole('button', { name: /continue/i });
    expect(btn).toBeDisabled();
  });

  it('selecting a swatch updates document.documentElement.dataset.accent', async () => {
    render(MakeItYours, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    const blueBtn = screen.getByRole('radio', { name: /blue/i });
    await fireEvent.click(blueBtn);
    await waitFor(() => expect(getAccentAttr()).toBe('blue'));
  });

  it('selecting different swatches each update data-accent correctly', async () => {
    render(MakeItYours, { props: { onadvance: vi.fn(), onback: vi.fn() } });

    const roseBtn = screen.getByRole('radio', { name: /rose/i });
    await fireEvent.click(roseBtn);
    await waitFor(() => expect(getAccentAttr()).toBe('rose'));

    const graphiteBtn = screen.getByRole('radio', { name: /graphite/i });
    await fireEvent.click(graphiteBtn);
    await waitFor(() => expect(getAccentAttr()).toBe('graphite'));
  });

  it('Back button fires onback', async () => {
    const onback = vi.fn();
    render(MakeItYours, { props: { onadvance: vi.fn(), onback } });
    const backBtn = screen.getByRole('button', { name: /back/i });
    await fireEvent.click(backBtn);
    expect(onback).toHaveBeenCalledOnce();
  });

  it('Continue calls onadvance after persisting (happy-path with mockIPC)', async () => {
    const onadvance = vi.fn();
    const setConfig = vi.fn().mockReturnValue(null);
    mockIPC((cmd, args) => {
      if (cmd === 'get_config') {
        return {
          theme: 'system',
          accent: 'purple',
          user_name: '',
          models: [],
          endpoints: {},
          voices: { host: '', guest: '' },
          tts: { provider: '', api_key: '' },
          enrichment: { enabled: false, coref_strategy: 'llm_inline', cloud_consent: false },
          paths: { data_dir: '' },
          tier_thresholds: { tier1_token_cap: 0, tier2_token_cap: 0 },
          onboarding_complete: false,
          embedding_model: ''
        };
      }
      if (cmd === 'set_config') {
        setConfig(args);
        return null;
      }
    });

    // Simulate running inside Tauri so updateConfig doesn't no-op.
    (globalThis as Record<string, unknown>).isTauri = true;

    render(MakeItYours, { props: { onadvance, onback: vi.fn() } });
    const input = screen.getByPlaceholderText(/e\.g\. Jamie or jdoe/i);
    await fireEvent.input(input, { target: { value: 'Jamie' } });

    const btn = screen.getByRole('button', { name: /continue/i });
    await waitFor(() => expect(btn).not.toBeDisabled());
    await fireEvent.click(btn);

    await waitFor(() => expect(onadvance).toHaveBeenCalledOnce());
    expect(setConfig).toHaveBeenCalledOnce();

    delete (globalThis as Record<string, unknown>).isTauri;
  });

  it('shows an inline error and stays on screen when persist fails', async () => {
    const onadvance = vi.fn();
    mockIPC((cmd) => {
      if (cmd === 'get_config') throw new Error('disk full');
    });

    (globalThis as Record<string, unknown>).isTauri = true;

    render(MakeItYours, { props: { onadvance, onback: vi.fn() } });
    const input = screen.getByPlaceholderText(/e\.g\. Jamie or jdoe/i);
    await fireEvent.input(input, { target: { value: 'Jamie' } });

    const btn = screen.getByRole('button', { name: /continue/i });
    await waitFor(() => expect(btn).not.toBeDisabled());
    await fireEvent.click(btn);

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/could not save/i));
    expect(onadvance).not.toHaveBeenCalled();

    delete (globalThis as Record<string, unknown>).isTauri;
  });

  it('all six accent swatches are rendered with the correct label', () => {
    render(MakeItYours, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    for (const label of ['Violet', 'Blue', 'Green', 'Amber', 'Rose', 'Graphite']) {
      expect(screen.getByRole('radio', { name: label })).toBeInTheDocument();
    }
  });
});
