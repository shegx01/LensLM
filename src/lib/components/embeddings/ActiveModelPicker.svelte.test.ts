import { render, screen, fireEvent, waitFor, within } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { ActiveModelSelection } from '$lib/models/types.js';
import { resetActiveModel } from '$lib/models/active-model.svelte.js';
import ActiveModelPicker from './ActiveModelPicker.svelte';

function setup(selection: ActiveModelSelection): {
  setActiveCalls: Array<{ provider: string; model: string }>;
} {
  const setActiveCalls: Array<{ provider: string; model: string }> = [];
  mockIPC((cmd, args) => {
    if (cmd === 'list_active_model_candidates') return selection;
    if (cmd === 'set_active_chat_model') {
      setActiveCalls.push(args as { provider: string; model: string });
      return null;
    }
    return undefined;
  });
  return { setActiveCalls };
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  resetActiveModel();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('ActiveModelPicker', () => {
  it('renders each candidate by label', async () => {
    setup({
      active: { provider: 'ollama', model: 'llama3.2:3b' },
      candidates: [
        {
          provider: 'ollama',
          model: 'llama3.2:3b',
          label: 'Ollama · llama3.2:3b',
          available: true,
          reason: null
        },
        {
          provider: 'openai',
          model: 'gpt-4o',
          label: 'OpenAI · gpt-4o',
          available: false,
          reason: 'cloud consent required'
        }
      ]
    });

    render(ActiveModelPicker);

    const select = await screen.findByLabelText('Active model');
    expect(
      within(select).getByRole('option', { name: 'Ollama · llama3.2:3b' })
    ).toBeInTheDocument();
    expect(within(select).getByRole('option', { name: 'OpenAI · gpt-4o' })).toBeInTheDocument();
  });

  it('disables unavailable candidates and surfaces the reason as help text', async () => {
    setup({
      active: { provider: 'ollama', model: 'llama3.2:3b' },
      candidates: [
        {
          provider: 'ollama',
          model: 'llama3.2:3b',
          label: 'Ollama · llama3.2:3b',
          available: true,
          reason: null
        },
        {
          provider: 'openai',
          model: 'gpt-4o',
          label: 'OpenAI · gpt-4o',
          available: false,
          reason: 'cloud consent required'
        }
      ]
    });

    render(ActiveModelPicker);

    const select = await screen.findByLabelText('Active model');
    const option = within(select).getByRole('option', { name: 'OpenAI · gpt-4o' });
    expect(option).toBeDisabled();
    expect(screen.getByText('OpenAI · gpt-4o — cloud consent required')).toBeInTheDocument();
  });

  it('selecting an available candidate persists it via set_active_chat_model', async () => {
    const calls = setup({
      active: { provider: 'ollama', model: 'llama3.2:3b' },
      candidates: [
        {
          provider: 'ollama',
          model: 'llama3.2:3b',
          label: 'Ollama · llama3.2:3b',
          available: true,
          reason: null
        },
        {
          provider: 'openai',
          model: 'gpt-4o',
          label: 'OpenAI · gpt-4o',
          available: true,
          reason: null
        }
      ]
    });

    render(ActiveModelPicker);
    const select = await screen.findByLabelText('Active model');

    await fireEvent.change(select, { target: { value: 'openai gpt-4o' } });

    await waitFor(() => expect(calls.setActiveCalls).toHaveLength(1));
    expect(calls.setActiveCalls[0]).toEqual({ provider: 'openai', model: 'gpt-4o' });
  });

  it('reflects the active pin as the selected option', async () => {
    setup({
      active: { provider: 'openai', model: 'gpt-4o' },
      candidates: [
        {
          provider: 'ollama',
          model: 'llama3.2:3b',
          label: 'Ollama · llama3.2:3b',
          available: true,
          reason: null
        },
        {
          provider: 'openai',
          model: 'gpt-4o',
          label: 'OpenAI · gpt-4o',
          available: true,
          reason: null
        }
      ]
    });

    render(ActiveModelPicker);
    const select = (await screen.findByLabelText('Active model')) as HTMLSelectElement;

    await waitFor(() => expect(select.value).toBe('openai gpt-4o'));
    expect(
      screen.queryByText(/without a pin|active model is unavailable/i)
    ).not.toBeInTheDocument();
  });

  it('represents a stale/removed pin as a disabled "no longer available" option', async () => {
    setup({
      active: { provider: 'openai', model: 'gpt-4o' },
      candidates: [
        {
          provider: 'ollama',
          model: 'llama3.2:3b',
          label: 'Ollama · llama3.2:3b',
          available: true,
          reason: null
        }
      ]
    });

    render(ActiveModelPicker);
    const select = (await screen.findByLabelText('Active model')) as HTMLSelectElement;

    // The select shows the stale pin's own option, never a silent fallback to the first
    // candidate.
    await waitFor(() => expect(select.value).toBe('openai gpt-4o'));
    expect(select.value).not.toBe('ollama llama3.2:3b');

    const staleOption = within(select).getByRole('option', {
      name: /openai · gpt-4o — no longer available/i
    });
    expect(staleOption).toBeDisabled();
    expect(screen.getByText(/active model is unavailable.*choose another/i)).toBeInTheDocument();
  });

  it('shows an explicit "none selected" state when active is null', async () => {
    setup({
      active: null,
      candidates: [
        {
          provider: 'ollama',
          model: 'llama3.2:3b',
          label: 'Ollama · llama3.2:3b',
          available: true,
          reason: null
        }
      ]
    });

    render(ActiveModelPicker);
    const select = (await screen.findByLabelText('Active model')) as HTMLSelectElement;

    await waitFor(() => expect(select.value).toBe(''));
    expect(screen.getByText(/no chat model configured/i)).toBeInTheDocument();
  });
});
