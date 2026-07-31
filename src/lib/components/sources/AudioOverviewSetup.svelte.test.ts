// AudioOverviewSetup.svelte.test.ts — the Audio Overview setup modal (#29 redesign).
// The TTS-catalog language gate and the AI focus-suggest command are mocked so the
// modal's own Format/Length/Language/Focus mapping is exercised without a native host.

import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { overviewLanguageOptions, suggestOverviewFocus } = vi.hoisted(() => ({
  overviewLanguageOptions: vi.fn<() => Promise<string[]>>(),
  suggestOverviewFocus: vi.fn<() => Promise<string>>()
}));

vi.mock('$lib/onboarding/system-check.js', () => ({ overviewLanguageOptions }));
vi.mock('$lib/sources/audio-ipc.js', () => ({ suggestOverviewFocus }));

import AudioOverviewSetup from './AudioOverviewSetup.svelte';

beforeEach(() => {
  vi.clearAllMocks();
  overviewLanguageOptions.mockResolvedValue([]);
  suggestOverviewFocus.mockResolvedValue('');
});

afterEach(() => {
  vi.clearAllMocks();
});

function open(onGenerate = vi.fn()) {
  render(AudioOverviewSetup, {
    props: { open: true, selectedCount: 2, notebookId: 'nb-001', onGenerate }
  });
  return onGenerate;
}

describe('AudioOverviewSetup — format & length', () => {
  it('generates with the default Deep dive / Medium setup', async () => {
    const onGenerate = open();

    await fireEvent.click(screen.getByRole('button', { name: 'Generate' }));

    expect(onGenerate).toHaveBeenCalledWith({
      format: 'deep_dive',
      length: 'medium',
      language: undefined,
      focus: undefined
    });
  });

  it('picking a format resets length to that format default (Brief → Short)', async () => {
    const onGenerate = open();

    await fireEvent.click(screen.getByRole('button', { name: 'Brief' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Generate' }));

    expect(onGenerate).toHaveBeenCalledWith(
      expect.objectContaining({ format: 'brief', length: 'short' })
    );
  });

  it('length stays adjustable after a format sets its default', async () => {
    const onGenerate = open();

    await fireEvent.click(screen.getByRole('button', { name: 'Brief' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Long' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Generate' }));

    expect(onGenerate).toHaveBeenCalledWith(
      expect.objectContaining({ format: 'brief', length: 'long' })
    );
  });

  it('shows the selected format description', async () => {
    open();
    await fireEvent.click(screen.getByRole('button', { name: 'Debate' }));
    expect(screen.getByText(/argue opposing positions/i)).toBeInTheDocument();
  });
});

describe('AudioOverviewSetup — language gating', () => {
  it('hides the Language picker when the active engine is single-language', async () => {
    overviewLanguageOptions.mockResolvedValue([]);
    open();
    await waitFor(() => expect(overviewLanguageOptions).toHaveBeenCalled());
    expect(screen.queryByText('Language')).not.toBeInTheDocument();
  });

  it('shows the Language picker when the active engine supports several', async () => {
    overviewLanguageOptions.mockResolvedValue(['spanish', 'french']);
    open();
    expect(await screen.findByText('Language')).toBeInTheDocument();
  });
});

describe('AudioOverviewSetup — focus', () => {
  it('passes a trimmed focus string through to the generate setup', async () => {
    const onGenerate = open();

    await fireEvent.input(screen.getByPlaceholderText(/executive-level/i), {
      target: { value: '  lead with the numbers  ' }
    });
    await fireEvent.click(screen.getByRole('button', { name: 'Generate' }));

    expect(onGenerate).toHaveBeenCalledWith(
      expect.objectContaining({ focus: 'lead with the numbers' })
    );
  });

  it('Suggest fills the focus field from the AI suggestion command', async () => {
    suggestOverviewFocus.mockResolvedValue('emphasise the Q3 revenue drivers');
    open();

    await fireEvent.click(screen.getByRole('button', { name: /Suggest/ }));

    const textarea = await screen.findByDisplayValue('emphasise the Q3 revenue drivers');
    expect(textarea).toBeInTheDocument();
    expect(suggestOverviewFocus).toHaveBeenCalledWith('nb-001');
  });
});
