// StudioPanel.svelte.test.ts — Component tests for the Audio Overview lifecycle card (#29).
// The audio-state store, notebook store, AudioPlayer, and Tauri core are all mocked
// so tests run without a native host and exercise StudioPanel's own state-to-UI mapping.

import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

type OverviewState = 'none' | 'generating' | 'ready' | 'stale' | 'failed' | 'missing';
type OverviewPhase = 'idle' | 'starting' | 'synthesizing' | 'stitching' | 'encoding';

const {
  mockAudioStore,
  mockSourcesStore,
  mockNotebookStore,
  generateOverview,
  cancelOverview,
  openSettings
} = vi.hoisted(() => {
  const state = {
    overviewStatus: 'none' as OverviewState,
    phase: 'idle' as OverviewPhase,
    turn: null as number | null,
    total: null as number | null,
    overviewPath: null as string | null,
    generatedAt: null as string | null,
    errorMessage: null as string | null,
    modelReady: true,
    canGenerate: true
  };

  const mockAudioStore = {
    get overviewStatus() {
      return state.overviewStatus;
    },
    get phase() {
      return state.phase;
    },
    get turn() {
      return state.turn;
    },
    get total() {
      return state.total;
    },
    get overviewPath() {
      return state.overviewPath;
    },
    get generatedAt() {
      return state.generatedAt;
    },
    get errorMessage() {
      return state.errorMessage;
    },
    get modelReady() {
      return state.modelReady;
    },
    get canGenerate() {
      return state.canGenerate;
    },
    _set(overrides: Partial<typeof state>) {
      Object.assign(state, overrides);
    },
    _reset() {
      Object.assign(state, {
        overviewStatus: 'none',
        phase: 'idle',
        turn: null,
        total: null,
        overviewPath: null,
        generatedAt: null,
        errorMessage: null,
        modelReady: true,
        canGenerate: true
      });
    }
  };

  const sourcesState = { selectedCount: 2 };
  const mockSourcesStore = {
    get selectedCount() {
      return sourcesState.selectedCount;
    },
    _setSelectedCount(n: number) {
      sourcesState.selectedCount = n;
    },
    _reset() {
      sourcesState.selectedCount = 2;
    }
  };

  const openSettings = vi.fn();
  const mockNotebookStore = {
    get activeNotebookId() {
      return 'nb-001';
    },
    openSettings
  };

  return {
    mockAudioStore,
    mockSourcesStore,
    mockNotebookStore,
    generateOverview: vi.fn().mockResolvedValue(undefined),
    cancelOverview: vi.fn().mockResolvedValue(undefined),
    openSettings
  };
});

vi.mock('$lib/sources/audio-state.svelte.js', () => ({
  audioOverviewStore: mockAudioStore,
  generateOverview,
  cancelOverview
}));

vi.mock('$lib/sources/sources-state.svelte.js', () => ({
  sourcesStore: mockSourcesStore
}));

vi.mock('$lib/notebooks/index.js', () => ({
  notebookStore: mockNotebookStore
}));

vi.mock('@tauri-apps/api/core', () => ({
  convertFileSrc: (path: string) => `asset://localhost/${path}`
}));

vi.mock('$lib/components/audio/AudioPlayer.svelte', async () => {
  const { default: Stub } = await import('./__fixtures__/AudioPlayerStub.svelte');
  return { default: Stub };
});

import StudioPanel from './StudioPanel.svelte';

beforeEach(() => {
  vi.clearAllMocks();
  mockAudioStore._reset();
  mockSourcesStore._reset();
});

afterEach(() => {
  mockAudioStore._reset();
  mockSourcesStore._reset();
});

describe('StudioPanel — Audio Overview idle state', () => {
  it('disables Generate and hints when zero sources are selected', () => {
    mockAudioStore._set({ canGenerate: false });
    mockSourcesStore._setSelectedCount(0);
    render(StudioPanel, { props: { selectedCount: 0, totalCount: 3 } });

    const generateBtn = screen.getByRole('button', { name: 'Generate Audio Overview' });
    expect(generateBtn).toBeDisabled();
    expect(
      screen.getByText('Select at least one source to generate an overview.')
    ).toBeInTheDocument();
  });

  it('disables Generate and links to Settings when the TTS model is not ready', () => {
    mockAudioStore._set({ modelReady: false, canGenerate: false });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    expect(screen.getByRole('button', { name: 'Generate Audio Overview' })).toBeDisabled();
    const link = screen.getByRole('button', { name: /Download a voice engine in Settings/ });
    expect(link).toBeInTheDocument();
  });

  it('clicking the Settings link opens the text_to_speech settings section', async () => {
    mockAudioStore._set({ modelReady: false, canGenerate: false });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    await fireEvent.click(screen.getByRole('button', { name: /Download a voice engine/ }));

    expect(openSettings).toHaveBeenCalledWith('text_to_speech');
  });

  it('prioritizes the no-sources hint over the model-not-ready hint when both apply', () => {
    mockAudioStore._set({ modelReady: false, canGenerate: false });
    mockSourcesStore._setSelectedCount(0);
    render(StudioPanel, { props: { selectedCount: 0, totalCount: 3 } });

    expect(
      screen.getByText('Select at least one source to generate an overview.')
    ).toBeInTheDocument();
    expect(screen.queryByText(/Download a voice engine/)).not.toBeInTheDocument();
  });

  it('enables Generate once sources are selected and the model is ready', () => {
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    expect(screen.getByRole('button', { name: 'Generate Audio Overview' })).toBeEnabled();
  });

  it('clicking Generate calls generateOverview with the notebook id and default (medium) length', async () => {
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    await fireEvent.click(screen.getByRole('button', { name: 'Generate Audio Overview' }));

    expect(generateOverview).toHaveBeenCalledWith('nb-001', 'medium');
  });

  it('switching the length segmented control changes the length passed to Generate', async () => {
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    await fireEvent.click(screen.getByRole('radio', { name: 'Long' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Generate Audio Overview' }));

    expect(generateOverview).toHaveBeenCalledWith('nb-001', 'long');
  });
});

describe('StudioPanel — generating state', () => {
  it('shows the preparing-script label and a Cancel action before any TtsPhase chunk arrives', () => {
    mockAudioStore._set({ overviewStatus: 'generating', phase: 'starting' });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    expect(screen.getByText('Preparing script…')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel generation' })).toBeInTheDocument();
  });

  it('shows a determinate turn/total label while synthesizing', () => {
    mockAudioStore._set({
      overviewStatus: 'generating',
      phase: 'synthesizing',
      turn: 2,
      total: 6
    });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    expect(screen.getByText('Synthesizing turn 2/6')).toBeInTheDocument();
  });

  it('shows stitching/encoding labels for those phases', () => {
    mockAudioStore._set({ overviewStatus: 'generating', phase: 'stitching' });
    const { unmount } = render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });
    expect(screen.getByText('Stitching audio…')).toBeInTheDocument();
    unmount();

    mockAudioStore._set({ overviewStatus: 'generating', phase: 'encoding' });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });
    expect(screen.getByText('Encoding…')).toBeInTheDocument();
  });

  it('clicking Cancel calls cancelOverview with the notebook id', async () => {
    mockAudioStore._set({ overviewStatus: 'generating', phase: 'starting' });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    await fireEvent.click(screen.getByRole('button', { name: 'Cancel generation' }));

    expect(cancelOverview).toHaveBeenCalledWith('nb-001');
  });

  it('does not render the length picker or a player while generating', () => {
    mockAudioStore._set({ overviewStatus: 'generating', phase: 'starting' });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    expect(screen.queryByRole('radiogroup')).not.toBeInTheDocument();
    expect(screen.queryByTestId('audio-player-stub')).not.toBeInTheDocument();
  });
});

describe('StudioPanel — ready / stale states', () => {
  it('renders the AudioPlayer with a convertFileSrc-derived src and a "Generated" caption', () => {
    mockAudioStore._set({
      overviewStatus: 'ready',
      overviewPath: '/data/notebooks/nb-001/overview.wav',
      generatedAt: new Date().toISOString()
    });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    const stub = screen.getByTestId('audio-player-stub');
    expect(stub).toHaveTextContent('asset://localhost//data/notebooks/nb-001/overview.wav');
    expect(screen.getByText(/Generated/)).toBeInTheDocument();
  });

  it('the regenerate icon button has an accessible name and fires generateOverview', async () => {
    mockAudioStore._set({
      overviewStatus: 'ready',
      overviewPath: '/data/notebooks/nb-001/overview.wav',
      generatedAt: new Date().toISOString()
    });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    const regenBtn = screen.getByRole('button', { name: 'Regenerate overview' });
    expect(regenBtn).toBeInTheDocument();

    await fireEvent.click(regenBtn);

    expect(generateOverview).toHaveBeenCalledWith('nb-001', 'medium');
  });

  it('disables the regenerate icon and explains why when generation is blocked', () => {
    mockAudioStore._set({
      overviewStatus: 'ready',
      overviewPath: '/data/notebooks/nb-001/overview.wav',
      generatedAt: new Date().toISOString(),
      canGenerate: false
    });
    mockSourcesStore._setSelectedCount(0);
    render(StudioPanel, { props: { selectedCount: 0, totalCount: 3 } });

    const regenBtn = screen.getByRole('button', { name: 'Regenerate overview' });
    expect(regenBtn).toBeDisabled();
    expect(regenBtn).toHaveAttribute('title', 'Select at least one source to regenerate');
  });

  it('shows a "Sources changed" hint beside the regenerate icon when stale', () => {
    mockAudioStore._set({
      overviewStatus: 'stale',
      overviewPath: '/data/notebooks/nb-001/overview.wav',
      generatedAt: new Date().toISOString()
    });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    expect(screen.getByText(/Sources changed/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Regenerate overview' })).toBeInTheDocument();
    expect(screen.getByTestId('audio-player-stub')).toBeInTheDocument();
  });

  it('does NOT show the stale hint when ready (not stale)', () => {
    mockAudioStore._set({
      overviewStatus: 'ready',
      overviewPath: '/data/notebooks/nb-001/overview.wav',
      generatedAt: new Date().toISOString()
    });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    expect(screen.queryByText(/Sources changed/)).not.toBeInTheDocument();
  });
});

describe('StudioPanel — failed / missing states', () => {
  it('shows the error message and a Retry button when failed', () => {
    mockAudioStore._set({ overviewStatus: 'failed', errorMessage: 'no TTS backend available' });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    expect(screen.getByRole('alert')).toHaveTextContent('no TTS backend available');
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('clicking Retry calls generateOverview again', async () => {
    mockAudioStore._set({ overviewStatus: 'failed', errorMessage: 'no TTS backend available' });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    await fireEvent.click(screen.getByRole('button', { name: 'Retry' }));

    expect(generateOverview).toHaveBeenCalledWith('nb-001', 'medium');
  });

  it('shows a missing-file message and a Generate button (no player) when missing', () => {
    mockAudioStore._set({ overviewStatus: 'missing' });
    render(StudioPanel, { props: { selectedCount: 2, totalCount: 3 } });

    expect(
      screen.getByText('The overview file is missing. Regenerate to create a new one.')
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Generate Audio Overview' })).toBeInTheDocument();
    expect(screen.queryByTestId('audio-player-stub')).not.toBeInTheDocument();
  });
});

describe('StudioPanel — study tools remain a disabled preview', () => {
  it('keeps the study-tool tiles aria-disabled', () => {
    render(StudioPanel, { props: { selectedCount: 0, totalCount: 0 } });

    const studyGuide = screen.getByRole('button', { name: /Study Guide/ });
    expect(studyGuide).toHaveAttribute('aria-disabled', 'true');
  });
});
