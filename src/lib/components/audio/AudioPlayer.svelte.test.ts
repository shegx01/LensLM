// Component tests for AudioPlayer.svelte (#29).
//
// The player loads its WAV into a Blob objectURL (see the component for why), so these
// tests stub `convertFileSrc`, global `fetch`, and `URL.createObjectURL`/`revokeObjectURL`
// and wait for the async load before asserting on the `<audio>` and controls.
//
// happy-dom's HTMLMediaElement doesn't implement real playback, so `play`/`pause`
// are stubbed to dispatch the native events the component listens for (mirrors how
// a real `<audio>` element would drive `playing` state) — see beforeEach. All
// synthetic DOM events are dispatched via `fireEvent(el, event)` (not raw
// `el.dispatchEvent`) so Svelte's reactive flush is awaited before assertions run.

import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AudioPlayer from './AudioPlayer.svelte';

vi.mock('@tauri-apps/api/core', () => ({
  convertFileSrc: (path: string) => `asset://localhost/${path}`
}));

const TEST_PATH = '/data/notebooks/nb-001/overview.wav';
const OBJECT_URL = 'blob:mock-audio-url';

async function setDuration(audio: HTMLAudioElement, seconds: number): Promise<void> {
  Object.defineProperty(audio, 'duration', { value: seconds, configurable: true });
  await fireEvent(audio, new Event('loadedmetadata'));
}

async function setCurrentTime(audio: HTMLAudioElement, seconds: number): Promise<void> {
  Object.defineProperty(audio, 'currentTime', {
    value: seconds,
    configurable: true,
    writable: true
  });
  await fireEvent(audio, new Event('timeupdate'));
}

/** Renders and waits for the blob load to resolve so the `<audio>` + controls exist. */
async function renderLoaded(path = TEST_PATH) {
  const utils = render(AudioPlayer, { props: { path } });
  const audio = (await waitFor(() => {
    const el = utils.container.querySelector('audio');
    if (!el) throw new Error('audio not rendered yet');
    return el;
  })) as HTMLAudioElement;
  return { ...utils, audio };
}

let originalPlay: typeof HTMLMediaElement.prototype.play;
let originalPause: typeof HTMLMediaElement.prototype.pause;

beforeEach(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn(() =>
      Promise.resolve({
        ok: true,
        blob: () => Promise.resolve(new Blob(['wav'], { type: 'audio/wav' }))
      } as Response)
    )
  );
  vi.spyOn(URL, 'createObjectURL').mockReturnValue(OBJECT_URL);
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

  originalPlay = HTMLMediaElement.prototype.play;
  originalPause = HTMLMediaElement.prototype.pause;
  HTMLMediaElement.prototype.play = vi.fn(function (this: HTMLMediaElement) {
    this.dispatchEvent(new Event('play'));
    return Promise.resolve();
  });
  HTMLMediaElement.prototype.pause = vi.fn(function (this: HTMLMediaElement) {
    this.dispatchEvent(new Event('pause'));
  });
});

afterEach(() => {
  HTMLMediaElement.prototype.play = originalPlay;
  HTMLMediaElement.prototype.pause = originalPause;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('AudioPlayer — blob loading', () => {
  it('shows a Loading state until the blob resolves', () => {
    let resolveFetch!: (r: Response) => void;
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockReturnValueOnce(
      new Promise<Response>((resolve) => {
        resolveFetch = resolve;
      })
    );

    render(AudioPlayer, { props: { path: TEST_PATH } });

    expect(screen.getByText('Loading…')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Play' })).not.toBeInTheDocument();
    // resolve so the pending promise doesn't leak into the next test
    resolveFetch({
      ok: true,
      blob: () => Promise.resolve(new Blob(['wav'], { type: 'audio/wav' }))
    } as Response);
  });

  it('fetches via convertFileSrc and binds the object URL to <audio>', async () => {
    const { audio } = await renderLoaded();

    expect(globalThis.fetch).toHaveBeenCalledWith(`asset://localhost/${TEST_PATH}`);
    expect(URL.createObjectURL).toHaveBeenCalledWith(expect.any(Blob));
    expect(audio.getAttribute('src')).toBe(OBJECT_URL);
  });

  it('shows a functional error state when the fetch fails', async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: false,
      status: 404
    } as Response);

    render(AudioPlayer, { props: { path: TEST_PATH } });

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/Couldn.t load the audio overview/);
    expect(screen.queryByRole('button', { name: 'Play' })).not.toBeInTheDocument();
  });

  it('resets playing + currentTime when the path changes to a new blob', async () => {
    const { audio, rerender, container } = await renderLoaded();
    await setDuration(audio, 100);
    await setCurrentTime(audio, 40);
    await fireEvent.click(screen.getByRole('button', { name: 'Play' }));
    expect(screen.getByRole('button', { name: 'Pause' })).toBeInTheDocument();

    await rerender({ path: '/data/notebooks/nb-002/overview.wav' });

    await waitFor(() => {
      if (!container.querySelector('audio')) throw new Error('audio not remounted');
    });
    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
    expect(screen.getByText(/^0:00 \//)).toBeInTheDocument();
  });

  it('revokes the object URL on unmount', async () => {
    const { unmount } = await renderLoaded();

    unmount();

    expect(URL.revokeObjectURL).toHaveBeenCalledWith(OBJECT_URL);
  });
});

describe('AudioPlayer', () => {
  it('renders a play button and a 0:00 / 0:00 time display once loaded', async () => {
    await renderLoaded();

    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
    expect(screen.getByText('0:00 / 0:00')).toBeInTheDocument();
  });

  it('the outer group carries an aria-label describing every keyboard shortcut', async () => {
    await renderLoaded();

    const group = screen.getByRole('group');
    expect(group).toHaveAttribute('aria-label', expect.stringContaining('Space plays or pauses'));
    expect(group).toHaveAttribute('aria-label', expect.stringContaining('J and L skip 15 seconds'));
  });

  it('clicking Play toggles the button to Pause (native play event flips state)', async () => {
    await renderLoaded();

    await fireEvent.click(screen.getByRole('button', { name: 'Play' }));

    expect(screen.getByRole('button', { name: 'Pause' })).toBeInTheDocument();
  });

  it('clicking Pause toggles back to Play', async () => {
    await renderLoaded();

    await fireEvent.click(screen.getByRole('button', { name: 'Play' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Pause' }));

    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
  });

  it('renders the current/total time from timeupdate + loadedmetadata events', async () => {
    const { audio } = await renderLoaded();

    await setDuration(audio, 125);
    await setCurrentTime(audio, 65);

    expect(screen.getByText('1:05 / 2:05')).toBeInTheDocument();
  });

  it('the seek range reflects duration as its max', async () => {
    const { audio } = await renderLoaded();
    await setDuration(audio, 200);

    const range = screen.getByRole('slider', { name: 'Seek' }) as HTMLInputElement;
    expect(range.max).toBe('200');
  });

  it('dragging the seek range sets audio.currentTime', async () => {
    const { audio } = await renderLoaded();
    await setDuration(audio, 200);
    Object.defineProperty(audio, 'currentTime', { value: 0, configurable: true, writable: true });

    const range = screen.getByRole('slider', { name: 'Seek' });
    await fireEvent.input(range, { target: { value: '80' } });

    expect(audio.currentTime).toBe(80);
    expect(screen.getByText('1:20 / 3:20')).toBeInTheDocument();
  });

  describe('keyboard shortcuts (scoped to the player group)', () => {
    it('Space toggles play/pause', async () => {
      await renderLoaded();
      const group = screen.getByRole('group');

      await fireEvent.keyDown(group, { key: ' ' });
      expect(screen.getByRole('button', { name: 'Pause' })).toBeInTheDocument();

      await fireEvent.keyDown(group, { key: ' ' });
      expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
    });

    it('ArrowRight/ArrowLeft seek by 5 seconds', async () => {
      const { audio } = await renderLoaded();
      await setDuration(audio, 200);
      await setCurrentTime(audio, 10);
      const group = screen.getByRole('group');

      await fireEvent.keyDown(group, { key: 'ArrowRight' });
      expect(audio.currentTime).toBe(15);

      await fireEvent.keyDown(group, { key: 'ArrowLeft' });
      expect(audio.currentTime).toBe(10);
    });

    it('J/L skip back/forward 15 seconds and announce it', async () => {
      const { audio } = await renderLoaded();
      await setDuration(audio, 200);
      await setCurrentTime(audio, 30);
      const group = screen.getByRole('group');

      await fireEvent.keyDown(group, { key: 'l' });
      expect(audio.currentTime).toBe(45);
      expect(screen.getByText('Skipped forward 15 seconds')).toBeInTheDocument();

      await fireEvent.keyDown(group, { key: 'j' });
      expect(audio.currentTime).toBe(30);
      expect(screen.getByText('Skipped back 15 seconds')).toBeInTheDocument();
    });

    it('[ and ] cycle the playback speed and surface a visible speed badge', async () => {
      const { container, audio } = await renderLoaded();
      await setDuration(audio, 200);
      const group = screen.getByRole('group');

      expect(container.querySelector('.speed-badge')).not.toBeInTheDocument();

      await fireEvent.keyDown(group, { key: ']' });
      expect(audio.playbackRate).toBe(1.25);
      expect(container.querySelector('.speed-badge')?.textContent).toBe('1.25× speed');

      await fireEvent.keyDown(group, { key: '[' });
      await fireEvent.keyDown(group, { key: '[' });
      expect(audio.playbackRate).toBe(0.75);
      expect(container.querySelector('.speed-badge')?.textContent).toBe('0.75× speed');
    });

    it('re-applies the current rate to a freshly-loaded <audio> on loadedmetadata', async () => {
      const { audio } = await renderLoaded();
      const group = screen.getByRole('group');

      await fireEvent.keyDown(group, { key: ']' });
      expect(audio.playbackRate).toBe(1.25);

      // A remount defaults playbackRate back to 1.0; the metadata handler must restore it
      // so the visible speed badge doesn't lie.
      audio.playbackRate = 1;
      await setDuration(audio, 100);
      expect(audio.playbackRate).toBe(1.25);
    });

    it('does not seek past 0 or past duration (clamped)', async () => {
      const { audio } = await renderLoaded();
      await setDuration(audio, 20);
      await setCurrentTime(audio, 2);
      const group = screen.getByRole('group');

      await fireEvent.keyDown(group, { key: 'ArrowLeft' });
      expect(audio.currentTime).toBe(0);
    });
  });
});
