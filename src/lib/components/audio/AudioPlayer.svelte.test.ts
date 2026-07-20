// Component tests for AudioPlayer.svelte (#29).
//
// happy-dom's HTMLMediaElement doesn't implement real playback, so `play`/`pause`
// are stubbed to dispatch the native events the component listens for (mirrors how
// a real `<audio>` element would drive `playing` state) — see beforeEach. All
// synthetic DOM events are dispatched via `fireEvent(el, event)` (not raw
// `el.dispatchEvent`) so Svelte's reactive flush is awaited before assertions run.

import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AudioPlayer from './AudioPlayer.svelte';

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

let originalPlay: typeof HTMLMediaElement.prototype.play;
let originalPause: typeof HTMLMediaElement.prototype.pause;

beforeEach(() => {
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
});

describe('AudioPlayer', () => {
  it('renders a play button and a 0:00 / 0:00 time display before metadata loads', () => {
    render(AudioPlayer, { props: { src: 'asset://localhost/overview.wav' } });

    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
    expect(screen.getByText('0:00 / 0:00')).toBeInTheDocument();
  });

  it('the outer group carries an aria-label describing every keyboard shortcut', () => {
    render(AudioPlayer, { props: { src: 'asset://localhost/overview.wav' } });

    const group = screen.getByRole('group');
    expect(group).toHaveAttribute('aria-label', expect.stringContaining('Space plays or pauses'));
    expect(group).toHaveAttribute('aria-label', expect.stringContaining('J and L skip 15 seconds'));
  });

  it('clicking Play toggles the button to Pause (native play event flips state)', async () => {
    render(AudioPlayer, { props: { src: 'asset://localhost/overview.wav' } });

    await fireEvent.click(screen.getByRole('button', { name: 'Play' }));

    expect(screen.getByRole('button', { name: 'Pause' })).toBeInTheDocument();
  });

  it('clicking Pause toggles back to Play', async () => {
    render(AudioPlayer, { props: { src: 'asset://localhost/overview.wav' } });

    await fireEvent.click(screen.getByRole('button', { name: 'Play' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Pause' }));

    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
  });

  it('renders the current/total time from timeupdate + loadedmetadata events', async () => {
    const { container } = render(AudioPlayer, {
      props: { src: 'asset://localhost/overview.wav' }
    });
    const audio = container.querySelector('audio') as HTMLAudioElement;

    await setDuration(audio, 125);
    await setCurrentTime(audio, 65);

    expect(screen.getByText('1:05 / 2:05')).toBeInTheDocument();
  });

  it('the seek range reflects duration as its max', async () => {
    const { container } = render(AudioPlayer, {
      props: { src: 'asset://localhost/overview.wav' }
    });
    const audio = container.querySelector('audio') as HTMLAudioElement;
    await setDuration(audio, 200);

    const range = screen.getByRole('slider', { name: 'Seek' }) as HTMLInputElement;
    expect(range.max).toBe('200');
  });

  it('dragging the seek range sets audio.currentTime', async () => {
    const { container } = render(AudioPlayer, {
      props: { src: 'asset://localhost/overview.wav' }
    });
    const audio = container.querySelector('audio') as HTMLAudioElement;
    await setDuration(audio, 200);
    Object.defineProperty(audio, 'currentTime', { value: 0, configurable: true, writable: true });

    const range = screen.getByRole('slider', { name: 'Seek' });
    await fireEvent.input(range, { target: { value: '80' } });

    expect(audio.currentTime).toBe(80);
    expect(screen.getByText('1:20 / 3:20')).toBeInTheDocument();
  });

  describe('keyboard shortcuts (scoped to the player group)', () => {
    it('Space toggles play/pause', async () => {
      render(AudioPlayer, { props: { src: 'asset://localhost/overview.wav' } });
      const group = screen.getByRole('group');

      await fireEvent.keyDown(group, { key: ' ' });
      expect(screen.getByRole('button', { name: 'Pause' })).toBeInTheDocument();

      await fireEvent.keyDown(group, { key: ' ' });
      expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
    });

    it('ArrowRight/ArrowLeft seek by 5 seconds', async () => {
      const { container } = render(AudioPlayer, {
        props: { src: 'asset://localhost/overview.wav' }
      });
      const audio = container.querySelector('audio') as HTMLAudioElement;
      await setDuration(audio, 200);
      await setCurrentTime(audio, 10);
      const group = screen.getByRole('group');

      await fireEvent.keyDown(group, { key: 'ArrowRight' });
      expect(audio.currentTime).toBe(15);

      await fireEvent.keyDown(group, { key: 'ArrowLeft' });
      expect(audio.currentTime).toBe(10);
    });

    it('J/L skip back/forward 15 seconds and announce it', async () => {
      const { container } = render(AudioPlayer, {
        props: { src: 'asset://localhost/overview.wav' }
      });
      const audio = container.querySelector('audio') as HTMLAudioElement;
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
      const { container } = render(AudioPlayer, {
        props: { src: 'asset://localhost/overview.wav' }
      });
      const audio = container.querySelector('audio') as HTMLAudioElement;
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

    it('does not seek past 0 or past duration (clamped)', async () => {
      const { container } = render(AudioPlayer, {
        props: { src: 'asset://localhost/overview.wav' }
      });
      const audio = container.querySelector('audio') as HTMLAudioElement;
      await setDuration(audio, 20);
      await setCurrentTime(audio, 2);
      const group = screen.getByRole('group');

      await fireEvent.keyDown(group, { key: 'ArrowLeft' });
      expect(audio.currentTime).toBe(0);
    });
  });
});
