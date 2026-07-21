import { render } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import ProgressBar from './ProgressBar.svelte';

describe('ProgressBar', () => {
  it('renders a determinate width when given a number', () => {
    const { container } = render(ProgressBar, { props: { value: 42 } });
    const fill = container.querySelector('.progress-fill') as HTMLElement;
    expect(fill.style.width).toBe('42%');
    expect(container.querySelector('.indeterminate-slide')).toBeNull();

    const track = container.querySelector('[role="progressbar"]') as HTMLElement;
    expect(track.getAttribute('aria-valuenow')).toBe('42');
  });

  it('clamps out-of-range determinate values to 0..100', () => {
    const { container } = render(ProgressBar, { props: { value: 250 } });
    const fill = container.querySelector('.progress-fill') as HTMLElement;
    expect(fill.style.width).toBe('100%');
  });

  it('renders the indeterminate slide + static branches when value is null', () => {
    const { container } = render(ProgressBar, { props: { value: null } });
    expect(container.querySelector('.indeterminate-slide')).not.toBeNull();
    expect(container.querySelector('.indeterminate-static')).not.toBeNull();

    const track = container.querySelector('[role="progressbar"]') as HTMLElement;
    expect(track.hasAttribute('aria-valuenow')).toBe(false);
  });

  it('defaults to indeterminate when value is omitted', () => {
    const { container } = render(ProgressBar, { props: {} });
    expect(container.querySelector('.indeterminate-slide')).not.toBeNull();
  });
});
