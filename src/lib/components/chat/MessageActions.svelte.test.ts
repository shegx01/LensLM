import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import MessageActions from './MessageActions.svelte';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('MessageActions', () => {
  it('copy invokes oncopy', async () => {
    const oncopy = vi.fn();
    render(MessageActions, {
      props: { feedback: null, oncopy, onregenerate: vi.fn(), onfeedback: vi.fn() }
    });
    await fireEvent.click(screen.getByLabelText('Copy answer'));
    expect(oncopy).toHaveBeenCalledOnce();
  });

  it('regenerate invokes onregenerate', async () => {
    const onregenerate = vi.fn();
    render(MessageActions, {
      props: { feedback: null, oncopy: vi.fn(), onregenerate, onfeedback: vi.fn() }
    });
    await fireEvent.click(screen.getByLabelText('Regenerate answer'));
    expect(onregenerate).toHaveBeenCalledOnce();
  });

  it('thumbs up invokes onfeedback with "up"', async () => {
    const onfeedback = vi.fn();
    render(MessageActions, {
      props: { feedback: null, oncopy: vi.fn(), onregenerate: vi.fn(), onfeedback }
    });
    await fireEvent.click(screen.getByLabelText('Good response'));
    expect(onfeedback).toHaveBeenCalledWith('up');
  });

  it('thumbs down invokes onfeedback with "down"', async () => {
    const onfeedback = vi.fn();
    render(MessageActions, {
      props: { feedback: null, oncopy: vi.fn(), onregenerate: vi.fn(), onfeedback }
    });
    await fireEvent.click(screen.getByLabelText('Bad response'));
    expect(onfeedback).toHaveBeenCalledWith('down');
  });

  it('reflects the current feedback state via aria-pressed', () => {
    render(MessageActions, {
      props: { feedback: 'up', oncopy: vi.fn(), onregenerate: vi.fn(), onfeedback: vi.fn() }
    });
    expect(screen.getByLabelText('Good response')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByLabelText('Bad response')).toHaveAttribute('aria-pressed', 'false');
  });

  it('disables regenerate when disabled prop is set', () => {
    render(MessageActions, {
      props: {
        feedback: null,
        disabled: true,
        oncopy: vi.fn(),
        onregenerate: vi.fn(),
        onfeedback: vi.fn()
      }
    });
    expect(screen.getByLabelText('Regenerate answer')).toBeDisabled();
  });
});
