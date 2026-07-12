import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import MessageActions from './MessageActions.svelte';

afterEach(() => {
  vi.restoreAllMocks();
});

function baseProps(overrides?: Record<string, unknown>) {
  return {
    feedback: null,
    saved: false,
    oncopy: vi.fn(),
    onregenerate: vi.fn(),
    onfeedback: vi.fn(),
    onsave: vi.fn(),
    ...overrides
  };
}

describe('MessageActions', () => {
  it('copy invokes oncopy', async () => {
    const oncopy = vi.fn();
    render(MessageActions, { props: baseProps({ oncopy }) });
    await fireEvent.click(screen.getByLabelText('Copy answer'));
    expect(oncopy).toHaveBeenCalledOnce();
  });

  it('regenerate invokes onregenerate', async () => {
    const onregenerate = vi.fn();
    render(MessageActions, { props: baseProps({ onregenerate }) });
    await fireEvent.click(screen.getByLabelText('Regenerate answer'));
    expect(onregenerate).toHaveBeenCalledOnce();
  });

  it('thumbs up invokes onfeedback with "up"', async () => {
    const onfeedback = vi.fn();
    render(MessageActions, { props: baseProps({ onfeedback }) });
    await fireEvent.click(screen.getByLabelText('Good response'));
    expect(onfeedback).toHaveBeenCalledWith('up');
  });

  it('thumbs down invokes onfeedback with "down"', async () => {
    const onfeedback = vi.fn();
    render(MessageActions, { props: baseProps({ onfeedback }) });
    await fireEvent.click(screen.getByLabelText('Bad response'));
    expect(onfeedback).toHaveBeenCalledWith('down');
  });

  it('reflects the current feedback state via aria-pressed', () => {
    render(MessageActions, { props: baseProps({ feedback: 'up' }) });
    expect(screen.getByLabelText('Good response')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByLabelText('Bad response')).toHaveAttribute('aria-pressed', 'false');
  });

  it('disables regenerate when disabled prop is set', () => {
    render(MessageActions, { props: baseProps({ disabled: true }) });
    expect(screen.getByLabelText('Regenerate answer')).toBeDisabled();
  });

  it('renders the save button and invokes onsave', async () => {
    const onsave = vi.fn();
    render(MessageActions, { props: baseProps({ onsave }) });
    await fireEvent.click(screen.getByLabelText('Save to notes'));
    expect(onsave).toHaveBeenCalledOnce();
  });

  it('reflects saved=true via aria-pressed and label', () => {
    render(MessageActions, { props: baseProps({ saved: true }) });
    const btn = screen.getByLabelText('Remove from notes');
    expect(btn).toHaveAttribute('aria-pressed', 'true');
  });

  it('reflects saved=false via aria-pressed and label', () => {
    render(MessageActions, { props: baseProps({ saved: false }) });
    const btn = screen.getByLabelText('Save to notes');
    expect(btn).toHaveAttribute('aria-pressed', 'false');
  });

  it('does not render the save button when finalized=false (streaming bubble)', () => {
    render(MessageActions, { props: baseProps({ finalized: false }) });
    expect(screen.queryByLabelText('Save to notes')).toBeNull();
    expect(screen.queryByLabelText('Remove from notes')).toBeNull();
  });
});
