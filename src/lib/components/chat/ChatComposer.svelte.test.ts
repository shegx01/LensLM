import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ChatComposer from './ChatComposer.svelte';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ChatComposer', () => {
  it('Enter sends the trimmed question', async () => {
    const onsend = vi.fn();
    render(ChatComposer, { props: { streaming: false, onsend, onstop: vi.fn() } });
    const textarea = screen.getByLabelText(/ask a question/i);
    await fireEvent.input(textarea, { target: { value: '  What is this?  ' } });
    await fireEvent.keyDown(textarea, { key: 'Enter' });
    expect(onsend).toHaveBeenCalledWith('What is this?');
  });

  it('Shift+Enter inserts a newline instead of sending', async () => {
    const onsend = vi.fn();
    render(ChatComposer, { props: { streaming: false, onsend, onstop: vi.fn() } });
    const textarea = screen.getByLabelText(/ask a question/i);
    await fireEvent.input(textarea, { target: { value: 'line one' } });
    await fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });
    expect(onsend).not.toHaveBeenCalled();
  });

  it('empty/whitespace-only input cannot send', async () => {
    const onsend = vi.fn();
    render(ChatComposer, { props: { streaming: false, onsend, onstop: vi.fn() } });
    const textarea = screen.getByLabelText(/ask a question/i);
    await fireEvent.input(textarea, { target: { value: '   ' } });
    await fireEvent.keyDown(textarea, { key: 'Enter' });
    expect(onsend).not.toHaveBeenCalled();

    const sendButton = screen.getByLabelText('Send question');
    expect(sendButton).toBeDisabled();
  });

  it('renders a Stop button instead of Send while streaming, and it calls onstop', async () => {
    const onstop = vi.fn();
    render(ChatComposer, { props: { streaming: true, onsend: vi.fn(), onstop } });
    expect(screen.queryByLabelText('Send question')).not.toBeInTheDocument();
    const stopButton = screen.getByLabelText('Stop generating');
    await fireEvent.click(stopButton);
    expect(onstop).toHaveBeenCalledOnce();
  });

  it('disables the textarea while streaming', () => {
    render(ChatComposer, { props: { streaming: true, onsend: vi.fn(), onstop: vi.fn() } });
    expect(screen.getByLabelText(/ask a question/i)).toBeDisabled();
  });

  it('renders the add-source and sources-scope tools', () => {
    render(ChatComposer, { props: { streaming: false, onsend: vi.fn(), onstop: vi.fn() } });
    expect(screen.getByLabelText('Add source')).toBeInTheDocument();
    // Empty store → honest "No sources yet" label on the scope chip.
    expect(screen.getByLabelText('Sources used for this question')).toHaveTextContent(
      /no sources yet/i
    );
  });

  it('offers voice as an honestly-disabled affordance (STT not yet shipped)', () => {
    render(ChatComposer, { props: { streaming: false, onsend: vi.fn(), onstop: vi.fn() } });
    expect(screen.getByLabelText(/dictate question/i)).toBeDisabled();
  });
});
