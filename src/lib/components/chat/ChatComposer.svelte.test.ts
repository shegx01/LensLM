import { render, screen, fireEvent } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { notebookStore } from '$lib/notebooks/index.js';
import { resetNotebookStore } from '$lib/notebooks/notebooks-state.svelte.js';
import { refreshChatProvider, resetChatProvider } from '$lib/models/chat-provider.svelte.js';
import ChatComposer from './ChatComposer.svelte';

// Drive the module-level chat-provider signal via the mocked command.
async function setProvider(available: boolean): Promise<void> {
  mockIPC((cmd) => {
    if (cmd === 'has_chat_provider') return available;
    return undefined;
  });
  await refreshChatProvider();
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  resetChatProvider();
  resetNotebookStore();
  delete (globalThis as { isTauri?: boolean }).isTauri;
  vi.restoreAllMocks();
});

describe('ChatComposer', () => {
  it('Enter sends the trimmed question', async () => {
    await setProvider(true);
    const onsend = vi.fn();
    render(ChatComposer, { props: { streaming: false, onsend, onstop: vi.fn() } });
    const textarea = screen.getByLabelText(/ask a question/i);
    await fireEvent.input(textarea, { target: { value: '  What is this?  ' } });
    await fireEvent.keyDown(textarea, { key: 'Enter' });
    expect(onsend).toHaveBeenCalledWith('What is this?');
  });

  it('Shift+Enter inserts a newline instead of sending', async () => {
    await setProvider(true);
    const onsend = vi.fn();
    render(ChatComposer, { props: { streaming: false, onsend, onstop: vi.fn() } });
    const textarea = screen.getByLabelText(/ask a question/i);
    await fireEvent.input(textarea, { target: { value: 'line one' } });
    await fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });
    expect(onsend).not.toHaveBeenCalled();
  });

  it('empty/whitespace-only input cannot send', async () => {
    await setProvider(true);
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
    await setProvider(true);
    const onstop = vi.fn();
    render(ChatComposer, { props: { streaming: true, onsend: vi.fn(), onstop } });
    expect(screen.queryByLabelText('Send question')).not.toBeInTheDocument();
    const stopButton = screen.getByLabelText('Stop generating');
    await fireEvent.click(stopButton);
    expect(onstop).toHaveBeenCalledOnce();
  });

  it('disables the textarea while streaming', async () => {
    await setProvider(true);
    render(ChatComposer, { props: { streaming: true, onsend: vi.fn(), onstop: vi.fn() } });
    expect(screen.getByLabelText(/ask a question/i)).toBeDisabled();
  });

  it('renders the add-source and sources-scope tools', async () => {
    await setProvider(true);
    render(ChatComposer, { props: { streaming: false, onsend: vi.fn(), onstop: vi.fn() } });
    expect(screen.getByLabelText('Add source')).toBeInTheDocument();
    // Empty store → honest "No sources yet" label on the scope chip.
    expect(screen.getByLabelText('Sources used for this question')).toHaveTextContent(
      /no sources yet/i
    );
  });

  it('offers voice as an honestly-disabled affordance (STT not yet shipped)', async () => {
    await setProvider(true);
    render(ChatComposer, { props: { streaming: false, onsend: vi.fn(), onstop: vi.fn() } });
    expect(screen.getByLabelText(/dictate question/i)).toBeDisabled();
  });

  describe('AC-11 — no usable chat provider', () => {
    it('disables Send and shows a Settings CTA that deep-links to AI Model', async () => {
      await setProvider(false);
      render(ChatComposer, { props: { streaming: false, onsend: vi.fn(), onstop: vi.fn() } });

      const textarea = screen.getByLabelText(/ask a question/i);
      await fireEvent.input(textarea, { target: { value: 'hello?' } });
      expect(screen.getByLabelText('Send question')).toBeDisabled();

      const cta = screen.getByRole('button', { name: /set up a model in settings/i });
      await fireEvent.click(cta);
      expect(notebookStore.settingsOpen).toBe(true);
      expect(notebookStore.settingsSection).toBe('ai');
    });

    it('a present-but-unusable entry (has_chat_provider=false) keeps Send disabled', async () => {
      // Engine returns false even though models[] may be non-empty upstream.
      await setProvider(false);
      const onsend = vi.fn();
      render(ChatComposer, { props: { streaming: false, onsend, onstop: vi.fn() } });
      const textarea = screen.getByLabelText(/ask a question/i);
      await fireEvent.input(textarea, { target: { value: 'question' } });
      await fireEvent.keyDown(textarea, { key: 'Enter' });
      expect(onsend).not.toHaveBeenCalled();
      expect(screen.getByLabelText('Send question')).toBeDisabled();
    });

    it('a usable provider enables Send and hides the CTA', async () => {
      await setProvider(true);
      render(ChatComposer, { props: { streaming: false, onsend: vi.fn(), onstop: vi.fn() } });
      await fireEvent.input(screen.getByLabelText(/ask a question/i), {
        target: { value: 'hello?' }
      });
      expect(screen.getByLabelText('Send question')).not.toBeDisabled();
      expect(
        screen.queryByRole('button', { name: /set up a model in settings/i })
      ).not.toBeInTheDocument();
    });
  });
});
