// Component wiring for per-block copy buttons. renderMarkdown is stubbed to a
// raw <pre><code> string because DOMPurify strips <pre> under happy-dom (an
// environment quirk, not real-app behavior); stubbing isolates the $effect +
// highlightCode gate from that quirk. Module-level behavior is covered in
// ../../chat/code-copy.test.ts.

import { render, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/chat/render-markdown.js', () => ({
  renderMarkdown: (source: string) => `<pre><code>${source}</code></pre>`
}));

import AssistantMessage from './AssistantMessage.svelte';
import { makeChatMessage } from '$lib/chat/test-fixtures.js';

let writeText: ReturnType<typeof vi.fn>;

function props(overrides?: Record<string, unknown>) {
  return {
    versions: [makeChatMessage({ role: 'assistant', content: 'const x = 1;\n' })],
    oncopy: vi.fn(),
    onregenerate: vi.fn(),
    onfeedback: vi.fn(),
    ...overrides
  };
}

beforeEach(() => {
  writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal('navigator', { clipboard: { writeText } });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('AssistantMessage code-copy', () => {
  it('adds a copy button to a fenced block for a settled (highlighted) answer', async () => {
    const { container } = render(AssistantMessage, { props: props() });
    await waitFor(() => expect(container.querySelectorAll('.code-copy-btn')).toHaveLength(1));
  });

  it('clicking the block copy button copies the block source', async () => {
    const { container } = render(AssistantMessage, { props: props() });
    const btn = await waitFor(() => {
      const b = container.querySelector<HTMLButtonElement>('.code-copy-btn');
      if (!b) throw new Error('no button yet');
      return b;
    });
    btn.click();
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('const x = 1;\n'));
  });

  it('renders no block copy button for a streaming answer (highlightCode=false)', async () => {
    const { container } = render(AssistantMessage, {
      props: props({ highlightCode: false })
    });
    await Promise.resolve();
    expect(container.querySelectorAll('.code-copy-btn')).toHaveLength(0);
  });
});
