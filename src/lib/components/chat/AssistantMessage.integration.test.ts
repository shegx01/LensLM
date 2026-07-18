// @vitest-environment jsdom
//
// End-to-end proof for #182: mounts the REAL AssistantMessage against hostile
// content and asserts the {@html} sink stays inert. renderMarkdown is spied but
// NOT replaced (it runs for real); only orthogonal async effects (mermaid,
// code-copy — covered by their own suites) are stubbed to keep this deterministic.

import { render } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/chat/mermaid.js', () => ({
  hydrateMermaid: vi.fn().mockResolvedValue(undefined)
}));

vi.mock('$lib/chat/code-copy.js', () => ({
  enhanceCodeBlocks: vi.fn(() => () => {})
}));

// Spy on renderMarkdown while keeping the real implementation, so the wiring
// assertion can prove the component routes `content` through the real sanitizer
// without coupling to the exact serialized DOM (which post-render effects mutate).
vi.mock('$lib/chat/render-markdown.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/chat/render-markdown.js')>();
  return { ...actual, renderMarkdown: vi.fn(actual.renderMarkdown) };
});

// Citation-free content, so the sources store is inert.
vi.mock('$lib/sources/sources-state.svelte.js', () => ({
  sourcesStore: {
    get sources() {
      return [];
    }
  },
  focusSource: vi.fn()
}));

import AssistantMessage from './AssistantMessage.svelte';
import { makeChatMessage } from '$lib/chat/test-fixtures.js';
import { renderMarkdown } from '$lib/chat/render-markdown.js';

// script + onerror handler + javascript: link + data: link + a fenced block, so
// the test also proves legitimate structure survives alongside the attack surface.
const HOSTILE_CONTENT = [
  '<script>window.__hostileExec = true;</script>',
  '<img src="x" onerror="window.__hostileExec = true">',
  '<a href="javascript:window.__hostileExec = true">click</a>',
  '<a href="data:text/html,<b>x</b>">data link</a>',
  '```js\nconst x = 1;\n```'
].join('\n\n');

function props(overrides?: Record<string, unknown>) {
  return {
    notebookId: 'nb-001',
    versions: [makeChatMessage({ role: 'assistant', content: HOSTILE_CONTENT })],
    oncopy: vi.fn(),
    onregenerate: vi.fn(),
    onfeedback: vi.fn(),
    ...overrides
  };
}

beforeEach(() => {
  vi.stubGlobal('navigator', {
    clipboard: { writeText: vi.fn().mockResolvedValue(undefined) }
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  delete (globalThis as unknown as { __hostileExec?: boolean }).__hostileExec;
});

describe('AssistantMessage — end-to-end never-execute invariant (#182)', () => {
  it('renders hostile content inert through the real {@html} sink', () => {
    const { container } = render(AssistantMessage, { props: props(), intro: false });

    expect(container.querySelector('script')).toBeNull();

    for (const el of Array.from(container.querySelectorAll('*'))) {
      for (const attr of Array.from(el.attributes)) {
        expect(attr.name.toLowerCase().startsWith('on')).toBe(false);
      }
    }

    for (const a of Array.from(container.querySelectorAll('a'))) {
      const href = a.getAttribute('href');
      if (href !== null) {
        expect(href.startsWith('javascript:')).toBe(false);
        expect(href.startsWith('data:')).toBe(false);
      }
    }

    expect(container.querySelector('pre code')).not.toBeNull();

    // Negative control: nothing in the payload executed. Vacuous under jsdom on its
    // own, but confirms the structural asserts above weren't reached via a no-op mount.
    expect((globalThis as unknown as { __hostileExec?: boolean }).__hostileExec).toBeUndefined();

    // Wiring: the component invoked the real renderMarkdown with the message content
    // (default highlightCode=true), proving it routes through the sanitizer rather
    // than a stubbed/short-circuited path — unprovable by a renderMarkdown unit test.
    expect(renderMarkdown).toHaveBeenCalledWith(HOSTILE_CONTENT, { highlight: true });
  });
});
