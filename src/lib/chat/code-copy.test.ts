import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { enhanceCodeBlocks } from './code-copy.js';

function makeContainer(html: string): HTMLElement {
  const el = document.createElement('div');
  el.innerHTML = html;
  document.body.appendChild(el);
  return el;
}

let writeText: ReturnType<typeof vi.fn>;

beforeEach(() => {
  writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal('navigator', { clipboard: { writeText } });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('enhanceCodeBlocks', () => {
  it('adds exactly one copy button to a single pre block', () => {
    const root = makeContainer('<pre><code>const x = 1;\n</code></pre>');
    enhanceCodeBlocks(root);
    expect(root.querySelectorAll('.code-copy-btn')).toHaveLength(1);
  });

  it('copies only the block source, not the message', async () => {
    const root = makeContainer('<pre><code>const x = 1;\n</code></pre>');
    enhanceCodeBlocks(root);
    root.querySelector<HTMLButtonElement>('.code-copy-btn')!.click();
    await vi.waitFor(() => expect(writeText).toHaveBeenCalledWith('const x = 1;\n'));
  });

  it('toggles data-copied true then reverts after the timeout', async () => {
    vi.useFakeTimers();
    const root = makeContainer('<pre><code>const x = 1;\n</code></pre>');
    enhanceCodeBlocks(root);
    const btn = root.querySelector<HTMLButtonElement>('.code-copy-btn')!;
    btn.click();
    await vi.waitFor(() => expect(btn.dataset.copied).toBe('true'));
    expect(btn.getAttribute('aria-label')).toBe('Copied');
    vi.advanceTimersByTime(1500);
    expect(btn.dataset.copied).toBeUndefined();
    expect(btn.getAttribute('aria-label')).toBe('Copy code');
    vi.useRealTimers();
  });

  it('is idempotent — a second call adds no second button', () => {
    const root = makeContainer('<pre><code>const x = 1;\n</code></pre>');
    enhanceCodeBlocks(root);
    enhanceCodeBlocks(root);
    expect(root.querySelectorAll('.code-copy-btn')).toHaveLength(1);
  });

  it('cleanup aborts listeners — a post-cleanup click does not copy', () => {
    const root = makeContainer('<pre><code>const x = 1;\n</code></pre>');
    const cleanup = enhanceCodeBlocks(root);
    cleanup();
    root.querySelector<HTMLButtonElement>('.code-copy-btn')!.click();
    expect(writeText).not.toHaveBeenCalled();
  });

  it('gives two pre blocks two independent buttons copying their own text', async () => {
    const root = makeContainer(
      '<pre><code>alpha();\n</code></pre><pre><code>beta();\n</code></pre>'
    );
    enhanceCodeBlocks(root);
    const btns = root.querySelectorAll<HTMLButtonElement>('.code-copy-btn');
    expect(btns).toHaveLength(2);
    btns[0].click();
    await vi.waitFor(() => expect(writeText).toHaveBeenCalledWith('alpha();\n'));
    btns[1].click();
    await vi.waitFor(() => expect(writeText).toHaveBeenCalledWith('beta();\n'));
  });

  it('does not add a button to inline code outside a pre', () => {
    const root = makeContainer('<p>use <code>const x = 1</code> here</p>');
    enhanceCodeBlocks(root);
    expect(root.querySelectorAll('.code-copy-btn')).toHaveLength(0);
  });

  it('does not enter the copied state when the clipboard write fails', async () => {
    writeText.mockRejectedValue(new Error('denied'));
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const root = makeContainer('<pre><code>const x = 1;\n</code></pre>');
    enhanceCodeBlocks(root);
    const btn = root.querySelector<HTMLButtonElement>('.code-copy-btn')!;
    btn.click();
    await vi.waitFor(() => expect(writeText).toHaveBeenCalled());
    expect(btn.dataset.copied).toBeUndefined();
  });
});
