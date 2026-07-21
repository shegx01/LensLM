import { describe, expect, it, vi, beforeEach } from 'vitest';
import { enhanceCitations, type CitationTarget } from './citation-inline.js';

vi.mock('$lib/sources/sources-state.svelte.js', () => ({
  focusSource: vi.fn()
}));

import { focusSource } from '$lib/sources/sources-state.svelte.js';

const resolve = (n: number): CitationTarget | null => {
  if (n === 1) return { source_id: 's1', title: 'Alpha', live: true, locators: [] };
  if (n === 2) return { source_id: 's2', title: 'Gone', live: false, locators: [] };
  return null;
};

function host(html: string): HTMLElement {
  const el = document.createElement('div');
  el.innerHTML = html;
  return el;
}

beforeEach(() => vi.clearAllMocks());

describe('enhanceCitations', () => {
  it('replaces a real [n] marker with a numbered chip button', () => {
    const el = host('<p>See [1].</p>');
    enhanceCitations(el, resolve);

    const chip = el.querySelector('button.citation-chip');
    expect(chip).not.toBeNull();
    expect(chip?.textContent).toBe('1');
    expect(chip?.getAttribute('aria-label')).toBe('Source 1: Alpha');
    expect(el.textContent).not.toContain('[1]');
  });

  it('leaves a non-citation bracket (resolver returns null) as literal text', () => {
    const el = host('<p>arr[3] and [1]</p>');
    enhanceCitations(el, resolve);

    expect(el.textContent).toContain('arr[3]');
    expect(el.querySelectorAll('button.citation-chip')).toHaveLength(1);
  });

  it('never touches markers inside code or pre', () => {
    const el = host('<p>text [1]</p><pre><code>arr[1]</code></pre>');
    enhanceCitations(el, resolve);

    expect(el.querySelector('pre')?.textContent).toBe('arr[1]');
    // Only the prose marker became a chip.
    expect(el.querySelectorAll('button.citation-chip')).toHaveLength(1);
  });

  it('is idempotent — re-running does not double-inject', () => {
    const el = host('<p>See [1] and [1].</p>');
    enhanceCitations(el, resolve);
    enhanceCitations(el, resolve);

    expect(el.querySelectorAll('button.citation-chip')).toHaveLength(2);
  });

  it('renders a disabled chip for a removed source and does not activate', () => {
    const el = host('<p>Gone [2].</p>');
    document.body.append(el);
    enhanceCitations(el, resolve);

    const chip = el.querySelector('button.citation-chip') as HTMLButtonElement;
    expect(chip.disabled).toBe(true);
    expect(chip.getAttribute('aria-label')).toBe('Source 2: Gone (unavailable)');
    chip.click();
    expect(focusSource).not.toHaveBeenCalled();
    el.remove();
  });

  it('reveals the source when a live chip is clicked', () => {
    const el = host('<p>See [1].</p>');
    document.body.append(el);
    enhanceCitations(el, resolve);

    (el.querySelector('button.citation-chip') as HTMLButtonElement).click();
    expect(focusSource).toHaveBeenCalledWith('s1');
    el.remove();
  });

  it('injects only elements — chip text is the number, never markup', () => {
    const el = host('<p>See [1].</p>');
    enhanceCitations(el, resolve);
    const chip = el.querySelector('button.citation-chip') as HTMLButtonElement;

    expect(chip.tagName).toBe('BUTTON');
    expect(chip.childNodes).toHaveLength(1);
    expect(chip.childNodes[0].nodeType).toBe(Node.TEXT_NODE);
  });

  it('strips a trailing (title) echo that exactly matches the source title', () => {
    const el = host('<p>See [1] (Alpha).</p>');
    enhanceCitations(el, resolve);

    expect(el.querySelector('button.citation-chip')?.textContent).toBe('1');
    expect(el.textContent).not.toContain('(Alpha)');
    // The chip hugs the preceding word (leading space consumed by design), so the
    // title echo is gone and only "See" + chip + "." remains.
    expect(el.textContent).toBe('See1.');
  });

  it('strips the echoed (title): label including the trailing colon', () => {
    const el = host('<p>See [1] (Alpha): the rest.</p>');
    enhanceCitations(el, resolve);

    expect(el.querySelector('button.citation-chip')?.textContent).toBe('1');
    expect(el.textContent).not.toContain('(Alpha)');
    expect(el.textContent).toBe('See1 the rest.');
  });

  it('never mangles surrounding prose like "open source" (title echo present)', () => {
    const el = host('<p>Use an open source [1] (Alpha) library.</p>');
    enhanceCitations(el, resolve);

    expect(el.querySelector('button.citation-chip')).not.toBeNull();
    // The (title) echo is dropped but the real words "open source" are untouched.
    expect(el.textContent).not.toContain('(Alpha)');
    expect(el.textContent).toContain('open source');
    expect(el.textContent).toBe('Use an open source1 library.');
  });

  it('does not strip a parenthetical that is not the source title', () => {
    const el = host('<p>See [1] (the diagram).</p>');
    enhanceCitations(el, resolve);

    expect(el.textContent).toContain('(the diagram)');
  });

  it('re-enhancing with a changed resolver flips a chip live→stale without duplicating (AC5)', () => {
    const el = host('<p>See [1].</p>');
    const live = (n: number) =>
      n === 1 ? { source_id: 's1', title: 'Alpha', live: true, locators: [] } : null;
    const stale = (n: number) =>
      n === 1 ? { source_id: 's1', title: 'Alpha', live: false, locators: [] } : null;

    enhanceCitations(el, live);
    expect((el.querySelector('button.citation-chip') as HTMLButtonElement).disabled).toBe(false);

    // Simulates the source being removed: the AssistantMessage $effect re-runs
    // enhanceCitations, which restores markers then re-injects with fresh liveness.
    enhanceCitations(el, stale);
    const chip = el.querySelector('button.citation-chip') as HTMLButtonElement;
    expect(chip.disabled).toBe(true);
    expect(el.querySelectorAll('button.citation-chip')).toHaveLength(1);
  });
});
