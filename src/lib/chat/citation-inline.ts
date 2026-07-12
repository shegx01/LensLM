// Post-render enhancer: turns inline `[n]` citation markers in already-sanitized
// assistant markdown into compact, clickable superscript chips, positioned exactly
// where the model cited. Chips are built with createElement + textContent only
// (never innerHTML), so model text cannot inject markup or script — a chip is
// display + focusSource only. Re-runnable: each call first restores any chips it
// previously injected back to `[n]` text, so it stays correct across re-renders and
// live source-list changes.

import { focusSource } from '$lib/sources/sources-state.svelte.js';

export interface CitationTarget {
  source_id: string;
  title: string;
  live: boolean;
}

const SKIP_TAGS = new Set(['CODE', 'PRE', 'A', 'BUTTON']);
const HAS_MARKER = /\[\d+\]/;
// A real `[n]` marker, not a markdown link `[1](url)` or reference def `[1]: url`.
// The optional leading space is consumed so the chip hugs the preceding word; the
// ordinal set (via `resolve` returning null) is the authoritative citation filter.
const MARKER = /[ \t]?\[(\d+)\](?!\(|\s*:)/g;

// dataset.marker holds the FULL original match (incl. any consumed leading space),
// so restoring is lossless; normalize() re-merges split text nodes for re-matching.
function restoreChips(container: HTMLElement): void {
  const chips = container.querySelectorAll<HTMLElement>('button.citation-chip');
  for (const chip of chips) {
    chip.replaceWith(document.createTextNode(chip.dataset.marker ?? `[${chip.textContent ?? ''}]`));
  }
  if (chips.length > 0) container.normalize();
}

export function enhanceCitations(
  container: HTMLElement,
  resolve: (ordinal: number) => CitationTarget | null
): () => void {
  restoreChips(container);

  const cleanups: Array<() => void> = [];

  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      for (let el = node.parentElement; el && el !== container; el = el.parentElement) {
        if (SKIP_TAGS.has(el.tagName)) return NodeFilter.FILTER_REJECT;
      }
      return HAS_MARKER.test(node.nodeValue ?? '')
        ? NodeFilter.FILTER_ACCEPT
        : NodeFilter.FILTER_REJECT;
    }
  });

  const textNodes: Text[] = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) textNodes.push(n as Text);

  for (const textNode of textNodes) {
    const text = textNode.nodeValue ?? '';
    MARKER.lastIndex = 0;
    const frag = document.createDocumentFragment();
    let lastIndex = 0;
    let matched = false;
    let m: RegExpExecArray | null;
    while ((m = MARKER.exec(text)) !== null) {
      const target = resolve(Number(m[1]));
      if (!target) continue;
      matched = true;
      frag.append(document.createTextNode(text.slice(lastIndex, m.index)));
      frag.append(buildChip(Number(m[1]), m[0], target, cleanups));
      lastIndex = m.index + m[0].length;
    }
    if (matched) {
      frag.append(document.createTextNode(text.slice(lastIndex)));
      textNode.replaceWith(frag);
    }
  }

  return () => {
    for (const c of cleanups) c();
  };
}

function buildChip(
  ordinal: number,
  marker: string,
  target: CitationTarget,
  cleanups: Array<() => void>
): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'citation-chip';
  btn.dataset.marker = marker;
  btn.textContent = String(ordinal);
  if (target.live) {
    btn.setAttribute('aria-label', `Source ${ordinal}: ${target.title}`);
    btn.title = target.title;
    const onClick = (): void => focusSource(target.source_id);
    btn.addEventListener('click', onClick);
    cleanups.push(() => btn.removeEventListener('click', onClick));
  } else {
    btn.disabled = true;
    btn.classList.add('citation-chip--stale');
    btn.setAttribute('aria-label', `Source ${ordinal}: ${target.title} (unavailable)`);
    btn.title = 'Source no longer available';
  }
  return btn;
}
