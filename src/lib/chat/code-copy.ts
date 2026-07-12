// Per-code-block "copy code" buttons for {@html}-rendered chat answers. Because
// the markdown is injected via {@html}, Svelte handlers can't bind to the inner
// <pre>; we enhance imperatively after render. A <pre><code>'s textContent
// reconstructs the original source (hljs <span>s are styling only), so that is
// what we copy — never the rendered/highlighted markup.

const COPY_ICON =
  '<svg class="code-copy-icon code-copy-icon--copy" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>';

const CHECK_ICON =
  '<svg class="code-copy-icon code-copy-icon--check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>';

const REVERT_MS = 1500;

/**
 * Adds a copy-code button to every un-enhanced `<pre>` under `root`. Idempotent
 * (guarded per-`<pre>` via a dataset flag). Returns a cleanup that aborts all
 * listeners and clears pending "Copied" reverts.
 */
export function enhanceCodeBlocks(root: HTMLElement): () => void {
  const controller = new AbortController();
  const timers = new Set<ReturnType<typeof setTimeout>>();

  for (const pre of root.querySelectorAll('pre')) {
    if (pre.dataset.copyEnhanced === 'true') continue;
    pre.dataset.copyEnhanced = 'true';
    pre.classList.add('code-block');

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'code-copy-btn';
    button.setAttribute('aria-label', 'Copy code');
    button.innerHTML = COPY_ICON + CHECK_ICON;

    button.addEventListener(
      'click',
      async () => {
        const text = pre.querySelector('code')?.textContent ?? pre.textContent ?? '';
        try {
          await navigator.clipboard.writeText(text);
        } catch (err) {
          console.warn('enhanceCodeBlocks: clipboard write failed', err);
          return;
        }
        button.dataset.copied = 'true';
        button.setAttribute('aria-label', 'Copied');
        const timer = setTimeout(() => {
          delete button.dataset.copied;
          button.setAttribute('aria-label', 'Copy code');
          timers.delete(timer);
        }, REVERT_MS);
        timers.add(timer);
      },
      { signal: controller.signal }
    );

    pre.appendChild(button);
  }

  return () => {
    controller.abort();
    for (const timer of timers) clearTimeout(timer);
    timers.clear();
  };
}
