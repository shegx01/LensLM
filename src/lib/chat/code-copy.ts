// Per-code-block enhancement for {@html}-rendered chat answers: wraps each <pre>
// in a collapsible panel (header with language + chevron) and adds a "copy code"
// button. Because the markdown is injected via {@html}, Svelte handlers can't bind
// to the inner <pre>; we enhance imperatively after render. A <pre><code>'s
// textContent reconstructs the original source (hljs <span>s are styling only), so
// that is what we copy — never the rendered/highlighted markup.

// lucide icon paths, hand-inlined because an imperatively-injected button can't mount a Svelte icon component.
const COPY_ICON =
  '<svg class="code-copy-icon code-copy-icon--copy" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>';

const CHECK_ICON =
  '<svg class="code-copy-icon code-copy-icon--check" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>';

// `braces` for JSON (the `{}` glyph genuinely denotes a JSON object); `code`
// (`</>`) for every other language so the icon doesn't misrepresent the block.
const BRACES_ICON =
  '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1"/><path d="M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1"/></svg>';

const CODE_ICON =
  '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m18 16 4-4-4-4"/><path d="m6 8-4 4 4 4"/><path d="m14.5 4-5 16"/></svg>';

const CHEVRON_ICON =
  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>';

const REVERT_MS = 1500;

let panelSeq = 0;

/** Friendly display label + matching icon kind for a fenced-code language hint. */
function languageInfo(codeEl: HTMLElement | null): {
  label: string;
  iconKind: 'braces' | 'code';
} {
  const match = /language-([\w-]+)/.exec(codeEl?.className ?? '');
  if (!match) return { label: 'Code', iconKind: 'code' };
  const raw = match[1].toLowerCase();
  const friendly: Record<string, string> = {
    js: 'JavaScript',
    javascript: 'JavaScript',
    ts: 'TypeScript',
    typescript: 'TypeScript',
    py: 'Python',
    python: 'Python',
    rs: 'Rust',
    rust: 'Rust',
    sh: 'Shell',
    bash: 'Shell',
    json: 'JSON',
    yaml: 'YAML',
    yml: 'YAML',
    html: 'HTML',
    css: 'CSS',
    sql: 'SQL',
    md: 'Markdown',
    markdown: 'Markdown'
  };
  return {
    label: friendly[raw] ?? raw.toUpperCase(),
    iconKind: raw === 'json' ? 'braces' : 'code'
  };
}

/**
 * Wraps every un-enhanced `<pre>` under `root` in a collapsible code panel (header
 * + copy button). Idempotent (guarded per-`<pre>` via a dataset flag). Returns a
 * cleanup that aborts all listeners and clears pending "Copied" reverts.
 */
export function enhanceCodeBlocks(root: HTMLElement): () => void {
  const controller = new AbortController();
  const timers = new Set<ReturnType<typeof setTimeout>>();

  for (const pre of root.querySelectorAll('pre')) {
    if (pre.dataset.copyEnhanced === 'true') continue;
    pre.dataset.copyEnhanced = 'true';

    const parent = pre.parentNode;
    if (!parent) continue;

    // Expanded by default (code is usually the answer); the header toggle collapses it.
    const panel = document.createElement('div');
    panel.className = 'code-block';
    panel.dataset.expanded = 'true';

    const bodyId = `code-block-body-${(panelSeq += 1)}`;

    const header = document.createElement('button');
    header.type = 'button';
    header.className = 'code-block__header';
    header.setAttribute('aria-expanded', 'true');
    header.setAttribute('aria-controls', bodyId);

    const info = languageInfo(pre.querySelector('code'));

    const icon = document.createElement('span');
    icon.className = 'code-block__icon';
    icon.dataset.icon = info.iconKind;
    icon.innerHTML = info.iconKind === 'braces' ? BRACES_ICON : CODE_ICON;

    const lang = document.createElement('span');
    lang.className = 'code-block__lang';
    lang.textContent = info.label;

    const chevron = document.createElement('span');
    chevron.className = 'code-block__chevron';
    chevron.innerHTML = CHEVRON_ICON;

    header.append(icon, lang, chevron);
    header.setAttribute('aria-label', `Toggle ${lang.textContent} code block`);

    const body = document.createElement('div');
    body.className = 'code-block__body';
    body.id = bodyId;

    parent.insertBefore(panel, pre);
    body.appendChild(pre);
    panel.append(header, body);

    header.addEventListener(
      'click',
      () => {
        const next = panel.dataset.expanded !== 'true';
        panel.dataset.expanded = String(next);
        header.setAttribute('aria-expanded', String(next));
      },
      { signal: controller.signal }
    );

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'code-copy-btn';
    button.setAttribute('aria-label', 'Copy code');
    button.innerHTML = COPY_ICON + CHECK_ICON;

    button.addEventListener(
      'click',
      async () => {
        const text = pre.querySelector('code')?.textContent ?? '';
        try {
          await navigator.clipboard.writeText(text);
        } catch (err) {
          console.warn('enhanceCodeBlocks: clipboard write failed', err);
          return;
        }
        if (controller.signal.aborted) return;
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
