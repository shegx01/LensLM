import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import SystemCheckRow from './SystemCheckRow.svelte';
import type { CheckResult } from '$lib/onboarding/system-check.js';

function row(over: Partial<CheckResult>): CheckResult {
  return {
    id: 'llm_runtime',
    label: 'LLM runtime',
    status: 'pass',
    detail: 'Ollama 0.3.2 detected',
    action: null,
    ...over
  };
}

// The icon badge is the leading aria-hidden span; the label is the text node.
function badge(container: HTMLElement): HTMLElement {
  const el = container.querySelector('span[aria-hidden="true"]');
  if (!el) throw new Error('icon badge not found');
  return el as HTMLElement;
}

describe('SystemCheckRow', () => {
  it('renders a Pass row with the green primary badge treatment', () => {
    const { container } = render(SystemCheckRow, { props: { result: row({ status: 'pass' }) } });
    expect(screen.getByText('LLM runtime')).toBeInTheDocument();
    const b = badge(container);
    expect(b.className).toContain('text-primary');
    expect(b.className).toContain('bg-primary/15');
  });

  it('renders a Fail row with the destructive badge and an action button', () => {
    render(SystemCheckRow, {
      props: {
        result: row({
          id: 'embedding_model',
          label: 'Embedding model',
          status: 'fail',
          detail: 'Select and install a model below',
          action: 'choose'
        })
      }
    });
    const b = badge(document.body);
    expect(b.className).toContain('text-destructive');
    expect(screen.getByRole('button', { name: /choose/i })).toBeInTheDocument();
  });

  it('renders configure DISABLED on non-expandable rows (Available in Settings), never a dead no-op', async () => {
    const onaction = vi.fn();
    // configure on text_to_speech (only expandable via `choose`, not `configure`) → disabled
    render(SystemCheckRow, {
      props: {
        result: row({ id: 'text_to_speech', status: 'fail', action: 'configure' }),
        onaction
      }
    });
    const cfg = screen.getByRole('button', { name: /configure/i });
    expect(cfg).toBeDisabled();
    expect(cfg).toHaveAttribute('title', 'Available in Settings');
    await fireEvent.click(cfg);
    expect(onaction).not.toHaveBeenCalled();
  });

  it('embedding_model + choose is expandable (enabled), not disabled', async () => {
    render(SystemCheckRow, {
      props: { result: row({ id: 'embedding_model', status: 'fail', action: 'choose' }) }
    });
    const choose = screen.getByRole('button', { name: /choose/i });
    expect(choose).not.toBeDisabled();
    expect(choose).not.toHaveAttribute('title', 'Available in Settings');
  });

  it('falls back to the Fail (non-Pass) treatment for an unknown status', () => {
    const { container } = render(SystemCheckRow, {
      props: {
        // Force a value outside the CheckStatus union to exercise the fallback.
        result: row({ status: 'unknown' as unknown as CheckResult['status'] })
      }
    });
    const b = badge(container);
    // The fallback is the Fail view — never the green Pass treatment.
    expect(b.className).not.toContain('text-primary');
    expect(b.className).not.toContain('bg-primary');
    expect(b.className).toContain('text-destructive');
    expect(b.className).toContain('bg-destructive');
  });

  it('detail copy carries NO internal milestone vocabulary', () => {
    render(SystemCheckRow, {
      props: {
        result: row({
          id: 'embedding_model',
          status: 'fail',
          detail: 'No embedding model installed',
          action: 'choose'
        })
      }
    });
    const detail = screen.getByText(/no embedding model installed/i);
    expect(detail.textContent).not.toMatch(/\bM\d\b/i);
    expect(detail.textContent?.toLowerCase()).not.toContain('milestone');
  });
});
