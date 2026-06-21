import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import SystemCheckRow from './SystemCheckRow.svelte';
import type { CheckResult } from '$lib/onboarding/system-check.js';

function row(over: Partial<CheckResult>): CheckResult {
  return {
    id: 'local_backend',
    label: 'Local backend',
    status: 'pass',
    detail: 'In-process engine ready',
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
    expect(screen.getByText('Local backend')).toBeInTheDocument();
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

  it('renders a Pending row visually DISTINCT from Pass (muted, not green)', () => {
    const { container } = render(SystemCheckRow, {
      props: {
        result: row({
          id: 'vector_database',
          label: 'Vector database',
          status: 'pending',
          detail: 'Built-in · set up automatically when you add your first source',
          action: null
        })
      }
    });
    const b = badge(container);
    // HARD REQUIREMENT (plan change #13): Pending must NOT use the Pass treatment.
    expect(b.className).not.toContain('text-primary');
    expect(b.className).not.toContain('bg-primary');
    // It uses the muted/neutral treatment instead.
    expect(b.className).toContain('text-muted-foreground');
    expect(b.className).toContain('bg-muted');
  });

  it('Pending detail copy carries NO internal milestone vocabulary', () => {
    render(SystemCheckRow, {
      props: {
        result: row({
          id: 'vector_database',
          status: 'pending',
          detail: 'Built-in · set up automatically when you add your first source',
          action: null
        })
      }
    });
    const detail = screen.getByText(/set up automatically/i);
    expect(detail.textContent).not.toMatch(/\bM\d\b/i);
    expect(detail.textContent?.toLowerCase()).not.toContain('milestone');
  });
});
