// Component tests for CitationChip: renders number + label; live chip fires
// onactivate on click; stale chip is disabled/aria-disabled and does not fire.

import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import CitationChip from './CitationChip.svelte';

describe('CitationChip', () => {
  it('renders the ordinal number and the label', () => {
    render(CitationChip, { n: 3, label: 'Market Report.pdf', live: true });
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('Market Report.pdf')).toBeInTheDocument();
  });

  it('live chip fires onactivate when clicked', async () => {
    const onactivate = vi.fn();
    render(CitationChip, { n: 1, label: 'Doc.md', live: true, onactivate });
    await fireEvent.click(screen.getByRole('button', { name: /source 1: doc\.md/i }));
    expect(onactivate).toHaveBeenCalledOnce();
  });

  it('live chip is not disabled and has aria-label without "(unavailable)"', () => {
    render(CitationChip, { n: 1, label: 'Doc.md', live: true });
    const btn = screen.getByRole('button', { name: 'Source 1: Doc.md' });
    expect(btn).not.toBeDisabled();
    expect(btn).toHaveAttribute('aria-disabled', 'false');
  });

  it('stale chip is disabled, aria-disabled, and does not fire onactivate', async () => {
    const onactivate = vi.fn();
    render(CitationChip, { n: 2, label: 'Removed source', live: false, onactivate });
    const btn = screen.getByRole('button', { name: /source 2: removed source \(unavailable\)/i });
    expect(btn).toBeDisabled();
    expect(btn).toHaveAttribute('aria-disabled', 'true');
    await fireEvent.click(btn);
    expect(onactivate).not.toHaveBeenCalled();
  });

  it('stale chip carries the unavailable title', () => {
    render(CitationChip, { n: 2, label: 'Removed source', live: false });
    const btn = screen.getByRole('button', { name: /unavailable/i });
    expect(btn).toHaveAttribute('title', 'Source no longer available');
  });
});
