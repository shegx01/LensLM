import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ReindexingNotice from './ReindexingNotice.svelte';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ReindexingNotice', () => {
  it('renders a calm, non-destructive retry notice (not an error)', () => {
    render(ReindexingNotice, { props: { onretry: vi.fn() } });
    expect(screen.getByText('Updating this notebook')).toBeInTheDocument();
    expect(screen.getByRole('status')).toBeInTheDocument();
    // Never surfaces an error kind — this is transient, not a failure.
    expect(screen.queryByText('Reindexing')).not.toBeInTheDocument();
  });

  it('Retry invokes onretry', async () => {
    const onretry = vi.fn();
    render(ReindexingNotice, { props: { onretry } });
    await fireEvent.click(screen.getByText('Retry'));
    expect(onretry).toHaveBeenCalledOnce();
  });
});
