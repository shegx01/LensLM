import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ErrorCard from './ErrorCard.svelte';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ErrorCard', () => {
  it('renders the sanitized {kind,message} text', () => {
    render(ErrorCard, {
      props: { error: { kind: 'Internal', message: 'Something went wrong' }, onretry: vi.fn() }
    });
    expect(screen.getByText('Internal')).toBeInTheDocument();
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('Retry invokes onretry', async () => {
    const onretry = vi.fn();
    render(ErrorCard, {
      props: { error: { kind: 'Internal', message: 'boom' }, onretry }
    });
    await fireEvent.click(screen.getByText('Retry'));
    expect(onretry).toHaveBeenCalledOnce();
  });
});
