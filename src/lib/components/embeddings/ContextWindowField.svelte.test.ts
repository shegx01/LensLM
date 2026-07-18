import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import ContextWindowField from './ContextWindowField.svelte';

describe('ContextWindowField', () => {
  it('renders presets and marks the active one', () => {
    render(ContextWindowField, { props: { value: 8192 } });
    expect(screen.getByRole('button', { name: '8K' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: '16K' })).toHaveAttribute('aria-pressed', 'false');
  });

  it('emits the preset value on click', async () => {
    const onchange = vi.fn();
    render(ContextWindowField, { props: { value: 8192, onchange } });
    await fireEvent.click(screen.getByRole('button', { name: '16K' }));
    expect(onchange).toHaveBeenCalledWith(16384);
  });

  it('emits a custom token count from the number field', async () => {
    const onchange = vi.fn();
    render(ContextWindowField, { props: { value: 8192, onchange } });
    await fireEvent.input(screen.getByLabelText(/custom context window/i), {
      target: { value: '64000' }
    });
    expect(onchange).toHaveBeenCalledWith(64000);
  });

  it('shows an advisory catalog hint when provided', () => {
    render(ContextWindowField, {
      props: { value: 8192, hint: 'Catalog limit: 128K tokens (advisory)' }
    });
    expect(screen.getByText(/catalog limit: 128K tokens/i)).toBeInTheDocument();
  });
});
