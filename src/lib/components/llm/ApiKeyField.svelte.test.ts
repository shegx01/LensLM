import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import ApiKeyField from './ApiKeyField.svelte';

describe('ApiKeyField', () => {
  it('masks a saved key and shows the replace hint', () => {
    render(ApiKeyField, { props: { id: 'k', hasSavedKey: true } });
    const input = screen.getByLabelText('API Key');
    expect(input).toHaveAttribute(
      'placeholder',
      expect.stringMatching(/saved — click to replace/i)
    );
    expect(screen.getByText(/a key is already saved/i)).toBeInTheDocument();
  });

  it('switches to edit mode on focus (key-wipe protection) and drops the masked placeholder', async () => {
    render(ApiKeyField, { props: { id: 'k', hasSavedKey: true } });
    const input = screen.getByLabelText('API Key');
    await fireEvent.focus(input);
    expect(input).toHaveAttribute('placeholder', expect.stringMatching(/paste api key/i));
    expect(screen.queryByText(/a key is already saved/i)).not.toBeInTheDocument();
  });

  it('is editable immediately when no key is saved', () => {
    render(ApiKeyField, { props: { id: 'k', hasSavedKey: false } });
    expect(screen.getByLabelText('API Key')).toHaveAttribute(
      'placeholder',
      expect.stringMatching(/paste api key/i)
    );
    expect(screen.queryByText(/a key is already saved/i)).not.toBeInTheDocument();
  });

  it('fires oncommit on blur so the caller can persist', async () => {
    const oncommit = vi.fn();
    render(ApiKeyField, { props: { id: 'k', oncommit } });
    await fireEvent.blur(screen.getByLabelText('API Key'));
    expect(oncommit).toHaveBeenCalledOnce();
  });
});
