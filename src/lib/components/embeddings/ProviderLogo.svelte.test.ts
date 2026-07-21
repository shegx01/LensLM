import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import ProviderLogo from './ProviderLogo.svelte';

describe('ProviderLogo', () => {
  it('renders the bundled inline SVG for a known provider id', () => {
    const { container } = render(ProviderLogo, { props: { id: 'anthropic', name: 'Anthropic' } });
    expect(container.querySelector('svg')).toBeInTheDocument();
  });

  it('falls back to a monogram for an id with no bundled mark', () => {
    render(ProviderLogo, {
      props: { id: 'openai-compatible', name: 'Custom (OpenAI-compatible)' }
    });
    expect(screen.getByText('C')).toBeInTheDocument();
  });
});
