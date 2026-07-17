import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import TtsLanguageGuardNotice, { type GuardVerdict } from './TtsLanguageGuardNotice.svelte';

function verdict(over: Partial<GuardVerdict>): GuardVerdict {
  return { allow: true, reason: null, offending: [], ...over };
}

describe('TtsLanguageGuardNotice', () => {
  it('renders nothing when allow=true', () => {
    const { container } = render(TtsLanguageGuardNotice, { props: { verdict: verdict({}) } });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(container.textContent?.trim()).toBe('');
  });

  it('renders the block reason and offending sources/languages when allow=false', () => {
    render(TtsLanguageGuardNotice, {
      props: {
        verdict: verdict({
          allow: false,
          reason: 'The selected engine cannot synthesize the language of: doc-1 (German)',
          offending: [
            { source_id: 'doc-1', language: 'german' },
            { source_id: 'doc-2', language: 'arabic' }
          ]
        })
      }
    });
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(
      screen.getByText('The selected engine cannot synthesize the language of: doc-1 (German)')
    ).toBeInTheDocument();
    expect(screen.getByText('doc-1 — German')).toBeInTheDocument();
    expect(screen.getByText('doc-2 — Arabic')).toBeInTheDocument();
  });

  it('renders the alert even with a null reason, when offending sources are present', () => {
    render(TtsLanguageGuardNotice, {
      props: {
        verdict: verdict({
          allow: false,
          reason: null,
          offending: [{ source_id: 'doc-3', language: 'russian' }]
        })
      }
    });
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText('doc-3 — Russian')).toBeInTheDocument();
  });
});
