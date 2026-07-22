import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import SystemCheckRow from './SystemCheckRow.svelte';
import type { CheckResult } from '$lib/onboarding/system-check.js';

// Both readiness gates now render an always-visible, purpose-built picker inline
// (no status-badge tile, no Choose/expand). SystemCheckRow only routes by id.
function row(over: Partial<CheckResult>): CheckResult {
  return {
    id: 'embedding_model',
    label: 'Embedding model',
    status: 'pass',
    detail: 'Embedding model installed',
    action: null,
    ...over
  };
}

describe('SystemCheckRow', () => {
  it('routes llm_runtime to the always-visible LLM picker', () => {
    render(SystemCheckRow, { props: { result: row({ id: 'llm_runtime', label: 'Local AI' }) } });
    expect(screen.getByLabelText('Endpoint')).toBeInTheDocument();
  });

  it('routes embedding_model to the inline embedding picker — provider tabs, no Choose/expand', () => {
    render(SystemCheckRow, { props: { result: row({ id: 'embedding_model' }) } });
    expect(screen.getByRole('radio', { name: 'On-device' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Ollama' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /choose/i })).not.toBeInTheDocument();
  });
});
