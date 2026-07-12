import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { EvalReportDto } from '$lib/embeddings/ipc.js';
import RetrievalSection from './RetrievalSection.svelte';

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

function report(overrides?: Partial<EvalReportDto>): EvalReportDto {
  return {
    graph_recall: 0.82,
    hybrid_recall: 0.71,
    delta_pp: 11,
    p95_ms: 340,
    passed: true,
    sample_n: 24,
    dropped_n: 0,
    graph_enabled: true,
    prompt_version: 'v1',
    ran_at: '2026-07-10T12:00:00Z',
    ...overrides
  };
}

describe('RetrievalSection', () => {
  it('renders the empty state with a Run benchmark CTA when there is no prior eval', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_notebook_graph_retrieval_enabled') return false;
      if (cmd === 'latest_notebook_eval') return null;
    });

    render(RetrievalSection, { props: { notebookId: 'nb1' } });

    expect(await screen.findByText(/no benchmark has run yet/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /run benchmark/i })).toBeEnabled();
    // Toggle stays usable in the empty state.
    expect(screen.getByRole('switch', { name: /use graph retrieval/i })).toBeEnabled();
  });

  it('renders the positive verdict and inline numbers for a passed report', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_notebook_graph_retrieval_enabled') return true;
      if (cmd === 'latest_notebook_eval') return report({ passed: true, delta_pp: 11 });
    });

    render(RetrievalSection, { props: { notebookId: 'nb1' } });

    expect(await screen.findByText(/graph retrieval improves this notebook/i)).toBeInTheDocument();
    expect(screen.getByText(/\+11\.0pp recall@5/i)).toBeInTheDocument();
    expect(screen.getByText('82%')).toBeInTheDocument();
    expect(screen.getByText('71%')).toBeInTheDocument();
    expect(screen.getByText('340ms')).toBeInTheDocument();
    expect(screen.getByText('24')).toBeInTheDocument();
  });

  it('renders the honest "no benefit" verdict for a failed report and keeps the toggle enabled', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_notebook_graph_retrieval_enabled') return false;
      if (cmd === 'latest_notebook_eval')
        return report({ passed: false, graph_recall: 0.7, hybrid_recall: 0.71, delta_pp: -1 });
    });

    render(RetrievalSection, { props: { notebookId: 'nb1' } });

    expect(await screen.findByText(/no measurable benefit/i)).toBeInTheDocument();
    expect(screen.getByRole('switch', { name: /use graph retrieval/i })).toBeEnabled();
  });

  it('optimistically toggles then reverts when the set command fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_notebook_graph_retrieval_enabled') return false;
      if (cmd === 'latest_notebook_eval') return null;
      if (cmd === 'set_notebook_graph_retrieval_enabled') throw new Error('write failed');
    });

    render(RetrievalSection, { props: { notebookId: 'nb1' } });

    const toggle = await screen.findByRole('switch', { name: /use graph retrieval/i });
    expect(toggle).toHaveAttribute('aria-checked', 'false');

    await fireEvent.click(toggle);

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/write failed/i));
    expect(screen.getByRole('switch', { name: /use graph retrieval/i })).toHaveAttribute(
      'aria-checked',
      'false'
    );
  });

  it('shows the "set up a chat model" CTA when the run fails with no configured provider', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_notebook_graph_retrieval_enabled') return false;
      if (cmd === 'latest_notebook_eval') return null;
      if (cmd === 'run_notebook_graph_eval') {
        const a = args as { onEvent?: { onmessage?: (m: unknown) => void } };
        a.onEvent?.onmessage?.({
          type: 'failed',
          data: { kind: 'Model', message: 'no chat model configured' }
        });
        return null;
      }
    });

    render(RetrievalSection, { props: { notebookId: 'nb1' } });
    await fireEvent.click(await screen.findByRole('button', { name: /run benchmark/i }));

    expect(
      await screen.findByText(/set up a chat model to run the benchmark/i)
    ).toBeInTheDocument();
  });

  it('shows the raw run-error alert (not the CTA) when the provider is unreachable', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'get_notebook_graph_retrieval_enabled') return false;
      if (cmd === 'latest_notebook_eval') return null;
      if (cmd === 'run_notebook_graph_eval') {
        const a = args as { onEvent?: { onmessage?: (m: unknown) => void } };
        a.onEvent?.onmessage?.({
          type: 'failed',
          data: { kind: 'Model', message: 'chat model unreachable' }
        });
        return null;
      }
    });

    render(RetrievalSection, { props: { notebookId: 'nb1' } });
    await fireEvent.click(await screen.findByRole('button', { name: /run benchmark/i }));

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/chat model unreachable/i)
    );
    expect(screen.queryByText(/set up a chat model/i)).not.toBeInTheDocument();
  });

  it('clears a stale verdict when a re-run is skipped', async () => {
    mockIPC((cmd) => {
      if (cmd === 'get_notebook_graph_retrieval_enabled') return false;
      if (cmd === 'latest_notebook_eval') return report({ passed: true, delta_pp: 11 });
      if (cmd === 'run_notebook_graph_eval')
        return { status: 'skipped', reason: 'not enough sources to benchmark' };
    });

    render(RetrievalSection, { props: { notebookId: 'nb1' } });
    expect(await screen.findByText(/graph retrieval improves this notebook/i)).toBeInTheDocument();

    await fireEvent.click(screen.getByRole('button', { name: /re-run benchmark/i }));

    expect(await screen.findByText(/not enough sources to benchmark/i)).toBeInTheDocument();
    expect(screen.queryByText(/graph retrieval improves this notebook/i)).not.toBeInTheDocument();
  });
});
