import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import CreateNotebook from './CreateNotebook.svelte';
import { draft, resetDraft } from '$lib/components/onboarding/onboarding-state.svelte.js';

function stubNotebook(id = 'nb-001') {
  mockIPC((cmd) => {
    if (cmd === 'create_notebook') return { id };
  });
}

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
  resetDraft();
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
  resetDraft();
});

describe('CreateNotebook — initial render', () => {
  it('renders title and subtitle', () => {
    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    expect(screen.getByText('Create your first notebook')).toBeInTheDocument();
    expect(
      screen.getByText('Name your knowledge space and choose a focus mode')
    ).toBeInTheDocument();
  });

  it('renders the three focus-mode cards', () => {
    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    expect(screen.getByRole('radio', { name: 'Research' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Coding' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Notes' })).toBeInTheDocument();
  });

  it('renders the Back button', () => {
    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    expect(screen.getByRole('button', { name: 'Back' })).toBeInTheDocument();
  });
});

describe('CreateNotebook — Next button gate', () => {
  it('Next is disabled when name is empty', () => {
    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    const btn = screen.getByRole('button', { name: 'Next — add sources' });
    expect(btn).toBeDisabled();
  });

  it('Next is enabled after typing a name', async () => {
    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    const input = screen.getByLabelText('Notebook name');
    await fireEvent.input(input, { target: { value: 'Q3 Earnings' } });
    expect(screen.getByRole('button', { name: 'Next — add sources' })).not.toBeDisabled();
  });

  it('Next becomes disabled again when name is cleared to whitespace', async () => {
    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    const input = screen.getByLabelText('Notebook name');
    await fireEvent.input(input, { target: { value: 'Something' } });
    expect(screen.getByRole('button', { name: 'Next — add sources' })).not.toBeDisabled();
    await fireEvent.input(input, { target: { value: '   ' } });
    expect(screen.getByRole('button', { name: 'Next — add sources' })).toBeDisabled();
  });
});

describe('CreateNotebook — focus mode selection', () => {
  it('"research" is selected by default (aria-checked=true)', () => {
    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    const research = screen.getByRole('radio', { name: 'Research' });
    expect(research).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByRole('radio', { name: 'Coding' })).toHaveAttribute('aria-checked', 'false');
    expect(screen.getByRole('radio', { name: 'Notes' })).toHaveAttribute('aria-checked', 'false');
  });

  it('clicking "coding" updates draft.focusMode and flips aria-checked', async () => {
    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    const coding = screen.getByRole('radio', { name: 'Coding' });
    await fireEvent.click(coding);
    expect(coding).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByRole('radio', { name: 'Research' })).toHaveAttribute(
      'aria-checked',
      'false'
    );
    expect(draft.focusMode).toBe('coding');
  });

  it('clicking "notes" updates draft.focusMode', async () => {
    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    await fireEvent.click(screen.getByRole('radio', { name: 'Notes' }));
    expect(draft.focusMode).toBe('notes');
    expect(screen.getByRole('radio', { name: 'Notes' })).toHaveAttribute('aria-checked', 'true');
  });

  it('only one mode is selected at a time', async () => {
    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    await fireEvent.click(screen.getByRole('radio', { name: 'Coding' }));
    await fireEvent.click(screen.getByRole('radio', { name: 'Notes' }));
    const checkedCount = screen
      .getAllByRole('radio')
      .filter((el) => el.getAttribute('aria-checked') === 'true').length;
    expect(checkedCount).toBe(1);
    expect(draft.focusMode).toBe('notes');
  });
});

describe('CreateNotebook — Back button', () => {
  it('fires onback when Back is clicked', async () => {
    const onback = vi.fn();
    render(CreateNotebook, { props: { onadvance: vi.fn(), onback } });
    await fireEvent.click(screen.getByRole('button', { name: 'Back' }));
    expect(onback).toHaveBeenCalledOnce();
  });

  it('does NOT fire onadvance when Back is clicked', async () => {
    const onadvance = vi.fn();
    const onback = vi.fn();
    render(CreateNotebook, { props: { onadvance, onback } });
    await fireEvent.click(screen.getByRole('button', { name: 'Back' }));
    expect(onadvance).not.toHaveBeenCalled();
  });
});

describe('CreateNotebook — create_notebook IPC', () => {
  it('calls create_notebook and fires onadvance when Next is clicked with a name', async () => {
    const createNotebook = vi.fn().mockReturnValue({ id: 'nb-123' });
    const onadvance = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'create_notebook') return createNotebook(args);
    });

    render(CreateNotebook, { props: { onadvance, onback: vi.fn() } });
    await fireEvent.input(screen.getByLabelText('Notebook name'), {
      target: { value: 'My Notebook' }
    });
    await fireEvent.click(screen.getByRole('button', { name: 'Next — add sources' }));

    await waitFor(() => expect(onadvance).toHaveBeenCalledOnce());
    expect(createNotebook).toHaveBeenCalledOnce();
  });

  it('stores the returned notebook id in draft.notebookId', async () => {
    stubNotebook('nb-xyz');
    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    await fireEvent.input(screen.getByLabelText('Notebook name'), {
      target: { value: 'Test Notebook' }
    });
    await fireEvent.click(screen.getByRole('button', { name: 'Next — add sources' }));
    await waitFor(() => expect(draft.notebookId).toBe('nb-xyz'));
  });

  it('passes the trimmed name, description, and focusMode to create_notebook', async () => {
    const createNotebook = vi.fn().mockReturnValue({ id: 'nb-abc' });
    mockIPC((cmd, args) => {
      if (cmd === 'create_notebook') return createNotebook(args);
    });

    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    await fireEvent.input(screen.getByLabelText('Notebook name'), {
      target: { value: '  Q3 Earnings  ' }
    });
    await fireEvent.input(screen.getByLabelText('Notebook description'), {
      target: { value: 'Financial review' }
    });
    await fireEvent.click(screen.getByRole('radio', { name: 'Coding' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Next — add sources' }));

    await waitFor(() => expect(createNotebook).toHaveBeenCalledOnce());
    expect(createNotebook).toHaveBeenCalledWith(
      expect.objectContaining({
        title: 'Q3 Earnings',
        description: 'Financial review',
        focusMode: 'coding'
      })
    );
  });

  it('skips create_notebook if draft.notebookId is already set (Back→Forward reuse)', async () => {
    draft.notebookId = 'existing-nb';
    const onadvance = vi.fn();
    const createNotebook = vi.fn();
    mockIPC((cmd, args) => {
      if (cmd === 'create_notebook') return createNotebook(args);
    });

    render(CreateNotebook, { props: { onadvance, onback: vi.fn() } });
    await fireEvent.input(screen.getByLabelText('Notebook name'), {
      target: { value: 'Existing notebook' }
    });
    await fireEvent.click(screen.getByRole('button', { name: 'Next — add sources' }));

    await waitFor(() => expect(onadvance).toHaveBeenCalledOnce());
    expect(createNotebook).not.toHaveBeenCalled();
  });

  it('shows an inline error and does NOT advance when create_notebook fails', async () => {
    mockIPC((cmd) => {
      if (cmd === 'create_notebook') throw new Error('disk full');
    });
    const onadvance = vi.fn();

    render(CreateNotebook, { props: { onadvance, onback: vi.fn() } });
    await fireEvent.input(screen.getByLabelText('Notebook name'), {
      target: { value: 'Test Notebook' }
    });
    await fireEvent.click(screen.getByRole('button', { name: 'Next — add sources' }));

    await waitFor(() =>
      expect(
        screen.getByText(/could not create your notebook\. please try again\./i)
      ).toBeInTheDocument()
    );
    expect(onadvance).not.toHaveBeenCalled();
  });

  it('passes null description when description textarea is empty', async () => {
    const createNotebook = vi.fn().mockReturnValue({ id: 'nb-999' });
    mockIPC((cmd, args) => {
      if (cmd === 'create_notebook') return createNotebook(args);
    });

    render(CreateNotebook, { props: { onadvance: vi.fn(), onback: vi.fn() } });
    await fireEvent.input(screen.getByLabelText('Notebook name'), {
      target: { value: 'No Description Notebook' }
    });
    await fireEvent.click(screen.getByRole('button', { name: 'Next — add sources' }));

    await waitFor(() => expect(createNotebook).toHaveBeenCalledOnce());
    expect(createNotebook).toHaveBeenCalledWith(expect.objectContaining({ description: null }));
  });
});
