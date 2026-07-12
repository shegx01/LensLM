// NotebookCreateDialog component tests.
//
// Covers: renders all fields including DESCRIPTION; focus-mode card selection
// toggles; Create button is disabled when name is empty and enabled when filled;
// clicking Create calls createNotebookAction(title, description, focusMode);
// Cancel calls onOpenChange(false); inline error renders on failure.
//
// The $lib/notebooks module is mocked so tests run without Tauri IPC.

import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { Notebook } from '$lib/notebooks/types.js';

const { mockCreate, storeState } = vi.hoisted(() => ({
  mockCreate:
    vi.fn<
      (
        title: string,
        description: string | null,
        focusMode: string | null
      ) => Promise<Notebook | null>
    >(),
  // Minimal stand-in for the reactive store; `handleCreate` reads `.error`
  // when the action returns null.
  storeState: { error: null as string | null }
}));

vi.mock('$lib/notebooks/index.js', () => ({
  createNotebookAction: mockCreate,
  notebookStore: storeState
}));

import NotebookCreateDialog from './NotebookCreateDialog.svelte';

/** A throwaway created-notebook fixture for the success path. */
const CREATED: Notebook = {
  id: 'nb-new',
  title: 'New',
  description: null,
  focus_mode: 'research',
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  trashed_at: null,
  last_activity_at: null,
  graph_retrieval_enabled: null,
  embedding_model: null,
  embedding_backend: null
};

function renderDialog(overrides: { open?: boolean; onOpenChange?: (v: boolean) => void } = {}) {
  const onOpenChange = vi.fn();
  const { component, ...rest } = render(NotebookCreateDialog, {
    props: {
      open: overrides.open ?? true,
      onOpenChange: overrides.onOpenChange ?? onOpenChange
    }
  });
  return { component, onOpenChange, ...rest };
}

beforeEach(() => {
  mockCreate.mockReset();
  mockCreate.mockResolvedValue(CREATED);
  storeState.error = null;
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('NotebookCreateDialog — field rendering', () => {
  it('renders the NAME label and input', () => {
    renderDialog();
    expect(screen.getByLabelText(/name/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Q3 Earnings Research/i)).toBeInTheDocument();
  });

  it('renders the DESCRIPTION label and textarea', () => {
    renderDialog();
    expect(screen.getByLabelText(/description/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/What's this notebook about/i)).toBeInTheDocument();
  });

  it('renders the FOCUS MODE section with all three options', () => {
    renderDialog();
    expect(screen.getByRole('radiogroup', { name: /focus mode/i })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /research/i })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /coding/i })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /notes/i })).toBeInTheDocument();
  });

  it('renders Cancel and Create notebook buttons', () => {
    renderDialog();
    expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /create notebook/i })).toBeInTheDocument();
  });

  it('renders the custom circular Close button', () => {
    renderDialog();
    expect(screen.getByRole('button', { name: /^close$/i })).toBeInTheDocument();
  });

  it('renders the header title and subtitle', () => {
    renderDialog();
    expect(screen.getByText('New notebook')).toBeInTheDocument();
    expect(screen.getByText(/create a new knowledge space/i)).toBeInTheDocument();
  });

  it('does NOT render when open is false', () => {
    renderDialog({ open: false });
    expect(screen.queryByLabelText(/name/i)).not.toBeInTheDocument();
  });
});

describe('NotebookCreateDialog — focus mode selection', () => {
  it('defaults to Research selected (aria-checked="true")', () => {
    renderDialog();
    const researchRadio = screen.getByRole('radio', { name: /research/i });
    expect(researchRadio).toHaveAttribute('aria-checked', 'true');
  });

  it('clicking Coding sets it as selected and deselects Research', async () => {
    renderDialog();
    const codingRadio = screen.getByRole('radio', { name: /coding/i });
    await fireEvent.click(codingRadio);
    expect(codingRadio).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByRole('radio', { name: /research/i })).toHaveAttribute(
      'aria-checked',
      'false'
    );
  });

  it('clicking Notes sets it as selected', async () => {
    renderDialog();
    const notesRadio = screen.getByRole('radio', { name: /notes/i });
    await fireEvent.click(notesRadio);
    expect(notesRadio).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByRole('radio', { name: /research/i })).toHaveAttribute(
      'aria-checked',
      'false'
    );
    expect(screen.getByRole('radio', { name: /coding/i })).toHaveAttribute('aria-checked', 'false');
  });

  it('only one radio is checked at a time after multiple selections', async () => {
    renderDialog();
    await fireEvent.click(screen.getByRole('radio', { name: /coding/i }));
    await fireEvent.click(screen.getByRole('radio', { name: /notes/i }));
    const checked = screen
      .getAllByRole('radio')
      .filter((r) => r.getAttribute('aria-checked') === 'true');
    expect(checked).toHaveLength(1);
    expect(checked[0]).toHaveAccessibleName(/notes/i);
  });
});

describe('NotebookCreateDialog — Create button state', () => {
  it('Create is disabled when name is empty', () => {
    renderDialog();
    const createBtn = screen.getByRole('button', { name: /create notebook/i });
    expect(createBtn).toBeDisabled();
  });

  it('Create is disabled when name is only whitespace', async () => {
    renderDialog();
    const nameInput = screen.getByLabelText(/name/i);
    await fireEvent.input(nameInput, { target: { value: '   ' } });
    expect(screen.getByRole('button', { name: /create notebook/i })).toBeDisabled();
  });

  it('Create is enabled when name has non-whitespace content', async () => {
    renderDialog();
    const nameInput = screen.getByLabelText(/name/i);
    await fireEvent.input(nameInput, { target: { value: 'My Notebook' } });
    expect(screen.getByRole('button', { name: /create notebook/i })).toBeEnabled();
  });
});

describe('NotebookCreateDialog — Create action', () => {
  it('calls createNotebookAction with (trimmed title, null description, focusMode)', async () => {
    renderDialog();
    const nameInput = screen.getByLabelText(/name/i);
    await fireEvent.input(nameInput, { target: { value: '  My Notebook  ' } });
    await fireEvent.click(screen.getByRole('button', { name: /create notebook/i }));
    await waitFor(() => expect(mockCreate).toHaveBeenCalledOnce());
    expect(mockCreate).toHaveBeenCalledWith('My Notebook', null, 'research');
  });

  it('calls createNotebookAction with trimmed description when provided', async () => {
    renderDialog();
    const nameInput = screen.getByLabelText(/name/i);
    const descInput = screen.getByLabelText(/description/i);
    await fireEvent.input(nameInput, { target: { value: 'Q3 Research' } });
    await fireEvent.input(descInput, { target: { value: '  Quarterly earnings analysis  ' } });
    await fireEvent.click(screen.getByRole('button', { name: /create notebook/i }));
    await waitFor(() => expect(mockCreate).toHaveBeenCalledOnce());
    expect(mockCreate).toHaveBeenCalledWith(
      'Q3 Research',
      'Quarterly earnings analysis',
      'research'
    );
  });

  it('calls createNotebookAction with null description when description is empty/whitespace', async () => {
    renderDialog();
    await fireEvent.input(screen.getByLabelText(/name/i), { target: { value: 'Alpha' } });
    await fireEvent.input(screen.getByLabelText(/description/i), { target: { value: '   ' } });
    await fireEvent.click(screen.getByRole('button', { name: /create notebook/i }));
    await waitFor(() => expect(mockCreate).toHaveBeenCalledOnce());
    expect(mockCreate).toHaveBeenCalledWith('Alpha', null, 'research');
  });

  it('calls createNotebookAction with selected focusMode', async () => {
    renderDialog();
    await fireEvent.input(screen.getByLabelText(/name/i), { target: { value: 'Code Notes' } });
    await fireEvent.click(screen.getByRole('radio', { name: /coding/i }));
    await fireEvent.click(screen.getByRole('button', { name: /create notebook/i }));
    await waitFor(() => expect(mockCreate).toHaveBeenCalledOnce());
    expect(mockCreate).toHaveBeenCalledWith('Code Notes', null, 'coding');
  });

  it('calls onOpenChange(false) after successful creation', async () => {
    const onOpenChange = vi.fn();
    render(NotebookCreateDialog, { props: { open: true, onOpenChange } });
    await fireEvent.input(screen.getByLabelText(/name/i), { target: { value: 'Test' } });
    await fireEvent.click(screen.getByRole('button', { name: /create notebook/i }));
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false));
  });

  it('shows the store error inline and keeps dialog open when createNotebookAction returns null', async () => {
    mockCreate.mockResolvedValue(null);
    storeState.error = 'IPC failure';
    const onOpenChange = vi.fn();
    render(NotebookCreateDialog, { props: { open: true, onOpenChange } });
    await fireEvent.input(screen.getByLabelText(/name/i), { target: { value: 'Test' } });
    await fireEvent.click(screen.getByRole('button', { name: /create notebook/i }));
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(screen.getByText(/IPC failure/i)).toBeInTheDocument();
    expect(onOpenChange).not.toHaveBeenCalledWith(false);
  });

  it('falls back to a generic inline error when the action returns null with no store error', async () => {
    mockCreate.mockResolvedValue(null);
    storeState.error = null;
    const onOpenChange = vi.fn();
    render(NotebookCreateDialog, { props: { open: true, onOpenChange } });
    await fireEvent.input(screen.getByLabelText(/name/i), { target: { value: 'Test' } });
    await fireEvent.click(screen.getByRole('button', { name: /create notebook/i }));
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(screen.getByText(/could not create the notebook/i)).toBeInTheDocument();
    expect(onOpenChange).not.toHaveBeenCalledWith(false);
  });
});

describe('NotebookCreateDialog — Cancel', () => {
  it('Cancel button calls onOpenChange(false)', async () => {
    const { onOpenChange } = renderDialog();
    await fireEvent.click(screen.getByRole('button', { name: /cancel/i }));
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('does NOT call createNotebookAction when Cancel is clicked', async () => {
    renderDialog();
    await fireEvent.input(screen.getByLabelText(/name/i), { target: { value: 'Test' } });
    await fireEvent.click(screen.getByRole('button', { name: /cancel/i }));
    expect(mockCreate).not.toHaveBeenCalled();
  });

  it('custom Close button calls onOpenChange(false)', async () => {
    const { onOpenChange } = renderDialog();
    await fireEvent.click(screen.getByRole('button', { name: /^close$/i }));
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});
