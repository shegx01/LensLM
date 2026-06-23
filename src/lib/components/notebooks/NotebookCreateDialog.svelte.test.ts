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

// ── Hoisted mocks ─────────────────────────────────────────────────────────────

const { mockCreate } = vi.hoisted(() => ({
  mockCreate:
    vi.fn<(title: string, description: string | null, focusMode: string | null) => Promise<void>>()
}));

vi.mock('$lib/notebooks/index.js', () => ({
  createNotebookAction: mockCreate
}));

import NotebookCreateDialog from './NotebookCreateDialog.svelte';

// ── Helpers ───────────────────────────────────────────────────────────────────

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

// ── Setup / teardown ─────────────────────────────────────────────────────────

beforeEach(() => {
  mockCreate.mockReset();
  mockCreate.mockResolvedValue(undefined);
});

afterEach(() => {
  vi.clearAllMocks();
});

// ── Tests: field rendering ─────────────────────────────────────────────────

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

  it('does NOT render when open is false', () => {
    renderDialog({ open: false });
    // Dialog portal content should not be present
    expect(screen.queryByLabelText(/name/i)).not.toBeInTheDocument();
  });
});

// ── Tests: focus mode selection ───────────────────────────────────────────────

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

// ── Tests: Create button disabled/enabled logic ──────────────────────────────

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

// ── Tests: Create action ─────────────────────────────────────────────────────

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

  it('shows inline error and keeps dialog open when createNotebookAction rejects', async () => {
    mockCreate.mockRejectedValue(new Error('IPC failure'));
    const onOpenChange = vi.fn();
    render(NotebookCreateDialog, { props: { open: true, onOpenChange } });
    await fireEvent.input(screen.getByLabelText(/name/i), { target: { value: 'Test' } });
    await fireEvent.click(screen.getByRole('button', { name: /create notebook/i }));
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(screen.getByText(/IPC failure/i)).toBeInTheDocument();
    // Dialog should remain open (onOpenChange not called with false)
    expect(onOpenChange).not.toHaveBeenCalledWith(false);
  });
});

// ── Tests: Cancel ─────────────────────────────────────────────────────────────

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
});
