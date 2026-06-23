// NotebookTopBar.svelte.test.ts
//
// Component tests for the center top-bar chrome.
//
// Covers:
//   - Renders the active notebook title
//   - Renders nothing when activeNotebook is null
//   - Chat/Notes toggle reflects activeTab from the store
//   - Clicking Chat/Notes buttons updates activeTab
//   - Share button is disabled
//   - Settings button is disabled
//   - Tooltip text is present on the disabled buttons
//
// The `$lib/notebooks` barrel is mocked with a minimal fake store so no real
// IPC or Tauri globals are needed. `activeTab` is exposed as a writable property.
// `resetNotebookStore` is called in afterEach to prevent cross-test bleed.

import { render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// ---------------------------------------------------------------------------
// Hoisted mock refs — must be created before vi.mock factory runs
// ---------------------------------------------------------------------------

const { mockStore } = vi.hoisted(() => {
  let _activeNotebook: { id: string; title: string } | null = {
    id: 'nb-001',
    title: 'Alpha Research'
  };
  let _activeTab: 'chat' | 'notes' = 'chat';

  return {
    mockStore: {
      get activeNotebook() {
        return _activeNotebook;
      },
      set activeNotebook(v: { id: string; title: string } | null) {
        _activeNotebook = v;
      },
      get activeTab() {
        return _activeTab;
      },
      set activeTab(v: 'chat' | 'notes') {
        _activeTab = v;
      },
      // Test helper: directly set the underlying value for setup
      _setActiveNotebook(v: { id: string; title: string } | null) {
        _activeNotebook = v;
      },
      _setActiveTab(v: 'chat' | 'notes') {
        _activeTab = v;
      }
    }
  };
});

// Mock the $lib/notebooks barrel — the component imports `notebookStore` from here.
vi.mock('$lib/notebooks/index.js', () => ({
  notebookStore: mockStore
}));

// Import component after mocks are set up.
import NotebookTopBar from './NotebookTopBar.svelte';

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

beforeEach(() => {
  // Reset to default state before each test
  mockStore._setActiveNotebook({ id: 'nb-001', title: 'Alpha Research' });
  mockStore._setActiveTab('chat');
});

afterEach(() => {
  vi.clearAllMocks();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('NotebookTopBar', () => {
  describe('when a notebook is active', () => {
    it('renders the active notebook title', () => {
      render(NotebookTopBar);
      expect(screen.getByText('Alpha Research')).toBeInTheDocument();
    });

    it('renders the Chat and Notes toggle buttons', () => {
      render(NotebookTopBar);
      expect(screen.getByRole('tab', { name: /chat/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /notes/i })).toBeInTheDocument();
    });

    it('Chat tab is aria-selected when activeTab is "chat"', () => {
      mockStore._setActiveTab('chat');
      render(NotebookTopBar);
      expect(screen.getByRole('tab', { name: /chat/i })).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByRole('tab', { name: /notes/i })).toHaveAttribute('aria-selected', 'false');
    });

    it('Notes tab is aria-selected when activeTab is "notes"', () => {
      mockStore._setActiveTab('notes');
      render(NotebookTopBar);
      expect(screen.getByRole('tab', { name: /notes/i })).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByRole('tab', { name: /chat/i })).toHaveAttribute('aria-selected', 'false');
    });

    it('clicking the Notes tab sets activeTab to "notes"', async () => {
      mockStore._setActiveTab('chat');
      render(NotebookTopBar);
      await fireEvent.click(screen.getByRole('tab', { name: /notes/i }));
      expect(mockStore.activeTab).toBe('notes');
    });

    it('clicking the Chat tab sets activeTab to "chat"', async () => {
      mockStore._setActiveTab('notes');
      render(NotebookTopBar);
      await fireEvent.click(screen.getByRole('tab', { name: /chat/i }));
      expect(mockStore.activeTab).toBe('chat');
    });

    it('Share button is disabled', () => {
      render(NotebookTopBar);
      const shareBtn = screen.getByRole('button', { name: /share notebook/i });
      expect(shareBtn).toBeDisabled();
    });

    it('Settings button is disabled', () => {
      render(NotebookTopBar);
      const settingsBtn = screen.getByRole('button', { name: /notebook settings/i });
      expect(settingsBtn).toBeDisabled();
    });

    it('Share button has aria-label mentioning availability', () => {
      render(NotebookTopBar);
      const shareBtn = screen.getByRole('button', { name: /share notebook/i });
      expect(shareBtn).toHaveAttribute('aria-label');
      expect(shareBtn.getAttribute('aria-label')).toMatch(/available soon/i);
    });

    it('Settings button has aria-label mentioning availability', () => {
      render(NotebookTopBar);
      const settingsBtn = screen.getByRole('button', { name: /notebook settings/i });
      expect(settingsBtn).toHaveAttribute('aria-label');
      expect(settingsBtn.getAttribute('aria-label')).toMatch(/available soon/i);
    });

    it('has the toolbar landmark', () => {
      render(NotebookTopBar);
      expect(screen.getByRole('toolbar', { name: /notebook toolbar/i })).toBeInTheDocument();
    });

    it('has a view toggle group', () => {
      render(NotebookTopBar);
      expect(screen.getByRole('group', { name: /view toggle/i })).toBeInTheDocument();
    });

    it('the outer bar row carries data-tauri-drag-region (the floating pill itself does not)', () => {
      render(NotebookTopBar);
      const toolbar = screen.getByRole('toolbar', { name: /notebook toolbar/i });
      // The floating pill is interactive and is NOT a drag region; its parent row is.
      expect(toolbar).not.toHaveAttribute('data-tauri-drag-region');
      expect(toolbar.parentElement).toHaveAttribute('data-tauri-drag-region');
    });
  });

  describe('when no notebook is active', () => {
    it('still renders the pill (header is always visible) but omits the title and tabs', () => {
      mockStore._setActiveNotebook(null);
      render(NotebookTopBar);
      // The header/pill is always present so share + settings stay reachable...
      expect(screen.getByRole('toolbar', { name: /notebook toolbar/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /share/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /settings/i })).toBeInTheDocument();
      // ...but the notebook-contextual title + Chat/Notes tabs are hidden.
      expect(screen.queryByRole('tab')).not.toBeInTheDocument();
    });
  });
});
