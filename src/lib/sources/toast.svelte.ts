// Minimal toast primitive for Lens.
//
// Module-level `$state` singleton — reactive array of active toasts.
// Import specifier for consumers:
//   - Same directory (e.g. dragDrop.ts):  './toast.svelte.js'
//   - From $lib path:                     '$lib/sources/toast.svelte.js'
//
// The ToastContainer component mounts once at the app-shell level (+layout.svelte)
// and renders this reactive array.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ToastMessage {
  id: string;
  message: string;
  /** Auto-dismiss duration in ms. Default 5000. */
  duration?: number;
}

// ---------------------------------------------------------------------------
// Reactive state (Svelte 5 runes — requires .svelte.ts extension)
// ---------------------------------------------------------------------------

let _toasts = $state<ToastMessage[]>([]);

/** Reactive store object — matches the getter pattern used throughout this codebase. */
export const toastStore = {
  get toasts(): ToastMessage[] {
    return _toasts;
  }
};

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/** Show a toast. Auto-dismissed after `duration` ms (default 5000). */
export function showToast(message: string, duration = 5000): void {
  const id = crypto.randomUUID();
  _toasts = [..._toasts, { id, message, duration }];
  setTimeout(() => {
    dismissToast(id);
  }, duration);
}

/** Dismiss a toast by id. */
export function dismissToast(id: string): void {
  _toasts = _toasts.filter((t) => t.id !== id);
}
