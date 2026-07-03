// Minimal toast primitive — module-level `$state` singleton consumed by `ToastContainer` in +layout.svelte.

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
// Reactive state
// ---------------------------------------------------------------------------

let _toasts = $state<ToastMessage[]>([]);

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
