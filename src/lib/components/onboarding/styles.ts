// Shared Tailwind class strings for the onboarding config panels.
//
// The native <select> elements in LlmConfigPanel (model picker) and TtsConfigPanel
// (host/guest voice pickers) share one visual treatment. Centralizing the class
// string here keeps them in lockstep (DRY — was triplicated inline).

/** Native <select> styling shared by the onboarding config-panel pickers. */
export const SELECT_CLASS =
  'border-input bg-transparent dark:bg-input/30 focus-visible:border-ring focus-visible:ring-ring/50 h-8 w-full min-w-0 rounded-lg border px-2.5 py-1 text-sm outline-none transition-colors focus-visible:ring-3 text-foreground';
