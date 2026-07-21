/** Passes a `{kind,message}` LensError through; wraps anything else without leaking a raw `Error:` prefix. */
export function toLensError(err: unknown): { kind: string; message: string } {
  if (err && typeof err === 'object' && 'kind' in err && 'message' in err) {
    return err as { kind: string; message: string };
  }
  return { kind: 'Internal', message: err instanceof Error ? err.message : String(err) };
}
