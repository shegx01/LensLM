# CLAUDE.md

Guidance for Claude Code (and any AI agent) working in this repository. Read this before making changes.

## Project

LensLM is a local-first, privacy-preserving NotebookLM clone: ingest documents, ground answers in their sources, and generate audio overviews — all on-device. Everything (embeddings, vector search, LLM routing, TTS) runs locally by default.

The stack is a **Tauri v2 desktop app** with a **SvelteKit + Svelte 5** frontend and a **Rust** backend, split into two crates:

- **`lens-core`** — the headless engine. Ingestion, extraction, chunking, embeddings, vector store, LLM routing, enrichment, TTS. No Tauri, no UI, no OS-window dependencies. Usable and testable standalone.
- **`src-tauri`** — the desktop bridge. Thin Tauri IPC layer that exposes `lens-core` to the frontend via commands, plus OS-specific concerns (file picker, offscreen webview rendering).

> **Architectural invariant:** `lens-core` MUST NOT depend on `tauri` or any UI/OS-window crate. All desktop-specific code lives in `src-tauri`. Keep the engine headless so it stays testable in isolation.

## Repository layout

```
Cargo.toml              # Root workspace: members, shared deps, lint policy
lens-core/              # Headless engine crate
  src/                  #   ingest, extract/, embedder/, vector_store, llm, tts, notebooks, error, ...
  migrations/           #   Sequential SQLite migrations (0001_*.sql .. NNNN_*.sql)
  tests/                #   Integration tests + insta snapshots/
src-tauri/              # Tauri bridge crate
  src/commands/         #   Per-domain IPC commands (config, notebooks, models, system, inspector)
  src/render/           #   Offscreen webview JS renderer
src/                    # SvelteKit frontend
  routes/               #   Filesystem-based routes
  lib/components/       #   UI components (ui/ = headless bits-ui primitives)
  lib/{notebooks,sources,embeddings,models,onboarding,inspector,theme}/  # State + logic
e2e/                    # Playwright e2e tests (*.e2e.ts)
scripts/                # Build/supply-chain scripts (pdfium fetch+verify, catalog, CSP check)
docs/                   # Project docs (see docs/ci.md for the CI pipeline)
.github/workflows/      # ci.yml (merge gate) + audit.yml (weekly supply-chain)
```

## Commands

Toolchain is pinned: Rust 1.94.1 (`rust-toolchain.toml`), Bun 1.2.15, Node 22.16.0 (`.tool-versions`). The JS package manager is **Bun** — use `bun run`, not npm/yarn.

### Develop

```bash
bun run tauri dev       # Run the full desktop app (frontend + Rust backend)
bun run dev             # Frontend-only Vite dev server (http://localhost:1420)
bun run build           # Build the static frontend
```

### Test

```bash
cargo nextest run --workspace       # Rust tests (fast, parallel — matches CI)
cargo test --workspace              # Rust tests (stock runner)
LENS_RUN_MODEL_TESTS=1 cargo test   # Include tests that download real models (slow, networked)
bun run test                        # Frontend unit tests (vitest)
bun run test:e2e                    # Playwright e2e (non-blocking in CI)
```

### Lint & format

```bash
cargo fmt --all -- --check                              # Rust format check
cargo clippy --workspace --all-targets -- -D warnings   # Rust lint (warnings are errors)
bun run format:check                                    # Prettier check (whole tree)
bun run check                                           # svelte-check / TS type check
```

## Before you push

CI blocks merge on all of these — run the full set locally first. It is easy to forget `cargo fmt` and the whole-tree `prettier` check after a batch of edits:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
bun run format:check && bun run check && bun run test
```

The Playwright e2e job runs `continue-on-error` and does not block merge (native Tauri e2e is not viable on the CI runner). `docs/ci.md` documents the pipeline in full.

## Workspace rules

These are project conventions. Follow them; they override generic habits.

### Comments

**No superfluous comments.** Comments exist to (1) give directives to human readers and (2) explain _complicated or non-obvious_ code — invariants, ordering constraints, why a workaround exists, ABI/version locks. Otherwise, do not comment the code. Never narrate what the code plainly says (`// increment i`, `// return the result`). If a comment only restates the line below it, delete it and let the code speak.

**Keep explanatory comments short and out of the way.** When a comment _is_ warranted, cap it at **1–3 lines** stating the essential "why" — not a paragraph. Do not write running, line-by-line narration that interleaves with the code and breaks up its flow; a reader should see the code, not a wall of prose around it. If explaining something needs more than three lines, the code (or its naming/structure) is the thing to fix, or the detail belongs in a doc/design note — not inline. This cap applies to `//` inline comments and `///` doc comments alike; keep doc comments to a tight summary.

### Rust

- **Errors:** use `thiserror`; propagate with `?`. Library/engine code returns `Result<T, LensError>` — the canonical error type serializes across the IPC boundary as `{kind, message}` (locked by a snapshot test). Never leak raw source errors, paths, or internal details across IPC.
- **No `.unwrap()` / `.expect()` / `panic!` in engine code.** They are acceptable only in tests. Handle the error or bubble it up.
- **Prefer enums over stringly-typed domains.** Kinds, statuses, backends, and block types are enums, not magic strings (e.g. `SourceKind`, `EmbeddingBackend`, `Compute`). This is enforced by convention across the codebase.
- **`pub` deliberately.** Expose domain entities, error, config, and the public engine API from `lib.rs`; keep infrastructure (`db`, `http`) private. Don't widen visibility without reason.
- **Borrow over clone;** reach for `.clone()` only when ownership genuinely requires it, and make it explicit.
- **Clippy is a gate:** the workspace lint policy is `clippy::all = "warn"` and CI runs `-D warnings`. Leave the tree clippy-clean.
- **Dependency versions live in `[workspace.dependencies]`** at the root — the single source of truth. Many deps are pinned with `=` for ABI/compat reasons (arrow↔lancedb, uuid, sqlx). Don't bump a pinned version without understanding the lock; the inline comments explain each.

### Database & migrations

- Migrations are sequential SQL files in `lens-core/migrations/` named `NNNN_description.sql`.
- **Never edit an applied migration** — it changes the checksum and breaks `sqlx`. Add a new numbered file instead.
- Use `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` (migrations must be idempotent).
- **Adding a migration means bumping the hard-coded migration count** asserted in `lens-core/tests/schema.rs` (and any mirrored count elsewhere). Validate with `cargo test --workspace`.
- Queries are **runtime-checked** `query`/`query_as` (no `DATABASE_URL` / offline cache). IDs are UUIDv7 stored as TEXT.

### Frontend

- Svelte 5 + TypeScript in **strict mode**; `<script lang="ts">`.
- Components in `lib/components/`; headless primitives from **bits-ui** under `ui/`. Styling is Tailwind v4 utilities via `tailwind-variants` — no ad-hoc CSS where a token or variant exists.
- Every surface must support **light mode, dark mode, and the user-selected accent** using theme tokens only — never hard-code colors.
- Colocate unit tests as `*.test.ts` / `*.svelte.test.ts` next to source (vitest + `@testing-library/svelte`, `happy-dom`).

### Adding an ingestable source format

Touch all of these together, or the format half-works: the backend `SourceKind`/extractor + `extract/`, `commands/notebooks.rs`, and the easy-to-miss frontend `lib/sources/dragDrop.ts` (`ACCEPTED_EXTENSIONS` / `PICKER_FILTERS`) and `types.ts`.

## Testing conventions

- Rust: prefer integration tests in `lens-core/tests/` (realistic full-engine setup) over inline `#[cfg(test)]`. Use `rstest` for parameterized cases, `insta` for snapshots, `tempfile` for scratch DBs, `wiremock` for HTTP.
- Keep tests **offline by default**: use the mock/counting embedder or hand-built vectors. Tests that hit real models are gated behind `LENS_RUN_MODEL_TESTS=1`.
- When practical, work test-first: red → green → refactor per step, rather than a trailing test pass.

## Things to avoid

- Adding a `tauri`/UI dependency to `lens-core` (breaks the headless invariant).
- `.unwrap()`/`.expect()`/`panic!` outside tests; leaking internal error detail across IPC.
- Editing an applied migration; forgetting the migration-count assertion.
- Hard-coded colors; skipping dark-mode or accent support on a new surface.
- Bumping a `=`-pinned dependency without reading why it's pinned.
- Comments that restate the code (see Workspace rules → Comments).
- Committing to `main` — work on a branch/worktree and open a PR.
