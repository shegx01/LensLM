# Continuous Integration

LensLM uses GitHub Actions for testing, linting, and dependency auditing.
Release automation is intentionally **not** set up yet — it will be added when the
app is ready to distribute.

## Workflows

### `CI` — `.github/workflows/ci.yml`

Runs on every pull request and on pushes to `main`. Linux-only (`ubuntu-latest`);
cross-platform bundling is verified later at release time.

| Job                        | What it runs                                                                   | Blocks merge? |
| -------------------------- | ------------------------------------------------------------------------------ | ------------- |
| **Rust (fmt)**             | `cargo fmt --all -- --check`                                                   | Yes           |
| **Rust (clippy)**          | `cargo clippy --workspace --all-targets -- -D warnings`                        | Yes           |
| **Rust (rig LLM backend)** | clippy + `cargo nextest run` for `lens-core` with `--features llm-backend-rig` | Yes           |
| **Frontend**               | `bun run format:check`, `bun run check`, `bun run test`                        | Yes           |
| **E2E**                    | Playwright against the SvelteKit dev server                                    | Yes           |
| **`signoff`**              | The Rust test suite, run locally — see below                                   | Yes           |

CI runs fmt + clippy (the Linux compile/lint canary; `clippy --all-targets` also
compiles the test code) plus the frontend and E2E suites. The rig LLM backend
(`llm-backend-rig`, epic #255) is off by default and excluded from the
`--workspace` clippy job, so it gets its own feature-gated clippy + `nextest`
job until Phase 2 flips the default. The **Rust test suite (feature-off)
runs locally** and is gated by the `signoff` commit status, not by a CI job —
dev hardware runs it faster than a shared runner, and the macOS-gated tests only
run there anyway. The shared `.github/actions/rust-env` composite installs the
Tauri v2 WebKitGTK system libraries (cached) so `src-tauri` compiles; clippy is
the sole Rust compile job, so it writes the shared cargo cache. Toolchains are
pinned: Rust `1.94.1` (`rust-toolchain.toml`), Bun `1.2.15` and Node `22.16.0`
(pinned in the workflow files, mirroring `.tool-versions`).

## Local test signoff

The Rust tests are gated by a `signoff` commit status posted with
[gh-signoff](https://github.com/basecamp/gh-signoff). One-time: `gh extension
install basecamp/gh-signoff`. Per change, after pushing:

```
bun run signoff   # runs `cargo test --workspace`, then `gh signoff` on green
```

`gh signoff` refuses to sign a dirty or unpushed tree, so the status always
matches pushed code. It is an honor-system attestation (no independent
verification) — appropriate for this trusted, single-maintainer repo.

> **Do NOT run `gh signoff install`.** It rewrites classic branch protection and
> is unaware of this repo's ruleset — it would clobber it. The ruleset already
> requires the `signoff` context (managed directly in repo settings).

### `Audit` — `.github/workflows/audit.yml`

Runs weekly (Mondays 06:00 UTC) and on manual `workflow_dispatch`. Kept off the
PR path so a newly-published advisory never blocks unrelated work.

| Job            | What it runs                                                   |
| -------------- | -------------------------------------------------------------- |
| **cargo-deny** | Advisories, licenses, bans, and source policy from `deny.toml` |
| **bun audit**  | Advisories in the JS dependency tree                           |

## Enabling required status checks (manual step)

The workflows fail (red) on real problems, but **GitHub does not block merges
until you enable branch protection** — that is a repository setting, not
something a workflow file can configure. To make the Rust and Frontend jobs
required:

1. Repository **Settings → Branches → Add branch ruleset** (or _Branch
   protection rules_) targeting `main`.
2. Enable **Require status checks to pass before merging**.
3. Add these checks (they appear after the first CI run):
   - `Rust (fmt)`
   - `Rust (clippy)`
   - `Rust (rig LLM backend, feature-gated)`
   - `Frontend (format + check + unit tests)`
   - `E2E (Playwright, non-blocking)`
   - `signoff` (posted locally — see [Local test signoff](#local-test-signoff))
4. Optionally enable **Require branches to be up to date before merging**.

## Dependency updates

`.github/dependabot.yml` opens weekly update PRs for the `cargo`,
`github-actions`, and `npm` (frontend manifest) ecosystems.

> **Note on `npm`/Bun PRs:** Dependabot updates `package.json` but does not
> understand `bun.lock`, so its frontend PRs leave the lockfile stale and CI's
> `bun install --frozen-lockfile` will fail. To land such a PR, check out its
> branch, run `bun install` to refresh `bun.lock`, and push the result.
