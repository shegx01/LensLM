# Continuous Integration

LensLM uses GitHub Actions for testing, linting, and dependency auditing.
Release automation is intentionally **not** set up yet — it will be added when the
app is ready to distribute.

## Workflows

### `CI` — `.github/workflows/ci.yml`

Runs on every pull request and on pushes to `main`. Linux-only (`ubuntu-latest`);
cross-platform bundling is verified later at release time.

| Job                 | What it runs                                                                    | Blocks merge?            |
| ------------------- | ------------------------------------------------------------------------------- | ------------------------ |
| **Rust (fmt)**      | `cargo fmt --all -- --check`                                                    | Yes                      |
| **Rust (clippy)**   | `cargo clippy --workspace --all-targets -- -D warnings`                         | Yes                      |
| **Rust (test/1–3)** | Compiles from the warm cache and runs its `cargo nextest run --partition` slice | Yes                      |
| **Frontend**        | `bun run format:check`, `bun run check`, `bun run test`                         | Yes                      |
| **E2E**             | Playwright against the SvelteKit dev server (`bun run test:e2e`)                | No (`continue-on-error`) |

The Rust pipeline is a fan-out: `fmt`, `clippy`, and 3 `test` shards run in
parallel, all sharing one warm cargo cache (the dependency graph is built once
and restored everywhere; shard 1 is the sole cache writer). Each shard compiles
the workspace test binaries from the warm cache and runs its partition of the
suite, so the test execution is split 3 ways and fmt/clippy give fast
fail-first feedback. The shared `.github/actions/rust-env` composite installs
the Tauri v2 WebKitGTK system libraries (cached) so `src-tauri` compiles and the
test binaries can load at runtime. Toolchains are pinned: Rust `1.94.1`
(`rust-toolchain.toml`), Bun `1.2.15` and Node `22.16.0` (pinned in the workflow
files, mirroring `.tool-versions`).

> **Scaling shards:** the shard count is duplicated in `ci.yml` — the `shard`
> matrix and `env.SHARD_TOTAL` (the nextest partition denominator). Update both
> together, and add the new `Rust (test/N)` required-check names below to match.

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
   - `Rust (test/1)`, `Rust (test/2)`, `Rust (test/3)`
   - `Frontend (format + check + unit tests)`
   - Leave **E2E** unselected — it is intentionally non-blocking.
4. Optionally enable **Require branches to be up to date before merging**.

> **Migrating an existing ruleset:** the Rust pipeline used to be a single
> `Rust (fmt + clippy + test)` check. That name no longer exists — remove it
> from the branch-protection ruleset and add the five Rust checks above, or
> merges will block on a check that never reports.

## Dependency updates

`.github/dependabot.yml` opens weekly update PRs for the `cargo`,
`github-actions`, and `npm` (frontend manifest) ecosystems.

> **Note on `npm`/Bun PRs:** Dependabot updates `package.json` but does not
> understand `bun.lock`, so its frontend PRs leave the lockfile stale and CI's
> `bun install --frozen-lockfile` will fail. To land such a PR, check out its
> branch, run `bun install` to refresh `bun.lock`, and push the result.
