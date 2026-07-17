# Continuous Integration

LensLM uses GitHub Actions for testing, linting, and dependency auditing.
Release automation is intentionally **not** set up yet — it will be added when the
app is ready to distribute.

## Workflows

### `CI` — `.github/workflows/ci.yml`

Runs on every pull request and on pushes to `main`. Linux-only (`ubuntu-latest`);
cross-platform bundling is verified later at release time.

| Job               | What it runs                                                     | Blocks merge? |
| ----------------- | ---------------------------------------------------------------- | ------------- |
| **Rust (fmt)**    | `cargo fmt --all -- --check`                                     | Yes           |
| **Rust (clippy)** | `cargo clippy --workspace --all-targets -- -D warnings`          | Yes           |
| **Rust (test/N)** | `cargo nextest run --workspace --partition count:N/3` (3 shards) | Yes           |
| **Frontend**      | `bun run format:check`, `bun run check`, `bun run test`          | Yes           |
| **E2E**           | Playwright against the SvelteKit dev server                      | No (advisory) |
| **`signoff`**     | The macOS-only Apple-native ASR compile proof — see below        | Yes           |

CI runs fmt + clippy plus the full Rust test suite (fanned out across 3
`ubuntu-latest` shards via `cargo nextest --partition`), the frontend suite, and
E2E. The shared `.github/actions/rust-env` composite installs the Tauri v2
WebKitGTK system libraries (cached) so `src-tauri` compiles; all Rust jobs share
one warm cargo cache and test shard 1 is its sole writer. Toolchains are pinned:
Rust `1.94.1` (`rust-toolchain.toml`), Bun `1.2.15` and Node `22.16.0` (pinned in
the workflow files, mirroring `.tool-versions`).

## Local test signoff

The bulk Rust suite runs in CI (above). The `signoff` commit status now covers
only the **macOS-gated Apple-native ASR bridge** (`--features apple-native-asr`,
aarch64-apple-darwin) — no Linux runner can build it, so it is proven on the
maintainer's Mac. Posted with [gh-signoff](https://github.com/basecamp/gh-signoff);
one-time: `gh extension install basecamp/gh-signoff`. Per change, after pushing:

```
bun run signoff   # compiles apple-native-asr (--no-run) on macOS, then `gh signoff`
```

On a non-macOS host the compile step is skipped and `signoff` is a pure
attestation. `gh signoff` refuses to sign a dirty or unpushed tree, so the status
always matches pushed code. It is an honor-system attestation (no independent
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
   - `Rust (test/1)`, `Rust (test/2)`, `Rust (test/3)`
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
