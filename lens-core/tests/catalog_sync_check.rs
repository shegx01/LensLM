//! SYNC-CHECK (issue #80, plan Step 9): the Rust `REGISTRY` and the TypeScript
//! `EMBEDDING_MODELS` catalog (`src/lib/embeddings/models.ts`) are two hand-
//! maintained mirrors of the same model set. They MUST agree on the id set and,
//! per id, on `dim` and `backends`. This test parses the TS file and diffs it
//! against `REGISTRY` so any drift (a model added on one side only, a wrong dim,
//! a mis-partitioned backend) fails the gate instead of shipping a picker that
//! offers a model the backend can't serve.
//!
//! Deliberately dependency-free (no `regex`): a tolerant brace/quote scanner over
//! the `EMBEDDING_MODELS` array literal. If the TS formatting changes materially
//! the parser will under-count and the length assertion will fail loudly — a
//! desync signal, not a silent pass. A future hardening (tracked as a non-goal)
//! is a shared `embedding_catalog.json` both sides read.

use std::collections::BTreeSet;

use lens_core::embedder::registry::REGISTRY;

/// One parsed TS `EmbeddingModelSpec` entry (only the SYNC-relevant fields).
#[derive(Debug, PartialEq, Eq)]
struct TsSpec {
    id: String,
    dim: usize,
    backends: BTreeSet<String>,
}

/// Extract the value of a `key: <...>` pair inside a single object literal body.
/// Returns the raw slice from just after the colon up to the next top-level
/// comma or the end of the body.
fn field_raw<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("{key}:");
    let start = body.find(&needle)? + needle.len();
    let rest = &body[start..];
    // Value ends at the first comma that is not inside [] or "" / ''.
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    for (i, ch) in rest.char_indices() {
        match quote {
            Some(q) => {
                if ch == q {
                    quote = None;
                }
            }
            None => match ch {
                '\'' | '"' => quote = Some(ch),
                '[' => depth += 1,
                ']' => depth -= 1,
                ',' if depth == 0 => return Some(rest[..i].trim()),
                _ => {}
            },
        }
    }
    Some(rest.trim())
}

/// Pull the first single/double-quoted string out of `s`.
fn first_quoted(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let q = s.find(['\'', '"'])?;
    let quote = bytes[q] as char;
    let after = &s[q + 1..];
    let end = after.find(quote)?;
    Some(after[..end].to_string())
}

/// All quoted tokens inside `s` (used for the `backends: [...]` array).
fn all_quoted(s: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut rest = s;
    while let Some(tok) = first_quoted(rest) {
        // Advance past this token's closing quote.
        let q = rest.find(['\'', '"']).unwrap();
        let quote = rest.as_bytes()[q] as char;
        let after = &rest[q + 1..];
        let end = after.find(quote).unwrap();
        rest = &after[end + 1..];
        out.insert(tok);
    }
    out
}

/// Parse the `EMBEDDING_MODELS` array literal into specs via brace matching.
fn parse_ts_catalog(src: &str) -> Vec<TsSpec> {
    // Anchor on `const EMBEDDING_MODELS` (a bare `EMBEDDING_MODELS` also matches
    // `ALLOWED_EMBEDDING_MODELS` in the header comment), then scan from the `=` to
    // the first `[` after it — the type annotation `EmbeddingModelSpec[]` has a
    // `[]` that precedes the real `= [ ... ]` array literal.
    let decl = src
        .find("const EMBEDDING_MODELS")
        .expect("const EMBEDDING_MODELS declaration not found");
    let eq = src[decl..]
        .find('=')
        .map(|j| decl + j)
        .expect("EMBEDDING_MODELS assignment not found");
    let arr_start = src[eq..]
        .find('[')
        .map(|j| eq + j + 1)
        .expect("EMBEDDING_MODELS array literal not found");

    let mut specs = Vec::new();
    let bytes = src.as_bytes();
    let mut i = arr_start;
    let mut depth = 0i32; // bracket depth relative to the array
    while i < bytes.len() {
        let ch = bytes[i] as char;
        match ch {
            ']' if depth == 0 => break, // end of the array literal
            '[' => depth += 1,
            ']' => depth -= 1,
            '{' => {
                // Object literal: find its matching brace.
                let mut brace = 1i32;
                let mut j = i + 1;
                while j < bytes.len() && brace > 0 {
                    match bytes[j] as char {
                        '{' => brace += 1,
                        '}' => brace -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                let body = &src[i + 1..j - 1];
                let id = field_raw(body, "id")
                    .and_then(first_quoted)
                    .expect("TS entry missing id");
                let dim = field_raw(body, "dims")
                    .expect("TS entry missing dims")
                    .trim()
                    .parse::<usize>()
                    .expect("TS dims not an integer");
                let backends =
                    all_quoted(field_raw(body, "backends").expect("TS entry missing backends"));
                specs.push(TsSpec { id, dim, backends });
                i = j;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    specs
}

#[test]
fn rust_registry_and_ts_catalog_are_in_sync() {
    let ts_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../src/lib/embeddings/models.ts"
    );
    let src =
        std::fs::read_to_string(ts_path).unwrap_or_else(|e| panic!("cannot read {ts_path}: {e}"));
    let ts = parse_ts_catalog(&src);

    // Same number of models on both sides.
    assert_eq!(
        ts.len(),
        REGISTRY.len(),
        "TS EMBEDDING_MODELS has {} entries but Rust REGISTRY has {}",
        ts.len(),
        REGISTRY.len()
    );

    // Every Rust spec has a TS twin with matching dim + backends.
    for spec in REGISTRY {
        let twin = ts
            .iter()
            .find(|t| t.id == spec.id)
            .unwrap_or_else(|| panic!("Rust model {:?} missing from TS models.ts", spec.id));

        assert_eq!(
            twin.dim, spec.dim,
            "dim mismatch for {:?}: TS {} vs Rust {}",
            spec.id, twin.dim, spec.dim
        );

        let rust_backends: BTreeSet<String> = spec
            .backends
            .iter()
            .map(|b| b.as_str().to_string())
            .collect();
        assert_eq!(
            twin.backends, rust_backends,
            "backends mismatch for {:?}: TS {:?} vs Rust {:?}",
            spec.id, twin.backends, rust_backends
        );
    }

    // And every TS id exists in Rust (catches a TS-only addition).
    let rust_ids: BTreeSet<&str> = REGISTRY.iter().map(|s| s.id).collect();
    for t in &ts {
        assert!(
            rust_ids.contains(t.id.as_str()),
            "TS model {:?} has no Rust REGISTRY twin",
            t.id
        );
    }
}
