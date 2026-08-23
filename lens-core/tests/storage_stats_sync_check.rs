//! SYNC-CHECK: `StorageStats` crosses IPC into `src/lib/theme/types.ts`, which
//! carries a "must match" comment that nothing enforced. Adding a field to one side
//! only is silent — the panel just never shows the bytes.
//!
//! Compares the REAL serialized shape (serde, not the struct definition) against the
//! TS interface, so a `#[serde(rename)]` or `skip` is caught too.

use std::collections::BTreeSet;
use std::path::Path;

use lens_core::StorageStats;

fn rust_fields() -> BTreeSet<String> {
    let zeroed = StorageStats {
        db_bytes: 0,
        vectors_bytes: 0,
        sources_bytes: 0,
        audio_bytes: 0,
        corpus_bytes: 0,
        reclaimable_cache_bytes: 0,
        retained_bytes: 0,
        sidecar_runtime_bytes: 0,
        total_bytes: 0,
    };
    let json = serde_json::to_value(zeroed).expect("serialize StorageStats");
    json.as_object()
        .expect("StorageStats serializes as an object")
        .keys()
        .cloned()
        .collect()
}

/// Field names from the `export interface StorageStats { … }` block.
fn ts_fields() -> BTreeSet<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("src/lib/theme/types.ts");
    let src = std::fs::read_to_string(&path).expect("read types.ts");

    let start = src
        .find("export interface StorageStats {")
        .expect("StorageStats interface not found — parser would pass vacuously");
    let body = &src[start..];
    let end = body.find('}').expect("unterminated interface");

    body[..end]
        .lines()
        .skip(1)
        .filter_map(|l| {
            let l = l.trim();
            if l.is_empty() || l.starts_with("//") {
                return None;
            }
            l.split(':').next().map(|n| n.trim().to_string())
        })
        .filter(|n| !n.is_empty())
        .collect()
}

#[test]
fn storage_stats_matches_its_typescript_mirror() {
    let rust = rust_fields();
    let ts = ts_fields();

    assert!(
        rust.len() >= 9,
        "serde produced only {} fields — check the fixture, not the mirror",
        rust.len()
    );
    assert!(
        ts.len() >= 9,
        "parsed only {} TS fields — types.ts formatting changed and this check \
         would pass vacuously: {ts:?}",
        ts.len()
    );

    let missing_in_ts: Vec<_> = rust.difference(&ts).collect();
    let missing_in_rust: Vec<_> = ts.difference(&rust).collect();
    assert!(
        missing_in_ts.is_empty(),
        "StorageStats fields absent from src/lib/theme/types.ts: {missing_in_ts:?}"
    );
    assert!(
        missing_in_rust.is_empty(),
        "types.ts declares fields StorageStats does not serialize: {missing_in_rust:?}"
    );
}
