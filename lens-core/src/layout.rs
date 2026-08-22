//! One descriptor for the engine's on-disk layout (#248 item 3).
//!
//! Relocation, old-dir cleanup and storage accounting each used to enumerate the
//! layout by hand, so adding a subsystem could silently miss one — the class of bug
//! that made the Qwen hf-cache invisible. They all derive from `LAYOUT` instead.

use std::path::{Path, PathBuf};

use crate::paths::StoragePaths;

/// Which root an entry hangs off. `Cache` follows an offloaded model cache;
/// `Data` never moves away from the corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Root {
    Data,
    Cache,
}

/// The `StorageStats` bucket a corpus entry's bytes count toward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CorpusBucket {
    Db,
    Vectors,
    Sources,
    Audio,
}

/// What an entry's bytes mean to accounting and to "Clear model cache".
/// Copy/cleanup behaviour is [`CopyPolicy`], deliberately a separate axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntryRole {
    /// User data that cannot be re-downloaded. Never reclaimable.
    Corpus(CorpusBucket),
    /// Re-downloadable bundle: summed into `reclaimable` and removed whole.
    CacheReclaimable,
    /// Parent of cache entries. Its declared children carry the accounting, so it
    /// is never summed and never deleted — it also holds the retained catalog.
    CacheContainer,
    /// Split per-model at runtime by `partition_backend_dir` into active-vs-rest.
    /// Never summed or deleted statically; a static role here would delete the
    /// active embedding model.
    CachePartitioned,
    /// On disk, in no accounting bucket, never cache-cleared.
    Unaccounted,
}

/// How relocation treats an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CopyPolicy {
    /// Bulk-copied by `copy_tree`.
    Copy,
    /// Reproduced by `VACUUM INTO`, so never bulk-copied.
    Snapshot,
    /// Lives in the OS anchor. Never copied, and never cleaned from an old dir —
    /// before the first relocation the anchor and the data dir are the same place.
    AnchorPointer,
}

pub(crate) struct LayoutEntry {
    /// The `StoragePaths` method returning this path, when one exists. Read only by
    /// the AC-3.2 sync-check, which is the point: it is how a new accessor without a
    /// LAYOUT row fails the gate instead of going silently unaccounted.
    #[allow(dead_code)]
    pub accessor: Option<&'static str>,
    pub root: Root,
    pub relpath: &'static str,
    pub role: EntryRole,
    pub copy: CopyPolicy,
}

use CopyPolicy::{AnchorPointer, Copy as BulkCopy, Snapshot};
use CorpusBucket::{Audio, Db, Sources, Vectors};
use EntryRole::{CacheContainer, CachePartitioned, CacheReclaimable, Corpus, Unaccounted};
use Root::{Cache, Data};

/// Order is load-bearing for the derived vectors: it reproduces the hand-written
/// order the accounting and relocation code used before the descriptor existed.
pub(crate) const LAYOUT: &[LayoutEntry] = &[
    e(Some("db_path"), Data, "lens.db", Corpus(Db), Snapshot),
    e(None, Data, "lens.db-wal", Corpus(Db), Snapshot),
    e(None, Data, "lens.db-shm", Corpus(Db), Snapshot),
    e(
        Some("lancedb_root"),
        Data,
        "lancedb",
        Corpus(Vectors),
        BulkCopy,
    ),
    e(
        Some("sources_dir"),
        Data,
        "sources",
        Corpus(Sources),
        BulkCopy,
    ),
    e(
        Some("notebooks_dir"),
        Data,
        "notebooks",
        Corpus(Audio),
        BulkCopy,
    ),
    e(None, Data, "config.json", Unaccounted, BulkCopy),
    e(None, Data, "location.json", Unaccounted, AnchorPointer),
    e(
        None,
        Data,
        "location.json.pending",
        Unaccounted,
        AnchorPointer,
    ),
    e(
        Some("models_dir"),
        Cache,
        "models",
        CacheContainer,
        BulkCopy,
    ),
    e(None, Cache, "models/orpheus", CacheReclaimable, BulkCopy),
    e(None, Cache, "models/snac", CacheReclaimable, BulkCopy),
    e(
        Some("whisper_dir"),
        Cache,
        "models/whisper",
        CacheReclaimable,
        BulkCopy,
    ),
    e(
        Some("fastembed_cache"),
        Cache,
        "models/fastembed",
        CachePartitioned,
        BulkCopy,
    ),
    e(
        Some("candle_cache"),
        Cache,
        "models/candle",
        CachePartitioned,
        BulkCopy,
    ),
    e(
        Some("hf_cache"),
        Cache,
        "hf-cache",
        CacheReclaimable,
        BulkCopy,
    ),
];

const fn e(
    accessor: Option<&'static str>,
    root: Root,
    relpath: &'static str,
    role: EntryRole,
    copy: CopyPolicy,
) -> LayoutEntry {
    LayoutEntry {
        accessor,
        root,
        relpath,
        role,
        copy,
    }
}

fn is_top_level(relpath: &str) -> bool {
    !relpath.contains('/')
}

/// Top-level names that make up a data dir, for copy and old-dir cleanup. Excludes
/// the anchor pointers: cleaning those would delete the live pointer whenever the
/// anchor and the data dir coincide.
pub(crate) fn data_entries() -> Vec<&'static str> {
    LAYOUT
        .iter()
        .filter(|x| is_top_level(x.relpath))
        .filter(|x| x.root == Cache || x.copy != AnchorPointer)
        .map(|x| x.relpath)
        .collect()
}

/// Re-downloadable roots, moved together on offload/reset.
pub(crate) fn cache_entries() -> Vec<&'static str> {
    LAYOUT
        .iter()
        .filter(|x| x.root == Cache && is_top_level(x.relpath))
        .map(|x| x.relpath)
        .collect()
}

/// Never bulk-copied: the DB files arrive via `VACUUM INTO`, the pointers are
/// anchor-only.
pub(crate) fn copy_skip() -> Vec<&'static str> {
    LAYOUT
        .iter()
        .filter(|x| matches!(x.copy, Snapshot | AnchorPointer))
        .map(|x| x.relpath)
        .collect()
}

fn root_of(paths: &StoragePaths, root: Root) -> PathBuf {
    match root {
        Data => paths.data_dir().to_path_buf(),
        Cache => paths.cache_root().to_path_buf(),
    }
}

fn resolve(paths: &StoragePaths, entry: &LayoutEntry) -> PathBuf {
    let mut p = root_of(paths, entry.root);
    for seg in entry.relpath.split('/') {
        p = p.join(seg);
    }
    p
}

/// Absolute paths for every entry carrying `role`.
pub(crate) fn paths_with_role(paths: &StoragePaths, role: EntryRole) -> Vec<PathBuf> {
    LAYOUT
        .iter()
        .filter(|x| x.role == role)
        .map(|x| resolve(paths, x))
        .collect()
}

/// True when `path` is under an entry whose root is [`Root::Data`] — the corpus and
/// settings that "Clear model cache" must never touch.
pub(crate) fn is_data_rooted(paths: &StoragePaths, path: &Path) -> bool {
    LAYOUT
        .iter()
        .filter(|x| x.root == Data)
        .any(|x| path.starts_with(resolve(paths, x)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_roots() -> StoragePaths {
        StoragePaths::new(Path::new("/data"), Some("/cache"))
    }

    /// AC-3.2 SYNC-CHECK: `paths.rs` and `LAYOUT` are two views of one layout. Rust
    /// has no reflection over an impl, so this parses the source; a new accessor
    /// without a LAYOUT row fails here rather than going silently unaccounted.
    #[test]
    fn every_storage_paths_accessor_is_registered_in_layout() {
        let src =
            std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/paths.rs"))
                .expect("read paths.rs");

        let declared: Vec<String> = src
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                let rest = l.strip_prefix("pub fn ")?;
                // Directory ROOTS return `&Path` and are not entries themselves.
                if !rest.contains("-> PathBuf") {
                    return None;
                }
                rest.split('(').next().map(str::to_string)
            })
            .collect();
        assert!(
            declared.len() >= 9,
            "parser found only {} accessors — paths.rs formatting changed and this \
             check would pass vacuously: {declared:?}",
            declared.len()
        );

        for name in &declared {
            assert!(
                LAYOUT.iter().any(|x| x.accessor == Some(name.as_str())),
                "new StoragePaths accessor `{name}` is not registered in LAYOUT"
            );
        }
        for entry in LAYOUT.iter().filter_map(|x| x.accessor) {
            assert!(
                declared.iter().any(|d| d == entry),
                "LAYOUT names accessor `{entry}`, which no longer exists in paths.rs"
            );
        }
    }

    /// PM-2: the #238 data-loss class. A mis-roled entry would put corpus or
    /// settings on the "Clear model cache" delete list.
    #[test]
    fn nothing_data_rooted_is_ever_reclaimable() {
        let paths = split_roots();
        for p in paths_with_role(&paths, CacheReclaimable) {
            assert!(
                !is_data_rooted(&paths, &p),
                "{} is under a Data root and must never be reclaimable",
                p.display()
            );
        }
        // Path comparison cannot see this when cache_root == data_dir (the default),
        // so pin the descriptor-level rule too.
        for x in LAYOUT.iter().filter(|x| x.role == CacheReclaimable) {
            assert_eq!(
                x.root, Cache,
                "`{}` is CacheReclaimable but Data-rooted",
                x.relpath
            );
        }
    }

    /// The derived vectors must reproduce the hand-written constants they replaced,
    /// element for element — this is what makes the refactor a no-op.
    #[test]
    fn derivations_match_the_enumerations_they_replaced() {
        assert_eq!(
            data_entries(),
            vec![
                "lens.db",
                "lens.db-wal",
                "lens.db-shm",
                "lancedb",
                "sources",
                "notebooks",
                "config.json",
                "models",
                "hf-cache",
            ]
        );
        assert_eq!(cache_entries(), vec!["models", "hf-cache"]);
        assert_eq!(
            copy_skip(),
            vec![
                "lens.db",
                "lens.db-wal",
                "lens.db-shm",
                "location.json",
                "location.json.pending",
            ]
        );
    }

    /// The anchor pointers must never reach old-dir cleanup: before the first
    /// relocation the anchor IS the data dir, so cleaning them deletes the live
    /// pointer to wherever the data just went.
    #[test]
    fn anchor_pointers_are_absent_from_data_entries() {
        for name in ["location.json", "location.json.pending"] {
            assert!(
                !data_entries().contains(&name),
                "{name} must not be cleaned"
            );
            assert!(copy_skip().contains(&name), "{name} must not be copied");
        }
    }
}
