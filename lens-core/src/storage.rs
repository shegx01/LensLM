//! On-disk storage accounting for the engine data directory: byte-usage figures
//! and clearing of the re-downloadable model cache.
//!
//! One private [`data_layout`] derives every path both [`storage_stats`] and
//! [`clear_model_cache`] act on, so the two can never disagree about what counts
//! as corpus, reclaimable cache, or retained state. The fs walk never follows
//! symlinks (cycle- and double-count-safe) and treats a missing path as 0 bytes.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::error::LensError;

/// Byte-usage breakdown of the engine data directory. Crosses the IPC boundary
/// as plain JSON numbers; consumed by the Storage settings panel.
///
/// `total_bytes == corpus_bytes + reclaimable_cache_bytes + retained_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StorageStats {
    /// User data that cannot be re-downloaded: DB, vectors, extracted sources,
    /// generated audio overviews.
    pub corpus_bytes: u64,
    /// Re-downloadable model bundles safe to clear: voice/ASR models and any
    /// embedding weights that are not the active model.
    pub reclaimable_cache_bytes: u64,
    /// Model state kept even when clearing: the active embedding model (needed
    /// for offline query) and the model catalog.
    pub retained_bytes: u64,
    /// Sum of the three buckets above.
    pub total_bytes: u64,
}

/// Path partition of the data dir. Private single source consumed by both
/// [`storage_stats`] and [`clear_model_cache`] so their notions of corpus /
/// reclaimable / retained can never drift.
struct DataLayout {
    corpus_paths: Vec<PathBuf>,
    reclaimable_cache_paths: Vec<PathBuf>,
    retained_paths: Vec<PathBuf>,
}

/// Partitions `{data_dir}` into corpus, reclaimable-cache, and retained paths.
///
/// Reads `models/{fastembed,candle}` to split each embedding backend's per-model
/// subdirs into the active model (retained) versus the rest (reclaimable); a
/// missing backend dir contributes nothing. `embedding_model` is the configured
/// id (empty resolves to the registry default at the embedder boundary).
fn data_layout(data_dir: &Path, embedding_model: &str) -> DataLayout {
    let corpus_paths = vec![
        data_dir.join("lens.db"),
        data_dir.join("lens.db-wal"),
        data_dir.join("lens.db-shm"),
        data_dir.join("lancedb"),
        data_dir.join("sources"),
        data_dir.join("notebooks"),
    ];

    let models = data_dir.join("models");

    // Re-downloadable bundles: voice + ASR weights are always reclaimable.
    let mut reclaimable_cache_paths = vec![
        models.join("orpheus"),
        models.join("snac"),
        models.join("whisper"),
    ];

    // Catalog is retained (tiny; its loss forces a network re-fetch and degrades
    // the Providers UI).
    let mut retained_paths = vec![data_dir.join(crate::model_catalog::MODELS_CATALOG_RELPATH)];

    let active_fastembed =
        crate::embedder::registry::resolve(embedding_model).fastembed_cache_subdir();
    partition_backend_dir(
        &models.join("fastembed"),
        active_fastembed.as_deref(),
        &mut reclaimable_cache_paths,
        &mut retained_paths,
    );

    // The candle backend (and its cache dir) exists only on the Metal build.
    #[cfg(feature = "native-ml-metal")]
    {
        let active_candle = crate::embedder::candle_cache_subdir(embedding_model);
        partition_backend_dir(
            &models.join("candle"),
            active_candle.as_deref(),
            &mut reclaimable_cache_paths,
            &mut retained_paths,
        );
    }

    DataLayout {
        corpus_paths,
        reclaimable_cache_paths,
        retained_paths,
    }
}

/// Splits an embedding backend's per-model subdirs: the active model's subdir is
/// retained, every other subdir is reclaimable. A missing/unreadable dir is a
/// no-op (fresh install), so both callers see it as 0 bytes.
fn partition_backend_dir(
    backend_dir: &Path,
    active_subdir: Option<&str>,
    reclaimable: &mut Vec<PathBuf>,
    retained: &mut Vec<PathBuf>,
) {
    let Ok(entries) = std::fs::read_dir(backend_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let is_active = entry
            .file_name()
            .to_str()
            .zip(active_subdir)
            .is_some_and(|(name, active)| name == active);
        if is_active {
            retained.push(entry.path());
        } else {
            reclaimable.push(entry.path());
        }
    }
}

/// Recursively sums the byte size of `path`. A missing path is 0 (not an error).
/// Symlinks are never followed — this both prevents cycle-induced infinite
/// recursion and avoids double-counting a symlinked target.
fn path_size_bytes(path: &Path) -> Result<u64, LensError> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(LensError::Io(format!("{}: {e}", path.display()))),
    };
    let file_type = meta.file_type();
    if file_type.is_symlink() {
        return Ok(0);
    }
    if !file_type.is_dir() {
        return Ok(meta.len());
    }
    let mut total = 0u64;
    let entries =
        std::fs::read_dir(path).map_err(|e| LensError::Io(format!("{}: {e}", path.display())))?;
    for entry in entries {
        let entry = entry.map_err(|e| LensError::Io(format!("{}: {e}", path.display())))?;
        total = total.saturating_add(path_size_bytes(&entry.path())?);
    }
    Ok(total)
}

fn sum_paths(paths: &[PathBuf]) -> Result<u64, LensError> {
    let mut total = 0u64;
    for path in paths {
        total = total.saturating_add(path_size_bytes(path)?);
    }
    Ok(total)
}

/// Blocking byte-usage scan. Runs off the async runtime via `spawn_blocking` at
/// the [`crate::LensEngine::storage_stats`] call site — a multi-GB walk must not
/// block an executor thread.
pub(crate) fn storage_stats_blocking(
    data_dir: &Path,
    embedding_model: &str,
) -> Result<StorageStats, LensError> {
    let layout = data_layout(data_dir, embedding_model);
    let corpus_bytes = sum_paths(&layout.corpus_paths)?;
    let reclaimable_cache_bytes = sum_paths(&layout.reclaimable_cache_paths)?;
    let retained_bytes = sum_paths(&layout.retained_paths)?;
    let total_bytes = corpus_bytes
        .saturating_add(reclaimable_cache_bytes)
        .saturating_add(retained_bytes);
    Ok(StorageStats {
        corpus_bytes,
        reclaimable_cache_bytes,
        retained_bytes,
        total_bytes,
    })
}

/// Blocking deletion of only the reclaimable-cache paths; returns bytes freed.
/// Never touches corpus, the active embedding model, or the catalog. Runs under
/// `spawn_blocking` and behind the engine `ingest_lock` at the call site.
pub(crate) fn clear_model_cache_blocking(
    data_dir: &Path,
    embedding_model: &str,
) -> Result<u64, LensError> {
    let layout = data_layout(data_dir, embedding_model);
    let mut freed = 0u64;
    for path in &layout.reclaimable_cache_paths {
        freed = freed.saturating_add(remove_path(path)?);
    }
    Ok(freed)
}

/// Removes `path` (file or directory subtree) and returns the bytes it held. A
/// missing path frees 0. Symlinks are removed as links (size 0), never followed.
fn remove_path(path: &Path) -> Result<u64, LensError> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(LensError::Io(format!("{}: {e}", path.display()))),
    };
    let freed = path_size_bytes(path)?;
    if meta.file_type().is_dir() {
        std::fs::remove_dir_all(path)
            .map_err(|e| LensError::Io(format!("{}: {e}", path.display())))?;
    } else {
        std::fs::remove_file(path)
            .map_err(|e| LensError::Io(format!("{}: {e}", path.display())))?;
    }
    Ok(freed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Default embedding model's fastembed subdir; the active model that must be
    /// retained across a cache clear.
    fn active_subdir() -> String {
        crate::embedder::registry::resolve("")
            .fastembed_cache_subdir()
            .expect("default embedding model has a fastembed cache subdir")
    }

    fn write_file(path: &Path, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(path, bytes).expect("write seed file");
    }

    /// Seeds one file in every bucket and returns their exact byte sizes.
    fn seed_all(data_dir: &Path) {
        let active = active_subdir();
        // Corpus.
        write_file(&data_dir.join("lens.db"), &[0u8; 10]);
        write_file(&data_dir.join("lancedb").join("data.lance"), &[0u8; 20]);
        write_file(
            &data_dir.join("sources").join("s1.extracted.txt"),
            &[0u8; 30],
        );
        write_file(
            &data_dir.join("notebooks").join("nb1").join("overview.wav"),
            &[0u8; 40],
        );
        // Reclaimable cache.
        write_file(
            &data_dir.join("models").join("orpheus").join("model.gguf"),
            &[0u8; 100],
        );
        write_file(
            &data_dir
                .join("models")
                .join("snac")
                .join("pytorch_model.bin"),
            &[0u8; 200],
        );
        write_file(
            &data_dir
                .join("models")
                .join("whisper")
                .join("ggml-base.bin"),
            &[0u8; 300],
        );
        write_file(
            &data_dir
                .join("models")
                .join("fastembed")
                .join("models--other--model")
                .join("w.onnx"),
            &[0u8; 400],
        );
        // Retained.
        write_file(
            &data_dir
                .join("models")
                .join("fastembed")
                .join(&active)
                .join("w.onnx"),
            &[0u8; 500],
        );
        write_file(
            &data_dir.join(crate::model_catalog::MODELS_CATALOG_RELPATH),
            &[0u8; 50],
        );
    }

    #[test]
    fn storage_stats_sums_each_bucket() {
        let dir = tempfile::tempdir().expect("tempdir");
        seed_all(dir.path());

        let stats = storage_stats_blocking(dir.path(), "").expect("stats");
        assert_eq!(stats.corpus_bytes, 10 + 20 + 30 + 40);
        assert_eq!(stats.reclaimable_cache_bytes, 100 + 200 + 300 + 400);
        assert_eq!(stats.retained_bytes, 500 + 50);
        assert_eq!(
            stats.total_bytes,
            stats.corpus_bytes + stats.reclaimable_cache_bytes + stats.retained_bytes
        );
    }

    #[test]
    fn storage_stats_missing_dirs_are_zero() {
        let dir = tempfile::tempdir().expect("tempdir");
        let stats = storage_stats_blocking(dir.path(), "").expect("stats");
        assert_eq!(stats.corpus_bytes, 0);
        assert_eq!(stats.reclaimable_cache_bytes, 0);
        assert_eq!(stats.retained_bytes, 0);
        assert_eq!(stats.total_bytes, 0);
    }

    #[test]
    fn clear_model_cache_removes_only_reclaimable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        seed_all(root);
        let active = active_subdir();

        let freed = clear_model_cache_blocking(root, "").expect("clear");
        assert_eq!(freed, 100 + 200 + 300 + 400);

        // Reclaimable subtrees gone.
        assert!(!root.join("models").join("orpheus").exists());
        assert!(!root.join("models").join("snac").exists());
        assert!(!root.join("models").join("whisper").exists());
        assert!(
            !root
                .join("models")
                .join("fastembed")
                .join("models--other--model")
                .exists()
        );

        // Corpus + active model + catalog intact.
        assert!(root.join("lens.db").exists());
        assert!(root.join("lancedb").join("data.lance").exists());
        assert!(root.join("sources").join("s1.extracted.txt").exists());
        assert!(
            root.join("notebooks")
                .join("nb1")
                .join("overview.wav")
                .exists()
        );
        assert!(
            root.join("models")
                .join("fastembed")
                .join(&active)
                .join("w.onnx")
                .exists()
        );
        assert!(
            root.join(crate::model_catalog::MODELS_CATALOG_RELPATH)
                .exists()
        );

        // Reclaimable figure drops to 0; retained unchanged.
        let stats = storage_stats_blocking(root, "").expect("stats after clear");
        assert_eq!(stats.reclaimable_cache_bytes, 0);
        assert_eq!(stats.retained_bytes, 500 + 50);
        assert_eq!(stats.corpus_bytes, 10 + 20 + 30 + 40);
    }

    #[test]
    fn clear_model_cache_on_fresh_install_frees_zero() {
        let dir = tempfile::tempdir().expect("tempdir");
        let freed = clear_model_cache_blocking(dir.path(), "").expect("clear");
        assert_eq!(freed, 0);
    }
}
