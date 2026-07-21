//! On-disk storage accounting for the engine data directory: byte-usage figures
//! and clearing of the re-downloadable model cache.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::error::LensError;
use crate::paths::StoragePaths;

/// Byte-usage breakdown of the engine data directory. Crosses the IPC boundary
/// as plain JSON numbers; consumed by the Storage settings panel.
///
/// `corpus_bytes == db_bytes + vectors_bytes + sources_bytes + audio_bytes` and
/// `total_bytes == corpus_bytes + reclaimable_cache_bytes + retained_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StorageStats {
    /// SQLite database (data + WAL + shared-memory index).
    pub db_bytes: u64,
    /// LanceDB vector store.
    pub vectors_bytes: u64,
    /// Extracted/managed source files.
    pub sources_bytes: u64,
    /// Generated audio overviews.
    pub audio_bytes: u64,
    /// User data that cannot be re-downloaded: the sum of the four buckets above.
    pub corpus_bytes: u64,
    /// Re-downloadable model bundles safe to clear: voice/ASR models and any
    /// embedding weights that are not the active model.
    pub reclaimable_cache_bytes: u64,
    /// Model state kept even when clearing: the active embedding model (needed
    /// for offline query) and the model catalog.
    pub retained_bytes: u64,
    pub total_bytes: u64,
}

/// Path partition of the data dir. Private single source consumed by both
/// [`storage_stats`] and [`clear_model_cache`] so their notions of corpus /
/// reclaimable / retained can never drift. Corpus is split into per-bottleneck
/// buckets (#238) for the usage breakdown.
struct DataLayout {
    db_paths: Vec<PathBuf>,
    vectors_paths: Vec<PathBuf>,
    sources_paths: Vec<PathBuf>,
    audio_paths: Vec<PathBuf>,
    reclaimable_cache_paths: Vec<PathBuf>,
    retained_paths: Vec<PathBuf>,
}

/// Partitions the engine directories into corpus, reclaimable-cache, and retained
/// paths. Corpus lives under `paths.data_dir()`; re-downloadable model/cache dirs
/// live under `paths.cache_root()` (the #238 offload root, else `data_dir`).
///
/// Reads `models/{fastembed,candle}` to split each embedding backend's per-model
/// subdirs into the active model (retained) versus the rest (reclaimable); a
/// missing backend dir contributes nothing. `embedding_model` is the configured
/// id (empty resolves to the registry default at the embedder boundary).
fn data_layout(paths: &StoragePaths, embedding_model: &str) -> DataLayout {
    let data_dir = paths.data_dir();
    let db_paths = vec![
        paths.db_path(),
        data_dir.join("lens.db-wal"),
        data_dir.join("lens.db-shm"),
    ];
    let vectors_paths = vec![paths.lancedb_root()];
    let sources_paths = vec![paths.sources_dir()];
    let audio_paths = vec![paths.notebooks_dir()];

    let models = paths.models_dir();

    // Re-downloadable bundles: voice + ASR weights are always reclaimable.
    // Qwen3-TTS (mlx-audio) caches its snapshot here; switching TTS backend re-downloads, so the whole dir is reclaimable.
    let mut reclaimable_cache_paths = vec![
        models.join("orpheus"),
        models.join("snac"),
        paths.whisper_dir(),
        paths.hf_cache(),
    ];

    // Catalog is retained (tiny; its loss forces a network re-fetch and degrades
    // the Providers UI).
    let mut retained_paths = vec![
        paths
            .cache_root()
            .join(crate::model_catalog::MODELS_CATALOG_RELPATH),
    ];

    let active_fastembed =
        crate::embedder::registry::resolve(embedding_model).fastembed_cache_subdir();
    partition_backend_dir(
        &paths.fastembed_cache(),
        active_fastembed.as_deref(),
        &mut reclaimable_cache_paths,
        &mut retained_paths,
    );

    // The candle backend (and its cache dir) exists only on the Metal build.
    #[cfg(feature = "native-ml-metal")]
    {
        let active_candle = crate::embedder::candle_cache_subdir(embedding_model);
        partition_backend_dir(
            &paths.candle_cache(),
            active_candle.as_deref(),
            &mut reclaimable_cache_paths,
            &mut retained_paths,
        );
    }

    DataLayout {
        db_paths,
        vectors_paths,
        sources_paths,
        audio_paths,
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
        Err(e) => {
            tracing::error!(path = %path.display(), error = %e, "stat during storage scan failed");
            return Err(LensError::Io("failed to read a storage directory".into()));
        }
    };
    let file_type = meta.file_type();
    if file_type.is_symlink() {
        return Ok(0);
    }
    if !file_type.is_dir() {
        return Ok(meta.len());
    }
    let mut total = 0u64;
    let entries = std::fs::read_dir(path).map_err(|e| {
        tracing::error!(path = %path.display(), error = %e, "read_dir during storage scan failed");
        LensError::Io("failed to read a storage directory".into())
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| {
            tracing::error!(path = %path.display(), error = %e, "dir entry during storage scan failed");
            LensError::Io("failed to read a storage directory".into())
        })?;
        total = total.saturating_add(path_size_bytes(&entry.path())?);
    }
    Ok(total)
}

/// Recursive byte size of a single path (symlink-skipping); a missing path is 0.
/// Shared with relocation/offload accounting (#238).
pub(crate) fn dir_size_bytes(path: &Path) -> Result<u64, LensError> {
    path_size_bytes(path)
}

fn sum_paths(paths: &[PathBuf]) -> Result<u64, LensError> {
    let mut total = 0u64;
    for path in paths {
        total = total.saturating_add(path_size_bytes(path)?);
    }
    Ok(total)
}

/// Blocking byte scan; invoked under `spawn_blocking` by [`crate::LensEngine::storage_stats`].
pub(crate) fn storage_stats_blocking(
    paths: &StoragePaths,
    embedding_model: &str,
) -> Result<StorageStats, LensError> {
    let layout = data_layout(paths, embedding_model);
    let db_bytes = sum_paths(&layout.db_paths)?;
    let vectors_bytes = sum_paths(&layout.vectors_paths)?;
    let sources_bytes = sum_paths(&layout.sources_paths)?;
    let audio_bytes = sum_paths(&layout.audio_paths)?;
    let corpus_bytes = db_bytes
        .saturating_add(vectors_bytes)
        .saturating_add(sources_bytes)
        .saturating_add(audio_bytes);
    let reclaimable_cache_bytes = sum_paths(&layout.reclaimable_cache_paths)?;
    let retained_bytes = sum_paths(&layout.retained_paths)?;
    let total_bytes = corpus_bytes
        .saturating_add(reclaimable_cache_bytes)
        .saturating_add(retained_bytes);
    Ok(StorageStats {
        db_bytes,
        vectors_bytes,
        sources_bytes,
        audio_bytes,
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
    paths: &StoragePaths,
    embedding_model: &str,
) -> Result<u64, LensError> {
    let layout = data_layout(paths, embedding_model);
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
        Err(e) => {
            tracing::error!(path = %path.display(), error = %e, "stat before cache removal failed");
            return Err(LensError::Io("failed to remove a cached model".into()));
        }
    };
    let freed = path_size_bytes(path)?;
    if meta.file_type().is_dir() {
        std::fs::remove_dir_all(path).map_err(|e| {
            tracing::error!(path = %path.display(), error = %e, "remove cache directory failed");
            LensError::Io("failed to remove a cached model".into())
        })?;
    } else {
        std::fs::remove_file(path).map_err(|e| {
            tracing::error!(path = %path.display(), error = %e, "remove cache file failed");
            LensError::Io("failed to remove a cached model".into())
        })?;
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

    /// Seeds one file in every bucket, placing corpus under `data_dir()` and cache
    /// under `cache_root()` so the same helper exercises both the default (equal
    /// roots) and offloaded (split roots) layouts.
    fn seed_all(paths: &StoragePaths) {
        let active = active_subdir();
        // Corpus (under data_dir).
        write_file(&paths.db_path(), &[0u8; 10]);
        write_file(&paths.lancedb_root().join("data.lance"), &[0u8; 20]);
        write_file(&paths.sources_dir().join("s1.extracted.txt"), &[0u8; 30]);
        write_file(
            &paths.notebooks_dir().join("nb1").join("overview.wav"),
            &[0u8; 40],
        );
        // Reclaimable cache (under cache_root).
        write_file(
            &paths.models_dir().join("orpheus").join("model.gguf"),
            &[0u8; 100],
        );
        write_file(
            &paths.models_dir().join("snac").join("pytorch_model.bin"),
            &[0u8; 200],
        );
        write_file(&paths.whisper_dir().join("ggml-base.bin"), &[0u8; 300]);
        write_file(
            &paths
                .fastembed_cache()
                .join("models--other--model")
                .join("w.onnx"),
            &[0u8; 400],
        );
        write_file(
            &paths
                .hf_cache()
                .join("hub")
                .join("models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")
                .join("weights.bin"),
            &[0u8; 700],
        );
        // Retained (under cache_root).
        write_file(
            &paths.fastembed_cache().join(&active).join("w.onnx"),
            &[0u8; 500],
        );
        write_file(
            &paths
                .cache_root()
                .join(crate::model_catalog::MODELS_CATALOG_RELPATH),
            &[0u8; 50],
        );
    }

    #[test]
    fn storage_stats_sums_each_bucket() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = StoragePaths::new(dir.path(), None);
        seed_all(&paths);

        let stats = storage_stats_blocking(&paths, "").expect("stats");
        assert_eq!(stats.db_bytes, 10);
        assert_eq!(stats.vectors_bytes, 20);
        assert_eq!(stats.sources_bytes, 30);
        assert_eq!(stats.audio_bytes, 40);
        assert_eq!(stats.corpus_bytes, 10 + 20 + 30 + 40);
        assert_eq!(
            stats.corpus_bytes,
            stats.db_bytes + stats.vectors_bytes + stats.sources_bytes + stats.audio_bytes
        );
        assert_eq!(stats.reclaimable_cache_bytes, 100 + 200 + 300 + 400 + 700);
        assert_eq!(stats.retained_bytes, 500 + 50);
        assert_eq!(
            stats.total_bytes,
            stats.corpus_bytes + stats.reclaimable_cache_bytes + stats.retained_bytes
        );
    }

    #[test]
    fn storage_stats_missing_dirs_are_zero() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = StoragePaths::new(dir.path(), None);
        let stats = storage_stats_blocking(&paths, "").expect("stats");
        assert_eq!(stats.corpus_bytes, 0);
        assert_eq!(stats.reclaimable_cache_bytes, 0);
        assert_eq!(stats.retained_bytes, 0);
        assert_eq!(stats.total_bytes, 0);
    }

    #[test]
    fn clear_model_cache_removes_only_reclaimable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = StoragePaths::new(dir.path(), None);
        seed_all(&paths);
        let active = active_subdir();

        let freed = clear_model_cache_blocking(&paths, "").expect("clear");
        assert_eq!(freed, 100 + 200 + 300 + 400 + 700);

        // Reclaimable subtrees gone.
        assert!(!paths.models_dir().join("orpheus").exists());
        assert!(!paths.models_dir().join("snac").exists());
        assert!(!paths.whisper_dir().exists());
        assert!(
            !paths
                .fastembed_cache()
                .join("models--other--model")
                .exists()
        );
        assert!(!paths.hf_cache().exists());

        // Corpus + active model + catalog intact.
        assert!(paths.db_path().exists());
        assert!(paths.lancedb_root().join("data.lance").exists());
        assert!(paths.sources_dir().join("s1.extracted.txt").exists());
        assert!(
            paths
                .notebooks_dir()
                .join("nb1")
                .join("overview.wav")
                .exists()
        );
        assert!(
            paths
                .fastembed_cache()
                .join(&active)
                .join("w.onnx")
                .exists()
        );
        assert!(
            paths
                .cache_root()
                .join(crate::model_catalog::MODELS_CATALOG_RELPATH)
                .exists()
        );

        // Reclaimable figure drops to 0; retained unchanged.
        let stats = storage_stats_blocking(&paths, "").expect("stats after clear");
        assert_eq!(stats.reclaimable_cache_bytes, 0);
        assert_eq!(stats.retained_bytes, 500 + 50);
        assert_eq!(stats.corpus_bytes, 10 + 20 + 30 + 40);
    }

    #[test]
    fn clear_model_cache_on_fresh_install_frees_zero() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = StoragePaths::new(dir.path(), None);
        let freed = clear_model_cache_blocking(&paths, "").expect("clear");
        assert_eq!(freed, 0);
    }

    #[test]
    fn clear_reclaims_qwen_hf_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = StoragePaths::new(dir.path(), None);
        seed_all(&paths);
        let active = active_subdir();

        // The Qwen3-TTS HF snapshot counts as reclaimable.
        let hf_cache = paths.hf_cache();
        assert!(hf_cache.exists());
        let before = storage_stats_blocking(&paths, "").expect("stats");
        assert_eq!(before.reclaimable_cache_bytes, 100 + 200 + 300 + 400 + 700);

        clear_model_cache_blocking(&paths, "").expect("clear");

        // hf-cache subtree gone; corpus + active model + catalog survive.
        assert!(!hf_cache.exists());
        assert!(paths.db_path().exists());
        assert!(
            paths
                .fastembed_cache()
                .join(&active)
                .join("w.onnx")
                .exists()
        );
        assert!(
            paths
                .cache_root()
                .join(crate::model_catalog::MODELS_CATALOG_RELPATH)
                .exists()
        );
    }

    #[test]
    fn clear_and_scan_ignore_symlink_into_corpus() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = StoragePaths::new(dir.path(), None);
        seed_all(&paths);

        // A symlink under a reclaimable dir pointing at a corpus file must be
        // neither counted by the scan nor followed by the clear.
        let target = paths.db_path();
        let link = paths.whisper_dir().join("link-to-db");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");

        let stats = storage_stats_blocking(&paths, "").expect("stats");
        assert_eq!(stats.reclaimable_cache_bytes, 100 + 200 + 300 + 400 + 700);

        clear_model_cache_blocking(&paths, "").expect("clear");
        assert!(!link.exists());
        assert!(target.exists());
    }

    /// #238: with an offloaded `cache_dir`, the model cache is scanned/cleared under
    /// the offload root while corpus stays under `data_dir`.
    #[test]
    fn offloaded_cache_dir_splits_corpus_from_cache() {
        let data = tempfile::tempdir().expect("data tempdir");
        let cache = tempfile::tempdir().expect("cache tempdir");
        let cache_str = cache.path().to_string_lossy().into_owned();
        let paths = StoragePaths::new(data.path(), Some(&cache_str));
        seed_all(&paths);

        // Cache lives under the offload root, not data_dir.
        assert!(cache.path().join("models").join("orpheus").exists());
        assert!(!data.path().join("models").exists());
        assert!(data.path().join("lens.db").exists());

        let stats = storage_stats_blocking(&paths, "").expect("stats");
        assert_eq!(stats.corpus_bytes, 10 + 20 + 30 + 40);
        assert_eq!(stats.reclaimable_cache_bytes, 100 + 200 + 300 + 400 + 700);
        assert_eq!(stats.retained_bytes, 500 + 50);

        let freed = clear_model_cache_blocking(&paths, "").expect("clear");
        assert_eq!(freed, 100 + 200 + 300 + 400 + 700);

        // Reclaimable cache removed from the offload root; corpus under data_dir intact.
        assert!(!cache.path().join("models").join("orpheus").exists());
        assert!(!cache.path().join("hf-cache").exists());
        assert!(data.path().join("lens.db").exists());
        assert!(data.path().join("lancedb").join("data.lance").exists());
    }
}
