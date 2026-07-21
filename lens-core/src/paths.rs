//! Typed resolver for the engine's on-disk directory roots (#238).

use std::path::{Path, PathBuf};

use crate::config::AppConfig;

/// Resolves on-disk locations, honoring an optional offloaded model-cache root
/// (`config.paths.cache_dir`, #238). Corpus lives under `data_dir`; re-downloadable
/// model/cache dirs live under `cache_root` (the offload target if set, else `data_dir`).
///
/// Owns DIRECTORY roots only — per-file names stay in the subsystem registries
/// (tts/asr), which receive the resolved directory.
#[derive(Debug, Clone)]
pub struct StoragePaths {
    data_dir: PathBuf,
    cache_root: PathBuf,
}

impl StoragePaths {
    /// `cache_root` is `cache_dir` when set and non-empty, else `data_dir` — an
    /// empty string is treated as unset so behavior matches the pre-#238 layout.
    pub fn new(data_dir: &Path, cache_dir: Option<&str>) -> Self {
        let cache_root = match cache_dir {
            Some(dir) if !dir.is_empty() => PathBuf::from(dir),
            _ => data_dir.to_path_buf(),
        };
        Self {
            data_dir: data_dir.to_path_buf(),
            cache_root,
        }
    }

    pub fn from_config(config: &AppConfig, data_dir: &Path) -> Self {
        Self::new(data_dir, config.paths.cache_dir.as_deref())
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    pub fn cache_root(&self) -> &Path {
        &self.cache_root
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("lens.db")
    }

    pub fn lancedb_root(&self) -> PathBuf {
        self.data_dir.join("lancedb")
    }

    pub fn sources_dir(&self) -> PathBuf {
        self.data_dir.join("sources")
    }

    pub fn notebooks_dir(&self) -> PathBuf {
        self.data_dir.join("notebooks")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.cache_root.join("models")
    }

    pub fn fastembed_cache(&self) -> PathBuf {
        self.models_dir().join("fastembed")
    }

    pub fn candle_cache(&self) -> PathBuf {
        self.models_dir().join("candle")
    }

    pub fn whisper_dir(&self) -> PathBuf {
        self.models_dir().join("whisper")
    }

    pub fn hf_cache(&self) -> PathBuf {
        self.cache_root.join("hf-cache")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cache_root_equals_data_dir() {
        let data = Path::new("/data");
        let p = StoragePaths::new(data, None);
        assert_eq!(p.data_dir(), data);
        assert_eq!(p.cache_root(), data);
        assert_eq!(p.models_dir(), data.join("models"));
        assert_eq!(p.fastembed_cache(), data.join("models").join("fastembed"));
        assert_eq!(p.hf_cache(), data.join("hf-cache"));
    }

    #[test]
    fn set_cache_dir_splits_corpus_from_cache() {
        let data = Path::new("/data");
        let cache = "/cache";
        let p = StoragePaths::new(data, Some(cache));

        // Corpus stays under data_dir.
        assert_eq!(p.db_path(), data.join("lens.db"));
        assert_eq!(p.lancedb_root(), data.join("lancedb"));
        assert_eq!(p.sources_dir(), data.join("sources"));
        assert_eq!(p.notebooks_dir(), data.join("notebooks"));

        // Cache moves under the offload root.
        let cache = Path::new(cache);
        assert_eq!(p.cache_root(), cache);
        assert_eq!(p.models_dir(), cache.join("models"));
        assert_eq!(p.fastembed_cache(), cache.join("models").join("fastembed"));
        assert_eq!(p.candle_cache(), cache.join("models").join("candle"));
        assert_eq!(p.whisper_dir(), cache.join("models").join("whisper"));
        assert_eq!(p.hf_cache(), cache.join("hf-cache"));
    }

    #[test]
    fn empty_cache_dir_treated_as_none() {
        let data = Path::new("/data");
        let p = StoragePaths::new(data, Some(""));
        assert_eq!(p.cache_root(), data);
        assert_eq!(p.models_dir(), data.join("models"));
    }
}
