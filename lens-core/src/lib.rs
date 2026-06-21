//! `lens-core` — the headless engine for LensLM.
//!
//! Pure Rust. Contains no Tauri, windowing, or UI dependencies. All localized
//! file-parsing, database routines, and inference tasks will be implemented here.

pub mod error;

pub use error::LensError;

use std::sync::Arc;
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Mutable engine resources live here: database connection pool, model cache,
/// and configuration will be added as fields as the engine grows.
#[derive(Default)]
pub struct LensEngineInner {
    // db: sqlx::Pool<Sqlite>,
    // model_cache: HashMap<String, ModelHandle>,
    // config: AppConfig,
}

/// Thread-safe, cheaply-cloneable handle to the LensLM engine state.
///
/// Cloning shares the same underlying state (`Arc`). Mutations go through an
/// async-aware `RwLock` so guards can be safely held across `.await` points —
/// this is the interior mutability Tauri's immutable `State<T>` requires.
#[derive(Clone, Default)]
pub struct LensEngine {
    inner: Arc<RwLock<LensEngineInner>>,
}

impl LensEngine {
    /// Initializes an empty instance of the core LensLM state framework.
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquires a shared read guard over the engine state.
    pub async fn read(&self) -> RwLockReadGuard<'_, LensEngineInner> {
        self.inner.read().await
    }

    /// Acquires an exclusive write guard over the engine state.
    pub async fn write(&self) -> RwLockWriteGuard<'_, LensEngineInner> {
        self.inner.write().await
    }
}
