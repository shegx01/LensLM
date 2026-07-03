//! Per-job compute-device selection for embedding (issue #91).
//!
//! The engine keeps TWO physically-distinct execution paths for the SAME
//! `fastembed` embedding coordinate, held in numerical parity (cosine 1.000000,
//! recall@5 identical — measured, see `.omc/plans/issue-91-candle-metal-spike-results.md`):
//!
//! - **CPU** — the mature `fastembed`/ONNX path ([`crate::embedder::FastembedEmbedder`]);
//!   the default everywhere, and the only path for interactive queries / non-Apple
//!   hardware / GPU-loser models.
//! - **Metal** — a `candle` engine running the whole forward pass on the Apple GPU
//!   ([`crate::embedder::CandleNomicEmbedder`], feature `native-ml-metal`); used for
//!   **bulk** jobs on Apple Silicon, where it frees ~99% of the CPU and runs ~2.6×
//!   faster.
//!
//! Because the two engines are parity-identical, the device is a **per-job runtime
//! choice**, NOT a persisted notebook property — no migration, no second vector
//! table. [`select_compute`] is the pure policy that maps
//! `(hardware, model, backend, workload)` to a [`Compute`].
//!
//! ## Polymorphic seam
//!
//! Hardware capability is discovered through the [`NativeAccelerator`] trait so the
//! selection policy is decoupled from the concrete accelerator, additional
//! accelerators (MLX-Swift, CUDA, …) can be added without touching the policy, and
//! tests can inject a fake probe. This is the generic seam issue #91 mandates;
//! [`crate::embedder::CandleNomicEmbedder`] is merely its first consumer.

use crate::embedder::EmbeddingBackend;
use crate::embedder::registry::EmbeddingModelSpec;

/// What kind of embedding job is asking for a device.
///
/// Decisive gate 3 of [`select_compute`]: bulk work is GPU-eligible (there is a
/// CPU to relieve and per-batch throughput matters); a single interactive query is
/// not (GPU launch/first-use latency loses for one vector, and there is no CPU load
/// to offload).
///
/// Deliberately NOT `Default`: every caller must state its workload explicitly. A
/// `Default` of `Bulk` would be a foot-gun for the future query path — a bare
/// `WorkloadKind::default()` would silently route a single query to the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    /// Large ingest, full-notebook re-embed, batch-select add, drag-drop add — all
    /// funnel through `run_ingest`/`reembed_notebook` and embed many chunks.
    Bulk,
    /// A single query embedded at retrieval time (reserved for the query path).
    Interactive,
}

/// The resolved execution device for one embed job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compute {
    /// The `fastembed`/ONNX CPU path (or Ollama). The default.
    #[default]
    Cpu,
    /// The `candle` Apple-Metal GPU path (implemented; Apple Silicon).
    Metal,
    /// The NVIDIA CUDA GPU path (issue #91 — INTERFACE ONLY; not yet implemented).
    /// Reserved so the seam accommodates Win/Linux NVIDIA without reshaping the
    /// policy. Until a candle-CUDA backend is wired, a job resolving to `Cuda` falls
    /// back to fastembed-CPU (see `LensEngine::embedder_for`).
    Cuda,
}

impl Compute {
    /// Stable token used in the embedder-cache key so embedders for the same
    /// `(model, backend)` on different devices occupy distinct slots.
    pub fn as_str(self) -> &'static str {
        match self {
            Compute::Cpu => "cpu",
            Compute::Metal => "metal",
            Compute::Cuda => "cuda",
        }
    }
}

/// Hardware acceleration a [`NativeAccelerator`] reports as available RIGHT NOW.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acceleration {
    /// An Apple Metal GPU is available (Apple Silicon + `native-ml-metal`).
    Metal,
    /// An NVIDIA CUDA GPU is available (Win/Linux + `native-ml-cuda`) — issue #91
    /// INTERFACE ONLY; no accelerator constructs this yet.
    Cuda,
    /// No usable accelerator — fall back to CPU.
    None,
}

impl Acceleration {
    /// Whether a Metal GPU is available.
    pub fn is_metal(self) -> bool {
        matches!(self, Acceleration::Metal)
    }

    /// The [`Compute`] device a **bulk** job should use for this acceleration, or
    /// `None` when there's no GPU (→ CPU). Keeps [`select_compute`] accelerator-
    /// agnostic: adding an accelerator is one arm here, not a policy rewrite.
    pub fn bulk_compute(self) -> Option<Compute> {
        match self {
            Acceleration::Metal => Some(Compute::Metal),
            Acceleration::Cuda => Some(Compute::Cuda),
            Acceleration::None => Option::None,
        }
    }
}

/// Capability probe for native ML acceleration — the polymorphic seam (issue #91).
///
/// `Send + Sync` so an `Arc<dyn NativeAccelerator>` can live on the engine and be
/// shared across threads. Implementations must be cheap to `probe` (it may be
/// called per job); cache any expensive discovery internally.
pub trait NativeAccelerator: Send + Sync {
    /// What acceleration is available on this machine at this moment.
    fn probe(&self) -> Acceleration;
}

/// Fallback accelerator: reports no acceleration. Used on non-Apple-Silicon, when
/// the `native-ml-metal` feature is off, or as an explicit CPU-only override.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuOnlyAccelerator;

impl NativeAccelerator for CpuOnlyAccelerator {
    fn probe(&self) -> Acceleration {
        Acceleration::None
    }
}

/// Apple-Metal accelerator: reports [`Acceleration::Metal`] unconditionally.
///
/// This is **feature-flag-driven, NOT a runtime device-capability check** (issue
/// #91 decision): the `native-ml-metal` feature is target-gated to
/// aarch64-apple-darwin in `Cargo.toml`, and every Apple-Silicon Mac has a Metal
/// GPU — so the feature being compiled IS the guarantee. We deliberately do NOT
/// call `Device::new_metal(0)` here: a compile-time flag (rather than a runtime
/// probe) keeps the two engines cleanly separable, so a feature-off build exercises
/// the fastembed path for ALL models. The (astronomically unlikely) case of a
/// Metal device that won't construct is still caught by the graceful fallback at
/// candle-embedder construction time — never by refusing to select it here.
///
/// Only built with `native-ml-metal`. Everywhere else the engine uses
/// [`CpuOnlyAccelerator`].
#[cfg(feature = "native-ml-metal")]
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalAccelerator;

#[cfg(feature = "native-ml-metal")]
impl NativeAccelerator for MetalAccelerator {
    fn probe(&self) -> Acceleration {
        Acceleration::Metal
    }
}

/// Whether the GPU (candle + Apple Metal) embedding path is compiled AND active on
/// this build — i.e. an aarch64-apple-darwin build with `native-ml-metal`. This is
/// the SAME condition that lets [`select_compute`] pick `Metal` and that makes
/// candle the default engine for a supported model.
///
/// The UI uses it to label the on-device provider ("On-device · Apple GPU") and to
/// tell the user it's the fastest local option. `false` on every other build (where
/// the on-device engine is fastembed/ONNX on the CPU).
pub const fn gpu_embedding_active() -> bool {
    cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        feature = "native-ml-metal"
    ))
}

/// The registry model ids that ACTUALLY run on the GPU on this build — the
/// candle-wired, GPU-eligible models when the Metal path is active (issue #91).
///
/// GPU acceleration is a PER-MODEL property, not a provider one: on Apple Silicon
/// today only `nomic-embed-text-v1.5` is candle-wired, so it alone runs on the GPU;
/// `mxbai`/`bge-m3` are GPU-eligible (`accelerate_hint`) but not yet wired, so they
/// fall back to fastembed-CPU, and `all-minilm` is CPU by design
/// (`accelerate_hint = false`). The UI badges exactly this set "Apple GPU". Empty on
/// every non-Apple-Silicon / feature-off build (fastembed-CPU serves everything).
pub fn gpu_accelerated_model_ids() -> Vec<&'static str> {
    #[cfg(feature = "native-ml-metal")]
    {
        if gpu_embedding_active() {
            return crate::embedder::registry::REGISTRY
                .iter()
                .filter(|s| s.accelerate_hint && crate::embedder::candle_supports_model(s.id))
                .map(|s| s.id)
                .collect();
        }
    }
    Vec::new()
}

/// NVIDIA-CUDA accelerator — issue #91 INTERFACE ONLY.
///
/// The polymorphic seam ([`NativeAccelerator`]) is meant to accommodate more than
/// Metal: this is the reserved slot for Win/Linux NVIDIA GPUs. It is feature-gated
/// to `native-ml-cuda` (which pulls NO dependencies yet — the candle-CUDA backend +
/// toolchain are the future implementation). It reports [`Acceleration::Cuda`] so
/// the selection policy is exercised end-to-end, but until a candle-CUDA embedder
/// is wired, a `Compute::Cuda` job falls back to fastembed-CPU in
/// [`crate::LensEngine::embedder_for`]. Implementing CUDA = fill in that arm +
/// a `CandleCudaEmbedder`, with ZERO changes to this policy.
#[cfg(feature = "native-ml-cuda")]
#[derive(Debug, Default, Clone, Copy)]
pub struct CudaAccelerator;

#[cfg(feature = "native-ml-cuda")]
impl NativeAccelerator for CudaAccelerator {
    fn probe(&self) -> Acceleration {
        Acceleration::Cuda
    }
}

/// The engine's default accelerator for this build target: [`MetalAccelerator`] on
/// aarch64-apple-darwin with `native-ml-metal`, [`CudaAccelerator`] with
/// `native-ml-cuda` (Win/Linux NVIDIA — interface only), else
/// [`CpuOnlyAccelerator`].
pub fn default_accelerator() -> std::sync::Arc<dyn NativeAccelerator> {
    #[cfg(feature = "native-ml-metal")]
    {
        std::sync::Arc::new(MetalAccelerator)
    }
    #[cfg(all(feature = "native-ml-cuda", not(feature = "native-ml-metal")))]
    {
        std::sync::Arc::new(CudaAccelerator)
    }
    #[cfg(not(any(feature = "native-ml-metal", feature = "native-ml-cuda")))]
    {
        std::sync::Arc::new(CpuOnlyAccelerator)
    }
}

/// The per-job device policy (issue #91) — a pure function of four inputs, in
/// priority order. Returns the device the embed job SHOULD run on; the caller is
/// responsible for the final graceful fallback to CPU if that device can't
/// actually be constructed (never fail a job over a device choice).
///
/// 1. **Backend gate** — only the `fastembed` coordinate has a parity-identical GPU
///    engine. Ollama (and anything else) is CPU.
/// 2. **Hardware gate** — no Metal device → CPU.
/// 3. **Model gate** — `!spec.accelerate_hint` (e.g. `all-minilm`, a measured GPU
///    loser) → CPU.
/// 4. **Workload gate** — `Interactive` (single query) → CPU; `Bulk` → Metal.
///
/// (A future soft "Metal-health / contention" gate — Kokoro TTS or the LLM already
/// saturating the GPU, or low memory — will further downgrade `Bulk` to CPU; for
/// now that is covered by the caller's construction-time fallback.)
pub fn select_compute(
    acceleration: Acceleration,
    spec: &EmbeddingModelSpec,
    backend: EmbeddingBackend,
    workload: WorkloadKind,
) -> Compute {
    if backend != EmbeddingBackend::Fastembed {
        return Compute::Cpu; // gate 1: only fastembed has a parity GPU twin
    }
    let Some(gpu) = acceleration.bulk_compute() else {
        return Compute::Cpu; // gate 2: hardware (no GPU → CPU)
    };
    if !spec.accelerate_hint {
        return Compute::Cpu; // gate 3: model (GPU-loser models stay CPU)
    }
    match workload {
        WorkloadKind::Interactive => Compute::Cpu, // gate 4: queries → CPU
        WorkloadKind::Bulk => gpu,                 // gate 4: bulk → the GPU (Metal/Cuda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::registry::resolve;

    fn nomic() -> &'static EmbeddingModelSpec {
        resolve("nomic-embed-text-v1.5")
    }
    fn minilm() -> &'static EmbeddingModelSpec {
        resolve("all-minilm")
    }

    #[test]
    fn compute_as_str_round_trips() {
        assert_eq!(Compute::Cpu.as_str(), "cpu");
        assert_eq!(Compute::Metal.as_str(), "metal");
    }

    #[test]
    fn compute_defaults_to_cpu() {
        // Compute has a safe default (CPU); WorkloadKind deliberately does NOT
        // (every caller must state its workload — see the enum doc).
        assert_eq!(Compute::default(), Compute::Cpu);
    }

    #[test]
    fn cpu_only_accelerator_reports_none() {
        assert_eq!(CpuOnlyAccelerator.probe(), Acceleration::None);
        assert!(!CpuOnlyAccelerator.probe().is_metal());
    }

    #[test]
    fn gpu_embedding_active_tracks_the_feature() {
        // The signal is exactly the aarch64-macOS + native-ml-metal build condition.
        assert_eq!(
            gpu_embedding_active(),
            cfg!(all(
                target_os = "macos",
                target_arch = "aarch64",
                feature = "native-ml-metal"
            ))
        );
        // Feature-off (default CI build) it MUST be false.
        #[cfg(not(feature = "native-ml-metal"))]
        assert!(!gpu_embedding_active());
    }

    #[test]
    fn gpu_accelerated_models_empty_without_the_feature() {
        // Feature-off: nothing runs on the GPU; fastembed-CPU serves everything.
        #[cfg(not(feature = "native-ml-metal"))]
        assert!(gpu_accelerated_model_ids().is_empty());
        // Feature-on: only candle-wired, GPU-eligible models — never all-minilm.
        #[cfg(feature = "native-ml-metal")]
        {
            let ids = gpu_accelerated_model_ids();
            assert!(ids.contains(&"nomic-embed-text-v1.5"));
            assert!(!ids.contains(&"all-minilm"));
        }
    }

    #[test]
    fn default_model_is_gpu_accelerated_on_the_flag_build() {
        // On the Apple-GPU build the DEFAULT model must itself be GPU-accelerated so
        // a fresh install gets the GPU path by default (issue #91) — guards the
        // default from silently drifting to a CPU-only model.
        #[cfg(feature = "native-ml-metal")]
        assert!(
            gpu_accelerated_model_ids()
                .contains(&crate::embedder::registry::DEFAULT_EMBED_MODEL_ID)
        );
    }

    // ── CUDA interface (issue #91, interface only — enum variants + routing exist
    // regardless of feature; no candle-CUDA impl yet) ──────────────────────────
    #[test]
    fn bulk_compute_maps_each_accelerator() {
        assert_eq!(Acceleration::Metal.bulk_compute(), Some(Compute::Metal));
        assert_eq!(Acceleration::Cuda.bulk_compute(), Some(Compute::Cuda));
        assert_eq!(Acceleration::None.bulk_compute(), None);
    }

    #[test]
    fn compute_cuda_cache_token() {
        assert_eq!(Compute::Cuda.as_str(), "cuda");
    }

    #[test]
    fn cuda_routes_symmetrically_to_metal() {
        // Bulk + fastembed + a GPU-eligible model + CUDA available → Compute::Cuda.
        assert_eq!(
            select_compute(
                Acceleration::Cuda,
                nomic(),
                EmbeddingBackend::Fastembed,
                WorkloadKind::Bulk
            ),
            Compute::Cuda
        );
        // Same gates as Metal: interactive → CPU, GPU-loser model → CPU.
        assert_eq!(
            select_compute(
                Acceleration::Cuda,
                nomic(),
                EmbeddingBackend::Fastembed,
                WorkloadKind::Interactive
            ),
            Compute::Cpu
        );
        assert_eq!(
            select_compute(
                Acceleration::Cuda,
                minilm(),
                EmbeddingBackend::Fastembed,
                WorkloadKind::Bulk
            ),
            Compute::Cpu
        );
    }

    #[test]
    fn gate1_non_fastembed_backend_is_cpu() {
        // Even bulk + Metal + a GPU-eligible model → CPU when backend isn't fastembed.
        assert_eq!(
            select_compute(
                Acceleration::Metal,
                nomic(),
                EmbeddingBackend::Ollama,
                WorkloadKind::Bulk
            ),
            Compute::Cpu
        );
    }

    #[test]
    fn gate2_no_metal_hardware_is_cpu() {
        assert_eq!(
            select_compute(
                Acceleration::None,
                nomic(),
                EmbeddingBackend::Fastembed,
                WorkloadKind::Bulk
            ),
            Compute::Cpu
        );
    }

    #[test]
    fn gate3_gpu_loser_model_is_cpu() {
        // all-minilm has accelerate_hint=false → CPU even on bulk + Metal + fastembed.
        assert_eq!(
            select_compute(
                Acceleration::Metal,
                minilm(),
                EmbeddingBackend::Fastembed,
                WorkloadKind::Bulk
            ),
            Compute::Cpu
        );
    }

    #[test]
    fn gate4_interactive_query_is_cpu() {
        assert_eq!(
            select_compute(
                Acceleration::Metal,
                nomic(),
                EmbeddingBackend::Fastembed,
                WorkloadKind::Interactive
            ),
            Compute::Cpu
        );
    }

    #[test]
    fn bulk_gpu_eligible_model_on_metal_is_metal() {
        // The one path that selects Metal: fastembed + Metal + accelerate_hint + Bulk.
        assert_eq!(
            select_compute(
                Acceleration::Metal,
                nomic(),
                EmbeddingBackend::Fastembed,
                WorkloadKind::Bulk
            ),
            Compute::Metal
        );
    }
}
