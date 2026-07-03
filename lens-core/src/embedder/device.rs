//! Per-job compute-device selection for embedding (issue #91).
//!
//! Two physically-distinct paths share the SAME embedding coordinate in numerical
//! parity (cosine 1.000000; measured): CPU (fastembed/ONNX, the default) and Metal
//! (candle, ~2.6× bulk throughput, ~99% CPU offload on Apple Silicon). Device is a
//! per-job runtime choice — NOT persisted — so no migration or second vector table.
//!
//! The [`NativeAccelerator`] trait decouples the selection policy from the concrete
//! accelerator so adding a new GPU engine (CUDA, MLX) requires zero policy changes.

use crate::embedder::EmbeddingBackend;
use crate::embedder::registry::EmbeddingModelSpec;

/// What kind of embedding job is requesting a device.
///
/// Deliberately NOT `Default`: every caller must state its workload explicitly.
/// A `Default` of `Bulk` would silently route a single interactive query to the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    Bulk,
    Interactive,
}

/// The resolved execution device for one embed job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compute {
    #[default]
    Cpu,
    Metal,
    /// NVIDIA CUDA (issue #91 — interface only; falls back to CPU until wired).
    Cuda,
}

impl Compute {
    /// Cache-key token; distinct per device so same `(model, backend)` on
    /// different devices occupies separate slots.
    pub fn as_str(self) -> &'static str {
        match self {
            Compute::Cpu => "cpu",
            Compute::Metal => "metal",
            Compute::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acceleration {
    Metal,
    /// NVIDIA CUDA (issue #91 — interface only).
    Cuda,
    None,
}

impl Acceleration {
    pub fn is_metal(self) -> bool {
        matches!(self, Acceleration::Metal)
    }

    /// Maps this accelerator to the [`Compute`] a bulk job should use.
    /// Keeps [`select_compute`] accelerator-agnostic: adding one is one arm here.
    pub fn bulk_compute(self) -> Option<Compute> {
        match self {
            Acceleration::Metal => Some(Compute::Metal),
            Acceleration::Cuda => Some(Compute::Cuda),
            Acceleration::None => Option::None,
        }
    }
}

/// Capability probe for native ML acceleration — the polymorphic seam (issue #91).
/// `Send + Sync` for `Arc<dyn NativeAccelerator>` on the engine. `probe` may be
/// called per job; cache expensive discovery internally.
pub trait NativeAccelerator: Send + Sync {
    /// What acceleration is available on this machine at this moment.
    fn probe(&self) -> Acceleration;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuOnlyAccelerator;

impl NativeAccelerator for CpuOnlyAccelerator {
    fn probe(&self) -> Acceleration {
        Acceleration::None
    }
}

/// Reports [`Acceleration::Metal`] unconditionally.
///
/// Feature-flag-driven, NOT a runtime probe (issue #91): `native-ml-metal` is
/// target-gated to aarch64-apple-darwin where every Mac has Metal, so the feature
/// being compiled IS the guarantee. Unlikely Metal device construction failures
/// are caught at candle-embedder init time via the caller's graceful fallback.
#[cfg(feature = "native-ml-metal")]
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalAccelerator;

#[cfg(feature = "native-ml-metal")]
impl NativeAccelerator for MetalAccelerator {
    fn probe(&self) -> Acceleration {
        Acceleration::Metal
    }
}

/// Whether the candle + Apple Metal embedding path is compiled and active.
/// Used by the UI to label the on-device provider ("On-device · Apple GPU").
pub const fn gpu_embedding_active() -> bool {
    cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        feature = "native-ml-metal"
    ))
}

/// Registry model ids that actually run on the GPU on this build — candle-wired
/// and GPU-eligible. The UI badges this set "Apple GPU". Empty on feature-off builds.
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

/// Reserved Win/Linux NVIDIA slot (issue #91 — interface only). Reports
/// `Acceleration::Cuda`; a `Compute::Cuda` job falls back to CPU until a
/// `CandleCudaEmbedder` is wired. Adding CUDA = implement that backend + extend
/// `LensEngine::embedder_for` with zero changes to this policy.
#[cfg(feature = "native-ml-cuda")]
#[derive(Debug, Default, Clone, Copy)]
pub struct CudaAccelerator;

#[cfg(feature = "native-ml-cuda")]
impl NativeAccelerator for CudaAccelerator {
    fn probe(&self) -> Acceleration {
        Acceleration::Cuda
    }
}

/// Default accelerator for this build target.
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

/// Per-job device policy (issue #91). Pure function of four priority-ordered gates:
/// 1. Backend — only fastembed has a parity GPU twin; Ollama is always CPU.
/// 2. Hardware — no GPU → CPU.
/// 3. Model — `!spec.accelerate_hint` (e.g. all-minilm, a measured loser) → CPU.
/// 4. Workload — `Interactive` → CPU; `Bulk` → GPU.
///
/// Caller is responsible for a graceful fallback if the device can't be constructed.
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
        assert_eq!(Compute::default(), Compute::Cpu);
    }

    #[test]
    fn cpu_only_accelerator_reports_none() {
        assert_eq!(CpuOnlyAccelerator.probe(), Acceleration::None);
        assert!(!CpuOnlyAccelerator.probe().is_metal());
    }

    #[test]
    fn gpu_embedding_active_tracks_the_feature() {
        assert_eq!(
            gpu_embedding_active(),
            cfg!(all(
                target_os = "macos",
                target_arch = "aarch64",
                feature = "native-ml-metal"
            ))
        );
        #[cfg(not(feature = "native-ml-metal"))]
        assert!(!gpu_embedding_active());
    }

    #[test]
    fn gpu_accelerated_models_empty_without_the_feature() {
        #[cfg(not(feature = "native-ml-metal"))]
        assert!(gpu_accelerated_model_ids().is_empty());
        #[cfg(feature = "native-ml-metal")]
        {
            let ids = gpu_accelerated_model_ids();
            assert!(ids.contains(&"nomic-embed-text-v1.5"));
            assert!(!ids.contains(&"all-minilm"));
        }
    }

    #[test]
    fn default_model_is_gpu_accelerated_on_the_flag_build() {
        #[cfg(feature = "native-ml-metal")]
        assert!(
            gpu_accelerated_model_ids()
                .contains(&crate::embedder::registry::DEFAULT_EMBED_MODEL_ID)
        );
    }

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
        assert_eq!(
            select_compute(
                Acceleration::Cuda,
                nomic(),
                EmbeddingBackend::Fastembed,
                WorkloadKind::Bulk
            ),
            Compute::Cuda
        );
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
