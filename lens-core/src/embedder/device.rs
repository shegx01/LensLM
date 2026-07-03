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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkloadKind {
    /// Large ingest, full-notebook re-embed, batch-select add, drag-drop add — all
    /// funnel through `run_ingest`/`reembed_notebook` and embed many chunks. The
    /// default: the only real embed call sites today are bulk.
    #[default]
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
    /// The `candle` Apple-Metal GPU path.
    Metal,
}

impl Compute {
    /// Stable token used in the embedder-cache key so a CPU and a Metal embedder
    /// for the same `(model, backend)` occupy distinct slots.
    pub fn as_str(self) -> &'static str {
        match self {
            Compute::Cpu => "cpu",
            Compute::Metal => "metal",
        }
    }
}

/// Hardware acceleration a [`NativeAccelerator`] reports as available RIGHT NOW.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acceleration {
    /// An Apple Metal GPU device is constructible (Apple Silicon + feature built).
    Metal,
    /// No usable accelerator — fall back to CPU.
    None,
}

impl Acceleration {
    /// Whether a Metal GPU is available.
    pub fn is_metal(self) -> bool {
        matches!(self, Acceleration::Metal)
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

/// Apple-Metal accelerator: reports [`Acceleration::Metal`] iff a candle Metal
/// device is actually constructible on this machine (not merely compiled in).
///
/// Only built with `native-ml-metal` (aarch64-apple-darwin). Everywhere else the
/// engine uses [`CpuOnlyAccelerator`].
#[cfg(feature = "native-ml-metal")]
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalAccelerator;

#[cfg(feature = "native-ml-metal")]
impl NativeAccelerator for MetalAccelerator {
    fn probe(&self) -> Acceleration {
        match candle_core::Device::new_metal(0) {
            Ok(_) => Acceleration::Metal,
            Err(_) => Acceleration::None,
        }
    }
}

/// The engine's default accelerator for this build target: [`MetalAccelerator`] on
/// aarch64-apple-darwin with `native-ml-metal`, else [`CpuOnlyAccelerator`].
pub fn default_accelerator() -> std::sync::Arc<dyn NativeAccelerator> {
    #[cfg(feature = "native-ml-metal")]
    {
        std::sync::Arc::new(MetalAccelerator)
    }
    #[cfg(not(feature = "native-ml-metal"))]
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
    if !acceleration.is_metal() {
        return Compute::Cpu; // gate 2: hardware
    }
    if !spec.accelerate_hint {
        return Compute::Cpu; // gate 3: model (GPU-loser models stay CPU)
    }
    match workload {
        WorkloadKind::Interactive => Compute::Cpu, // gate 4: queries → CPU
        WorkloadKind::Bulk => Compute::Metal,      // gate 4: bulk → GPU
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
    fn defaults_are_cpu_and_bulk() {
        assert_eq!(Compute::default(), Compute::Cpu);
        assert_eq!(WorkloadKind::default(), WorkloadKind::Bulk);
    }

    #[test]
    fn cpu_only_accelerator_reports_none() {
        assert_eq!(CpuOnlyAccelerator.probe(), Acceleration::None);
        assert!(!CpuOnlyAccelerator.probe().is_metal());
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
