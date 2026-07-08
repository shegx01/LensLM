//! Background enrichment infrastructure: the [`EnrichmentJob`] message, the bounded
//! `mpsc` queue, and the dedicated background [`worker`] task (M4 Phase-3 Steps 3–5).
//!
//! Concurrency invariant (lock #3): the worker holds NO `ingest_lock` during its job
//! body. Enqueue is a non-blocking `try_send` issued after the ingest permit is
//! released, so a full channel can never deadlock. Dropped jobs are recovered by the
//! startup/rescan queue-rebuild.

mod batching;
pub mod coref;
pub mod embedding_text;
pub mod map;
pub mod meta;
pub mod reembed;
pub mod worker;

/// Shared enrichment test mock. Available in-crate (`#[cfg(test)]`) and to the
/// integration-test crate via the `test-util` feature.
#[cfg(any(test, feature = "test-util"))]
pub mod test_util;

pub use coref::{ChunkCoref, CorefResponse, CorefSub, apply_substitutions, resolve_coref_batch};
pub use embedding_text::{
    CorefStrategy, RelationsStrategy, compose_embedding_text, compose_prefix,
};
pub use map::{MapError, MapOutcome, build_structural_map};
pub use meta::{
    Budget, BudgetCheck, CacheKeyParts, ENRICHMENT_PROMPT_VERSION, ENRICHMENT_SIZE_GATE_TOKENS,
    EnrichmentMeta, MAP_QUALITY_FALLBACK, MAP_QUALITY_OK, MAP_QUALITY_SKIPPED, SessionBudget,
    StructuralMap,
};
pub use worker::{EnrichmentJob, spawn_worker};

/// Bounded capacity of the enrichment `mpsc` queue. A full channel drops the job;
/// the startup/rescan queue-rebuild re-enqueues any missed source (self-healing).
pub const ENRICHMENT_QUEUE_CAPACITY: usize = 1024;

/// Block types that skip the structural map (non-prose). Shared by the worker's
/// prose gate and the M13 graph builder's prose-leaf filter so they agree.
pub fn is_nonprose_block(block_type: Option<&str>) -> bool {
    matches!(block_type, Some("code") | Some("table") | Some("html"))
}
