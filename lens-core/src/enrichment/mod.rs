//! Background enrichment infrastructure for the M4 Phase-3 pass.
//!
//! This module owns the *wiring* (Step 3): the [`EnrichmentJob`] message, the
//! bounded `mpsc` queue, and the dedicated background [`worker`] task spawned in
//! [`crate::LensEngine::init`]. It deliberately does NOT yet implement the LLM
//! structural-map / `embedding_text` jobs (Step 4) or the re-embed new-table-flip
//! (Step 5): the worker's job body is a CLEARLY-MARKED stub that walks a source
//! through the [`EnrichmentStatus`](crate::notebooks::EnrichmentStatus) lifecycle
//! (`pending → enriching → enriched`) so the queue + worker lifecycle + recovery
//! are observable and testable now.
//!
//! ## Concurrency contract (the load-bearing part of Step 3)
//!
//! The worker holds NO [`ingest_lock`](crate::LensEngine) permit during its job
//! body (lock #3 of the plan). Enqueue is a non-blocking `try_send` issued by the
//! ingest path AFTER the source reaches `Indexed` AND the ingest permit has been
//! released — so it never awaits the lock and a full channel can never deadlock
//! against the held permit (a dropped job is recovered by the startup/rescan
//! queue-rebuild). See [`crate::LensEngine::enqueue_enrichment`].

pub mod coref;
pub mod embedding_text;
pub mod map;
pub mod meta;
pub mod reembed;
pub mod worker;

#[cfg(test)]
pub(crate) mod test_util;

pub use coref::{ChunkCoref, CorefResponse, CorefSub, apply_substitutions, resolve_coref_batch};
pub use embedding_text::{CorefStrategy, compose_embedding_text, compose_prefix};
pub use map::{MapError, MapOutcome, build_structural_map};
pub use meta::{
    Budget, BudgetCheck, CacheKeyParts, ENRICHMENT_PROMPT_VERSION, ENRICHMENT_SIZE_GATE_TOKENS,
    EnrichmentMeta, MAP_QUALITY_FALLBACK, MAP_QUALITY_OK, MAP_QUALITY_SKIPPED, SessionBudget,
    StructuralMap,
};
pub use worker::{EnrichmentJob, spawn_worker};

/// Bounded capacity of the enrichment `mpsc` queue (plan Step 3: ~1024).
///
/// `try_send` into a full channel logs and drops the job; the startup/rescan
/// queue-rebuild ([`crate::LensEngine::rebuild_enrichment_queue`]) re-enqueues any
/// `Indexed && enrichment_status IN (none, failed, pending)` source, so an
/// overflow is self-healing rather than a lost-update.
pub const ENRICHMENT_QUEUE_CAPACITY: usize = 1024;
