//! Parent-child chunker: converts a flat [`Block`] list into a two-level
//! adjacency-list hierarchy of [`Chunk`] rows ready for insertion into the
//! `chunks` SQLite table.
//!
//! ## Hierarchy
//!
//! * **Parents** (`level = 0`, `kind = "parent"`, `parent_id = None`): pack
//!   adjacent blocks into windows of approximately **512 tokens** each.
//! * **Children** (`level = 1`, `kind = "child"`, `parent_id = Some(parent_id)`):
//!   split each parent text into windows of approximately **128 tokens** each,
//!   with a small overlap between adjacent children (see [`CHILD_OVERLAP_TOKENS`]).
//!
//! Both parents and children are returned in one `Vec<Chunk>` (interleaved:
//! each parent followed immediately by its children).  They all become rows in
//! the `chunks` table keyed by `source_id` — the caller (the ingest pipeline)
//! fills in `source_id`, `page`, `enrichment`, and `created_at` before inserting.
//!
//! ## Byte-identity invariant
//!
//! Every returned chunk satisfies:
//! ```text
//! src[chunk.char_start as usize..chunk.char_end as usize] == chunk.text
//! ```
//! The public functions take `src: &str` so they can slice the original bytes
//! directly; blocks are joined by preserving the source bytes between them rather
//! than inserting synthetic `\n` separators.  This holds even with child overlap:
//! overlapping children share source bytes, but each child's `text` is still a
//! single verbatim contiguous source span.
//!
//! ## Token counting
//!
//! Token counts are performed by the nomic `tokenizer.json` passed as a
//! `tokenizers::Tokenizer`.  This is the **same tokenizer** that
//! `fastembed`/`nomic-embed-text-v1.5` uses internally, so the 512/128 windows
//! are measured with the same vocabulary as the eventual embedding step.
//!
//! Window *packing* uses an **incremental** count: each block is tokenized once
//! and a running per-block sum drives the "would this overflow?" decision.  The
//! per-block sum is a conservative over-estimate of the merged-span count
//! (`encode(.., true)` adds CLS/SEP special tokens per call, and cross-block
//! subword merges only ever *reduce* the count), so packing never *under*-counts
//! and never exceeds the bound.  At each flush boundary the exact merged span is
//! re-encoded once to record the precise `token_start`/`token_end`.  This avoids
//! the previous O(n²) behaviour of re-tokenizing the whole growing window on
//! every block.
//!
//! ## Tolerance
//!
//! The token-window targets are **soft**:
//!
//! ```text
//! child token span  ≤ 128 + CHILD_TOKEN_TOLERANCE
//! parent token span ≤ 512 + PARENT_TOKEN_TOLERANCE
//! ```
//!
//! `CHILD_TOKEN_TOLERANCE = 1` and `PARENT_TOKEN_TOLERANCE = 1` allow the
//! tokenizer to produce a single boundary token that would otherwise push the
//! count one over the target (e.g. a newline that merges with the previous
//! token only in context).  The tolerances are deliberately narrow; chunks are
//! split at block boundaries first, so in practice most windows land well
//! below the target.
//!
//! ## Deterministic ids for the eval harness
//!
//! [`chunk_blocks`] uses `uuid::Uuid::now_v7()` (time-ordered, non-deterministic).
//! The eval harness needs stable ids so `gold_chunk_ids` in `queries.json` stay
//! valid run-to-run.  Use [`chunk_blocks_deterministic`] for fixture corpora: it
//! assigns a content-derived id computed as
//! `sha2::Sha256(level_bytes || "\x00" || section_path || "\x00" || text || "\x00" || ordinal_bytes)`,
//! hex-encoded.  Do NOT use deterministic ids in production — two chunks with
//! identical text but different `section_path`s (or different ordinals) would
//! still get distinct ids, but the scheme couples ids to content which is
//! unsuitable for mutable production data.

use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use uuid::Uuid;

use crate::LensError;
use crate::parse::Block;

/// Chunk-kind labels (the `Chunk::kind` / `chunks.kind` column values).
///
/// Single source of truth for the two-level hierarchy's kind literals.
pub(crate) mod kind {
    /// A level-0 parent chunk.
    pub const PARENT: &str = "parent";
    /// A level-1 child chunk.
    pub const CHILD: &str = "child";
    /// A level-2 doc-summary RAPTOR node (`parent_id = NULL`, `source_id` set),
    /// emitted by the M4 Phase-3 enrichment pass. Consumed by the enrichment
    /// worker (Step 4/5); declared here as the schema-level kind constant.
    pub const SUMMARY: &str = "summary";
}

/// Soft upper bound for child-chunk token span (target = 128, tolerance = 1).
pub const CHILD_TOKEN_TARGET: usize = 128;
/// Soft upper bound for parent-chunk token span (target = 512, tolerance = 1).
pub const PARENT_TOKEN_TARGET: usize = 512;
/// Allowed overshoot above the child target due to boundary tokenization.
pub const CHILD_TOKEN_TOLERANCE: usize = 1;
/// Allowed overshoot above the parent target due to boundary tokenization.
pub const PARENT_TOKEN_TOLERANCE: usize = 1;

/// Number of tokens each child re-shares with the tail of the previous child.
///
/// Children are otherwise non-overlapping windows.  A concept whose phrasing
/// straddles a child boundary would, without overlap, be split across two
/// children such that *neither* embeds the whole concept — hurting recall.  By
/// having each child (after the first within a parent) start ~`CHILD_OVERLAP_TOKENS`
/// tokens back into the previous child's tail, the boundary concept appears
/// intact in at least one child.
///
/// `16` is ≈12% of the 128-token child target — large enough to capture a short
/// phrase spanning the seam, small enough to keep redundancy (and embedding
/// cost) low.  Parents stay **non-overlapping**: overlap is a child-level
/// retrieval aid, and overlapping parents would inflate the context the LLM
/// re-reads after a child hit.
///
/// Byte-identity is preserved: the overlap means adjacent child spans *share*
/// source bytes, but each child's `text` is still one verbatim contiguous slice
/// of `src`.
pub const CHILD_OVERLAP_TOKENS: usize = 16;

/// A chunk row ready for insertion into the `chunks` SQLite table.
///
/// Fields map 1-to-1 to the table columns defined in
/// `lens-core/migrations/0001_init.sql:40-58`.  The caller (ingest pipeline)
/// fills in the remaining table columns: `source_id`, `page`, `enrichment`,
/// and `created_at`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// UUIDv7 primary key (production) or content-derived hex id (eval fixtures).
    pub id: String,
    /// `None` for level-0 parent chunks; `Some(parent.id)` for level-1 children.
    pub parent_id: Option<String>,
    /// `"parent"` for level-0, `"child"` for level-1.
    pub kind: String,
    /// `0` = parent, `1` = child.
    pub level: i32,
    /// Heading trail in force at the first block of this chunk (e.g. `"A > B"`).
    pub section_path: String,
    /// Verbatim bytes from the source document.
    ///
    /// Invariant: `src[char_start as usize..char_end as usize] == text`
    /// (bytes, not chars — consistent with [`Block::char_start`]).
    pub text: String,
    /// Cumulative token index of the first token in this chunk across the whole
    /// source.  Used by the retrieval layer to reconstruct reading order.
    ///
    /// Token offsets **tile** their container: parents partition the whole
    /// source, and a parent's children partition that parent's token range
    /// (`children[0].token_start == parent.token_start` and
    /// `children[last].token_end == parent.token_end`). Overlapping children
    /// share source *bytes* but are assigned non-overlapping token tiles, so
    /// these offsets stay contiguous and never overshoot the parent (see
    /// [`retile_child_token_offsets`]).
    pub token_start: i64,
    /// Cumulative token index one-past the last token in this chunk.
    pub token_end: i64,
    /// Byte offset of the first byte of this chunk in the source string.
    pub char_start: i64,
    /// Byte offset one-past the last byte of this chunk in the source string.
    pub char_end: i64,
    /// Block type of the first (or dominant) block composing this chunk.
    ///
    /// **Lossy for multi-block parents.**  A parent packed from several blocks
    /// carries only the **first** (dominant) block's `block_type` — there is no
    /// single value that describes a mixed window, so downstream consumers must
    /// treat a parent's `block_type` as the leading block's type, not an
    /// invariant over every byte in the parent.  Child chunks inherit the type
    /// of the block they were split from.  `None` is valid (the column is
    /// nullable in SQLite).
    pub block_type: Option<String>,
    /// JSON-serialized [`SourceAnchor`](crate::extract::SourceAnchor) for this
    /// chunk — the format-native coordinates of the first (or dominant) block
    /// that composed this chunk (consistent with how `block_type` / `section_path`
    /// inherit from the first block).
    ///
    /// Stored as a pre-serialized JSON `String` rather than the enum itself so
    /// `Chunk` stays format-agnostic and keeps its `Eq` derivation (not all
    /// anchor variants are total-order comparable). `None` means no anchor was
    /// supplied (possible for chunks produced before Step 4, or via
    /// `chunk_blocks` called directly without an anchor map). The string is
    /// persisted verbatim into the `chunks.source_anchor` TEXT column.
    pub source_anchor: Option<String>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Converts a flat [`Block`] slice into a two-level hierarchy of [`Chunk`]s.
///
/// `src` must be the original source string from which `blocks` were parsed.
/// Each returned chunk satisfies
/// `src[chunk.char_start as usize..chunk.char_end as usize] == chunk.text`.
///
/// Uses `uuid::Uuid::now_v7()` for ids — suitable for production but
/// **non-deterministic** across runs.  For eval fixtures use
/// [`chunk_blocks_deterministic`] instead.
///
/// Returns `Err(LensError::Parse(_))` if the tokenizer fails to encode any
/// text segment.
pub fn chunk_blocks(
    src: &str,
    blocks: &[Block],
    tokenizer: &Tokenizer,
) -> Result<Vec<Chunk>, LensError> {
    chunk_blocks_inner(src, blocks, tokenizer, IdStrategy::V7)
}

/// Like [`chunk_blocks`] but assigns **content-derived ids** so that ids are
/// stable across runs for the same input.
///
/// Intended for the `cargo run --bin eval` eval harness only.  See the module
/// documentation for the id derivation scheme.
pub fn chunk_blocks_deterministic(
    src: &str,
    blocks: &[Block],
    tokenizer: &Tokenizer,
) -> Result<Vec<Chunk>, LensError> {
    chunk_blocks_inner(src, blocks, tokenizer, IdStrategy::ContentDerived)
}

// ---------------------------------------------------------------------------
// Internal implementation
// ---------------------------------------------------------------------------

/// Controls how [`Chunk::id`] is generated.
#[derive(Clone, Copy)]
enum IdStrategy {
    /// UUIDv7 — time-ordered, suitable for production inserts.
    V7,
    /// SHA-256 of `level_bytes || "\x00" || section_path || "\x00" || text || "\x00" || ordinal_bytes`
    /// — stable across runs for the same input, suitable for eval fixture gold sets.
    ContentDerived,
}

impl IdStrategy {
    fn make_id(&self, level: i32, section_path: &str, text: &str, ordinal: usize) -> String {
        match self {
            IdStrategy::V7 => Uuid::now_v7().to_string(),
            IdStrategy::ContentDerived => {
                let mut hasher = Sha256::new();
                hasher.update(level.to_le_bytes());
                hasher.update(b"\x00");
                hasher.update(section_path.as_bytes());
                hasher.update(b"\x00");
                hasher.update(text.as_bytes());
                hasher.update(b"\x00");
                hasher.update(ordinal.to_le_bytes());
                format!("{:x}", hasher.finalize())
            }
        }
    }
}

/// Tokenizes `text`, returning the token-id count.  Maps tokenizer errors to
/// `LensError::Parse`.
fn token_count(tokenizer: &Tokenizer, text: &str) -> Result<usize, LensError> {
    tokenizer
        .encode(text, true)
        .map(|enc| enc.get_ids().len())
        .map_err(|e| LensError::Parse(format!("tokenizer encode failed: {e}")))
}

/// Shared implementation backing both [`chunk_blocks`] and
/// [`chunk_blocks_deterministic`].
fn chunk_blocks_inner(
    src: &str,
    blocks: &[Block],
    tokenizer: &Tokenizer,
    strategy: IdStrategy,
) -> Result<Vec<Chunk>, LensError> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }

    // ── Step 1: pack blocks into parent windows ───────────────────────────
    // A parent window is a contiguous run of blocks whose combined raw source
    // span fits within PARENT_TOKEN_TARGET tokens.  We use the verbatim source
    // bytes (from char_start of the first block to char_end of the last) as the
    // parent text so the byte-identity invariant holds.
    let parent_windows = build_parent_windows(src, blocks, tokenizer)?;

    // ── Step 2: split each parent into child windows ──────────────────────
    let mut result: Vec<Chunk> = Vec::new();
    let mut running_token_offset: i64 = 0;
    let mut child_ordinal: usize = 0;

    for (parent_ordinal, win) in parent_windows.iter().enumerate() {
        let parent_token_count = token_count(tokenizer, &win.text)? as i64;
        let parent_id = strategy.make_id(0, &win.section_path, &win.text, parent_ordinal);

        let parent_token_start = running_token_offset;
        let parent_token_end = running_token_offset + parent_token_count;

        // Build children by splitting the parent text at block boundaries or
        // at a hard character split if a single block exceeds the child target.
        let mut children = build_child_chunks(
            src,
            win,
            tokenizer,
            &parent_id,
            parent_token_start,
            &strategy,
            &mut child_ordinal,
        )?;

        // Retile child token offsets so children TILE the parent token range:
        // children may overlap (sharing source bytes), which would inflate the
        // cumulative `token_end` past the parent end (B4). We recompute each
        // child's `token_start`/`token_end` from its NON-OVERLAPPING tile measured
        // as prefix token counts of the parent text, so the last child's
        // `token_end` lands exactly on `parent_token_end`.
        retile_child_token_offsets(
            src,
            win,
            tokenizer,
            parent_token_start,
            parent_token_end,
            &mut children,
        )?;

        // Emit the parent chunk.
        result.push(Chunk {
            id: parent_id,
            parent_id: None,
            kind: kind::PARENT.to_string(),
            level: 0,
            section_path: win.section_path.clone(),
            text: win.text.clone(),
            token_start: parent_token_start,
            token_end: parent_token_end,
            char_start: win.char_start as i64,
            char_end: win.char_end as i64,
            block_type: win.block_type.clone(),
            // Anchor is attached post-chunking by the ingest pipeline (it aligns
            // chunks to source blocks by char offset). `chunk_blocks` itself stays
            // format-agnostic and leaves this field as None.
            source_anchor: None,
        });

        // Emit the children immediately after their parent.
        result.extend(children);

        running_token_offset = parent_token_end;
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Oversized-block splitting (shared by parent and child levels)
// ---------------------------------------------------------------------------

/// Splits a single oversized `block` whose text exceeds `limit` tokens into
/// bounded contiguous sub-spans, invoking `emit` once per sub-span.
///
/// This is the single source of truth for "find split point → slice → emit"
/// at both the parent and child levels; the two callers differ only in `limit`
/// and what they build from each `(seg_text, abs_start, abs_end)` triple
/// (a [`ParentWindow`] vs a child [`Chunk`]).  `block.text` is a verbatim slice
/// of `src` anchored at `block.char_start`, so slicing `src[abs_start..abs_end]`
/// yields byte-identical sub-span text.
///
/// `emit` receives `(seg_text, abs_start, abs_end)` where `abs_*` are absolute
/// byte offsets in `src` and `src[abs_start..abs_end] == seg_text`.
fn split_oversized<F>(
    src: &str,
    block: &Block,
    limit: usize,
    tokenizer: &Tokenizer,
    mut emit: F,
) -> Result<(), LensError>
where
    F: FnMut(&str, usize, usize) -> Result<(), LensError>,
{
    let limit = limit.max(1);
    let text = &block.text;
    let mut seg_start_byte: usize = 0; // byte offset within `block.text`

    while seg_start_byte < text.len() {
        let remaining = &text[seg_start_byte..];
        let seg_len = find_split(remaining, tokenizer, limit)?;
        let mut seg_end_byte = (seg_start_byte + seg_len).min(text.len());
        // Ensure progress — never produce a zero-length segment.
        if seg_end_byte <= seg_start_byte {
            seg_end_byte = (seg_start_byte + 1).min(text.len());
        }

        // Absolute byte offsets in the source string.
        let abs_start = block.char_start + seg_start_byte;
        let abs_end = (block.char_start + seg_end_byte).min(src.len());
        let seg_text = &src[abs_start..abs_end];

        emit(seg_text, abs_start, abs_end)?;

        seg_start_byte = seg_end_byte;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Parent window building
// ---------------------------------------------------------------------------

/// A packed window of blocks that will become one parent chunk.
struct ParentWindow {
    /// Verbatim source bytes: `src[char_start..char_end]`.
    text: String,
    /// Heading trail from the first block in this window.
    section_path: String,
    /// Block type of the first block in this window (see [`Chunk::block_type`]).
    block_type: Option<String>,
    /// Byte offset of the first byte of the first block.
    char_start: usize,
    /// Byte offset one-past the last byte of the last block.
    char_end: usize,
    /// The individual blocks that make up this window (used for child splitting).
    blocks: Vec<Block>,
}

/// Packs `blocks` into [`ParentWindow`]s of at most
/// `PARENT_TOKEN_TARGET + PARENT_TOKEN_TOLERANCE` tokens.
///
/// The window text is `src[first.char_start..last.char_end]` — verbatim source
/// bytes including any whitespace between blocks — so
/// `src[window.char_start..window.char_end] == window.text` always holds.
///
/// Token counting is **incremental**: each block is tokenized once and a running
/// per-block sum drives the overflow decision (see the module-level "Token
/// counting" note).  The exact merged-span count is recorded later, in
/// `chunk_blocks_inner`, when each window's parent chunk is emitted.
fn build_parent_windows(
    src: &str,
    blocks: &[Block],
    tokenizer: &Tokenizer,
) -> Result<Vec<ParentWindow>, LensError> {
    let mut windows: Vec<ParentWindow> = Vec::new();
    let limit = PARENT_TOKEN_TARGET + PARENT_TOKEN_TOLERANCE;

    let mut current_blocks: Vec<Block> = Vec::new();
    let mut current_char_start: usize = 0;
    let mut current_char_end: usize = 0;
    let mut current_tokens: usize = 0; // running per-block sum for the open window

    for block in blocks {
        // Tokenize this block exactly once.
        let block_tokens = token_count(tokenizer, &block.text)?;

        // If this single block already exceeds the limit on its own, split it
        // (after flushing any in-progress window) into multiple BOUNDED parent
        // windows so no parent ever exceeds PARENT_TOKEN_TARGET + tolerance. A
        // single oversized block previously became one unbounded parent, which
        // violated the AC "parent ≤ 512 + tolerance".
        if block_tokens > limit {
            if !current_blocks.is_empty() {
                flush_parent_window(
                    src,
                    &current_blocks,
                    current_char_start,
                    current_char_end,
                    &mut windows,
                );
                current_blocks.clear();
                current_tokens = 0;
            }
            split_oversized_parent_block(src, block, tokenizer, limit, &mut windows)?;
            continue;
        }

        // If adding this block would overflow the current window, flush it.
        // The running sum is a conservative over-estimate of the exact merged
        // count, so this never under-counts.
        if !current_blocks.is_empty() && current_tokens + block_tokens > limit {
            flush_parent_window(
                src,
                &current_blocks,
                current_char_start,
                current_char_end,
                &mut windows,
            );
            current_blocks.clear();
            current_tokens = 0;
        }

        if current_blocks.is_empty() {
            current_char_start = block.char_start;
        }
        current_char_end = block.char_end;
        current_tokens += block_tokens;
        current_blocks.push(block.clone());
    }

    // Flush the final window.
    if !current_blocks.is_empty() {
        flush_parent_window(
            src,
            &current_blocks,
            current_char_start,
            current_char_end,
            &mut windows,
        );
    }

    Ok(windows)
}

/// Splits a single block whose text exceeds the PARENT token limit into multiple
/// bounded [`ParentWindow`]s via the shared [`split_oversized`] helper at the
/// parent limit.
///
/// Each emitted window carries a synthetic single-block `blocks` vector that is a
/// verbatim sub-slice of the original block, so the downstream child splitter
/// re-splits each bounded parent into its own ≤ `CHILD_TOKEN_TARGET` children and
/// the byte-identity invariant holds for both levels
/// (`src[char_start..char_end] == text`).
fn split_oversized_parent_block(
    src: &str,
    block: &Block,
    tokenizer: &Tokenizer,
    limit: usize,
    windows: &mut Vec<ParentWindow>,
) -> Result<(), LensError> {
    split_oversized(
        src,
        block,
        limit,
        tokenizer,
        |seg_text, abs_start, abs_end| {
            let sub_block = Block {
                block_type: block.block_type.clone(),
                section_path: block.section_path.clone(),
                text: seg_text.to_string(),
                char_start: abs_start,
                char_end: abs_end,
            };
            windows.push(ParentWindow {
                text: seg_text.to_string(),
                section_path: block.section_path.clone(),
                block_type: Some(block.block_type.clone()),
                char_start: abs_start,
                char_end: abs_end,
                blocks: vec![sub_block],
            });
            Ok(())
        },
    )
}

/// Finalises a parent window and appends it to `windows`.
fn flush_parent_window(
    src: &str,
    blocks: &[Block],
    char_start: usize,
    char_end: usize,
    windows: &mut Vec<ParentWindow>,
) {
    let end = char_end.min(src.len());
    let text = src[char_start..end].to_string();
    let section_path = blocks
        .first()
        .map(|b| b.section_path.as_str())
        .unwrap_or("")
        .to_string();
    let block_type = blocks.first().map(|b| b.block_type.clone());
    windows.push(ParentWindow {
        text,
        section_path,
        block_type,
        char_start,
        char_end,
        blocks: blocks.to_vec(),
    });
}

// ---------------------------------------------------------------------------
// Child chunk building
// ---------------------------------------------------------------------------

/// Mutable state threaded through child packing, mirroring the locals that
/// [`flush_parent_window`] takes by argument.  Holds the open child window's
/// blocks, its byte span, and the running per-block token sum (incremental
/// count — see the module "Token counting" note).
struct ChildAccumulator<'a> {
    blocks: Vec<&'a Block>,
    char_start: usize,
    char_end: usize,
    /// Running per-block token sum for the open window (conservative).
    tokens: usize,
}

impl<'a> ChildAccumulator<'a> {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            char_start: 0,
            char_end: 0,
            tokens: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// Read-only context for the child splitter, grouping the parameters that do
/// not change across a single parent's children.
struct ChildCtx<'a> {
    src: &'a str,
    tokenizer: &'a Tokenizer,
    parent_id: &'a str,
    strategy: &'a IdStrategy,
}

/// Splits a parent window into child chunks of ≤ `CHILD_TOKEN_TARGET +
/// CHILD_TOKEN_TOLERANCE` tokens, with a [`CHILD_OVERLAP_TOKENS`]-token overlap
/// between adjacent children.
///
/// Splitting follows block boundaries first: blocks are packed into child
/// windows exactly as parent windows are packed from blocks (incremental token
/// count).  If a single block exceeds the child limit it is split at whitespace
/// boundaries (best effort — guaranteed to terminate).
///
/// Each child's text is `src[child.char_start..child.char_end]`, preserving the
/// byte-identity invariant.  Overlapping children share source bytes, but each
/// child's span is still a single verbatim contiguous slice.
fn build_child_chunks(
    src: &str,
    win: &ParentWindow,
    tokenizer: &Tokenizer,
    parent_id: &str,
    parent_token_start: i64,
    strategy: &IdStrategy,
    child_ordinal: &mut usize,
) -> Result<Vec<Chunk>, LensError> {
    let limit = CHILD_TOKEN_TARGET + CHILD_TOKEN_TOLERANCE;
    let ctx = ChildCtx {
        src,
        tokenizer,
        parent_id,
        strategy,
    };
    let mut children: Vec<Chunk> = Vec::new();
    let mut running_token_offset = parent_token_start;
    let mut acc = ChildAccumulator::new();
    // Byte offset within the parent span at which the *next* child window should
    // begin so it overlaps the tail of the just-flushed child.  `None` means no
    // overlap seed is pending (first child, or after an oversized-block split).
    let mut pending_overlap_start: Option<usize> = None;

    for block in &win.blocks {
        // Tokenize each block exactly once (incremental count — no re-encoding
        // of the growing window).
        let block_tokens = token_count(tokenizer, &block.text)?;

        // Oversized single block: flush the current child window first, then
        // hard-split the block at whitespace boundaries.  The hard-split path
        // applies the same `CHILD_OVERLAP_TOKENS` overlap as the packing path,
        // so adjacent hard-split children also share boundary bytes (this is
        // the most common long-prose case — a single oversized paragraph).
        if block_tokens > limit {
            running_token_offset = flush_child_window(
                &ctx,
                &mut acc,
                &mut children,
                running_token_offset,
                child_ordinal,
            )?;
            pending_overlap_start = None;
            split_oversized_child_block(
                &ctx,
                win,
                block,
                limit,
                &mut children,
                &mut running_token_offset,
                child_ordinal,
            )?;
            continue;
        }

        // When opening a fresh window, fold any pending overlap seed in. Keep
        // the seed only if its tail token cost plus this block still fits the
        // limit; otherwise drop the overlap for this window and start at the
        // block boundary so the window never exceeds the bound. The seeded
        // tokens become the window's starting count (conservative incremental
        // sum: overlap-tail tokens + per-block sums).
        if acc.is_empty() {
            let overlap_tokens = match pending_overlap_start {
                Some(ov) => seed_overlap_tokens(src, ov, block.char_start, tokenizer)?,
                None => 0,
            };
            if overlap_tokens + block_tokens > limit {
                pending_overlap_start = None;
                acc.char_start = block.char_start;
                acc.tokens = 0;
            } else {
                acc.char_start = pending_overlap_start.unwrap_or(block.char_start);
                acc.tokens = overlap_tokens;
            }
            acc.char_end = block.char_end;
            acc.tokens += block_tokens;
            acc.blocks.push(block);
            continue;
        }

        // Incremental overflow check: running sum + this block. The running sum
        // is a conservative over-estimate of the exact merged-span count, so we
        // never under-count and never exceed the bound.
        if acc.tokens + block_tokens > limit {
            running_token_offset = flush_child_window(
                &ctx,
                &mut acc,
                &mut children,
                running_token_offset,
                child_ordinal,
            )?;
            pending_overlap_start = overlap_back_off(src, win, children.last(), tokenizer)?;

            // Reopen the window with the (possibly overlap-seeded) block.
            let overlap_tokens = match pending_overlap_start {
                Some(ov) => seed_overlap_tokens(src, ov, block.char_start, tokenizer)?,
                None => 0,
            };
            if overlap_tokens + block_tokens > limit {
                pending_overlap_start = None;
                acc.char_start = block.char_start;
                acc.tokens = 0;
            } else {
                acc.char_start = pending_overlap_start.unwrap_or(block.char_start);
                acc.tokens = overlap_tokens;
            }
            acc.char_end = block.char_end;
            acc.tokens += block_tokens;
            acc.blocks.push(block);
            continue;
        }

        // Block fits in the open window.
        acc.char_end = block.char_end;
        acc.tokens += block_tokens;
        acc.blocks.push(block);
    }

    // Flush the final child window.
    flush_child_window(
        &ctx,
        &mut acc,
        &mut children,
        running_token_offset,
        child_ordinal,
    )?;

    Ok(children)
}

/// Recomputes the `token_start`/`token_end` of each child so the children TILE
/// the parent's token range: the last child's `token_end` equals
/// `parent_token_end` and the first child's `token_start` equals
/// `parent_token_start` (B4).
///
/// Children may share source bytes via [`CHILD_OVERLAP_TOKENS`]. Summing each
/// child's full (overlapping) token count would push the cumulative offset past
/// the parent end. Instead we assign token positions from the child's
/// NON-OVERLAPPING tile, measured as a prefix token count of the parent text:
///
/// * `tile_start(0)   = parent.char_start`
/// * `tile_start(i>0) = children[i-1].char_end`  (where the previous tile ended)
/// * `token_start(i)  = parent_token_start + tokens(src[parent.char_start..tile_start(i)])`
/// * `token_end(i)    = parent_token_start + tokens(src[parent.char_start..children[i].char_end])`
///
/// Because the tiles partition the parent byte span and the last child ends at
/// `parent.char_end`, the prefix counts tile the parent token range exactly. The
/// stored `char_start`/`char_end` (and `text`) are left untouched, so overlap and
/// byte-identity are preserved — only the reading-order token offsets change.
fn retile_child_token_offsets(
    src: &str,
    win: &ParentWindow,
    tokenizer: &Tokenizer,
    parent_token_start: i64,
    parent_token_end: i64,
    children: &mut [Chunk],
) -> Result<(), LensError> {
    let Some((last, head)) = children.split_last_mut() else {
        return Ok(());
    };

    let parent_char_start = win.char_start;
    // Prefix token count of the parent text from its start up to `boundary`.
    // An empty prefix is 0 tokens (a bare `encode("", true)` would otherwise
    // return the 2 CLS/SEP special tokens and offset every child by 2).
    let prefix_tokens = |boundary: usize| -> Result<i64, LensError> {
        let end = boundary.min(src.len());
        if end <= parent_char_start {
            return Ok(0);
        }
        Ok(token_count(tokenizer, &src[parent_char_start..end])? as i64)
    };

    // `tile_start` walks forward through the non-overlapping tile boundaries.
    let mut tile_start = parent_char_start;
    for child in head.iter_mut() {
        let child_end = (child.char_end as usize).min(src.len());
        child.token_start = parent_token_start + prefix_tokens(tile_start)?;
        child.token_end = parent_token_start + prefix_tokens(child_end)?;
        // The next tile begins where this child's bytes end (its non-overlapping
        // contribution ends here; the next child's overlap re-shares earlier
        // bytes but contributes no new tile tokens before this point).
        tile_start = child_end.max(tile_start);
    }

    // The last child closes the parent exactly: its token_end snaps to the
    // parent end so the children tile with no overshoot or gap.
    last.token_start = parent_token_start + prefix_tokens(tile_start)?;
    last.token_end = parent_token_end;

    Ok(())
}

/// Finalises the open child window in `acc` (if any), appends the resulting
/// child [`Chunk`] to `children`, clears `acc`, and returns the updated running
/// token offset.  Symmetric with [`flush_parent_window`].
///
/// The exact token count of the flushed span is computed once here, so the
/// child's `token_end - token_start` is the true span length (satisfying the
/// `≤ CHILD_TOKEN_TARGET + tolerance` contract).
fn flush_child_window(
    ctx: &ChildCtx<'_>,
    acc: &mut ChildAccumulator<'_>,
    children: &mut Vec<Chunk>,
    running_token_offset: i64,
    child_ordinal: &mut usize,
) -> Result<i64, LensError> {
    if acc.is_empty() {
        return Ok(running_token_offset);
    }

    let end = acc.char_end.min(ctx.src.len());
    let text = ctx.src[acc.char_start..end].to_string();
    let toks = token_count(ctx.tokenizer, &text)? as i64;
    let section_path = acc
        .blocks
        .first()
        .map(|b| b.section_path.as_str())
        .unwrap_or("")
        .to_string();
    let block_type_val = acc.blocks.first().map(|b| b.block_type.clone());
    let t_start = running_token_offset;
    let t_end = t_start + toks;
    let id = ctx
        .strategy
        .make_id(1, &section_path, &text, *child_ordinal);
    *child_ordinal += 1;

    children.push(Chunk {
        id,
        parent_id: Some(ctx.parent_id.to_string()),
        kind: kind::CHILD.to_string(),
        level: 1,
        section_path,
        text,
        token_start: t_start,
        token_end: t_end,
        char_start: acc.char_start as i64,
        char_end: end as i64,
        block_type: block_type_val,
        source_anchor: None,
    });

    acc.blocks.clear();
    acc.char_start = 0;
    acc.char_end = 0;
    acc.tokens = 0;

    Ok(t_end)
}

/// Computes the byte offset within the parent span at which the *next* child
/// should begin so it re-shares ~[`CHILD_OVERLAP_TOKENS`] tokens with the tail
/// of `last_child`.  Returns `None` if no meaningful overlap is possible (no
/// previous child, or the previous child is itself ≤ the overlap budget).
///
/// The returned offset is clamped to a `char_boundary` of `src` and is `>=`
/// the previous child's `char_start` (so windows never run backward past the
/// child they overlap) — preserving byte-identity for the next contiguous span.
fn overlap_back_off(
    src: &str,
    win: &ParentWindow,
    last_child: Option<&Chunk>,
    tokenizer: &Tokenizer,
) -> Result<Option<usize>, LensError> {
    let Some(prev) = last_child else {
        return Ok(None);
    };
    if CHILD_OVERLAP_TOKENS == 0 {
        return Ok(None);
    }
    let prev_start = prev.char_start as usize;
    let prev_end = (prev.char_end as usize).min(src.len());
    if prev_end <= prev_start {
        return Ok(None);
    }

    // Find the largest byte offset `b` in (prev_start, prev_end) such that the
    // tail `src[b..prev_end]` is ~CHILD_OVERLAP_TOKENS tokens.  We want the
    // overlap span to hold *at most* CHILD_OVERLAP_TOKENS tokens of real
    // content, so walk whitespace boundaries from the end.
    let mut candidates: Vec<usize> = Vec::new();
    for (idx, ch) in src[prev_start..prev_end].char_indices() {
        if ch.is_whitespace() {
            let abs = prev_start + idx + ch.len_utf8();
            if abs > prev_start && abs < prev_end {
                candidates.push(abs);
            }
        }
    }
    // Pick the earliest candidate whose tail is ≤ CHILD_OVERLAP_TOKENS so we
    // capture as much of the boundary concept as the budget allows.
    let mut chosen: Option<usize> = None;
    for &b in &candidates {
        let tail_tokens = token_count(tokenizer, &src[b..prev_end])?;
        if tail_tokens <= CHILD_OVERLAP_TOKENS {
            chosen = Some(b);
            break;
        }
    }

    // The next child must extend strictly beyond the previous child, otherwise
    // the overlap seed alone would reproduce the previous child verbatim and
    // stall progress.  `win` bounds the parent span; if the overlap start is at
    // or past the parent end there is nothing left to overlap into.
    match chosen {
        Some(b) if b < win.char_end => Ok(Some(b)),
        _ => Ok(None),
    }
}

/// Token cost of the overlap tail `src[overlap_start..block_start]` that a
/// fresh, overlap-seeded child window would carry before its first block.
/// Returns `0` when the seed does not actually precede the block.
fn seed_overlap_tokens(
    src: &str,
    overlap_start: usize,
    block_start: usize,
    tokenizer: &Tokenizer,
) -> Result<usize, LensError> {
    let end = block_start.min(src.len());
    if overlap_start >= end {
        return Ok(0);
    }
    token_count(tokenizer, &src[overlap_start..end])
}

/// Splits a single oversized child block into bounded child [`Chunk`]s, applying
/// the same [`CHILD_OVERLAP_TOKENS`] overlap as the multi-block packing path so
/// adjacent hard-split children share boundary bytes.
///
/// This is the most common long-prose case — a single oversized paragraph — so
/// the overlap must engage here, not only when children pack from multiple
/// blocks.  Each child after the first re-starts ~`CHILD_OVERLAP_TOKENS` tokens
/// back into the previous child's tail (computed by the shared
/// [`overlap_back_off`] machinery), while the contiguous *consumed frontier*
/// still advances by a full window each step.
///
/// Invariants preserved:
/// * **Byte-identity** — absolute `char_start/char_end` slice `src` verbatim, so
///   `src[char_start..char_end] == chunk.text` for every emitted sub-chunk.
///   Overlap means adjacent CHAR spans share bytes (that is the point), but each
///   child's `text` is still one contiguous source slice.
/// * **Token bound** — each child still packs ≤ `limit` (= `CHILD_TOKEN_TARGET +
///   CHILD_TOKEN_TOLERANCE`) tokens: the overlap window is re-split by
///   [`find_split`] under the same `limit`.
/// * **Termination** — the contiguous frontier (`consumed_end`) strictly advances
///   by ≥ 1 byte each iteration regardless of overlap; overlap only moves a
///   window's *start* earlier, never the frontier.  If a candidate overlap start
///   is not strictly before `consumed_end` it is dropped (no-overlap fallback),
///   so a window can never start at/after the previous frontier and stall.
fn split_oversized_child_block(
    ctx: &ChildCtx<'_>,
    win: &ParentWindow,
    block: &Block,
    limit: usize,
    children: &mut Vec<Chunk>,
    running_token_offset: &mut i64,
    child_ordinal: &mut usize,
) -> Result<(), LensError> {
    let src = ctx.src;
    let limit = limit.max(1);
    let block_text = &block.text;
    // `consumed_end` is the *contiguous* frontier in absolute source bytes: the
    // point past which no source byte has yet been covered.  It only ever moves
    // forward, which is what guarantees termination.  `win_start` is where the
    // next window begins; it may be pulled back before `consumed_end` to create
    // overlap, but never before the previous window's start.
    let mut consumed_end = block.char_start; // absolute byte offset in `src`
    let block_end = (block.char_start + block_text.len()).min(src.len());

    while consumed_end < block_end {
        // Choose the window start: overlap back into the previous child's tail
        // when possible, otherwise start at the contiguous frontier.
        let win_start = match overlap_back_off(src, win, children.last(), ctx.tokenizer)? {
            // The overlap start must lie strictly before the frontier so the new
            // window covers fresh bytes (terminating); otherwise fall back to no
            // overlap.  It is already `>= prev.char_start` and `< win.char_end`.
            Some(ov) if ov < consumed_end => ov,
            _ => consumed_end,
        };

        // Split the remaining window text under the child token limit.  The
        // window text begins at `win_start` (the overlap seed) but the contiguous
        // frontier advances by the part of this window that lies at/after
        // `consumed_end`.
        let remaining = &src[win_start..block_end];
        let seg_len = find_split(remaining, ctx.tokenizer, limit)?;
        let mut abs_end = (win_start + seg_len).min(block_end);

        // Guarantee the frontier advances: this window must extend strictly past
        // `consumed_end`.  Because `win_start <= consumed_end` and `find_split`
        // returns ≥ 1, `abs_end` could in a pathological case (a tiny overlap
        // segment whose token budget is exhausted by the overlap tail itself)
        // land at/before `consumed_end`.  Force at least one fresh byte.
        if abs_end <= consumed_end {
            abs_end = (consumed_end + 1).min(block_end);
        }

        let seg_text = &src[win_start..abs_end];
        let seg_tokens = token_count(ctx.tokenizer, seg_text)? as i64;
        let t_start = *running_token_offset;
        let t_end = t_start + seg_tokens;
        let id = ctx
            .strategy
            .make_id(1, &block.section_path, seg_text, *child_ordinal);
        *child_ordinal += 1;
        children.push(Chunk {
            id,
            parent_id: Some(ctx.parent_id.to_string()),
            kind: kind::CHILD.to_string(),
            level: 1,
            section_path: block.section_path.clone(),
            text: seg_text.to_string(),
            token_start: t_start,
            token_end: t_end,
            char_start: win_start as i64,
            char_end: abs_end as i64,
            block_type: Some(block.block_type.clone()),
            source_anchor: None,
        });
        *running_token_offset = t_end;

        // Advance the contiguous frontier (strictly forward — termination).
        consumed_end = abs_end;
    }

    Ok(())
}

/// Returns the byte offset (within `text`) at which to split to stay under
/// `limit` tokens.  Tries to split at a whitespace boundary.
///
/// Uses a **binary search** over whitespace-boundary candidates instead of a
/// linear re-tokenize-every-boundary scan: token count is monotonic in prefix
/// length, so the longest prefix that fits can be found in `O(log n)` encodes.
/// Falls back to a per-character binary search when the text has no usable
/// whitespace boundary.
///
/// Guaranteed to return a value in `1..=text.len()` so the caller never loops
/// infinitely on a non-empty `text`.
fn find_split(text: &str, tokenizer: &Tokenizer, limit: usize) -> Result<usize, LensError> {
    // Fast path: the whole segment fits.
    if token_count(tokenizer, text)? <= limit {
        return Ok(text.len());
    }

    // Collect whitespace-boundary byte offsets (split *before* the whitespace,
    // matching the previous behaviour where `&text[..byte_idx]` was the prefix).
    let ws_boundaries: Vec<usize> = text
        .char_indices()
        .filter(|(idx, ch)| *idx > 0 && ch.is_whitespace())
        .map(|(idx, _)| idx)
        .collect();

    // Binary search for the largest whitespace boundary whose prefix fits.
    // Monotonicity: token_count(text[..a]) <= token_count(text[..b]) for a <= b.
    if !ws_boundaries.is_empty() {
        let mut lo = 0usize;
        let mut hi = ws_boundaries.len(); // exclusive
        let mut best: Option<usize> = None;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let cut = ws_boundaries[mid];
            if token_count(tokenizer, &text[..cut])? <= limit {
                best = Some(cut);
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if let Some(cut) = best {
            return Ok(cut);
        }
    }

    // No whitespace boundary fits — hard split at the longest char-boundary
    // prefix under the limit, found by binary search over char boundaries.
    let char_boundaries: Vec<usize> = text
        .char_indices()
        .map(|(idx, _)| idx)
        .skip(1) // skip 0 (empty prefix)
        .collect();
    let mut lo = 0usize;
    let mut hi = char_boundaries.len();
    let mut safe_end = char_boundaries.first().copied().unwrap_or(text.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let cut = char_boundaries[mid];
        if token_count(tokenizer, &text[..cut])? <= limit {
            safe_end = cut;
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Ok(safe_end.max(1))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{SourceKind, parse_blocks};

    /// Returns a nomic tokenizer loaded from the environment variable
    /// `NOMIC_TOKENIZER_PATH`, or `None` if the variable is not set.
    ///
    /// In CI / local dev, set `NOMIC_TOKENIZER_PATH` to the path of the
    /// `tokenizer.json` downloaded by fastembed into
    /// `{data_dir}/models/fastembed/nomic-ai/…`.  Tests that call this helper
    /// are intentionally skipped when the tokenizer is not present so the
    /// unit-test suite stays runnable offline.
    fn load_tokenizer() -> Option<Tokenizer> {
        let path = std::env::var("NOMIC_TOKENIZER_PATH").ok()?;
        Tokenizer::from_file(&path).ok()
    }

    /// Verify the byte-identity invariant for every chunk returned by the chunker.
    fn assert_byte_identity(src: &str, chunks: &[Chunk]) {
        for (i, c) in chunks.iter().enumerate() {
            let s = c.char_start as usize;
            let e = c.char_end as usize;
            assert!(
                e <= src.len(),
                "chunk[{i}] char_end {e} > src.len() {}",
                src.len()
            );
            assert_eq!(
                &src[s..e],
                c.text,
                "byte-identity invariant violated for chunk[{i}] ({} level={})",
                c.kind,
                c.level
            );
        }
    }

    #[test]
    fn chunk_empty_input() {
        if let Some(tok) = load_tokenizer() {
            let result = chunk_blocks("", &[], &tok).unwrap();
            assert!(result.is_empty());
        }
    }

    #[test]
    fn chunk_hierarchy_invariants() {
        let tok = match load_tokenizer() {
            Some(t) => t,
            None => return, // skip if tokenizer unavailable
        };

        // Build a moderately long markdown document so we get at least one
        // parent and multiple children.
        let mut src = String::new();
        src.push_str("# Introduction\n\n");
        for i in 0..60 {
            src.push_str(&format!(
                "This is sentence number {i} in the introduction. It adds some tokens to help exercise chunking boundaries.\n\n"
            ));
        }

        let blocks = parse_blocks(&src, SourceKind::Markdown);
        let chunks = chunk_blocks(&src, &blocks, &tok).unwrap();

        assert_byte_identity(&src, &chunks);

        // There must be at least one parent and at least one child.
        let parents: Vec<_> = chunks.iter().filter(|c| c.level == 0).collect();
        let children: Vec<_> = chunks.iter().filter(|c| c.level == 1).collect();
        assert!(!parents.is_empty(), "expected at least one parent chunk");
        assert!(!children.is_empty(), "expected at least one child chunk");

        // Every child must have a parent_id that resolves to a real parent id.
        let parent_ids: std::collections::HashSet<&str> =
            parents.iter().map(|c| c.id.as_str()).collect();
        for child in &children {
            let pid = child
                .parent_id
                .as_deref()
                .expect("child must have parent_id");
            assert!(
                parent_ids.contains(pid),
                "child.parent_id {pid} not in parent id set"
            );
        }

        // Parent token spans must be within the allowed bound.
        for p in &parents {
            let span = (p.token_end - p.token_start) as usize;
            assert!(
                span <= PARENT_TOKEN_TARGET + PARENT_TOKEN_TOLERANCE,
                "parent token span {span} exceeds limit"
            );
        }

        // Child token spans must be within the allowed bound.
        for c in &children {
            let span = (c.token_end - c.token_start) as usize;
            assert!(
                span <= CHILD_TOKEN_TARGET + CHILD_TOKEN_TOLERANCE,
                "child token span {span} exceeds limit"
            );
        }
    }

    #[test]
    fn child_overlap_shares_boundary_bytes() {
        let tok = match load_tokenizer() {
            Some(t) => t,
            None => return,
        };

        // One long paragraph (single block) under the parent limit but well over
        // the child limit, forcing multiple overlapping children within one
        // parent. A single block means children are packed by overlap, not by
        // block boundaries — exactly the seam the overlap is meant to bridge.
        let mut src = String::from("# Doc\n\n");
        for i in 0..220 {
            src.push_str(&format!("token{i} "));
        }

        let blocks = parse_blocks(&src, SourceKind::Markdown);
        let chunks = chunk_blocks(&src, &blocks, &tok).unwrap();
        assert_byte_identity(&src, &chunks);

        let children: Vec<_> = chunks.iter().filter(|c| c.level == 1).collect();
        assert!(
            children.len() >= 2,
            "expected multiple children to exercise overlap (got {})",
            children.len()
        );

        // At least one adjacent child pair must share source bytes (overlap):
        // child[n+1].char_start < child[n].char_end. This single-block fixture
        // is hard-split, so this asserts the hard-split path applies overlap.
        let mut saw_overlap = false;
        for w in children.windows(2) {
            let prev = w[0];
            let next = w[1];
            let prev_start = prev.char_start as usize;
            let prev_end = prev.char_end as usize;
            let next_start = next.char_start as usize;

            // Windows must always make forward progress (termination): the next
            // child starts after the previous one and ends past it.
            assert!(
                next_start > prev_start,
                "child must advance: next.char_start {next_start} <= prev.char_start {prev_start}"
            );
            assert!(
                next.char_end as usize > prev_end,
                "child end must advance: {} <= {prev_end}",
                next.char_end as usize
            );

            if next_start < prev_end {
                saw_overlap = true;
                // The shared region must be BYTE-IDENTICAL on both sides: the
                // bytes `src[next.char_start..prev.char_end]` are exactly the
                // matching tail of the previous child's text (and the matching
                // head of the next child's text). This is the whole point of
                // overlap — the boundary concept appears intact in both children.
                let shared = &src[next_start..prev_end];
                let tail_off = next_start - prev_start; // offset into prev.text
                assert_eq!(
                    shared,
                    &prev.text[tail_off..],
                    "shared bytes must equal the previous child's tail"
                );
                let head_len = prev_end - next_start; // length of overlap in next
                assert_eq!(
                    shared,
                    &next.text[..head_len],
                    "shared bytes must equal the next child's head"
                );
            }
        }
        assert!(
            saw_overlap,
            "expected at least one overlapping adjacent child pair with shared boundary bytes"
        );

        // Every child still respects the token bound.
        for c in &children {
            let span = (c.token_end - c.token_start) as usize;
            assert!(
                span <= CHILD_TOKEN_TARGET + CHILD_TOKEN_TOLERANCE,
                "child token span {span} exceeds limit"
            );
        }
    }

    #[test]
    fn section_path_inherited() {
        let tok = match load_tokenizer() {
            Some(t) => t,
            None => return,
        };
        // Build enough body under `## Sub` that the content there spills into a
        // child whose first block is the `Top > Sub` paragraph (a tiny doc would
        // pack every block into one child keyed by the leading `# Top` heading,
        // and section_path is taken from a chunk's FIRST block — so the trail
        // would never surface). Several paragraphs guarantee a `Top > Sub` child.
        let mut src = String::from("# Top\n\n## Sub\n\n");
        for i in 0..40 {
            src.push_str(&format!(
                "Paragraph {i} under sub adds tokens to push the child window past the leading heading block.\n\n"
            ));
        }
        let blocks = parse_blocks(&src, SourceKind::Markdown);
        let chunks = chunk_blocks(&src, &blocks, &tok).unwrap();
        assert_byte_identity(&src, &chunks);
        let has_sub = chunks.iter().any(|c| c.section_path.contains("Sub"));
        assert!(
            has_sub,
            "expected a chunk with section_path containing 'Sub'"
        );
    }

    #[test]
    fn deterministic_ids_stable() {
        let tok = match load_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let src = "# A\n\nHello world.\n\n## B\n\nAnother paragraph.\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        let run1 = chunk_blocks_deterministic(src, &blocks, &tok).unwrap();
        let run2 = chunk_blocks_deterministic(src, &blocks, &tok).unwrap();
        let ids1: Vec<_> = run1.iter().map(|c| c.id.as_str()).collect();
        let ids2: Vec<_> = run2.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids1, ids2, "deterministic ids must be stable across runs");
    }

    #[test]
    fn v7_ids_are_uuid_format() {
        let tok = match load_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let src = "# A\n\nSome text here.\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        let chunks = chunk_blocks(src, &blocks, &tok).unwrap();
        for chunk in &chunks {
            // UUIDv7 strings are 36 chars with hyphens in the standard layout.
            assert_eq!(chunk.id.len(), 36, "UUID string should be 36 chars");
            assert!(
                Uuid::parse_str(&chunk.id).is_ok(),
                "chunk id {} should be a valid UUID",
                chunk.id
            );
        }
    }

    #[test]
    fn oversized_single_block_splits_into_bounded_parents() {
        let tok = match load_tokenizer() {
            Some(t) => t,
            None => return, // skip if tokenizer unavailable
        };

        // A single paragraph (one block) far larger than PARENT_TOKEN_TARGET.
        // "word " repeated yields well over 512 tokens with no blank line, so
        // the parser emits exactly one block that must be split at the parent
        // level into multiple bounded parents.
        let mut src = String::new();
        for i in 0..1200 {
            src.push_str(&format!("word{i} "));
        }

        let blocks = parse_blocks(&src, SourceKind::Text);
        assert_eq!(
            blocks.len(),
            1,
            "fixture must be a single oversized block (got {})",
            blocks.len()
        );

        let chunks = chunk_blocks(&src, &blocks, &tok).unwrap();
        assert_byte_identity(&src, &chunks);

        let parents: Vec<_> = chunks.iter().filter(|c| c.level == 0).collect();
        assert!(
            parents.len() > 1,
            "an oversized single block must split into multiple parents (got {})",
            parents.len()
        );

        // Every parent must respect the documented bound.
        for p in &parents {
            let span = (p.token_end - p.token_start) as usize;
            assert!(
                span <= PARENT_TOKEN_TARGET + PARENT_TOKEN_TOLERANCE,
                "parent token span {span} exceeds limit {}",
                PARENT_TOKEN_TARGET + PARENT_TOKEN_TOLERANCE
            );
        }

        // And every child still respects its bound, with a resolvable parent.
        let parent_ids: std::collections::HashSet<&str> =
            parents.iter().map(|c| c.id.as_str()).collect();
        for c in chunks.iter().filter(|c| c.level == 1) {
            let span = (c.token_end - c.token_start) as usize;
            assert!(
                span <= CHILD_TOKEN_TARGET + CHILD_TOKEN_TOLERANCE,
                "child token span {span} exceeds limit"
            );
            let pid = c.parent_id.as_deref().expect("child must have parent_id");
            assert!(
                parent_ids.contains(pid),
                "child parent_id not in parent set"
            );
        }
    }

    /// B4 — children must TILE their parent's token range: the last child's
    /// `token_end` must equal the parent's `token_end`. Before the fix, child
    /// overlap inflated cumulative offsets so the last child overshot the parent
    /// end (parent 0..480 but children reaching ~544).
    #[test]
    fn children_tile_parent_token_range() {
        let tok = match load_tokenizer() {
            Some(t) => t,
            None => return,
        };

        // One long single-block paragraph that fits in a parent but spans several
        // overlapping children (the case where overlap inflated offsets).
        let mut src = String::from("# Doc\n\n");
        for i in 0..220 {
            src.push_str(&format!("token{i} "));
        }
        let blocks = parse_blocks(&src, SourceKind::Markdown);
        let chunks = chunk_blocks(&src, &blocks, &tok).unwrap();
        assert_byte_identity(&src, &chunks);

        // Group children under each parent and assert the last child of each
        // parent ends exactly at the parent's token_end (perfect tiling), and the
        // first child starts at the parent's token_start.
        let parents: Vec<&Chunk> = chunks.iter().filter(|c| c.level == 0).collect();
        assert!(!parents.is_empty(), "expected at least one parent");

        for p in &parents {
            let kids: Vec<&Chunk> = chunks
                .iter()
                .filter(|c| c.level == 1 && c.parent_id.as_deref() == Some(p.id.as_str()))
                .collect();
            assert!(!kids.is_empty(), "parent {} should have children", p.id);
            assert_eq!(
                kids.first().unwrap().token_start,
                p.token_start,
                "first child token_start must equal parent token_start"
            );
            assert_eq!(
                kids.last().unwrap().token_end,
                p.token_end,
                "last child token_end ({}) must equal parent token_end ({}) — children must tile the parent token range",
                kids.last().unwrap().token_end,
                p.token_end
            );
        }
    }

    #[test]
    fn plain_text_byte_identity() {
        let tok = match load_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let src = "First paragraph with some content.\n\nSecond paragraph also has content here.\n";
        let blocks = parse_blocks(src, SourceKind::Text);
        let chunks = chunk_blocks(src, &blocks, &tok).unwrap();
        assert_byte_identity(src, &chunks);
    }
}
