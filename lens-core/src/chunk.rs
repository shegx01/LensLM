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
pub(crate) mod kind {
    pub const PARENT: &str = "parent";
    pub const CHILD: &str = "child";
    /// Doc-summary RAPTOR node emitted by the enrichment pass (M4 Phase-3).
    pub const SUMMARY: &str = "summary";
}

pub const CHILD_TOKEN_TARGET: usize = 128;
pub const PARENT_TOKEN_TARGET: usize = 512;
/// Allowed overshoot above the child/parent targets due to boundary tokenization.
pub const CHILD_TOKEN_TOLERANCE: usize = 1;
pub const PARENT_TOKEN_TOLERANCE: usize = 1;

/// Tokens each child re-shares with the tail of the previous child.
///
/// Overlap ensures a concept straddling a child boundary appears intact in at
/// least one child. 16 ≈ 12% of the 128-token target: enough to capture a
/// short boundary phrase, cheap enough to keep embedding cost low. Parents are
/// non-overlapping; byte-identity is preserved for each child's contiguous span.
pub const CHILD_OVERLAP_TOKENS: usize = 16;

/// A chunk row ready for insertion into the `chunks` SQLite table.
///
/// Caller fills in `source_id`, `page`, `enrichment`, and `created_at` before
/// inserting; all other fields come from the chunker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// UUIDv7 primary key (production) or content-derived hex id (eval fixtures).
    pub id: String,
    pub parent_id: Option<String>,
    pub kind: String,
    pub level: i32,
    pub section_path: String,
    /// Verbatim source bytes; invariant: `src[char_start..char_end] == text`.
    pub text: String,
    /// Cumulative token index of the first token in this chunk (reading order).
    ///
    /// Children **tile** the parent's token range via non-overlapping tiles (see
    /// `retile_child_token_offsets`) even though adjacent child byte spans overlap.
    pub token_start: i64,
    pub token_end: i64,
    pub char_start: i64,
    pub char_end: i64,
    /// Block type of the first (dominant) block; lossy for multi-block parents.
    pub block_type: Option<String>,
    /// Pre-serialized JSON `SourceAnchor`; `None` when no anchor was supplied.
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

    let parent_windows = build_parent_windows(src, blocks, tokenizer)?;

    let mut result: Vec<Chunk> = Vec::new();
    let mut running_token_offset: i64 = 0;
    let mut child_ordinal: usize = 0;

    for (parent_ordinal, win) in parent_windows.iter().enumerate() {
        let parent_token_count = token_count(tokenizer, &win.text)? as i64;
        let parent_id = strategy.make_id(0, &win.section_path, &win.text, parent_ordinal);

        let parent_token_start = running_token_offset;
        let parent_token_end = running_token_offset + parent_token_count;

        let mut children = build_child_chunks(
            src,
            win,
            tokenizer,
            &parent_id,
            parent_token_start,
            &strategy,
            &mut child_ordinal,
        )?;

        retile_child_token_offsets(
            src,
            win,
            tokenizer,
            parent_token_start,
            parent_token_end,
            &mut children,
        )?;

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
            // Anchor attached post-chunking by the ingest pipeline; chunk_blocks
            // itself stays format-agnostic.
            source_anchor: None,
        });

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
    text: String,
    section_path: String,
    block_type: Option<String>,
    char_start: usize,
    char_end: usize,
    blocks: Vec<Block>,
}

/// Packs `blocks` into [`ParentWindow`]s of at most
/// `PARENT_TOKEN_TARGET + PARENT_TOKEN_TOLERANCE` tokens using an incremental
/// per-block token sum (see module-level "Token counting" note).
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
    let mut current_tokens: usize = 0;

    for block in blocks {
        let block_tokens = token_count(tokenizer, &block.text)?;

        // A single block that exceeds the limit must be split into bounded parent
        // windows so no parent ever exceeds PARENT_TOKEN_TARGET + tolerance.
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

/// Mutable state for the open child window (mirrors `flush_parent_window` locals).
struct ChildAccumulator<'a> {
    blocks: Vec<&'a Block>,
    char_start: usize,
    char_end: usize,
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

/// Read-only context shared across all children of one parent.
struct ChildCtx<'a> {
    src: &'a str,
    tokenizer: &'a Tokenizer,
    parent_id: &'a str,
    strategy: &'a IdStrategy,
}

/// Splits a parent window into child chunks of ≤ `CHILD_TOKEN_TARGET +
/// CHILD_TOKEN_TOLERANCE` tokens with `CHILD_OVERLAP_TOKENS`-token overlap.
/// Blocks are packed at block boundaries; single oversized blocks are hard-split.
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
    // Start offset for the next child's overlap seed; None = no pending overlap.
    let mut pending_overlap_start: Option<usize> = None;

    for block in &win.blocks {
        let block_tokens = token_count(tokenizer, &block.text)?;

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

        // Opening a fresh window: fold in any pending overlap seed when it fits.
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

        acc.char_end = block.char_end;
        acc.tokens += block_tokens;
        acc.blocks.push(block);
    }

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
    // Empty prefix returns 0 (not 2 CLS/SEP tokens from `encode("", true)`).
    let prefix_tokens = |boundary: usize| -> Result<i64, LensError> {
        let end = boundary.min(src.len());
        if end <= parent_char_start {
            return Ok(0);
        }
        Ok(token_count(tokenizer, &src[parent_char_start..end])? as i64)
    };

    let mut tile_start = parent_char_start;
    for child in head.iter_mut() {
        let child_end = (child.char_end as usize).min(src.len());
        child.token_start = parent_token_start + prefix_tokens(tile_start)?;
        child.token_end = parent_token_start + prefix_tokens(child_end)?;
        tile_start = child_end.max(tile_start);
    }

    // Last child's token_end snaps to parent_token_end — no overshoot or gap.
    last.token_start = parent_token_start + prefix_tokens(tile_start)?;
    last.token_end = parent_token_end;

    Ok(())
}

/// Finalises the open child window in `acc`, appends to `children`, clears
/// `acc`, and returns the updated running token offset.
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

/// Returns the byte offset at which the next child should start to re-share
/// ~`CHILD_OVERLAP_TOKENS` tokens with `last_child`'s tail, or `None` if no
/// meaningful overlap is possible (no prior child, or prior child too short).
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

    // Find the earliest whitespace boundary whose tail `src[b..prev_end]` is
    // ≤ CHILD_OVERLAP_TOKENS tokens (maximises overlap within budget).
    let mut candidates: Vec<usize> = Vec::new();
    for (idx, ch) in src[prev_start..prev_end].char_indices() {
        if ch.is_whitespace() {
            let abs = prev_start + idx + ch.len_utf8();
            if abs > prev_start && abs < prev_end {
                candidates.push(abs);
            }
        }
    }
    let mut chosen: Option<usize> = None;
    for &b in &candidates {
        let tail_tokens = token_count(tokenizer, &src[b..prev_end])?;
        if tail_tokens <= CHILD_OVERLAP_TOKENS {
            chosen = Some(b);
            break;
        }
    }

    // Overlap start must be strictly within the parent span to avoid stalling.
    match chosen {
        Some(b) if b < win.char_end => Ok(Some(b)),
        _ => Ok(None),
    }
}

/// Token cost of `src[overlap_start..block_start]`; `0` when seed doesn't precede block.
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
    // `consumed_end`: the contiguous frontier in absolute source bytes — only
    // moves forward, guaranteeing termination regardless of overlap.
    let mut consumed_end = block.char_start;
    let block_end = (block.char_start + block_text.len()).min(src.len());

    while consumed_end < block_end {
        // The overlap start must be strictly before the frontier to cover fresh bytes.
        let win_start = match overlap_back_off(src, win, children.last(), ctx.tokenizer)? {
            Some(ov) if ov < consumed_end => ov,
            _ => consumed_end,
        };

        let remaining = &src[win_start..block_end];
        let seg_len = find_split(remaining, ctx.tokenizer, limit)?;
        let mut abs_end = (win_start + seg_len).min(block_end);

        // Force the frontier to advance by at least one byte (termination guard).
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
        consumed_end = abs_end;
    }

    Ok(())
}

/// Returns the byte offset (within `text`) at which to split to stay under
/// `limit` tokens, using binary search over whitespace boundaries (O(log n)).
/// Falls back to a per-character binary search when no whitespace boundary fits.
/// Guaranteed to return a value in `1..=text.len()`.
fn find_split(text: &str, tokenizer: &Tokenizer, limit: usize) -> Result<usize, LensError> {
    if token_count(tokenizer, text)? <= limit {
        return Ok(text.len());
    }

    let ws_boundaries: Vec<usize> = text
        .char_indices()
        .filter(|(idx, ch)| *idx > 0 && ch.is_whitespace())
        .map(|(idx, _)| idx)
        .collect();

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

    // No whitespace boundary fits — hard split at char boundaries.
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

    /// Loads the nomic tokenizer from `NOMIC_TOKENIZER_PATH`; returns `None` when
    /// the variable is unset so the suite stays runnable offline.
    fn load_tokenizer() -> Option<Tokenizer> {
        let path = std::env::var("NOMIC_TOKENIZER_PATH").ok()?;
        Tokenizer::from_file(&path).ok()
    }

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

        let parents: Vec<_> = chunks.iter().filter(|c| c.level == 0).collect();
        let children: Vec<_> = chunks.iter().filter(|c| c.level == 1).collect();
        assert!(!parents.is_empty(), "expected at least one parent chunk");
        assert!(!children.is_empty(), "expected at least one child chunk");

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

        for p in &parents {
            let span = (p.token_end - p.token_start) as usize;
            assert!(
                span <= PARENT_TOKEN_TARGET + PARENT_TOKEN_TOLERANCE,
                "parent token span {span} exceeds limit"
            );
        }

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

        for p in &parents {
            let span = (p.token_end - p.token_start) as usize;
            assert!(
                span <= PARENT_TOKEN_TARGET + PARENT_TOKEN_TOLERANCE,
                "parent token span {span} exceeds limit {}",
                PARENT_TOKEN_TARGET + PARENT_TOKEN_TOLERANCE
            );
        }

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

        let mut src = String::from("# Doc\n\n");
        for i in 0..220 {
            src.push_str(&format!("token{i} "));
        }
        let blocks = parse_blocks(&src, SourceKind::Markdown);
        let chunks = chunk_blocks(&src, &blocks, &tok).unwrap();
        assert_byte_identity(&src, &chunks);

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
