//! Parent-child chunker: converts a flat [`Block`] list into a two-level
//! adjacency-list hierarchy of [`Chunk`] rows ready for insertion into the
//! `chunks` SQLite table.
//!
//! ## Hierarchy
//!
//! * **Parents** (`level = 0`, `kind = "parent"`, `parent_id = None`): pack
//!   adjacent blocks into windows of approximately **512 tokens** each.
//! * **Children** (`level = 1`, `kind = "child"`, `parent_id = Some(parent_id)`):
//!   split each parent text into windows of approximately **128 tokens** each.
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
//! than inserting synthetic `\n` separators.
//!
//! ## Token counting
//!
//! Token counts are performed by the nomic `tokenizer.json` passed as a
//! `tokenizers::Tokenizer`.  This is the **same tokenizer** that
//! `fastembed`/`nomic-embed-text-v1.5` uses internally, so the 512/128 windows
//! are measured with the same vocabulary as the eventual embedding step.
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
}

/// Soft upper bound for child-chunk token span (target = 128, tolerance = 1).
pub const CHILD_TOKEN_TARGET: usize = 128;
/// Soft upper bound for parent-chunk token span (target = 512, tolerance = 1).
pub const PARENT_TOKEN_TARGET: usize = 512;
/// Allowed overshoot above the child target due to boundary tokenization.
pub const CHILD_TOKEN_TOLERANCE: usize = 1;
/// Allowed overshoot above the parent target due to boundary tokenization.
pub const PARENT_TOKEN_TOLERANCE: usize = 1;

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
    pub token_start: i64,
    /// Cumulative token index one-past the last token in this chunk.
    pub token_end: i64,
    /// Byte offset of the first byte of this chunk in the source string.
    pub char_start: i64,
    /// Byte offset one-past the last byte of this chunk in the source string.
    pub char_end: i64,
    /// Block type of the first (or dominant) block composing this chunk.
    /// Maps to the `chunks.block_type` column.  Parent chunks carry the block
    /// type of their first constituent block; child chunks inherit from the
    /// parent block.  `None` is valid (the column is nullable in SQLite).
    pub block_type: Option<String>,
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
        let children = build_child_chunks(
            src,
            win,
            tokenizer,
            &parent_id,
            parent_token_start,
            &strategy,
            &mut child_ordinal,
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
        });

        // Emit the children immediately after their parent.
        result.extend(children);

        running_token_offset = parent_token_end;
    }

    Ok(result)
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
    /// Block type of the first block in this window.
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

    for block in blocks {
        // Compute tentative text if we add this block to the current window.
        let tentative_start = if current_blocks.is_empty() {
            block.char_start
        } else {
            current_char_start
        };
        let tentative_end = block.char_end;
        let tentative_text = &src[tentative_start..tentative_end.min(src.len())];
        let tentative_tokens = token_count(tokenizer, tentative_text)?;

        // If this single block already exceeds the limit on its own, split it
        // (after flushing any in-progress window) into multiple BOUNDED parent
        // windows so no parent ever exceeds PARENT_TOKEN_TARGET + tolerance. A
        // single oversized block previously became one unbounded parent, which
        // violated the AC "parent ≤ 512 + tolerance".
        if current_blocks.is_empty() && token_count(tokenizer, &block.text)? > limit {
            split_oversized_parent_block(src, block, tokenizer, limit, &mut windows)?;
            continue;
        }

        // If adding this block would overflow the current window, flush it.
        if !current_blocks.is_empty() && tentative_tokens > limit {
            flush_parent_window(
                src,
                &current_blocks,
                current_char_start,
                current_char_end,
                &mut windows,
            );
            current_blocks.clear();
        }

        if current_blocks.is_empty() {
            current_char_start = block.char_start;
        }
        current_char_end = block.char_end;
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
/// bounded [`ParentWindow`]s, mirroring the child-level
/// [`split_oversized_block`] logic (whitespace-boundary scan via
/// [`find_child_split`]) but at the parent limit.
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
    let limit = limit.max(1);
    let text = &block.text;
    let mut seg_start_byte: usize = 0; // byte offset within `block.text`

    while seg_start_byte < text.len() {
        let remaining = &text[seg_start_byte..];
        let seg_len = find_child_split(remaining, tokenizer, limit)?;
        let mut seg_end_byte = (seg_start_byte + seg_len).min(text.len());
        // Ensure progress — never produce a zero-length segment.
        if seg_end_byte <= seg_start_byte {
            seg_end_byte = (seg_start_byte + 1).min(text.len());
        }

        // Absolute byte offsets in the source string (block.text is a verbatim
        // slice of src anchored at block.char_start). Slice `src` directly so the
        // window text is byte-identical to the source span.
        let abs_start = block.char_start + seg_start_byte;
        let abs_end = (block.char_start + seg_end_byte).min(src.len());
        let seg_text = src[abs_start..abs_end].to_string();

        let sub_block = Block {
            block_type: block.block_type.clone(),
            section_path: block.section_path.clone(),
            text: seg_text.clone(),
            char_start: abs_start,
            char_end: abs_end,
        };
        windows.push(ParentWindow {
            text: seg_text,
            section_path: block.section_path.clone(),
            block_type: Some(block.block_type.clone()),
            char_start: abs_start,
            char_end: abs_end,
            blocks: vec![sub_block],
        });

        seg_start_byte = seg_end_byte;
    }

    Ok(())
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

/// Splits a parent window into child chunks of ≤ `CHILD_TOKEN_TARGET +
/// CHILD_TOKEN_TOLERANCE` tokens.
///
/// Splitting follows block boundaries first: blocks are packed into child
/// windows exactly as parent windows are packed from blocks.  If a single
/// block exceeds the child limit it is split at whitespace boundaries (best
/// effort — guaranteed to terminate).
///
/// Each child's text is `src[child.char_start..child.char_end]`, preserving
/// the byte-identity invariant.
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
    let mut children: Vec<Chunk> = Vec::new();
    let mut running_token_offset = parent_token_start;

    let mut child_blocks: Vec<&Block> = Vec::new();
    let mut child_char_start: usize = 0;
    let mut child_char_end: usize = 0;

    // Helper: emit a child chunk from the current child_blocks accumulation.
    // Evaluates to the updated token offset (unchanged if nothing was flushed).
    macro_rules! flush_child {
        () => {{
            if !child_blocks.is_empty() {
                let end = child_char_end.min(src.len());
                let text = src[child_char_start..end].to_string();
                let toks = token_count(tokenizer, &text)? as i64;
                let section_path = child_blocks
                    .first()
                    .map(|b| b.section_path.as_str())
                    .unwrap_or("")
                    .to_string();
                let block_type_val = child_blocks.first().map(|b| b.block_type.clone());
                let t_start = running_token_offset;
                let t_end = t_start + toks;
                let id = strategy.make_id(1, &section_path, &text, *child_ordinal);
                *child_ordinal += 1;
                children.push(Chunk {
                    id,
                    parent_id: Some(parent_id.to_string()),
                    kind: kind::CHILD.to_string(),
                    level: 1,
                    section_path,
                    text,
                    token_start: t_start,
                    token_end: t_end,
                    char_start: child_char_start as i64,
                    char_end: end as i64,
                    block_type: block_type_val,
                });
                child_blocks.clear();
                t_end
            } else {
                running_token_offset
            }
        }};
    }

    for block in &win.blocks {
        let block_alone_tokens = token_count(tokenizer, &block.text)?;

        // Oversized single block: flush the current child window first, then
        // split the block at whitespace boundaries.
        if block_alone_tokens > limit {
            running_token_offset = flush_child!();
            let sub = split_oversized_block(
                SplitCtx {
                    block,
                    tokenizer,
                    parent_id,
                    strategy,
                    limit,
                },
                &mut running_token_offset,
                child_ordinal,
            )?;
            children.extend(sub);
            continue;
        }

        // Compute tentative span if we add this block to the current child window.
        let tentative_start = if child_blocks.is_empty() {
            block.char_start
        } else {
            child_char_start
        };
        let tentative_text = &src[tentative_start..block.char_end.min(src.len())];
        let tentative_tokens = token_count(tokenizer, tentative_text)?;

        if !child_blocks.is_empty() && tentative_tokens > limit {
            running_token_offset = flush_child!();
        }

        if child_blocks.is_empty() {
            child_char_start = block.char_start;
        }
        child_char_end = block.char_end;
        child_blocks.push(block);
    }

    // Update running_token_offset with the final child's token end so the
    // accumulated offset is consistent; the caller doesn't use this return
    // value, but the variable must be live here to satisfy `unused_assignments`.
    let _final_child_token_end = flush_child!();

    Ok(children)
}

/// Context passed to [`split_oversized_block`], grouping the read-only
/// parameters to stay within the 7-argument clippy limit.
struct SplitCtx<'a> {
    block: &'a Block,
    tokenizer: &'a Tokenizer,
    parent_id: &'a str,
    strategy: &'a IdStrategy,
    limit: usize,
}

/// Splits a single block whose text exceeds the child token limit into multiple
/// child chunks by scanning for whitespace boundaries.
///
/// Byte-identity is preserved: absolute `char_start/char_end` are anchored to
/// `block.char_start + seg_start_byte` so `src[char_start..char_end] == chunk.text`
/// holds for every emitted sub-chunk (since `block.text` is a verbatim slice of `src`).
fn split_oversized_block(
    ctx: SplitCtx<'_>,
    running_token_offset: &mut i64,
    child_ordinal: &mut usize,
) -> Result<Vec<Chunk>, LensError> {
    let SplitCtx {
        block,
        tokenizer,
        parent_id,
        strategy,
        limit,
    } = ctx;
    let limit = limit.max(1);
    let text = &block.text;
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut seg_start_byte: usize = 0; // byte offset within `text`

    while seg_start_byte < text.len() {
        let remaining = &text[seg_start_byte..];
        let seg_len = find_child_split(remaining, tokenizer, limit)?;
        let seg_end_byte = (seg_start_byte + seg_len).min(text.len());
        // Ensure progress — never produce a zero-length segment.
        let seg_end_byte = if seg_end_byte <= seg_start_byte {
            (seg_start_byte + 1).min(text.len())
        } else {
            seg_end_byte
        };

        let seg_text = &text[seg_start_byte..seg_end_byte];
        let seg_tokens = token_count(tokenizer, seg_text)? as i64;
        let t_start = *running_token_offset;
        let t_end = t_start + seg_tokens;

        // Absolute byte offsets in the source string.
        let abs_start = block.char_start + seg_start_byte;
        let abs_end = block.char_start + seg_end_byte;

        let id = strategy.make_id(1, &block.section_path, seg_text, *child_ordinal);
        *child_ordinal += 1;

        chunks.push(Chunk {
            id,
            parent_id: Some(parent_id.to_string()),
            kind: kind::CHILD.to_string(),
            level: 1,
            section_path: block.section_path.clone(),
            text: seg_text.to_string(),
            token_start: t_start,
            token_end: t_end,
            char_start: abs_start as i64,
            char_end: abs_end as i64,
            block_type: Some(block.block_type.clone()),
        });

        *running_token_offset = t_end;
        seg_start_byte = seg_end_byte;
    }

    Ok(chunks)
}

/// Returns the byte offset (within `text`) at which to split to stay under
/// `limit` tokens.  Tries to split at a whitespace boundary.
///
/// Guaranteed to return a value in `1..=text.len()` so the caller never loops
/// infinitely on a non-empty `text`.
fn find_child_split(text: &str, tokenizer: &Tokenizer, limit: usize) -> Result<usize, LensError> {
    // Fast path: the whole segment fits.
    if token_count(tokenizer, text)? <= limit {
        return Ok(text.len());
    }

    // Walk forward finding the last whitespace boundary before the limit.
    let mut last_ws_byte: usize = 0;
    let mut prev_byte: usize = 0;

    for (byte_idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            let candidate = &text[..byte_idx];
            let toks = token_count(tokenizer, candidate)?;
            if toks <= limit {
                last_ws_byte = byte_idx;
            } else {
                // Past the limit — split at last good whitespace boundary.
                if last_ws_byte == 0 {
                    // No whitespace found before limit; split here.
                    last_ws_byte = prev_byte.max(1);
                }
                return Ok(last_ws_byte);
            }
        }
        prev_byte = byte_idx + ch.len_utf8();
    }

    // Scanned full string without hitting the limit (shouldn't happen after the
    // fast path, but be safe).
    if last_ws_byte > 0 {
        return Ok(text.len());
    }

    // No whitespace at all — hard split: find the last character boundary that
    // keeps us under the limit.
    let mut safe_end: usize = 1;
    for (byte_idx, _ch) in text.char_indices() {
        if byte_idx == 0 {
            continue;
        }
        if token_count(tokenizer, &text[..byte_idx])? <= limit {
            safe_end = byte_idx;
        } else {
            break;
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
    fn section_path_inherited() {
        let tok = match load_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let src = "# Top\n\n## Sub\n\nParagraph under sub.\n";
        let blocks = parse_blocks(src, SourceKind::Markdown);
        let chunks = chunk_blocks(src, &blocks, &tok).unwrap();
        assert_byte_identity(src, &chunks);
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
