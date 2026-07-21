//! Citation read-back (issue #237): resolves a citation's persisted byte offsets
//! against the retained source buffer into display-ready segments — never raw
//! offsets across IPC (see [`SnippetSegments`]).
//!
//! Retention: the canonical buffer lives until purge, never removed after
//! ingest/enrichment (see `lib.rs::remove_managed_source_file`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use crate::LensError;
use crate::parse::SourceKind;

/// Context bytes expanded on each side of a cited span before boundary-snapping.
const WINDOW: usize = 240;
/// Byte ceiling on a snippet's total length; caps the outward sentence snap so a
/// span with no nearby boundary cannot drag in an unbounded slice.
const MAX_SNIPPET: usize = 1200;
/// Byte ceiling on a [`SourceView`] payload; larger sources window the returned
/// text around the span (or the head) and set `truncated` to bound the webview.
const VIEW_CAP: usize = 256 * 1024;
/// Byte ceiling on the `marked` slice itself; a stale/oversized cited span must
/// not copy the whole buffer regardless of the surrounding context caps.
const MAX_MARKED: usize = 4 * 1024;

/// A cited span split into display segments (three `textContent` nodes on the
/// frontend). `truncated_*` drive the ellipsis affordances at the buffer edges.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnippetSegments {
    pub before: String,
    pub marked: String,
    pub after: String,
    pub truncated_before: bool,
    pub truncated_after: bool,
}

/// Full-source view for the "view in source" affordance. `before` holds the
/// pre-span text (or the WHOLE text when there is no span), `marked` the cited
/// span (empty with no span), `after` the remainder. `truncated` is set when a
/// large source was windowed to bound the payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceView {
    pub before: String,
    pub marked: String,
    pub after: String,
    pub title: String,
    pub kind: String,
    pub truncated: bool,
}

/// Loads the retained canonical buffer for `source_id`, plus its title and kind.
///
/// Errors stay path-free across IPC: absent id → `Validation`, unreadable file →
/// `Io`, non-UTF-8 → `Internal`. Read runs under `spawn_blocking`.
async fn load_source_buffer(
    pool: &SqlitePool,
    data_dir: &Path,
    source_id: &str,
) -> Result<(String, String, SourceKind), LensError> {
    let row = sqlx::query_as::<_, (String, String, String)>(
        "SELECT kind, title, locator FROM sources WHERE id = ?",
    )
    .bind(source_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| LensError::Validation("source not found".into()))?;
    let (kind_str, title, locator) = row;
    let kind: SourceKind = kind_str.parse()?;

    let path = if kind.is_text_like() {
        PathBuf::from(locator)
    } else {
        crate::ingest::extracted_sibling_path(data_dir, source_id)
    };

    let buffer = tokio::task::spawn_blocking(move || read_buffer_blocking(&path)).await??;
    Ok((buffer, title, kind))
}

/// Blocking read + UTF-8 validation of the canonical buffer. Never leaks the path:
/// the detail is logged for operators, the IPC error stays generic.
fn read_buffer_blocking(path: &Path) -> Result<String, LensError> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            tracing::error!(error = %e, "citation source buffer read failed");
            return Err(LensError::Io("citation source is unavailable".into()));
        }
    };
    String::from_utf8(bytes)
        .map_err(|_| LensError::Internal("citation source is not valid text".into()))
}

/// Largest char boundary `<= i`.
fn floor_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Smallest char boundary `>= i`.
fn ceil_boundary(s: &str, mut i: usize) -> usize {
    let n = s.len();
    if i >= n {
        return n;
    }
    while i < n && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Whether index `i` begins a sentence/paragraph (buffer start, after `\n\n`, or
/// after a sentence terminator + space).
fn is_clean_start(s: &str, i: usize) -> bool {
    if i == 0 {
        return true;
    }
    let b = s.as_bytes();
    if i >= 2 && b[i - 1] == b'\n' && b[i - 2] == b'\n' {
        return true;
    }
    i >= 2 && b[i - 1] == b' ' && matches!(b[i - 2], b'.' | b'?' | b'!')
}

/// Whether index `i` ends a sentence/paragraph (buffer end, after a terminator
/// with a following space/newline, or at the first newline of a `\n\n` break).
fn is_clean_end(s: &str, i: usize) -> bool {
    if i == s.len() {
        return true;
    }
    let b = s.as_bytes();
    if i >= 1 && matches!(b[i - 1], b'.' | b'?' | b'!') && matches!(b[i], b' ' | b'\n') {
        return true;
    }
    b[i] == b'\n' && i + 1 < b.len() && b[i + 1] == b'\n'
}

/// Nearest clean start at or left of `from`, searching outward down to `floor`.
/// Falls back to `from` (char-snapped) when no boundary is within the budget.
fn snap_start_outward(s: &str, from: usize, floor: usize) -> usize {
    let mut i = from;
    loop {
        if s.is_char_boundary(i) && is_clean_start(s, i) {
            return i;
        }
        if i <= floor {
            return floor_boundary(s, from);
        }
        i -= 1;
    }
}

/// Nearest clean end at or right of `from`, searching outward up to `ceil`.
/// Falls back to `from` (char-snapped) when no boundary is within the budget.
fn snap_end_outward(s: &str, from: usize, ceil: usize) -> usize {
    let mut i = from;
    loop {
        if s.is_char_boundary(i) && is_clean_end(s, i) {
            return i;
        }
        if i >= ceil {
            return ceil_boundary(s, from);
        }
        i += 1;
    }
}

/// Clamps `[char_start, char_end]` to `0..=buffer.len()` with `start <= end`, then
/// snaps both cut points to char boundaries (start floored, end ceiled) BEFORE any
/// slice — a shrunk re-fetched buffer or drift can leave offsets mid-codepoint.
fn clamp_span(buffer: &str, char_start: usize, char_end: usize) -> (usize, usize) {
    let len = buffer.len();
    let end = char_end.min(len);
    let start = char_start.min(end);
    (floor_boundary(buffer, start), ceil_boundary(buffer, end))
}

/// Builds a bounded, boundary-snapped snippet around `buffer[char_start..char_end]`.
/// Pure and panic-free: offsets are clamped and char-snapped first, so a stale or
/// shrunk buffer yields valid UTF-8 segments rather than a slice panic.
/// `char_start`/`char_end` are byte offsets, not char indices.
pub fn compute_snippet(buffer: &str, char_start: usize, char_end: usize) -> SnippetSegments {
    let (start, end) = clamp_span(buffer, char_start, char_end);
    let len = buffer.len();

    // A huge/stale span must not copy the whole buffer into `marked`.
    let (end, marked_truncated) = if end - start > MAX_MARKED {
        (ceil_boundary(buffer, start + MAX_MARKED), true)
    } else {
        (end, false)
    };

    // Per-side budget keeps before + marked + after within MAX_SNIPPET while never
    // shrinking below the base WINDOW.
    let side_budget = (MAX_SNIPPET.saturating_sub(end - start) / 2).max(WINDOW);

    let ctx_start = snap_start_outward(
        buffer,
        start.saturating_sub(WINDOW),
        start.saturating_sub(side_budget),
    );
    let ctx_end = snap_end_outward(
        buffer,
        (end + WINDOW).min(len),
        (end + side_budget).min(len),
    );

    SnippetSegments {
        before: buffer[ctx_start..start].to_string(),
        marked: buffer[start..end].to_string(),
        after: buffer[end..ctx_end].to_string(),
        truncated_before: ctx_start > 0,
        truncated_after: marked_truncated || ctx_end < len,
    }
}

/// Builds a [`SourceView`]. With no span the whole buffer sits in `before`
/// (windowed to the head when it exceeds `VIEW_CAP`). With a span the buffer splits
/// into before/marked/after at the clamped, char-snapped span; oversized sources
/// window before/after around the span. Pure and panic-free.
pub fn compute_source_view(
    buffer: &str,
    span: Option<(usize, usize)>,
    title: &str,
    kind: &str,
) -> SourceView {
    let mk = |before: String, marked: String, after: String, truncated: bool| SourceView {
        before,
        marked,
        after,
        title: title.to_string(),
        kind: kind.to_string(),
        truncated,
    };

    match span {
        None => {
            if buffer.len() <= VIEW_CAP {
                mk(buffer.to_string(), String::new(), String::new(), false)
            } else {
                let cut = floor_boundary(buffer, VIEW_CAP);
                mk(
                    buffer[..cut].to_string(),
                    String::new(),
                    String::new(),
                    true,
                )
            }
        }
        Some((cs, ce)) => {
            let (start, end) = clamp_span(buffer, cs, ce);
            // A huge/stale span must not copy the whole buffer into `marked`.
            let (end, marked_truncated) = if end - start > MAX_MARKED {
                (ceil_boundary(buffer, start + MAX_MARKED), true)
            } else {
                (end, false)
            };
            let len = buffer.len();
            let marked = buffer[start..end].to_string();
            if len <= VIEW_CAP {
                mk(
                    buffer[..start].to_string(),
                    marked,
                    buffer[end..].to_string(),
                    marked_truncated,
                )
            } else {
                let half = VIEW_CAP.saturating_sub(end - start) / 2;
                let bstart = floor_boundary(buffer, start.saturating_sub(half));
                let aend = ceil_boundary(buffer, (end + half).min(len));
                mk(
                    buffer[bstart..start].to_string(),
                    marked,
                    buffer[end..aend].to_string(),
                    true,
                )
            }
        }
    }
}

impl crate::LensEngine {
    /// Resolves a citation's persisted byte offsets against the retained source
    /// buffer, returning bounded [`SnippetSegments`] for the inline affordance.
    /// The blocking buffer read runs under `spawn_blocking` inside the loader.
    /// `char_start`/`char_end` are byte offsets, not char indices.
    pub async fn citation_snippet(
        &self,
        source_id: &str,
        char_start: usize,
        char_end: usize,
    ) -> Result<SnippetSegments, LensError> {
        let pool = self.pool().await;
        let data_dir = self.data_dir().await;
        let (buffer, _title, _kind) = load_source_buffer(&pool, &data_dir, source_id).await?;
        Ok(compute_snippet(&buffer, char_start, char_end))
    }

    /// Loads a source for the "view in source" viewer. `span` is optional: `None`
    /// returns the whole text (older chat history may carry null offsets); `Some`
    /// splits the text around the cited span. Large sources are windowed.
    pub async fn source_view(
        &self,
        source_id: &str,
        span: Option<(usize, usize)>,
    ) -> Result<SourceView, LensError> {
        let pool = self.pool().await;
        let data_dir = self.data_dir().await;
        let (buffer, title, kind) = load_source_buffer(&pool, &data_dir, source_id).await?;
        Ok(compute_source_view(&buffer, span, &title, kind.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_marked_is_exact_span() {
        let buf = "Alpha beta. The cited sentence here. Gamma delta.";
        let start = buf.find("The cited").unwrap();
        let end = start + "The cited sentence here.".len();
        let seg = compute_snippet(buf, start, end);
        assert_eq!(seg.marked, "The cited sentence here.");
        assert_eq!(
            format!("{}{}{}", seg.before, seg.marked, seg.after),
            buf,
            "small buffer reassembles losslessly"
        );
        assert!(!seg.truncated_before && !seg.truncated_after);
    }

    #[test]
    fn snippet_clamps_out_of_range_without_panic() {
        let buf = "short text";
        let seg = compute_snippet(buf, 1000, 5000);
        assert_eq!(seg.marked, "");
        assert!(!seg.truncated_after);
    }

    #[test]
    fn snippet_multibyte_edge_is_valid_utf8() {
        // Emoji + CJK straddling the window edge must never split a codepoint.
        let mut buf = String::new();
        buf.push_str(&"😀".repeat(200));
        let marker_start = buf.len();
        buf.push_str("中文标记片段");
        let marker_end = buf.len();
        buf.push_str(&"漢字".repeat(200));
        let seg = compute_snippet(&buf, marker_start, marker_end);
        assert_eq!(seg.marked, "中文标记片段");
        assert!(seg.before.is_char_boundary(0));
        // Reassembly around marked stays valid UTF-8 by construction.
        let _ = format!("{}{}{}", seg.before, seg.marked, seg.after);
    }

    #[test]
    fn snippet_truncation_flags_at_edges() {
        let long = "x".repeat(5000);
        let buf = format!("{long}. MARK. {long}");
        let start = buf.find("MARK").unwrap();
        let seg = compute_snippet(&buf, start, start + 4);
        assert!(seg.truncated_before && seg.truncated_after);
        assert!(seg.before.len() <= MAX_SNIPPET && seg.after.len() <= MAX_SNIPPET);
    }

    #[test]
    fn snippet_bounds_oversized_marked_span() {
        // Span wider than VIEW_CAP (and MAX_MARKED) must not copy the whole
        // buffer into `marked`, independent of the context caps.
        let buf = "M".repeat(VIEW_CAP + 10_000);
        let seg = compute_snippet(&buf, 0, buf.len());
        assert!(seg.marked.len() <= MAX_MARKED);
        assert!(seg.truncated_after);
    }

    #[test]
    fn source_view_none_returns_whole_text() {
        let buf = "the whole document body";
        let view = compute_source_view(buf, None, "Doc", "text");
        assert_eq!(view.before, buf);
        assert!(view.marked.is_empty() && view.after.is_empty());
        assert!(!view.truncated);
        assert_eq!(view.title, "Doc");
        assert_eq!(view.kind, "text");
    }

    #[test]
    fn source_view_span_splits_losslessly() {
        let buf = "before-part MARKED after-part";
        let start = buf.find("MARKED").unwrap();
        let view = compute_source_view(buf, Some((start, start + 6)), "T", "pdf");
        assert_eq!(view.marked, "MARKED");
        assert_eq!(format!("{}{}{}", view.before, view.marked, view.after), buf);
    }

    #[test]
    fn source_view_large_source_windows_and_truncates() {
        let big = "a".repeat(VIEW_CAP + 10_000);
        let view = compute_source_view(&big, None, "Big", "pdf");
        assert!(view.truncated);
        assert!(view.before.len() <= VIEW_CAP);
    }

    #[test]
    fn source_view_bounds_oversized_marked_span() {
        // Span wider than VIEW_CAP (and MAX_MARKED) must not copy the whole
        // buffer into `marked`, independent of the windowing cap.
        let buf = "M".repeat(VIEW_CAP + 10_000);
        let view = compute_source_view(&buf, Some((0, buf.len())), "Big", "pdf");
        assert!(view.marked.len() <= MAX_MARKED);
        assert!(view.truncated);
    }
}
