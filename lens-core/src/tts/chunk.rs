//! Scene-boundary chunking for whole-exchange (dialogue) cloud TTS engines (#40).
//!
//! A dialogue provider (ElevenLabs, Gemini) renders the whole exchange in one
//! request to preserve cross-turn dynamics, but each caps its input size. When a
//! script exceeds that cap, [`scene_chunks`] splits it into the fewest contiguous
//! multi-turn ranges that each fit — splitting ONLY between turns, so a chunk is
//! always a whole number of turns. A single turn larger than the cap is rejected
//! up front (never split mid-turn, never silently degraded to per-turn rendering
//! — issue #40 AC4).

use std::ops::Range;

use crate::dialogue::Turn;
use crate::error::LensError;

/// A single turn's rendered text exceeds the provider input cap. Actionable and
/// IPC-safe (no provider/host detail); surfaced instead of letting the request
/// fail as an opaque provider 413.
const TURN_TOO_LONG: &str =
    "a dialogue turn is too long for the cloud voice provider; choose a shorter overview length";

/// Splits `turns` into contiguous index ranges, each whose measured length sums to
/// at most `char_limit`. Greedy: accumulate whole turns until the next would
/// exceed the cap, then start a new chunk. Splits only between turns.
///
/// `sized_len` measures one turn's *submitted* length — callers pass the
/// post-markup size (emotion cues / speaker labels a provider prepends) so a chunk
/// computed here still fits the cap at request time. A single turn whose
/// `sized_len` exceeds `char_limit` is a [`LensError::Validation`], not a chunk.
pub fn scene_chunks(
    turns: &[Turn],
    char_limit: usize,
    sized_len: impl Fn(&Turn) -> usize,
) -> Result<Vec<Range<usize>>, LensError> {
    let mut chunks: Vec<Range<usize>> = Vec::new();
    let mut start = 0usize;
    let mut acc = 0usize;
    for (i, turn) in turns.iter().enumerate() {
        let len = sized_len(turn);
        if len > char_limit {
            return Err(LensError::Validation(TURN_TOO_LONG.into()));
        }
        if i > start && acc + len > char_limit {
            chunks.push(start..i);
            start = i;
            acc = 0;
        }
        acc += len;
    }
    if start < turns.len() {
        chunks.push(start..turns.len());
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialogue::Speaker;

    fn turn(text: &str) -> Turn {
        Turn {
            speaker: Speaker::Host,
            text: text.to_string(),
            emotion: None,
            source_ids: Vec::new(),
        }
    }

    fn by_text(t: &Turn) -> usize {
        t.text.chars().count()
    }

    #[test]
    fn empty_script_yields_no_chunks() {
        assert!(scene_chunks(&[], 100, by_text).unwrap().is_empty());
    }

    #[test]
    fn single_under_limit_is_one_chunk() {
        let turns = vec![turn("hello")];
        assert_eq!(scene_chunks(&turns, 100, by_text).unwrap(), vec![0..1]);
    }

    #[test]
    fn exact_limit_stays_one_chunk() {
        // 1000 + 1000 == 2000 == limit: must NOT split.
        let turns = vec![turn(&"a".repeat(1000)), turn(&"b".repeat(1000))];
        assert_eq!(scene_chunks(&turns, 2000, by_text).unwrap(), vec![0..2]);
    }

    #[test]
    fn one_over_limit_splits_at_the_turn_boundary() {
        // 1000 + 1001 == 2001 > 2000: split before the second turn.
        let turns = vec![turn(&"a".repeat(1000)), turn(&"b".repeat(1001))];
        assert_eq!(
            scene_chunks(&turns, 2000, by_text).unwrap(),
            vec![0..1, 1..2]
        );
    }

    #[test]
    fn greedy_packs_whole_turns_per_chunk() {
        let turns = vec![
            turn(&"a".repeat(800)),
            turn(&"b".repeat(800)),
            turn(&"c".repeat(800)),
        ];
        // 800+800=1600 fits; +800=2400 doesn't → [0..2],[2..3].
        assert_eq!(
            scene_chunks(&turns, 2000, by_text).unwrap(),
            vec![0..2, 2..3]
        );
    }

    #[test]
    fn single_turn_over_limit_is_validation_error() {
        let turns = vec![turn(&"x".repeat(2001))];
        assert!(matches!(
            scene_chunks(&turns, 2000, by_text),
            Err(LensError::Validation(_))
        ));
    }

    #[test]
    fn sizing_closure_accounts_for_prepended_markup() {
        // Each turn is 1000 chars of text but the closure adds a 30-char cue, so
        // 2 turns = 2060 > 2000 and must split — proving the cap is measured
        // post-expansion, not on raw text.
        let turns = vec![turn(&"a".repeat(1000)), turn(&"b".repeat(1000))];
        let sized = |t: &Turn| t.text.chars().count() + 30;
        assert_eq!(scene_chunks(&turns, 2000, sized).unwrap(), vec![0..1, 1..2]);
    }

    #[test]
    fn conserves_order_and_covers_every_turn_exactly_once() {
        let turns: Vec<Turn> = (0..10).map(|i| turn(&"z".repeat(500 + i))).collect();
        let chunks = scene_chunks(&turns, 2000, by_text).unwrap();
        let mut covered = Vec::new();
        for c in &chunks {
            covered.extend(c.clone());
        }
        assert_eq!(covered, (0..10).collect::<Vec<_>>());
    }
}
