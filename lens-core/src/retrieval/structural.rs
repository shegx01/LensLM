//! Structural-query detection (#279): recognizes positional queries ("chapter 2",
//! "the introduction") so the router can resolve them against the `sections` outline.
//! Pure and deterministic; unclear queries return `None` and fall through to normal
//! retrieval. Precision over recall — a false positive scopes to the wrong section.

/// The structural unit a positional query names — user vocabulary the resolver maps onto
/// the document's actual heading depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralKind {
    Chapter,
    Section,
    Part,
}

impl StructuralKind {
    /// The lowercase noun that names this unit in a heading title ("Chapter 2").
    pub fn title_keyword(self) -> &'static str {
        match self {
            Self::Chapter => "chapter",
            Self::Section => "section",
            Self::Part => "part",
        }
    }
}

/// A named front/back-matter section addressable by name rather than number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamedSection {
    Introduction,
    Conclusion,
}

impl NamedSection {
    /// Lowercase substrings that identify this section inside a heading title.
    pub fn title_keywords(self) -> &'static [&'static str] {
        match self {
            Self::Introduction => &["introduction", "intro"],
            Self::Conclusion => &["conclusion", "concluding"],
        }
    }
}

/// A structural reference extracted from a query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralTarget {
    /// A numbered unit, e.g. "chapter 2" → `{ kind: Chapter, ordinal: 2 }`.
    Ordinal { kind: StructuralKind, ordinal: u32 },
    /// A named section, e.g. "the introduction".
    Named(NamedSection),
}

/// Extracts a structural target from `query`, or `None` when the query is not a
/// clear positional request. Detection is case-insensitive and order-tolerant
/// ("chapter 2" and "the 2nd chapter" both match).
pub fn detect_structural_target(query: &str) -> Option<StructuralTarget> {
    let lower = query.to_lowercase();
    let toks: Vec<&str> = lower
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|t| !t.is_empty())
        .collect();

    // Ordinal form: a structural noun adjacent to a cardinal, in either order.
    for i in 0..toks.len() {
        let Some(kind) = structural_noun(toks[i]) else {
            continue;
        };
        if let Some(next) = toks.get(i + 1).and_then(|t| parse_cardinal(t)) {
            return Some(StructuralTarget::Ordinal {
                kind,
                ordinal: next,
            });
        }
        if i > 0
            && let Some(prev) = parse_cardinal(toks[i - 1])
        {
            return Some(StructuralTarget::Ordinal {
                kind,
                ordinal: prev,
            });
        }
    }

    // Named form: "the <named>" only, to avoid firing on prose like "in conclusion".
    for w in toks.windows(2) {
        if w[0] == "the"
            && let Some(named) = named_section(w[1])
        {
            return Some(StructuralTarget::Named(named));
        }
    }

    None
}

fn structural_noun(tok: &str) -> Option<StructuralKind> {
    match tok {
        "chapter" | "ch" => Some(StructuralKind::Chapter),
        "section" => Some(StructuralKind::Section),
        "part" => Some(StructuralKind::Part),
        _ => None,
    }
}

fn named_section(tok: &str) -> Option<NamedSection> {
    match tok {
        "introduction" | "intro" => Some(NamedSection::Introduction),
        "conclusion" => Some(NamedSection::Conclusion),
        _ => None,
    }
}

/// Parses a cardinal from digits ("2", "2nd"), English words ("two"), or roman
/// numerals ("ii"). Returns `None` for anything else — including dotted sub-section
/// numbers ("3.1"), which are intentionally not handled (see module non-goals).
fn parse_cardinal(tok: &str) -> Option<u32> {
    let digits: String = tok.chars().take_while(|c| c.is_ascii_digit()).collect();
    if !digits.is_empty() {
        // Accept a trailing ordinal suffix ("2nd", "3rd") but reject a dotted or
        // embedded form ("3.1", "2a") so sub-sections fall through to normal search.
        let rest = &tok[digits.len()..];
        if rest.is_empty() || matches!(rest, "st" | "nd" | "rd" | "th") {
            return digits.parse().ok();
        }
        return None;
    }
    if let Some(n) = word_cardinal(tok) {
        return Some(n);
    }
    roman_cardinal(tok)
}

fn word_cardinal(tok: &str) -> Option<u32> {
    const WORDS: [&str; 20] = [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ];
    const ORDINALS: [&str; 20] = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
    ];
    WORDS
        .iter()
        .position(|w| *w == tok)
        .or_else(|| ORDINALS.iter().position(|w| *w == tok))
        .map(|i| i as u32 + 1)
}

fn roman_cardinal(tok: &str) -> Option<u32> {
    if tok.is_empty() || !tok.chars().all(|c| "ivxlcdm".contains(c)) {
        return None;
    }
    let val = |c: char| match c {
        'i' => 1,
        'v' => 5,
        'x' => 10,
        'l' => 50,
        'c' => 100,
        'd' => 500,
        'm' => 1000,
        _ => 0,
    };
    let chars: Vec<char> = tok.chars().collect();
    let mut total = 0i64;
    for i in 0..chars.len() {
        let cur = val(chars[i]);
        let next = chars.get(i + 1).map(|c| val(*c)).unwrap_or(0);
        if cur < next {
            total -= cur;
        } else {
            total += cur;
        }
    }
    (total > 0).then_some(total as u32)
}

#[cfg(test)]
mod tests {
    use super::StructuralKind::*;
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("what is the summary of chapter 2?", Some(StructuralTarget::Ordinal { kind: Chapter, ordinal: 2 }))]
    #[case("summarize chapter two", Some(StructuralTarget::Ordinal { kind: Chapter, ordinal: 2 }))]
    #[case("tell me about the 2nd chapter", Some(StructuralTarget::Ordinal { kind: Chapter, ordinal: 2 }))]
    #[case("what does the second chapter cover", Some(StructuralTarget::Ordinal { kind: Chapter, ordinal: 2 }))]
    #[case("overview of part II", Some(StructuralTarget::Ordinal { kind: Part, ordinal: 2 }))]
    #[case("section 5 please", Some(StructuralTarget::Ordinal { kind: Section, ordinal: 5 }))]
    #[case("ch 3 summary", Some(StructuralTarget::Ordinal { kind: Chapter, ordinal: 3 }))]
    #[case(
        "summarize the introduction",
        Some(StructuralTarget::Named(NamedSection::Introduction))
    )]
    #[case(
        "what's in the conclusion?",
        Some(StructuralTarget::Named(NamedSection::Conclusion))
    )]
    fn detects_positional_queries(#[case] q: &str, #[case] want: Option<StructuralTarget>) {
        assert_eq!(detect_structural_target(q), want);
    }

    #[rstest]
    // Ordinary questions — no structural reference.
    #[case("what is photosynthesis?")]
    #[case("summarize the document")]
    #[case("how many chapters are there?")] // plural noun, no ordinal
    #[case("explain the concept in this chapter")] // noun but no cardinal
    #[case("the second law of thermodynamics")] // cardinal but no structural noun
    #[case("in conclusion the author argues")] // named word not preceded by "the"
    // Cross-level / sub-section: intentionally not scoped.
    #[case("summarize section 3.1")]
    fn ignores_non_structural_queries(#[case] q: &str) {
        assert_eq!(detect_structural_target(q), None);
    }

    #[test]
    fn distinguishes_kinds_at_same_ordinal() {
        // "chapter 2" and "section 2" must not collapse to the same target.
        let ch = detect_structural_target("chapter 2");
        let sec = detect_structural_target("section 2");
        assert_ne!(ch, sec);
        assert_eq!(
            ch,
            Some(StructuralTarget::Ordinal {
                kind: Chapter,
                ordinal: 2
            })
        );
        assert_eq!(
            sec,
            Some(StructuralTarget::Ordinal {
                kind: Section,
                ordinal: 2
            })
        );
    }
}
