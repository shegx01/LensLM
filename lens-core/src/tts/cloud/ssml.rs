//! W3C SSML builder + plain-text instruction mapping for cloud TTS.
//!
//! `build_ssml` produces markup for an SSML-capable provider (see
//! [`super::super::CloudTtsKind::supports_ssml`]). The OpenAI-compatible path does
//! NOT accept SSML in `input`; it uses [`emotion_to_instruction`] for the optional
//! `instructions` field instead.

use crate::dialogue::{Emotion, Turn};

/// Builds a W3C `<speak>` document for `turn`, mapping [`Emotion`] to
/// `<prosody>`/`<emphasis>` and appending a `<break>` where a beat helps delivery.
pub fn build_ssml(turn: &Turn) -> String {
    let text = escape_xml(&turn.text);
    match turn.emotion {
        None | Some(Emotion::Neutral) => format!("<speak>{text}</speak>"),
        Some(Emotion::Excited) => format!(
            "<speak><prosody rate=\"fast\" pitch=\"high\"><emphasis level=\"strong\">{text}</emphasis></prosody></speak>"
        ),
        Some(Emotion::Thoughtful) => {
            format!("<speak><prosody rate=\"slow\">{text}</prosody></speak>")
        }
        Some(Emotion::Laugh) => format!(
            "<speak><emphasis level=\"moderate\">{text}</emphasis><break time=\"300ms\"/></speak>"
        ),
        Some(Emotion::Sigh) => format!(
            "<speak><prosody rate=\"slow\" pitch=\"low\">{text}</prosody><break time=\"500ms\"/></speak>"
        ),
    }
}

/// Maps an [`Emotion`] to a plain-text delivery instruction for providers that take
/// an `instructions` field (OpenAI `gpt-4o-mini-tts`). `Neutral` maps to `None` so
/// the field is omitted (strict OpenAI-compatible servers may reject an unknown or
/// empty field).
pub fn emotion_to_instruction(emotion: Emotion) -> Option<String> {
    let s = match emotion {
        Emotion::Neutral => return None,
        Emotion::Laugh => "Speak with warm, light laughter in your voice.",
        Emotion::Sigh => "Speak with a soft, weary sigh.",
        Emotion::Excited => "Speak with bright, energetic excitement.",
        Emotion::Thoughtful => "Speak in a measured, thoughtful tone.",
    };
    Some(s.to_string())
}

fn escape_xml(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialogue::Speaker;

    fn turn(text: &str, emotion: Option<Emotion>) -> Turn {
        Turn {
            speaker: Speaker::Host,
            text: text.to_string(),
            emotion,
            source_ids: Vec::new(),
        }
    }

    #[test]
    fn build_ssml_neutral_and_none_are_bare_speak() {
        assert_eq!(build_ssml(&turn("hello", None)), "<speak>hello</speak>");
        assert_eq!(
            build_ssml(&turn("hello", Some(Emotion::Neutral))),
            "<speak>hello</speak>"
        );
    }

    #[test]
    fn build_ssml_golden_strings_per_emotion() {
        assert_eq!(
            build_ssml(&turn("hi", Some(Emotion::Excited))),
            "<speak><prosody rate=\"fast\" pitch=\"high\"><emphasis level=\"strong\">hi</emphasis></prosody></speak>"
        );
        assert_eq!(
            build_ssml(&turn("hi", Some(Emotion::Thoughtful))),
            "<speak><prosody rate=\"slow\">hi</prosody></speak>"
        );
        assert_eq!(
            build_ssml(&turn("hi", Some(Emotion::Laugh))),
            "<speak><emphasis level=\"moderate\">hi</emphasis><break time=\"300ms\"/></speak>"
        );
        assert_eq!(
            build_ssml(&turn("hi", Some(Emotion::Sigh))),
            "<speak><prosody rate=\"slow\" pitch=\"low\">hi</prosody><break time=\"500ms\"/></speak>"
        );
    }

    #[test]
    fn build_ssml_escapes_markup_in_text() {
        assert_eq!(
            build_ssml(&turn("a<b>&\"'", None)),
            "<speak>a&lt;b&gt;&amp;&quot;&apos;</speak>"
        );
    }

    #[test]
    fn emotion_to_instruction_neutral_is_none() {
        assert!(emotion_to_instruction(Emotion::Neutral).is_none());
    }

    #[test]
    fn emotion_to_instruction_non_neutral_is_some_non_empty() {
        for e in [
            Emotion::Laugh,
            Emotion::Sigh,
            Emotion::Excited,
            Emotion::Thoughtful,
        ] {
            let s = emotion_to_instruction(e).expect("non-neutral maps to an instruction");
            assert!(!s.is_empty());
        }
    }
}
