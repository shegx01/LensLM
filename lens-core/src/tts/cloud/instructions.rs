//! Plain-text delivery-instruction mapping for cloud TTS providers that take an
//! `instructions` field (OpenAI-compatible `gpt-4o-mini-tts`). Derives the sentence
//! from the central emotion table so it cannot drift from the other adapters.

use crate::dialogue::Emotion;

/// Maps an [`Emotion`] to a plain-text delivery instruction. `Neutral` (and any
/// emotion with no style) maps to `None` so the field is omitted — a strict
/// OpenAI-compatible server may reject an unknown/empty field.
pub fn emotion_to_instruction(emotion: Emotion) -> Option<String> {
    crate::tts::emotion_render(emotion)
        .style
        .map(|s| format!("Speak with {s}."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_is_none() {
        assert!(emotion_to_instruction(Emotion::Neutral).is_none());
    }

    #[test]
    fn non_neutral_is_some_sentence() {
        for e in [
            Emotion::Laugh,
            Emotion::Sigh,
            Emotion::Excited,
            Emotion::Thoughtful,
            Emotion::Curious,
            Emotion::Serious,
        ] {
            let s = emotion_to_instruction(e).expect("non-neutral maps to an instruction");
            assert!(s.starts_with("Speak with "));
            assert!(s.ends_with('.'));
        }
    }
}
