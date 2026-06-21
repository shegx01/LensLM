//! Streaming primitive shared by every long-running command.
//!
//! A single adjacently-tagged envelope is sent repeatedly over a
//! [`tauri::ipc::Channel`] the frontend passes into the command. Adjacent
//! tagging (`type` + `data`) is required because data-carrying variants like
//! `Chunk(T)` / `Failed(LensError)` are newtype variants — internally-tagged
//! serde would reject the non-map payload of `Chunk(String)`.

use lens_core::LensError;
use serde::{Deserialize, Serialize};
#[cfg(debug_assertions)]
use tauri::ipc::Channel;

/// One event in a stream. `T` is the per-chunk payload type (e.g. `String` for
/// LLM tokens). Serializes as `{"type": <snake_case variant>, "data": <payload>}`
/// for data-carrying variants, and `{"type": <variant>}` for unit variants.
///
/// This is foundational streaming scaffolding. In release builds its only M0
/// consumer (`stream_demo`) is gated out, so the type is unused until the first
/// real streaming command lands in M1 — hence the release-only `dead_code`
/// exemption (it is exercised by tests and the dev-only demo in debug builds).
#[cfg_attr(not(debug_assertions), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum StreamEvent<T> {
    /// The stream has begun.
    Started,
    /// A payload chunk.
    Chunk(T),
    /// Progress update; `total` is `None` when the upper bound is unknown.
    Progress {
        /// Units completed so far.
        done: u64,
        /// Total units, if known.
        total: Option<u64>,
    },
    /// The stream finished successfully.
    Done,
    /// The stream failed; carries the structured error (nests `{kind,message}`).
    Failed(LensError),
}

/// Sends one event over the channel, mapping the transport error into a
/// [`LensError::Internal`] so callers can use `?`. Centralizes the otherwise
/// repeated `.map_err(|e| LensError::Internal(e.to_string()))` boilerplate.
///
/// Gated to `debug_assertions` alongside its only current caller (`stream_demo`);
/// the gate lifts as soon as a release-surface streaming command lands.
#[cfg(debug_assertions)]
pub fn send_event<T: Serialize + Clone>(
    channel: &Channel<StreamEvent<T>>,
    event: StreamEvent<T>,
) -> Result<(), LensError> {
    channel
        .send(event)
        .map_err(|e| LensError::Internal(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_serializes_to_adjacent_tagged_shape() {
        let event = StreamEvent::Chunk("hello".to_string());
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(
            json,
            serde_json::json!({ "type": "chunk", "data": "hello" })
        );
        let back: StreamEvent<String> = serde_json::from_value(json).unwrap();
        assert_eq!(back, event);
    }

    #[test]
    fn failed_nests_lens_error_envelope() {
        let event: StreamEvent<String> = StreamEvent::Failed(LensError::Internal("boom".into()));
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "type": "failed",
                "data": { "kind": "Internal", "message": "boom" }
            })
        );
        let back: StreamEvent<String> = serde_json::from_value(json).unwrap();
        assert_eq!(back, event);
    }

    #[test]
    fn unit_variants_emit_no_data_key() {
        let json = serde_json::to_value(StreamEvent::<String>::Started).unwrap();
        assert_eq!(json, serde_json::json!({ "type": "started" }));
        let json = serde_json::to_value(StreamEvent::<String>::Done).unwrap();
        assert_eq!(json, serde_json::json!({ "type": "done" }));
    }
}
