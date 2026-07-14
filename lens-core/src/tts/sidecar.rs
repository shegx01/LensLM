//! TTS sidecar seam (#190) — the tauri-free trait boundary for an out-of-process
//! synthesis engine (#193 MOSS-TTS). Mirrors the `render.rs` `JsRenderer` seam:
//! `lens-core` defines the trait and holds it behind an
//! `Arc<RwLock<Option<Arc<dyn _>>>>` DI cell injected via
//! [`set_tts_sidecar`](crate::LensEngine::set_tts_sidecar); the concrete process
//! manager lives in `src-tauri`. No implementation ships in #190.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::config::VoiceConfig;
use crate::dialogue::Turn;
use crate::error::LensError;
use crate::tts::AudioBuffer;

/// An async, object-safe out-of-process TTS engine held behind
/// `Arc<dyn TtsSidecar>`. Explicit `start`/`stop` lifecycle (#193 owns a spawned
/// process that must be booted and torn down), a cheap `health` probe used by the
/// `synthesize_overview` resolution precedence, and per-turn synthesis producing a
/// canonical [`AudioBuffer`] the engine stitches with the same `stitch_turns`.
#[async_trait]
pub trait TtsSidecar: Send + Sync {
    /// Boots the sidecar process. Idempotent; returns once it is ready to serve.
    async fn start(&self) -> Result<(), LensError>;

    /// Tears the sidecar process down. Idempotent.
    async fn stop(&self) -> Result<(), LensError>;

    /// Whether the sidecar is up and ready to synthesize.
    async fn health(&self) -> bool;

    /// Synthesizes one dialogue turn into a canonical [`AudioBuffer`], racing the
    /// call against `cancel`.
    async fn synthesize_turn(
        &self,
        turn: &Turn,
        voices: &VoiceConfig,
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer, LensError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LensEngine;
    use crate::tts::audio::TARGET_RATE;
    use std::sync::Arc;

    /// A fake sidecar proving `Arc<dyn TtsSidecar>` is object-safe and the engine
    /// DI seam round-trips (set → get → clear), mirroring the `JsRenderer` test.
    struct FakeSidecar;

    #[async_trait]
    impl TtsSidecar for FakeSidecar {
        async fn start(&self) -> Result<(), LensError> {
            Ok(())
        }
        async fn stop(&self) -> Result<(), LensError> {
            Ok(())
        }
        async fn health(&self) -> bool {
            true
        }
        async fn synthesize_turn(
            &self,
            _turn: &Turn,
            _voices: &VoiceConfig,
            _cancel: &CancellationToken,
        ) -> Result<AudioBuffer, LensError> {
            Ok(AudioBuffer::mono(vec![0.0; 8], TARGET_RATE))
        }
    }

    #[tokio::test]
    async fn tts_sidecar_seam_set_get_clear_round_trip() {
        let engine = LensEngine::for_test().await;
        assert!(
            engine.tts_sidecar().await.is_none(),
            "sidecar cell must start empty"
        );

        let fake: Arc<dyn TtsSidecar> = Arc::new(FakeSidecar);
        engine.set_tts_sidecar(Some(fake)).await;
        let got = engine.tts_sidecar().await.expect("sidecar must be set");
        assert!(got.health().await);

        engine.set_tts_sidecar(None).await;
        assert!(engine.tts_sidecar().await.is_none());
    }
}
