use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::config::VoiceConfig;
use crate::dialogue::Turn;
use crate::error::LensError;
use crate::tts::AudioBuffer;

#[async_trait]
pub trait TtsSidecar: Send + Sync {
    async fn start(&self) -> Result<(), LensError>;

    async fn stop(&self) -> Result<(), LensError>;

    async fn health(&self) -> bool;

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
