//! The real [`WhisperEngine`] — whisper.cpp (via `whisper-rs`) behind the
//! `local-whisper` feature. Loads a ggml model offline and implements the
//! headless [`AsrEngine`] seam.
//!
//! whisper.cpp's `full()` does its own internal 30 s windowing over
//! arbitrary-length PCM, so this engine feeds the whole #41 buffer at once and
//! reads back segments; no manual framing here. Inference is CPU-blocking, so
//! `transcribe_pcm` runs it on `spawn_blocking` — the context lives behind an
//! `Arc<Mutex<..>>` so the blocking task can own a `'static` handle to it.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::LensError;
use crate::asr::{AsrEngine, Lang, TranscribeConfig, TranscriptSegment};

/// A loaded whisper.cpp model. The context sits behind `Arc<Mutex<..>>` so a
/// clone can move into `spawn_blocking`; the `Mutex` serialises inference (a
/// fresh state is created per call, but the context is shared).
pub struct WhisperEngine {
    ctx: Arc<Mutex<WhisperContext>>,
    model_id: String,
}

impl WhisperEngine {
    /// Loads a ggml model from `model_path` (must already be on disk — download
    /// happens via the registry, not here). Maps any load failure to
    /// [`LensError::Transcription`].
    pub fn load(model_path: &Path) -> Result<Self, LensError> {
        let model_id = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let path_str = model_path.to_str().ok_or_else(|| {
            LensError::Transcription("whisper model path is not valid UTF-8".into())
        })?;
        let ctx = WhisperContext::new_with_params(path_str, WhisperContextParameters::default())
            .map_err(|e| LensError::Transcription(format!("failed to load whisper model: {e}")))?;
        Ok(Self {
            ctx: Arc::new(Mutex::new(ctx)),
            model_id,
        })
    }

    /// The id (ggml file stem, e.g. `ggml-base`) of the loaded model.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Maps a [`Lang`] to the whisper language code (`"en"`, `"de"`, …). The
/// `Other` escape hatch passes its BCP-47-ish code through unchanged.
fn lang_to_whisper_code(lang: &Lang) -> &str {
    match lang {
        Lang::En => "en",
        Lang::De => "de",
        Lang::Fr => "fr",
        Lang::Es => "es",
        Lang::It => "it",
        Lang::Pt => "pt",
        Lang::Nl => "nl",
        Lang::Ru => "ru",
        Lang::Zh => "zh",
        Lang::Ja => "ja",
        Lang::Ko => "ko",
        Lang::Other(code) => code.as_str(),
    }
}

/// Runs the whole (blocking) inference: create a state, set params from config,
/// call `full()`, and read segments back into [`TranscriptSegment`]s.
fn run_inference(
    ctx: &Mutex<WhisperContext>,
    pcm: &[f32],
    language: Option<&str>,
    translate: bool,
    progress_tx: Option<tokio::sync::mpsc::UnboundedSender<f32>>,
) -> Result<Vec<TranscriptSegment>, LensError> {
    let ctx = ctx
        .lock()
        .map_err(|e| LensError::Transcription(format!("whisper context lock poisoned: {e}")))?;
    let mut state = ctx
        .create_state()
        .map_err(|e| LensError::Transcription(format!("failed to create whisper state: {e}")))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(language);
    params.set_translate(translate);
    // Keep whisper.cpp quiet: no stdout progress/realtime/timestamp spam.
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    if let Some(tx) = progress_tx {
        // whisper reports 0..=100; the trait convention is 0.0..=1.0.
        params.set_progress_callback_safe(move |percent: i32| {
            let _ = tx.send((percent as f32 / 100.0).clamp(0.0, 1.0));
        });
    }

    state
        .full(params, pcm)
        .map_err(|e| LensError::Transcription(format!("whisper transcription failed: {e}")))?;

    let n: usize = state
        .full_n_segments()
        .try_into()
        .map_err(|e| LensError::Transcription(format!("invalid segment count: {e}")))?;
    let mut segments = Vec::with_capacity(n);
    for seg in state.as_iter() {
        let text = seg
            .to_str_lossy()
            .map_err(|e| LensError::Transcription(format!("segment text decode failed: {e}")))?
            .trim()
            .to_string();
        // whisper timestamps are centiseconds (hundredths of a second) → seconds.
        let start_second = seg.start_timestamp() as f32 / 100.0;
        let end_second = seg.end_timestamp() as f32 / 100.0;
        segments.push(TranscriptSegment {
            text,
            start_second,
            end_second,
        });
    }
    Ok(segments)
}

#[async_trait]
impl AsrEngine for WhisperEngine {
    async fn transcribe_pcm(
        &self,
        pcm: &[f32],
        config: &TranscribeConfig,
        progress_tx: Option<tokio::sync::mpsc::UnboundedSender<f32>>,
    ) -> Result<Vec<TranscriptSegment>, LensError> {
        // Move owned copies of the inputs into the blocking task; whisper.full is
        // CPU-blocking and must not run on the async runtime.
        let pcm = pcm.to_vec();
        let language = config
            .language
            .as_ref()
            .map(|l| lang_to_whisper_code(l).to_string());
        let translate = config.translate;
        let ctx = Arc::clone(&self.ctx);

        tokio::task::spawn_blocking(move || {
            run_inference(&ctx, &pcm, language.as_deref(), translate, progress_tx)
        })
        .await
        .map_err(|e| LensError::Transcription(format!("whisper inference task failed: {e}")))?
    }
}
