//! Orpheus-3B GGUF adapter (issue #191 [161c]).
//!
//! Generates SNAC audio-codec tokens from the Q4_K_M GGUF via a bundled
//! llama.cpp binding, then decodes them to 24 kHz mono PCM with [`SnacDecoder`].
//! Construction is cheap (paths only); the ~2 GB model loads lazily on the first
//! `synthesize_turn` inside `spawn_blocking` and is cached for the rest of the
//! script. The `!Send` `LlamaContext` is created and dropped entirely inside the
//! blocking closure so the `synthesize_turn` future stays `Send`.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::config::{VoiceConfig, VoiceRef};
use crate::dialogue::{Speaker, Turn};
use crate::error::LensError;
use crate::tts::snac::SnacDecoder;
use crate::tts::{
    AudioBuffer, Gender, TtsBackend, TtsProvider, TtsProviderInfo, TtsVoice, emotion_tag,
};

pub const ORPHEUS_MODEL_ID: &str = "orpheus";
pub const ORPHEUS_MODEL_URL: &str = "https://huggingface.co/unsloth/orpheus-3b-0.1-ft-GGUF/resolve/main/orpheus-3b-0.1-ft-Q4_K_M.gguf";
pub const ORPHEUS_MODEL_SHA256_HEX: &str =
    "4e94a925593c8fc1c20ee50e5b0e4bf915ea106a0bda3b917ff69b02559f7ab9";
pub const ORPHEUS_MODEL_RELPATH: &str = "models/orpheus/orpheus-3b-0.1-ft-Q4_K_M.gguf";

// Orpheus prompt framing (Llama-3 special ids): a turn is
// `[SOH] + tokenize("{voice}: {tag?}{text}") + [EOT, EOH]`; generation runs until
// the end-of-audio marker (or any EOG). SNAC's `extract_codes` skips the
// interleaved audio markers (128257/128261) by range, so we collect raw ids as-is.
const TOKEN_SOH: i32 = 128259;
const TOKEN_EOT: i32 = 128009;
const TOKEN_EOH: i32 = 128260;
const TOKEN_EOA: i32 = 128258;

const N_CTX: u32 = 8192;
const BATCH_CAP: usize = 512;
const MAX_NEW_TOKENS: usize = 4096;
// Cancel is polled every N generated tokens so a long detached blocking loop
// aborts promptly, not only at turn boundaries (AC2.3).
const CANCEL_POLL_INTERVAL: usize = 16;
const BASE_SEED: u32 = 1234;
// Early-EOS guard (AC2.6): estimate a token floor from word count and regenerate
// (fresh seed) up to K times when a run falls well short, then accept the longest.
const MAX_RETRIES: usize = 2;
const TOKENS_PER_WORD_EST: usize = 30;
const EARLY_EOS_FRACTION: f64 = 0.35;

// Backend-specific cancel message; intentionally distinct from the pipeline-level
// `tts::CANCELLED_MSG` so the source of the cancellation is legible.
const CANCELLED_MSG: &str = "orpheus generation cancelled";

/// Single source of truth for the Orpheus named voices: (id, display name, gender).
/// Both `voices()` (catalog) and `orpheus_voice()` (id resolution) derive from it.
const CATALOG: &[(&str, &str, Gender)] = &[
    ("tara", "Tara", Gender::Female),
    ("leah", "Leah", Gender::Female),
    ("jess", "Jess", Gender::Female),
    ("leo", "Leo", Gender::Male),
    ("dan", "Dan", Gender::Male),
    ("mia", "Mia", Gender::Female),
    ("zac", "Zac", Gender::Male),
    ("zoe", "Zoe", Gender::Female),
];

/// Locked Orpheus sampling (spike-tuned; exposure deferred to 161f settings).
#[derive(Debug, Clone, Copy)]
struct Sampling {
    temp: f32,
    top_p: f32,
    top_k: i32,
    repeat_penalty: f32,
    repeat_last_n: i32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temp: 0.6,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

pub struct OrpheusAdapter {
    model_path: PathBuf,
    snac_path: PathBuf,
    sampling: Sampling,
    loaded: OnceCell<Arc<LoadedOrpheus>>,
}

struct LoadedOrpheus {
    backend: Arc<LlamaBackend>,
    model: LlamaModel,
    snac: SnacDecoder,
}

impl LoadedOrpheus {
    fn load(model_path: &Path, snac_path: &Path) -> Result<Arc<Self>, LensError> {
        let backend = llama_backend()?;
        // Offload to Metal on Apple Silicon; CPU everywhere else (cross-platform floor).
        let n_gpu_layers = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            1000
        } else {
            0
        };
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        // Generic message: the underlying llama.cpp error embeds the model path,
        // which must not cross the IPC boundary.
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|_| LensError::Tts("orpheus: model load failed".into()))?;
        let snac = SnacDecoder::load(snac_path)?;
        Ok(Arc::new(Self {
            backend,
            model,
            snac,
        }))
    }
}

// The llama backend can be initialized only once per process; cache it (never
// dropped) so multiple adapter instances share one global init. Any `&LlamaBackend`
// satisfies `new_context`, so a single global suffices. The init `Result` is cached:
// a failed init is recorded once and never retried for the process lifetime.
static LLAMA_BACKEND: OnceLock<Result<Arc<LlamaBackend>, String>> = OnceLock::new();

fn llama_backend() -> Result<Arc<LlamaBackend>, LensError> {
    LLAMA_BACKEND
        .get_or_init(|| {
            LlamaBackend::init()
                .map(Arc::new)
                .map_err(|e| format!("orpheus: llama backend init failed: {e}"))
        })
        .clone()
        .map_err(LensError::Tts)
}

impl OrpheusAdapter {
    pub fn new(model_path: PathBuf, snac_path: PathBuf) -> Self {
        Self {
            model_path,
            snac_path,
            sampling: Sampling::default(),
            loaded: OnceCell::new(),
        }
    }

    async fn loaded(&self) -> Result<Arc<LoadedOrpheus>, LensError> {
        self.loaded
            .get_or_try_init(|| async {
                let model_path = self.model_path.clone();
                let snac_path = self.snac_path.clone();
                tokio::task::spawn_blocking(move || LoadedOrpheus::load(&model_path, &snac_path))
                    .await
                    .map_err(|e| LensError::Tts(format!("orpheus: load task failed: {e}")))?
            })
            .await
            .map(Arc::clone)
    }
}

#[async_trait]
impl TtsProvider for OrpheusAdapter {
    fn info(&self) -> TtsProviderInfo {
        TtsProviderInfo {
            backend: TtsBackend::Orpheus,
            model: "orpheus-3b-0.1-ft-Q4_K_M".to_string(),
        }
    }

    fn voices(&self) -> Vec<TtsVoice> {
        CATALOG
            .iter()
            .map(|&(id, name, gender)| TtsVoice::new(id, name, gender))
            .collect()
    }

    async fn synthesize_turn(
        &self,
        turn: &Turn,
        voices: &VoiceConfig,
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer, LensError> {
        let loaded = self.loaded().await?;
        // Resolve + validate the voice synchronously so an unsupported voice errors
        // before the (expensive) blocking generation is scheduled.
        let voice_ref = match turn.speaker {
            Speaker::Host => &voices.host,
            Speaker::Guest => &voices.guest,
        };
        let voice = voice_id(voice_ref, turn.speaker)?;
        let tag = turn
            .emotion
            .and_then(|e| emotion_tag(e, TtsBackend::Orpheus))
            .unwrap_or_default();
        let prompt_text = format!("{voice}: {tag}{}", turn.text);
        let floor = expected_token_floor(&turn.text);
        let sampling = self.sampling;
        let cancel = cancel.clone();

        tokio::task::spawn_blocking(move || -> Result<AudioBuffer, LensError> {
            let prompt_tokens = build_prompt_tokens(&loaded.model, &prompt_text)?;
            let audio_ids = generate_with_retry(floor, &cancel, MAX_RETRIES, |seed| {
                run_generation(&loaded, &prompt_tokens, &sampling, seed, &cancel)
            })?;
            loaded.snac.decode(&audio_ids)
        })
        .await
        .map_err(|e| LensError::Tts(format!("orpheus: synth task failed: {e}")))?
    }
}

fn build_prompt_tokens(model: &LlamaModel, text: &str) -> Result<Vec<LlamaToken>, LensError> {
    let mut tokens = Vec::new();
    tokens.push(LlamaToken(TOKEN_SOH));
    let body = model
        .str_to_token(text, AddBos::Never)
        .map_err(|e| LensError::Tts(format!("orpheus: tokenize failed: {e}")))?;
    tokens.extend(body);
    tokens.push(LlamaToken(TOKEN_EOT));
    tokens.push(LlamaToken(TOKEN_EOH));
    Ok(tokens)
}

/// One end-to-end generation attempt: primes a fresh context with the prompt,
/// then samples until the end-of-audio marker (or budget), returning the raw
/// audio-token ids. The `LlamaContext` lives and dies inside this fn.
fn run_generation(
    loaded: &LoadedOrpheus,
    prompt_tokens: &[LlamaToken],
    sampling: &Sampling,
    seed: u32,
    cancel: &CancellationToken,
) -> Result<Vec<u32>, LensError> {
    let n_ctx =
        NonZeroU32::new(N_CTX).ok_or_else(|| LensError::Tts("orpheus: invalid n_ctx".into()))?;
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
    let mut ctx = loaded
        .model
        .new_context(&loaded.backend, ctx_params)
        .map_err(|e| LensError::Tts(format!("orpheus: context init failed: {e}")))?;

    let mut batch = LlamaBatch::new(BATCH_CAP, 1);
    let last = prompt_tokens.len().saturating_sub(1) as i32;
    for (i, &token) in prompt_tokens.iter().enumerate() {
        let pos = i as i32;
        batch
            .add(token, pos, &[0], pos == last)
            .map_err(|e| LensError::Tts(format!("orpheus: batch add failed: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| LensError::Tts(format!("orpheus: prompt decode failed: {e}")))?;

    let mut sampler = build_sampler(sampling, seed);
    let model = &loaded.model;
    let mut n_cur = prompt_tokens.len() as i32;

    let next = || -> Result<TokenStep, LensError> {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if token.0 == TOKEN_EOA || model.is_eog_token(token) {
            return Ok(TokenStep::Stop);
        }
        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| LensError::Tts(format!("orpheus: batch add failed: {e}")))?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|e| LensError::Tts(format!("orpheus: decode failed: {e}")))?;
        Ok(TokenStep::Audio(token.0 as u32))
    };

    run_token_loop(next, MAX_NEW_TOKENS, CANCEL_POLL_INTERVAL, cancel)
}

fn build_sampler(s: &Sampling, seed: u32) -> LlamaSampler {
    LlamaSampler::chain_simple([
        LlamaSampler::penalties(s.repeat_last_n, s.repeat_penalty, 0.0, 0.0),
        LlamaSampler::top_k(s.top_k),
        LlamaSampler::top_p(s.top_p, 1),
        LlamaSampler::temp(s.temp),
        LlamaSampler::dist(seed),
    ])
}

enum TokenStep {
    Audio(u32),
    Stop,
}

/// Drives a token generator until it stops, the budget is hit, or cancellation.
/// Cancellation is polled every `poll_interval` tokens so a detached blocking run
/// aborts promptly. Generic over the step source so the loop (incl. cancel) is
/// unit-testable with a fake generator (AC2.3).
fn run_token_loop<N>(
    mut next: N,
    max_new: usize,
    poll_interval: usize,
    cancel: &CancellationToken,
) -> Result<Vec<u32>, LensError>
where
    N: FnMut() -> Result<TokenStep, LensError>,
{
    let mut ids = Vec::new();
    for produced in 0..max_new {
        if produced % poll_interval == 0 && cancel.is_cancelled() {
            return Err(LensError::Cancelled(CANCELLED_MSG.into()));
        }
        match next()? {
            TokenStep::Stop => break,
            TokenStep::Audio(id) => ids.push(id),
        }
    }
    Ok(ids)
}

/// Retries generation up to `max_retries` times when a run falls below the
/// early-EOS floor, then accepts the longest. Generic over the per-attempt
/// generator so the retry/decision policy is unit-testable (AC2.6).
fn generate_with_retry<G>(
    floor: usize,
    cancel: &CancellationToken,
    max_retries: usize,
    mut run_once: G,
) -> Result<Vec<u32>, LensError>
where
    G: FnMut(u32) -> Result<Vec<u32>, LensError>,
{
    let mut best: Vec<u32> = Vec::new();
    for attempt in 0..=max_retries {
        if cancel.is_cancelled() {
            return Err(LensError::Cancelled(CANCELLED_MSG.into()));
        }
        let ids = run_once(BASE_SEED + attempt as u32)?;
        if ids.len() > best.len() {
            best = ids;
        }
        if let EosDecision::Accept = early_eos_decision(best.len(), floor) {
            return Ok(best);
        }
    }
    Ok(best)
}

#[derive(Debug, PartialEq, Eq)]
enum EosDecision {
    Retry,
    Accept,
}

/// Pure early-EOS policy: a run producing fewer than `EARLY_EOS_FRACTION` of the
/// expected token floor is a suspected early stop and should be retried.
fn early_eos_decision(produced: usize, floor: usize) -> EosDecision {
    if (produced as f64) < EARLY_EOS_FRACTION * floor as f64 {
        EosDecision::Retry
    } else {
        EosDecision::Accept
    }
}

fn expected_token_floor(text: &str) -> usize {
    let words = text.split_whitespace().count().max(1);
    words * TOKENS_PER_WORD_EST
}

/// Resolves a turn's [`VoiceRef`] to a named Orpheus voice. Cloning
/// (`VoiceRef::Reference`) and unknown/non-Orpheus names error — no silent
/// fallback (AC2.4).
fn voice_id(voice: &VoiceRef, speaker: Speaker) -> Result<&'static str, LensError> {
    match voice {
        VoiceRef::Reference { .. } => Err(LensError::Tts(
            "voice cloning (VoiceRef::Reference) is unsupported by the Orpheus backend; \
             use the MOSS backend (#161e) or a named Orpheus voice \
             (tara/leah/jess/leo/dan/mia/zac/zoe)"
                .into(),
        )),
        VoiceRef::Named(name) if name.is_empty() => Ok(default_voice(speaker)),
        VoiceRef::Named(name) => orpheus_voice(name).ok_or_else(|| {
            LensError::Tts(format!(
                "unknown Orpheus voice {name:?}; valid voices are \
                 tara/leah/jess/leo/dan/mia/zac/zoe (use the MOSS backend #161e for cloning)"
            ))
        }),
    }
}

fn default_voice(speaker: Speaker) -> &'static str {
    match speaker {
        Speaker::Host => "tara",
        Speaker::Guest => "leo",
    }
}

fn orpheus_voice(name: &str) -> Option<&'static str> {
    CATALOG
        .iter()
        .find(|&&(id, _, _)| id == name)
        .map(|&(id, _, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn voices_catalog_covers_all_named_voices() {
        let adapter = OrpheusAdapter::new(PathBuf::from("/x/orpheus"), PathBuf::from("/x/snac"));
        let voices = adapter.voices();
        assert_eq!(voices.len(), CATALOG.len());
        for &(id, _, _) in CATALOG {
            assert!(voices.iter().any(|v| v.id == id), "missing voice {id}");
        }
        let female = voices.iter().filter(|v| v.gender == Gender::Female).count();
        let male = voices.iter().filter(|v| v.gender == Gender::Male).count();
        assert_eq!(female, 5);
        assert_eq!(male, 3);
    }

    #[test]
    fn voice_id_named_resolves_all_eight() {
        for &(id, _, _) in CATALOG {
            let r = VoiceRef::Named(id.to_string());
            assert_eq!(voice_id(&r, Speaker::Host).unwrap(), id);
        }
    }

    #[test]
    fn voice_id_defaults_when_unset() {
        let unset = VoiceRef::default();
        assert_eq!(voice_id(&unset, Speaker::Host).unwrap(), "tara");
        assert_eq!(voice_id(&unset, Speaker::Guest).unwrap(), "leo");
    }

    #[test]
    fn voice_id_reference_errors() {
        let r = VoiceRef::Reference {
            clip_path: PathBuf::from("/tmp/clip.wav"),
            transcript: "hi".into(),
        };
        let err = voice_id(&r, Speaker::Host).unwrap_err();
        assert!(matches!(err, LensError::Tts(_)), "got {err:?}");
    }

    #[test]
    fn voice_id_unknown_named_errors() {
        let r = VoiceRef::Named("not_a_voice".into());
        let err = voice_id(&r, Speaker::Guest).unwrap_err();
        assert!(matches!(err, LensError::Tts(_)), "got {err:?}");
    }

    #[test]
    fn early_eos_decision_table() {
        // floor 100: below 35 -> retry; at/above 35 -> accept.
        assert_eq!(early_eos_decision(0, 100), EosDecision::Retry);
        assert_eq!(early_eos_decision(34, 100), EosDecision::Retry);
        assert_eq!(early_eos_decision(35, 100), EosDecision::Accept);
        assert_eq!(early_eos_decision(200, 100), EosDecision::Accept);
        // A zero floor never triggers a retry.
        assert_eq!(early_eos_decision(0, 0), EosDecision::Accept);
    }

    #[test]
    fn expected_token_floor_counts_words() {
        assert_eq!(
            expected_token_floor("one two three"),
            3 * TOKENS_PER_WORD_EST
        );
        assert_eq!(expected_token_floor(""), TOKENS_PER_WORD_EST);
        assert_eq!(expected_token_floor("   "), TOKENS_PER_WORD_EST);
    }

    #[test]
    fn retry_accepts_longest_after_cap() {
        // Every attempt falls below the floor -> exhausts retries, returns longest.
        let floor = 1000;
        let cancel = CancellationToken::new();
        let calls = AtomicUsize::new(0);
        let out = generate_with_retry(floor, &cancel, MAX_RETRIES, |_seed| {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            // lengths 1, 3, 2 -> best is the 3-length attempt.
            Ok(vec![0u32; [1usize, 3, 2][n]])
        })
        .unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(calls.load(Ordering::SeqCst), MAX_RETRIES + 1);
    }

    #[test]
    fn retry_accepts_early_when_floor_met() {
        let floor = 2;
        let cancel = CancellationToken::new();
        let calls = AtomicUsize::new(0);
        let out = generate_with_retry(floor, &cancel, MAX_RETRIES, |_seed| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0u32; 10])
        })
        .unwrap();
        assert_eq!(out.len(), 10);
        // First attempt clears the floor -> no retry.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn retry_aborts_when_cancelled_before_attempt() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let calls = AtomicUsize::new(0);
        let err = generate_with_retry(100, &cancel, MAX_RETRIES, |_seed| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0u32; 1])
        })
        .unwrap_err();
        assert!(matches!(err, LensError::Cancelled(_)), "got {err:?}");
        assert_eq!(calls.load(Ordering::SeqCst), 0, "generator must not run");
    }

    #[test]
    fn token_loop_aborts_mid_generation_on_cancel() {
        // A fake generator yields audio forever; cancelling mid-stream must stop
        // the loop promptly (AC2.3), not run to the budget.
        let cancel = CancellationToken::new();
        let produced = AtomicUsize::new(0);
        let next = || {
            let n = produced.fetch_add(1, Ordering::SeqCst);
            if n == 5 {
                cancel.cancel();
            }
            Ok(TokenStep::Audio(1))
        };
        // poll_interval 1 => cancel is observed on the next iteration.
        let err = run_token_loop(next, 10_000, 1, &cancel).unwrap_err();
        assert!(matches!(err, LensError::Cancelled(_)), "got {err:?}");
        assert!(
            produced.load(Ordering::SeqCst) < 10_000,
            "loop must abort well before the budget"
        );
    }

    #[test]
    fn token_loop_stops_on_stop_step() {
        let cancel = CancellationToken::new();
        let n = AtomicUsize::new(0);
        let next = || {
            let i = n.fetch_add(1, Ordering::SeqCst);
            if i < 3 {
                Ok(TokenStep::Audio(i as u32 + 1))
            } else {
                Ok(TokenStep::Stop)
            }
        };
        let ids = run_token_loop(next, 100, CANCEL_POLL_INTERVAL, &cancel).unwrap();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    // Gated end-to-end (AC2.9): needs the real 1.9 GB GGUF + SNAC weights. Set
    // `LENS_RUN_MODEL_TESTS=1`, `LENS_ORPHEUS_GGUF=<...Q4_K_M.gguf>`, and
    // `LENS_SNAC_WEIGHTS=<...pytorch_model.bin>`. Asserts a short script
    // synthesizes to a non-silent 24 kHz mono buffer.
    #[tokio::test]
    async fn orpheus_e2e_synthesizes_non_silent_audio() {
        use crate::dialogue::DialogueScript;
        use crate::tts::TtsPhase;
        use crate::tts::audio::TARGET_RATE;

        if std::env::var("LENS_RUN_MODEL_TESTS").is_err() {
            eprintln!("skipping orpheus_e2e (set LENS_RUN_MODEL_TESTS=1)");
            return;
        }
        let (Ok(gguf), Ok(snac)) = (
            std::env::var("LENS_ORPHEUS_GGUF"),
            std::env::var("LENS_SNAC_WEIGHTS"),
        ) else {
            eprintln!("skipping: set LENS_ORPHEUS_GGUF and LENS_SNAC_WEIGHTS");
            return;
        };
        let adapter = OrpheusAdapter::new(PathBuf::from(gguf), PathBuf::from(snac));
        let script = DialogueScript {
            turns: vec![
                Turn {
                    speaker: Speaker::Host,
                    text: "Welcome to the overview.".into(),
                    emotion: None,
                    source_ids: Vec::new(),
                },
                Turn {
                    speaker: Speaker::Guest,
                    text: "Glad to be here.".into(),
                    emotion: Some(crate::dialogue::Emotion::Laugh),
                    source_ids: Vec::new(),
                },
            ],
        };
        let voices = VoiceConfig::default();
        let cancel = CancellationToken::new();
        let noop = |_p: TtsPhase| {};
        let provider: Arc<dyn TtsProvider> = Arc::new(adapter);
        let buf = provider
            .synthesize_script(&script, &voices, &noop, &cancel)
            .await
            .expect("synthesis");
        assert_eq!(buf.sample_rate, TARGET_RATE);
        assert_eq!(buf.channels, 1);
        assert!(!buf.samples.is_empty());
        let rms =
            (buf.samples.iter().map(|s| s * s).sum::<f32>() / buf.samples.len() as f32).sqrt();
        assert!(rms > 1e-4, "synthesized audio is silent (rms {rms})");
    }
}
