use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::config::{TtsConfig, VoiceConfig};
use crate::dialogue::{DialogueScript, Emotion, Speaker, Turn};
use crate::error::LensError;

pub mod audio;
pub mod catalog;
pub mod chunk;
pub mod cloud;
pub mod orpheus;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod qwen;
pub mod registry;
pub mod sidecar;
pub mod snac;

pub(crate) use audio::write_wav_16bit;
pub use audio::{AudioBuffer, read_wav_mono16};
// Voice metadata + capability catalog live in the non-cfg-gated `catalog` module.
pub use catalog::{
    EngineCapability, EngineCatalogEntry, GuardVerdict, Lang, LanguageSupport, OffendingSource,
    Platform, QwenVoice, TtsEngineId, code_to_lang, evaluate_language_guard, lang_to_qwen_name,
    qwen_voice, tts_catalog, tts_catalog_serialized, validate_qwen_language,
};
pub use registry::{
    TTS_REGISTRY, TtsModelSpec, download_tts_model, resolve_tts, tts_model_downloaded,
    tts_model_file_present, tts_model_path,
};
pub use sidecar::TtsSidecar;

/// Speaker gender. Serializes lowercase to match the `'male' | 'female'` union in the Svelte
/// client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gender {
    Male,
    Female,
}

/// One selectable named voice. Frozen IPC contract — mirrored in the Svelte client as
/// `TtsVoice { id, name, gender }`. The catalog is adapter-driven via [`TtsProvider::voices`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsVoice {
    pub id: String,
    pub name: String,
    pub gender: Gender,
}

impl TtsVoice {
    pub(crate) fn new(id: &str, name: &str, gender: Gender) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            gender,
        }
    }
}

/// Download progress. Frozen IPC contract — mirrored in the Svelte client as
/// `{ received, total, done }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub received: u64,
    pub total: Option<u64>,
    pub done: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsBackend {
    #[default]
    Orpheus,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    Qwen3Local,
    Cloud(CloudTtsKind),
}

// `Qwen3Local` is cfg-gated to Apple Silicon; the derived impl would reject
// `"qwen3_local"` as unknown off-target. Route strings through `from_opt_str`
// (unknown -> default) and keep `{"cloud": ...}` for the kind.
impl<'de> Deserialize<'de> for TtsBackend {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Tag(String),
            Cloud { cloud: CloudTtsKind },
        }
        Ok(match Wire::deserialize(deserializer)? {
            Wire::Tag(s) => TtsBackend::from_opt_str(Some(&s)),
            Wire::Cloud { cloud } => TtsBackend::Cloud(cloud),
        })
    }
}

impl TtsBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            TtsBackend::Orpheus => "orpheus",
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            TtsBackend::Qwen3Local => "qwen3_local",
            TtsBackend::Cloud(_) => "cloud",
        }
    }

    pub fn from_opt_str(s: Option<&str>) -> Self {
        match s.unwrap_or("") {
            "orpheus" => TtsBackend::Orpheus,
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            "qwen3_local" => TtsBackend::Qwen3Local,
            "cloud" => TtsBackend::Cloud(CloudTtsKind::default()),
            _ => TtsBackend::default(),
        }
    }

    /// Registry ids of every model artifact this backend needs on disk to be
    /// usable. Non-embedded backends (cloud, not-yet-wired local) return `&[]`.
    /// Qwen3Local has none: `mlx-audio` fetches its model lazily on first synth,
    /// not via the Rust registry.
    pub fn required_model_ids(&self) -> &'static [&'static str] {
        match self {
            TtsBackend::Orpheus => &["orpheus", "snac"],
            TtsBackend::Cloud(_) => &[],
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            TtsBackend::Qwen3Local => &[],
        }
    }
}

// `Ord`/`Hash` so it can key the per-provider `TtsConfig.clouds` map (#40).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum CloudTtsKind {
    #[default]
    OpenAiCompatible,
    Deepgram,
    ElevenLabs,
    /// Google's multi-speaker dialogue TTS. Targets the Gemini API
    /// `generateContent` surface (`generativelanguage.googleapis.com`) — the one
    /// Google multi-speaker path callable with a plain API key. The OAuth-only
    /// Cloud TTS `text:synthesize` product is deliberately not used (#40).
    GoogleCloud,
}

impl CloudTtsKind {
    /// Whether this provider accepts W3C SSML in the synthesis input. No wired cloud
    /// engine does: OpenAI takes an `instructions` hint, ElevenLabs takes inline
    /// bracketed audio tags, and Gemini takes natural-language style cues in the turn
    /// text. Consulted per-provider — not in the static catalog DTO (see #195 ADR).
    pub fn supports_ssml(self) -> bool {
        match self {
            CloudTtsKind::OpenAiCompatible
            | CloudTtsKind::Deepgram
            | CloudTtsKind::ElevenLabs
            | CloudTtsKind::GoogleCloud => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsProviderInfo {
    pub backend: TtsBackend,
    pub model: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsPhase {
    Synthesizing { turn: usize, total: usize },
    Stitching,
    Encoding,
}

const CANCELLED_MSG: &str = "tts synthesis cancelled";

type TurnSynthFuture<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<AudioBuffer, LensError>> + Send + 'a>,
>;

pub(crate) async fn synthesize_and_stitch<'t, F>(
    turns: &'t [Turn],
    on_phase: &(dyn Fn(TtsPhase) + Send + Sync),
    cancel: &CancellationToken,
    mut synth_turn: F,
) -> Result<AudioBuffer, LensError>
where
    F: FnMut(&'t Turn) -> TurnSynthFuture<'t>,
{
    let total = turns.len();
    let mut buffers: Vec<(Speaker, AudioBuffer)> = Vec::with_capacity(total);
    for (i, turn) in turns.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(LensError::Cancelled(CANCELLED_MSG.into()));
        }
        on_phase(TtsPhase::Synthesizing { turn: i + 1, total });
        let buf = tokio::select! {
            r = synth_turn(turn) => r?,
            _ = cancel.cancelled() => {
                return Err(LensError::Cancelled(CANCELLED_MSG.into()));
            }
        };
        buffers.push((turn.speaker, buf));
    }
    on_phase(TtsPhase::Stitching);
    audio::stitch_turns(&buffers)
}

#[async_trait]
pub trait TtsProvider: Send + Sync {
    fn info(&self) -> TtsProviderInfo;

    /// Adapter-driven named-voice catalog. Empty when the backend enumerates no
    /// fixed voices (e.g. a clone-only backend).
    fn voices(&self) -> Vec<TtsVoice>;

    async fn synthesize_turn(
        &self,
        turn: &Turn,
        voices: &VoiceConfig,
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer, LensError>;

    // `&dyn Fn` (not `impl Fn`) for object-safety; `+ Sync` because it is held across `.await` in a `Send` future.
    async fn synthesize_script(
        &self,
        script: &DialogueScript,
        voices: &VoiceConfig,
        on_phase: &(dyn Fn(TtsPhase) + Send + Sync),
        cancel: &CancellationToken,
    ) -> Result<AudioBuffer, LensError> {
        synthesize_and_stitch(&script.turns, on_phase, cancel, |turn| {
            self.synthesize_turn(turn, voices, cancel)
        })
        .await
    }
}

/// Whether the user has consented to sending text to a cloud TTS provider
/// (`AppConfig::tts_cloud_consent`). Withheld is the default and a saved API key
/// never implies Granted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudTtsConsent {
    Granted,
    Withheld,
}

impl CloudTtsConsent {
    pub fn from_flag(granted: bool) -> Self {
        if granted {
            Self::Granted
        } else {
            Self::Withheld
        }
    }
}

/// The ONE predicate every cloud-TTS gate consults: resolution, availability, the
/// system check, the voice list and the engine catalog. Sites cannot disagree
/// because there is a single boolean (#273).
pub fn cloud_tts_usable(cfg: &TtsConfig, kind: CloudTtsKind, consent: CloudTtsConsent) -> bool {
    consent == CloudTtsConsent::Granted
        && cfg.clouds.get(&kind).is_some_and(|c| !c.api_key.is_empty())
}

/// User-facing reason a cloud engine is unselectable because consent is withheld.
/// Carries its own remediation — the reason surface renders it verbatim.
pub const CLOUD_TTS_CONSENT_REASON: &str = "Cloud text-to-speech is off. Turn on \"Allow cloud text-to-speech\" in Privacy settings to use this voice.";

/// Resolves a [`TtsProvider`] for `backend`, given an optional injected `sidecar`.
/// Single dispatch path — [`resolve_tts_provider`] and `synthesize_overview` both
/// route through it so the two entry points cannot diverge.
pub fn resolve_tts_provider_full(
    backend: TtsBackend,
    cfg: &TtsConfig,
    consent: CloudTtsConsent,
    cache_root: &Path,
    // Consumed only by the Apple-Silicon-gated `Qwen3Local` arm below.
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(unused_variables)
    )]
    sidecar: Option<Arc<dyn TtsSidecar>>,
) -> Option<Arc<dyn TtsProvider>> {
    match backend {
        TtsBackend::Orpheus => {
            let orpheus = tts_model_path(cache_root, "orpheus")?;
            let snac = tts_model_path(cache_root, "snac")?;
            Some(Arc::new(orpheus::OrpheusAdapter::new(orpheus, snac)))
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        TtsBackend::Qwen3Local => {
            sidecar.map(|s| Arc::new(qwen::QwenLocalAdapter::new(s)) as Arc<dyn TtsProvider>)
        }
        TtsBackend::Cloud(kind) => {
            let creds = cfg.clouds.get(&kind);
            // #40 AC6: unusable (no key, or consent withheld) -> fall back to offline
            // Orpheus iff its weights are on disk (agrees with the availability gate);
            // else None. Selection-time only — a mid-request network failure is a
            // Network error, not a silent swap.
            if !cloud_tts_usable(cfg, kind, consent) {
                if !orpheus_ready(cache_root) {
                    return None;
                }
                let orpheus = tts_model_path(cache_root, "orpheus")?;
                let snac = tts_model_path(cache_root, "snac")?;
                return Some(Arc::new(orpheus::OrpheusAdapter::new(orpheus, snac)));
            }
            let api_key = creds.map(|c| c.api_key.clone()).unwrap_or_default();
            let base_url = creds
                .map(|c| c.base_url.clone())
                .filter(|b| !b.is_empty())
                .unwrap_or_else(|| cloud::default_base_url(kind).to_string());
            let model = if cfg.model.is_empty() {
                cloud::default_model(kind).to_string()
            } else {
                cfg.model.clone()
            };
            Some(Arc::new(cloud::CloudTtsAdapter::new(
                kind, base_url, api_key, model,
            )))
        }
    }
}

/// Whether the offline Orpheus default's weights are fully on disk (exact-size
/// probe). The no-key cloud fallback (`resolve_tts_provider_full`) and the cloud
/// availability gate (`LensEngine::tts_backend_available`) both consult this so the
/// two never disagree about whether a keyless Cloud config can synthesize.
pub(crate) fn orpheus_ready(cache_root: &Path) -> bool {
    TtsBackend::Orpheus
        .required_model_ids()
        .iter()
        .all(|id| tts_model_downloaded(cache_root, id))
}

/// Thin wrapper over [`resolve_tts_provider_full`] with no sidecar; sidecar-backed
/// backends (Qwen3Local) return `None` here by design — call `_full` when one is needed.
pub fn resolve_tts_provider(
    backend: TtsBackend,
    cfg: &TtsConfig,
    consent: CloudTtsConsent,
    cache_root: &Path,
) -> Option<Arc<dyn TtsProvider>> {
    resolve_tts_provider_full(backend, cfg, consent, cache_root, None)
}

/// Single source of truth for rendering an abstract [`Emotion`] per TTS modality:
/// each adapter reads the field for its own capability, so the engine maps can't
/// drift. Unsupported emotions are `None` and degrade to plain delivery.
pub(crate) struct EmotionRender {
    /// Orpheus inline paralinguistic tag (a discrete sound), e.g. `<laugh>`.
    pub orpheus: Option<&'static str>,
    /// ElevenLabs v3 audio tag — DOCUMENTED tags only; an undocumented tag is spoken
    /// literally rather than performed.
    pub elevenlabs: Option<&'static str>,
    /// Natural-language style, fitting `"Speak with {..}."` for the OpenAI
    /// `instructions` field and a per-line parenthetical for Gemini.
    pub style: Option<&'static str>,
}

pub(crate) fn emotion_render(emotion: Emotion) -> EmotionRender {
    let (orpheus, elevenlabs, style) = match emotion {
        Emotion::Neutral => (None, None, None),
        Emotion::Laugh => (
            Some("<laugh>"),
            Some("[laughs]"),
            Some("warm, light laughter"),
        ),
        Emotion::Sigh => (Some("<sigh>"), Some("[sighs]"), Some("a soft, weary sigh")),
        Emotion::Excited => (
            None,
            Some("[excited]"),
            Some("bright, energetic excitement"),
        ),
        Emotion::Thoughtful => (None, None, Some("a measured, thoughtful tone")),
        Emotion::Curious => (None, Some("[curious]"), Some("genuine, engaged curiosity")),
        Emotion::Serious => (None, None, Some("a serious, grounded tone")),
    };
    EmotionRender {
        orpheus,
        elevenlabs,
        style,
    }
}

pub fn emotion_tag(emotion: Emotion, backend: TtsBackend) -> Option<String> {
    match backend {
        TtsBackend::Orpheus => emotion_render(emotion).orpheus.map(str::to_string),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_returns_none_for_non_embedded_backends() {
        // Default config has empty `clouds`, so every Cloud kind dispatches to `None`.
        let cfg = TtsConfig::default();
        let data_dir = Path::new("/data");
        for backend in [
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            TtsBackend::Qwen3Local,
            TtsBackend::Cloud(CloudTtsKind::OpenAiCompatible),
            TtsBackend::Cloud(CloudTtsKind::Deepgram),
            TtsBackend::Cloud(CloudTtsKind::ElevenLabs),
            TtsBackend::Cloud(CloudTtsKind::GoogleCloud),
        ] {
            assert!(
                resolve_tts_provider(backend, &cfg, CloudTtsConsent::Granted, data_dir).is_none()
            );
        }
    }

    fn cloud_cfg(kind: CloudTtsKind, api_key: &str, base_url: &str) -> TtsConfig {
        use crate::config::CloudTtsCreds;
        TtsConfig {
            version: 1,
            backend: TtsBackend::Cloud(kind),
            model: String::new(),
            clouds: std::collections::BTreeMap::from([(
                kind,
                CloudTtsCreds {
                    api_key: api_key.into(),
                    base_url: base_url.into(),
                },
            )]),
        }
    }

    #[test]
    fn resolve_returns_some_for_cloud_with_config() {
        let data_dir = Path::new("/data");
        for kind in [
            CloudTtsKind::OpenAiCompatible,
            CloudTtsKind::Deepgram,
            CloudTtsKind::ElevenLabs,
            CloudTtsKind::GoogleCloud,
        ] {
            let cfg = cloud_cfg(kind, "sk-test", "https://api.example.com");
            let provider = resolve_tts_provider(
                TtsBackend::Cloud(kind),
                &cfg,
                CloudTtsConsent::Granted,
                data_dir,
            )
            .expect("cloud resolves with a cloud config");
            assert_eq!(provider.info().backend, TtsBackend::Cloud(kind));
            // Empty `cfg.model` falls back to the PER-KIND default cloud model.
            assert_eq!(provider.info().model, cloud::default_model(kind));
        }
    }

    /// Fakes both Orpheus weights at their exact pinned sizes (sparse files) so
    /// `orpheus_ready` is true without a real multi-GB download.
    fn fake_orpheus_on_disk(cache_root: &Path) {
        for id in ["orpheus", "snac"] {
            let spec = resolve_tts(id).unwrap();
            let path = tts_model_path(cache_root, id).unwrap();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(&path)
                .unwrap()
                .set_len(spec.size_bytes)
                .unwrap();
        }
    }

    #[test]
    fn resolve_cloud_empty_key_falls_back_to_orpheus_when_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        fake_orpheus_on_disk(dir.path());
        let kind = CloudTtsKind::ElevenLabs;
        let cfg = cloud_cfg(kind, "", "");
        let provider = resolve_tts_provider(
            TtsBackend::Cloud(kind),
            &cfg,
            CloudTtsConsent::Granted,
            dir.path(),
        )
        .expect("empty key falls back to Orpheus when weights are on disk");
        assert_eq!(provider.info().backend, TtsBackend::Orpheus);
    }

    #[test]
    fn resolve_cloud_empty_key_none_when_no_orpheus_on_disk() {
        let kind = CloudTtsKind::GoogleCloud;
        let cfg = cloud_cfg(kind, "", "");
        // No Orpheus weights under /data → no fallback → None.
        assert!(
            resolve_tts_provider(
                TtsBackend::Cloud(kind),
                &cfg,
                CloudTtsConsent::Granted,
                Path::new("/data")
            )
            .is_none()
        );
    }

    #[test]
    fn cloud_tts_usable_requires_both_consent_and_a_key() {
        let kind = CloudTtsKind::OpenAiCompatible;
        let keyed = cloud_cfg(kind, "sk-test", "https://api.example.com");
        let keyless = cloud_cfg(kind, "", "https://api.example.com");
        for (cfg, consent, expected) in [
            (&keyed, CloudTtsConsent::Granted, true),
            (&keyed, CloudTtsConsent::Withheld, false),
            (&keyless, CloudTtsConsent::Granted, false),
            (&keyless, CloudTtsConsent::Withheld, false),
        ] {
            assert_eq!(
                cloud_tts_usable(cfg, kind, consent),
                expected,
                "consent {consent:?} with key {:?}",
                cfg.clouds.get(&kind).map(|c| c.api_key.as_str())
            );
        }
        // A key saved for a DIFFERENT provider never enables this one.
        assert!(!cloud_tts_usable(
            &keyed,
            CloudTtsKind::Deepgram,
            CloudTtsConsent::Granted
        ));
    }

    #[test]
    fn resolve_cloud_with_key_but_consent_withheld_takes_the_orpheus_fallback() {
        let dir = tempfile::tempdir().unwrap();
        fake_orpheus_on_disk(dir.path());
        let kind = CloudTtsKind::OpenAiCompatible;
        let cfg = cloud_cfg(kind, "sk-test", "https://api.example.com");
        let provider = resolve_tts_provider(
            TtsBackend::Cloud(kind),
            &cfg,
            CloudTtsConsent::Withheld,
            dir.path(),
        )
        .expect("withheld consent degrades to Orpheus when its weights are on disk");
        assert_eq!(provider.info().backend, TtsBackend::Orpheus);
    }

    #[test]
    fn resolve_cloud_with_key_but_consent_withheld_is_none_without_orpheus() {
        let kind = CloudTtsKind::OpenAiCompatible;
        let cfg = cloud_cfg(kind, "sk-test", "https://api.example.com");
        assert!(
            resolve_tts_provider(
                TtsBackend::Cloud(kind),
                &cfg,
                CloudTtsConsent::Withheld,
                Path::new("/data")
            )
            .is_none(),
            "a keyed cloud backend must never resolve to a cloud provider without consent"
        );
    }

    #[test]
    fn resolve_cloud_with_key_applies_per_kind_base_url_and_model_defaults() {
        let kind = CloudTtsKind::GoogleCloud;
        let cfg = cloud_cfg(kind, "key", "");
        let provider = resolve_tts_provider(
            TtsBackend::Cloud(kind),
            &cfg,
            CloudTtsConsent::Granted,
            Path::new("/data"),
        )
        .expect("keyed cloud resolves");
        assert_eq!(provider.info().model, cloud::default_model(kind));
    }

    #[test]
    fn resolve_returns_orpheus_adapter_cheaply() {
        // Cheap construct: an adapter is returned even when the weights are
        // absent (paths only, no load); availability is a separate file probe.
        let cfg = TtsConfig::default();
        let provider = resolve_tts_provider(
            TtsBackend::Orpheus,
            &cfg,
            CloudTtsConsent::Granted,
            Path::new("/data"),
        )
        .expect("orpheus resolves to an adapter");
        assert_eq!(provider.info().backend, TtsBackend::Orpheus);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn resolve_full_qwen3_local_needs_sidecar() {
        let cfg = TtsConfig::default();
        let data_dir = Path::new("/data");
        assert!(
            resolve_tts_provider_full(
                TtsBackend::Qwen3Local,
                &cfg,
                CloudTtsConsent::Granted,
                data_dir,
                None
            )
            .is_none()
        );
        assert!(
            resolve_tts_provider(
                TtsBackend::Qwen3Local,
                &cfg,
                CloudTtsConsent::Granted,
                data_dir
            )
            .is_none()
        );

        let sidecar: Arc<dyn TtsSidecar> = Arc::new(NoopSidecar);
        let provider = resolve_tts_provider_full(
            TtsBackend::Qwen3Local,
            &cfg,
            CloudTtsConsent::Granted,
            data_dir,
            Some(sidecar),
        )
        .expect("qwen3_local resolves with a sidecar");
        assert_eq!(provider.info().backend, TtsBackend::Qwen3Local);
        assert_eq!(provider.info().model, "qwen3-tts-customvoice");
    }

    #[test]
    fn resolve_orpheus_via_wrapper_ignores_absent_sidecar() {
        let cfg = TtsConfig::default();
        let provider = resolve_tts_provider(
            TtsBackend::Orpheus,
            &cfg,
            CloudTtsConsent::Granted,
            Path::new("/data"),
        )
        .expect("orpheus resolves without a sidecar");
        assert_eq!(provider.info().backend, TtsBackend::Orpheus);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    struct NoopSidecar;

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[async_trait]
    impl crate::tts::sidecar::TtsSidecar for NoopSidecar {
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
            Ok(AudioBuffer::mono(vec![0.0; 8], audio::TARGET_RATE))
        }
    }

    #[test]
    fn emotion_tag_orpheus_table() {
        assert_eq!(emotion_tag(Emotion::Neutral, TtsBackend::Orpheus), None);
        assert_eq!(
            emotion_tag(Emotion::Laugh, TtsBackend::Orpheus).as_deref(),
            Some("<laugh>")
        );
        assert_eq!(
            emotion_tag(Emotion::Sigh, TtsBackend::Orpheus).as_deref(),
            Some("<sigh>")
        );
        assert_eq!(emotion_tag(Emotion::Excited, TtsBackend::Orpheus), None);
        assert_eq!(emotion_tag(Emotion::Thoughtful, TtsBackend::Orpheus), None);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn emotion_tag_none_for_non_orpheus_backends() {
        for emotion in [
            Emotion::Neutral,
            Emotion::Laugh,
            Emotion::Sigh,
            Emotion::Excited,
            Emotion::Thoughtful,
        ] {
            assert!(emotion_tag(emotion, TtsBackend::Qwen3Local).is_none());
        }
    }

    #[test]
    fn backend_default_is_orpheus() {
        assert_eq!(TtsBackend::default(), TtsBackend::Orpheus);
    }

    #[test]
    fn cloud_kind_ssml_capability() {
        // No wired cloud engine consumes W3C SSML (ElevenLabs + Gemini use inline
        // bracketed audio cues; OpenAI uses an instructions hint).
        assert!(!CloudTtsKind::OpenAiCompatible.supports_ssml());
        assert!(!CloudTtsKind::Deepgram.supports_ssml());
        assert!(!CloudTtsKind::ElevenLabs.supports_ssml());
        assert!(!CloudTtsKind::GoogleCloud.supports_ssml());
    }

    #[test]
    fn gender_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Gender::Male).unwrap(), "\"male\"");
        assert_eq!(
            serde_json::to_string(&Gender::Female).unwrap(),
            "\"female\""
        );
    }

    #[test]
    fn backend_as_str_and_from_opt_str_round_trip() {
        // The Qwen3Local variant is cfg-gated, so off Apple Silicon this array is
        // a single element — the loop shape is target-dependent, not a mistake.
        #[allow(clippy::single_element_loop)]
        for b in [
            TtsBackend::Orpheus,
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            TtsBackend::Qwen3Local,
        ] {
            assert_eq!(TtsBackend::from_opt_str(Some(b.as_str())), b);
        }
        assert_eq!(
            TtsBackend::Cloud(CloudTtsKind::ElevenLabs).as_str(),
            "cloud"
        );
        assert_eq!(
            TtsBackend::from_opt_str(Some("cloud")),
            TtsBackend::Cloud(CloudTtsKind::default())
        );
        assert_eq!(TtsBackend::from_opt_str(None), TtsBackend::Orpheus);
        assert_eq!(TtsBackend::from_opt_str(Some("")), TtsBackend::Orpheus);
        assert_eq!(TtsBackend::from_opt_str(Some("nope")), TtsBackend::Orpheus);
    }

    #[test]
    fn backend_serde_round_trips_including_cloud() {
        for b in [
            TtsBackend::Orpheus,
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            TtsBackend::Qwen3Local,
            TtsBackend::Cloud(CloudTtsKind::ElevenLabs),
        ] {
            let json = serde_json::to_string(&b).unwrap();
            let back: TtsBackend = serde_json::from_str(&json).unwrap();
            assert_eq!(b, back);
        }
    }

    struct FakeProvider;

    #[async_trait]
    impl TtsProvider for FakeProvider {
        fn info(&self) -> TtsProviderInfo {
            TtsProviderInfo {
                backend: TtsBackend::Orpheus,
                model: "fake".to_string(),
            }
        }
        fn voices(&self) -> Vec<TtsVoice> {
            Vec::new()
        }
        async fn synthesize_turn(
            &self,
            _turn: &Turn,
            _voices: &VoiceConfig,
            _cancel: &CancellationToken,
        ) -> Result<AudioBuffer, LensError> {
            Ok(AudioBuffer::mono(vec![0.5; 1000], audio::TARGET_RATE))
        }
    }

    #[tokio::test]
    async fn default_synthesize_script_stitches_behind_arc_dyn() {
        let provider: Arc<dyn TtsProvider> = Arc::new(FakeProvider);
        let script = DialogueScript {
            turns: vec![
                Turn {
                    speaker: Speaker::Host,
                    text: "a".into(),
                    emotion: None,
                    source_ids: Vec::new(),
                },
                Turn {
                    speaker: Speaker::Guest,
                    text: "b".into(),
                    emotion: None,
                    source_ids: Vec::new(),
                },
            ],
        };
        let voices = VoiceConfig::default();
        let cancel = CancellationToken::new();
        let phases = std::sync::Mutex::new(Vec::new());
        let on_phase = |p: TtsPhase| phases.lock().unwrap().push(p);
        let out = provider
            .synthesize_script(&script, &voices, &on_phase, &cancel)
            .await
            .unwrap();
        assert_eq!(out.samples.len(), 1000 + 10_800 + 1000);
        let recorded = phases.lock().unwrap();
        assert_eq!(recorded[0], TtsPhase::Synthesizing { turn: 1, total: 2 });
        assert_eq!(recorded[1], TtsPhase::Synthesizing { turn: 2, total: 2 });
        assert_eq!(recorded[2], TtsPhase::Stitching);
    }

    #[tokio::test]
    async fn synthesize_script_honors_cancel() {
        let provider: Arc<dyn TtsProvider> = Arc::new(FakeProvider);
        let script = DialogueScript {
            turns: vec![Turn {
                speaker: Speaker::Host,
                text: "a".into(),
                emotion: None,
                source_ids: Vec::new(),
            }],
        };
        let voices = VoiceConfig::default();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let noop = |_p: TtsPhase| {};
        let err = provider
            .synthesize_script(&script, &voices, &noop, &cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, LensError::Cancelled(_)));
    }
}
