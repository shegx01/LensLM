//! Static per-engine TTS capability catalog + language model (#194).
//!
//! Single source of truth for the engine selector and the language guard;
//! keyed by a non-cfg-gated [`TtsEngineId`] so every engine is enumerable on
//! every platform (the cfg-gated [`TtsBackend`] is used only for dispatch).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::error::LensError;
use crate::tts::{
    CLOUD_TTS_CONSENT_REASON, CloudTtsConsent, CloudTtsKind, Gender, TtsBackend, TtsVoice,
};

/// Lives here (not the Apple-Silicon-gated `qwen` adapter) so the catalog can
/// enumerate presets on every platform.
pub struct QwenVoice {
    pub id: &'static str,
    pub display_name: &'static str,
    pub gender: Gender,
    pub instruct: &'static str,
}

/// Default instruct applied to every preset until per-preset tuning lands: an
/// energetic podcast-host delivery (the same string benchmarked in the spike).
const DEFAULT_INSTRUCT: &str = "Upbeat, energetic podcast host, conversational and lively.";

/// The four surfaced Qwen3-TTS CustomVoice presets (the model supports more via
/// `get_supported_speakers()`). Ids are the model's canonical lowercase speaker
/// ids; the sidecar resolves them case-insensitively.
pub static QWEN_VOICES: &[QwenVoice] = &[
    QwenVoice {
        id: "dylan",
        display_name: "Dylan",
        gender: Gender::Male,
        instruct: DEFAULT_INSTRUCT,
    },
    QwenVoice {
        id: "aiden",
        display_name: "Aiden",
        gender: Gender::Male,
        instruct: DEFAULT_INSTRUCT,
    },
    QwenVoice {
        id: "serena",
        display_name: "Serena",
        gender: Gender::Female,
        instruct: DEFAULT_INSTRUCT,
    },
    QwenVoice {
        // Canonical model speaker id is "ono_anna" (not "anna") — do not normalize.
        id: "ono_anna",
        display_name: "Anna",
        gender: Gender::Female,
        instruct: DEFAULT_INSTRUCT,
    },
];

/// Used by `src-tauri` to map a `VoiceRef::Named(id)` to its speaker id + instruct string.
pub fn qwen_voice(id: &str) -> Option<&'static QwenVoice> {
    QWEN_VOICES.iter().find(|v| v.id == id)
}

/// A guard-comparable language: Qwen3-TTS's 10 plus a few common others. Anything
/// outside this set maps to `None` and is treated permissively (see
/// [`evaluate_language_guard`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Lang {
    // Qwen3-TTS supported set (runtime-authoritative source is the model's own
    // `get_supported_languages()`; this catalog hardcodes the shipped 10).
    English,
    Chinese,
    German,
    Italian,
    Portuguese,
    Spanish,
    Japanese,
    Korean,
    French,
    Russian,
    // Common languages whatlang confirms but no LOCAL engine supports yet. Present
    // so the guard blocks them (rather than silently passing) and so Qwen-language
    // validation has a reachable "unsupported" branch.
    Dutch,
    Arabic,
    Hindi,
}

const ORPHEUS_LANGS: &[Lang] = &[Lang::English];

const QWEN_LANGS: &[Lang] = &[
    Lang::Chinese,
    Lang::English,
    Lang::German,
    Lang::Italian,
    Lang::Portuguese,
    Lang::Spanish,
    Lang::Japanese,
    Lang::Korean,
    Lang::French,
    Lang::Russian,
];

/// Maps a whatlang ISO 639-3 code to a guard-comparable [`Lang`]. Unknown codes
/// (a language outside our capability set) return `None` → permissive.
pub fn code_to_lang(code: &str) -> Option<Lang> {
    Some(match code {
        "eng" => Lang::English,
        "cmn" => Lang::Chinese,
        "deu" => Lang::German,
        "ita" => Lang::Italian,
        "por" => Lang::Portuguese,
        "spa" => Lang::Spanish,
        "jpn" => Lang::Japanese,
        "kor" => Lang::Korean,
        "fra" => Lang::French,
        "rus" => Lang::Russian,
        "nld" => Lang::Dutch,
        "arb" => Lang::Arabic,
        "hin" => Lang::Hindi,
        _ => return None,
    })
}

/// The lowercase full name Qwen3-TTS expects for its `language=` param, or `None`
/// if this language is outside Qwen's supported set.
pub fn lang_to_qwen_name(lang: Lang) -> Option<&'static str> {
    Some(match lang {
        Lang::English => "english",
        Lang::Chinese => "chinese",
        Lang::German => "german",
        Lang::Italian => "italian",
        Lang::Portuguese => "portuguese",
        Lang::Spanish => "spanish",
        Lang::Japanese => "japanese",
        Lang::Korean => "korean",
        Lang::French => "french",
        Lang::Russian => "russian",
        Lang::Dutch | Lang::Arabic | Lang::Hindi => return None,
    })
}

/// Validates a language against Qwen3-TTS's supported set at the trust boundary:
/// supported → the lowercase Qwen name; unsupported → [`LensError::Tts`]. Pure,
/// no IO. Once #28/#161 threads a real `Turn.language`, the adapter WILL call this
/// before the sidecar; until then the request sends `"auto"`.
pub fn validate_qwen_language(lang: Lang) -> Result<&'static str, LensError> {
    lang_to_qwen_name(lang).ok_or_else(|| {
        LensError::Tts(format!(
            "language {lang:?} is not supported by the Qwen3-TTS engine"
        ))
    })
}

/// A non-cfg-gated engine identity, enumerable on every platform. Distinct from
/// the cfg-gated [`TtsBackend`] dispatch enum. Each cloud provider is its own
/// engine (#40) so the selector lists them alongside Orpheus/Qwen and each is
/// configured separately; the cloud variants' serde names match [`CloudTtsKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsEngineId {
    Orpheus,
    Qwen3Local,
    OpenAiCompatible,
    Deepgram,
    ElevenLabs,
    GoogleCloud,
}

impl TtsEngineId {
    /// The cloud provider kind this engine dispatches to, or `None` for a local
    /// engine. The one place the engine↔kind mapping lives.
    fn cloud_kind(self) -> Option<CloudTtsKind> {
        match self {
            TtsEngineId::OpenAiCompatible => Some(CloudTtsKind::OpenAiCompatible),
            TtsEngineId::Deepgram => Some(CloudTtsKind::Deepgram),
            TtsEngineId::ElevenLabs => Some(CloudTtsKind::ElevenLabs),
            TtsEngineId::GoogleCloud => Some(CloudTtsKind::GoogleCloud),
            TtsEngineId::Orpheus | TtsEngineId::Qwen3Local => None,
        }
    }

    fn from_cloud_kind(kind: CloudTtsKind) -> TtsEngineId {
        match kind {
            CloudTtsKind::OpenAiCompatible => TtsEngineId::OpenAiCompatible,
            CloudTtsKind::Deepgram => TtsEngineId::Deepgram,
            CloudTtsKind::ElevenLabs => TtsEngineId::ElevenLabs,
            CloudTtsKind::GoogleCloud => TtsEngineId::GoogleCloud,
        }
    }

    /// Bridge to the cfg-gated dispatch enum. PARTIAL, not a bijection: off Apple
    /// Silicon `TtsBackend::Qwen3Local` does not exist → `None`.
    pub fn to_backend(self) -> Option<TtsBackend> {
        match self {
            TtsEngineId::Orpheus => Some(TtsBackend::Orpheus),
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            TtsEngineId::Qwen3Local => Some(TtsBackend::Qwen3Local),
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            TtsEngineId::Qwen3Local => None,
            other => other.cloud_kind().map(TtsBackend::Cloud),
        }
    }

    /// Collapse a dispatch backend to its catalog identity. `Cloud(kind)` maps to
    /// the matching per-provider engine id.
    pub fn from_backend(backend: &TtsBackend) -> TtsEngineId {
        match backend {
            TtsBackend::Orpheus => TtsEngineId::Orpheus,
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            TtsBackend::Qwen3Local => TtsEngineId::Qwen3Local,
            TtsBackend::Cloud(kind) => TtsEngineId::from_cloud_kind(*kind),
        }
    }

    fn language_support(self) -> LanguageSupport {
        match self {
            TtsEngineId::Orpheus => LanguageSupport::Set(ORPHEUS_LANGS),
            TtsEngineId::Qwen3Local => LanguageSupport::Set(QWEN_LANGS),
            TtsEngineId::OpenAiCompatible
            | TtsEngineId::Deepgram
            | TtsEngineId::ElevenLabs
            | TtsEngineId::GoogleCloud => LanguageSupport::Multilingual,
        }
    }

    /// Preset named voices for this engine's selector display. Derived from the
    /// canonical voice lists (no duplication): Orpheus/Qwen from their adapter
    /// catalogs, and each cloud provider from its own curated voice set.
    pub fn preset_voices(self) -> Vec<TtsVoice> {
        fn from_tuples(voices: &[(&'static str, &'static str, Gender)]) -> Vec<TtsVoice> {
            voices
                .iter()
                .map(|&(id, name, gender)| TtsVoice::new(id, name, gender))
                .collect()
        }
        match self {
            TtsEngineId::Orpheus => from_tuples(crate::tts::orpheus::ORPHEUS_VOICES),
            TtsEngineId::Qwen3Local => QWEN_VOICES
                .iter()
                .map(|v| TtsVoice::new(v.id, v.display_name, v.gender))
                .collect(),
            TtsEngineId::OpenAiCompatible | TtsEngineId::Deepgram => {
                from_tuples(crate::tts::cloud::OPENAI_VOICES)
            }
            TtsEngineId::ElevenLabs => {
                from_tuples(crate::tts::cloud::elevenlabs::ELEVENLABS_VOICES)
            }
            TtsEngineId::GoogleCloud => from_tuples(crate::tts::cloud::google::GEMINI_VOICES),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Platform {
    CrossPlatform,
    AppleSilicon,
}

/// An engine's language capability: a concrete supported set, or the multilingual
/// (provider-defined) marker used by the Cloud reserved slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageSupport {
    Set(&'static [Lang]),
    Multilingual,
}

/// Static per-engine capability. The single source of truth for the selector and
/// the guard. `preset_voices` is derived on demand via [`TtsEngineId::preset_voices`]
/// (avoids duplicating the canonical voice lists into a `&'static` field).
#[derive(Debug, Clone, Copy)]
pub struct EngineCapability {
    pub id: TtsEngineId,
    pub platform: Platform,
    pub needs_key: bool,
    pub languages: LanguageSupport,
    /// Approximate on-disk download size for the always-visible size label.
    pub model_size_bytes: Option<u64>,
    pub language_capability_label: &'static str,
}

// Orpheus = the Q4_K_M 3B GGUF (~2.3 GB; the paired SNAC decoder is small).
// Qwen3Local = the mlx-community CustomVoice weights mlx-audio fetches lazily (~4.5 GB).
const ORPHEUS_SIZE_BYTES: u64 = 2_300_000_000;
const QWEN_SIZE_BYTES: u64 = 4_500_000_000;

static CATALOG: &[EngineCapability] = &[
    EngineCapability {
        id: TtsEngineId::Orpheus,
        platform: Platform::CrossPlatform,
        needs_key: false,
        languages: LanguageSupport::Set(ORPHEUS_LANGS),
        model_size_bytes: Some(ORPHEUS_SIZE_BYTES),
        language_capability_label: "English only",
    },
    EngineCapability {
        id: TtsEngineId::Qwen3Local,
        platform: Platform::AppleSilicon,
        needs_key: false,
        languages: LanguageSupport::Set(QWEN_LANGS),
        model_size_bytes: Some(QWEN_SIZE_BYTES),
        language_capability_label: "10 languages",
    },
    EngineCapability {
        id: TtsEngineId::OpenAiCompatible,
        platform: Platform::CrossPlatform,
        needs_key: true,
        languages: LanguageSupport::Multilingual,
        model_size_bytes: None,
        language_capability_label: "OpenAI-compatible · multilingual",
    },
    EngineCapability {
        id: TtsEngineId::ElevenLabs,
        platform: Platform::CrossPlatform,
        needs_key: true,
        languages: LanguageSupport::Multilingual,
        model_size_bytes: None,
        language_capability_label: "ElevenLabs dialogue · multilingual",
    },
    EngineCapability {
        id: TtsEngineId::GoogleCloud,
        platform: Platform::CrossPlatform,
        needs_key: true,
        languages: LanguageSupport::Multilingual,
        model_size_bytes: None,
        language_capability_label: "Google multi-speaker · multilingual",
    },
];

/// All engines, for the selector and the guard. Contains every engine on every
/// platform (Qwen included off Apple Silicon so it can be shown "unavailable").
pub fn tts_catalog() -> &'static [EngineCapability] {
    CATALOG
}

/// A serialized catalog entry for the frontend selector (IPC DTO). Carries the
/// runtime-resolved `available`/`unavailable_reason` (Qwen off Apple Silicon,
/// Cloud without a key) plus the display metadata the selector needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EngineCatalogEntry {
    pub id: TtsEngineId,
    pub platform: Platform,
    pub needs_key: bool,
    /// Selectable on this build with the current config.
    pub available: bool,
    /// Why not, when `available` is false (platform or missing key).
    pub unavailable_reason: Option<String>,
    /// `true` for the Cloud reserved slot (provider-defined language set).
    pub multilingual: bool,
    /// Concrete supported languages; empty when `multilingual`.
    pub supported_languages: Vec<Lang>,
    pub preset_voices: Vec<TtsVoice>,
    pub model_size_bytes: Option<u64>,
    pub language_capability_label: String,
    /// Registry model ids this engine needs on disk (authority: [`TtsBackend::required_model_ids`]).
    pub required_model_ids: Vec<String>,
}

impl EngineCatalogEntry {
    fn from_capability(
        cap: &EngineCapability,
        keyed_cloud_kinds: &BTreeSet<CloudTtsKind>,
        consent: CloudTtsConsent,
    ) -> Self {
        let platform_available = cap.id.to_backend().is_some();
        // A cloud engine is available once ITS OWN provider kind has a saved key.
        let has_key = cap
            .id
            .cloud_kind()
            .is_some_and(|k| keyed_cloud_kinds.contains(&k));
        // Consent precedes the key gate: with consent withheld, "Requires an API key"
        // would send the user to add a key that still would not enable the engine.
        let (available, unavailable_reason) = if !platform_available {
            (false, Some("Requires Apple Silicon".to_string()))
        } else if cap.needs_key && consent == CloudTtsConsent::Withheld {
            (false, Some(CLOUD_TTS_CONSENT_REASON.to_string()))
        } else if cap.needs_key && !has_key {
            (false, Some("Requires an API key".to_string()))
        } else {
            (true, None)
        };
        let (multilingual, supported_languages) = match cap.languages {
            LanguageSupport::Multilingual => (true, Vec::new()),
            LanguageSupport::Set(set) => (false, set.to_vec()),
        };
        EngineCatalogEntry {
            id: cap.id,
            platform: cap.platform,
            needs_key: cap.needs_key,
            available,
            unavailable_reason,
            multilingual,
            supported_languages,
            preset_voices: cap.id.preset_voices(),
            model_size_bytes: cap.model_size_bytes,
            language_capability_label: cap.language_capability_label.to_string(),
            required_model_ids: cap
                .id
                .to_backend()
                .map(|b| {
                    b.required_model_ids()
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

/// Serializes the full catalog for the frontend selector, resolving availability
/// against this build (Qwen platform) and `keyed_cloud_kinds` — the set of cloud
/// provider kinds that currently have a saved key (each gates its own engine row).
pub fn tts_catalog_serialized(
    keyed_cloud_kinds: &BTreeSet<CloudTtsKind>,
    consent: CloudTtsConsent,
) -> Vec<EngineCatalogEntry> {
    CATALOG
        .iter()
        .map(|cap| EngineCatalogEntry::from_capability(cap, keyed_cloud_kinds, consent))
        .collect()
}

/// A source whose confirmed language is outside the selected engine's set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OffendingSource {
    pub source_id: String,
    pub language: Lang,
}

/// The engine-aware language-guard outcome. Serde-serializable for IPC/UI reuse
/// (the #28/#161 synthesis button mounts the inline-reason component on it).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardVerdict {
    pub allow: bool,
    pub reason: Option<String>,
    pub offending: Vec<OffendingSource>,
}

/// Allow iff every source's CONFIRMED language is in `engine`'s supported set.
/// `None`/unknown is PERMISSIVE (never blocks): pre-migration sources have no
/// detected language, so a blocking default would disable synthesis everywhere.
pub fn evaluate_language_guard(
    engine: TtsEngineId,
    sources: &[(String, Option<Lang>)],
) -> GuardVerdict {
    let offending: Vec<OffendingSource> = match engine.language_support() {
        LanguageSupport::Multilingual => Vec::new(),
        LanguageSupport::Set(set) => sources
            .iter()
            .filter_map(|(id, lang)| match lang {
                Some(l) if !set.contains(l) => Some(OffendingSource {
                    source_id: id.clone(),
                    language: *l,
                }),
                _ => None,
            })
            .collect(),
    };

    if offending.is_empty() {
        GuardVerdict {
            allow: true,
            reason: None,
            offending,
        }
    } else {
        let names: Vec<String> = offending
            .iter()
            .map(|o| format!("{} ({:?})", o.source_id, o.language))
            .collect();
        GuardVerdict {
            allow: false,
            reason: Some(format!(
                "The selected engine cannot synthesize the language of: {}",
                names.join(", ")
            )),
            offending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_lists_the_selectable_engines_with_expected_language_sets() {
        let catalog = tts_catalog();
        // Orpheus + Qwen + three user-selectable cloud providers (Deepgram reserved).
        assert_eq!(catalog.len(), 5);
        assert!(!catalog.iter().any(|c| c.id == TtsEngineId::Deepgram));

        let orpheus = catalog
            .iter()
            .find(|c| c.id == TtsEngineId::Orpheus)
            .unwrap();
        assert_eq!(orpheus.languages, LanguageSupport::Set(&[Lang::English]));
        assert_eq!(orpheus.language_capability_label, "English only");
        assert!(!orpheus.needs_key);
        assert_eq!(orpheus.platform, Platform::CrossPlatform);

        let qwen = catalog
            .iter()
            .find(|c| c.id == TtsEngineId::Qwen3Local)
            .unwrap();
        assert_eq!(qwen.platform, Platform::AppleSilicon);
        assert!(!qwen.needs_key);
        match qwen.languages {
            LanguageSupport::Set(set) => assert_eq!(set.len(), 10),
            LanguageSupport::Multilingual => panic!("qwen must be a concrete set"),
        }

        for id in [
            TtsEngineId::OpenAiCompatible,
            TtsEngineId::ElevenLabs,
            TtsEngineId::GoogleCloud,
        ] {
            let cloud = catalog.iter().find(|c| c.id == id).unwrap();
            assert_eq!(cloud.languages, LanguageSupport::Multilingual);
            assert!(cloud.needs_key);
            assert!(cloud.model_size_bytes.is_none());
        }
    }

    #[test]
    fn engine_id_from_backend_maps_each_cloud_kind() {
        assert_eq!(
            TtsEngineId::from_backend(&TtsBackend::Orpheus),
            TtsEngineId::Orpheus
        );
        assert_eq!(
            TtsEngineId::from_backend(&TtsBackend::Cloud(CloudTtsKind::ElevenLabs)),
            TtsEngineId::ElevenLabs
        );
        assert_eq!(
            TtsEngineId::from_backend(&TtsBackend::Cloud(CloudTtsKind::GoogleCloud)),
            TtsEngineId::GoogleCloud
        );
        assert_eq!(
            TtsEngineId::from_backend(&TtsBackend::Cloud(CloudTtsKind::Deepgram)),
            TtsEngineId::Deepgram
        );
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(
            TtsEngineId::from_backend(&TtsBackend::Qwen3Local),
            TtsEngineId::Qwen3Local
        );
    }

    #[test]
    fn engine_id_to_backend_round_trips_cloud_kinds_and_is_partial_for_qwen() {
        assert_eq!(TtsEngineId::Orpheus.to_backend(), Some(TtsBackend::Orpheus));
        for kind in [
            CloudTtsKind::OpenAiCompatible,
            CloudTtsKind::Deepgram,
            CloudTtsKind::ElevenLabs,
            CloudTtsKind::GoogleCloud,
        ] {
            assert_eq!(
                TtsEngineId::from_cloud_kind(kind).to_backend(),
                Some(TtsBackend::Cloud(kind))
            );
        }
        // Qwen3Local resolves only on Apple Silicon (cfg-gated backend variant).
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert!(TtsEngineId::Qwen3Local.to_backend().is_some());
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        assert!(TtsEngineId::Qwen3Local.to_backend().is_none());
    }

    #[test]
    fn code_to_lang_maps_known_and_rejects_unknown() {
        assert_eq!(code_to_lang("eng"), Some(Lang::English));
        assert_eq!(code_to_lang("cmn"), Some(Lang::Chinese));
        assert_eq!(code_to_lang("deu"), Some(Lang::German));
        assert_eq!(code_to_lang("nld"), Some(Lang::Dutch));
        assert_eq!(code_to_lang("zzz"), None);
        assert_eq!(code_to_lang(""), None);
    }

    #[test]
    fn lang_to_qwen_name_covers_the_ten_lowercase() {
        for (lang, name) in [
            (Lang::English, "english"),
            (Lang::Chinese, "chinese"),
            (Lang::German, "german"),
            (Lang::Italian, "italian"),
            (Lang::Portuguese, "portuguese"),
            (Lang::Spanish, "spanish"),
            (Lang::Japanese, "japanese"),
            (Lang::Korean, "korean"),
            (Lang::French, "french"),
            (Lang::Russian, "russian"),
        ] {
            assert_eq!(lang_to_qwen_name(lang), Some(name));
        }
        assert_eq!(lang_to_qwen_name(Lang::Dutch), None);
    }

    #[test]
    fn validate_qwen_language_accepts_supported_rejects_others() {
        assert_eq!(validate_qwen_language(Lang::German).unwrap(), "german");
        assert_eq!(validate_qwen_language(Lang::English).unwrap(), "english");
        assert!(matches!(
            validate_qwen_language(Lang::Dutch),
            Err(LensError::Tts(_))
        ));
    }

    #[test]
    fn guard_orpheus_blocks_non_english_names_offenders() {
        let sources = vec![
            ("s1".to_string(), Some(Lang::English)),
            ("s2".to_string(), Some(Lang::German)),
        ];
        let v = evaluate_language_guard(TtsEngineId::Orpheus, &sources);
        assert!(!v.allow);
        assert_eq!(v.offending.len(), 1);
        assert_eq!(v.offending[0].source_id, "s2");
        assert_eq!(v.offending[0].language, Lang::German);
        assert!(v.reason.as_ref().unwrap().contains("s2"));
    }

    #[test]
    fn guard_orpheus_allows_all_english() {
        let sources = vec![
            ("s1".to_string(), Some(Lang::English)),
            ("s2".to_string(), Some(Lang::English)),
        ];
        let v = evaluate_language_guard(TtsEngineId::Orpheus, &sources);
        assert!(v.allow);
        assert!(v.offending.is_empty());
    }

    #[test]
    fn guard_qwen_allows_its_set_blocks_others() {
        let allowed = vec![
            ("s1".to_string(), Some(Lang::German)),
            ("s2".to_string(), Some(Lang::Japanese)),
            ("s3".to_string(), Some(Lang::English)),
        ];
        assert!(evaluate_language_guard(TtsEngineId::Qwen3Local, &allowed).allow);

        let blocked = vec![("s4".to_string(), Some(Lang::Dutch))];
        let v = evaluate_language_guard(TtsEngineId::Qwen3Local, &blocked);
        assert!(!v.allow);
        assert_eq!(v.offending.len(), 1);
    }

    #[test]
    fn guard_cloud_allows_multilingual() {
        let sources = vec![
            ("s1".to_string(), Some(Lang::Dutch)),
            ("s2".to_string(), Some(Lang::Arabic)),
            ("s3".to_string(), None),
        ];
        for id in [
            TtsEngineId::OpenAiCompatible,
            TtsEngineId::ElevenLabs,
            TtsEngineId::GoogleCloud,
        ] {
            let v = evaluate_language_guard(id, &sources);
            assert!(v.allow);
            assert!(v.offending.is_empty());
        }
    }

    #[test]
    fn guard_null_and_unknown_are_permissive() {
        // All-unknown never blocks, for any engine (permissive default).
        let all_unknown = vec![("s1".to_string(), None), ("s2".to_string(), None)];
        assert!(evaluate_language_guard(TtsEngineId::Orpheus, &all_unknown).allow);

        let mixed = vec![
            ("s1".to_string(), Some(Lang::English)),
            ("s2".to_string(), None),
        ];
        assert!(evaluate_language_guard(TtsEngineId::Orpheus, &mixed).allow);
    }

    #[test]
    fn preset_voices_derive_from_canonical_lists() {
        let orpheus = TtsEngineId::Orpheus.preset_voices();
        assert_eq!(orpheus.len(), crate::tts::orpheus::ORPHEUS_VOICES.len());

        let qwen = TtsEngineId::Qwen3Local.preset_voices();
        assert_eq!(qwen.len(), QWEN_VOICES.len());
        assert!(qwen.iter().any(|v| v.id == "dylan"));

        let openai = TtsEngineId::OpenAiCompatible.preset_voices();
        assert_eq!(openai.len(), crate::tts::cloud::OPENAI_VOICES.len());
        assert!(openai.iter().any(|v| v.id == "alloy"));

        // Each cloud provider surfaces ITS OWN curated voices, not OpenAI's (#40).
        let eleven = TtsEngineId::ElevenLabs.preset_voices();
        assert_eq!(
            eleven.len(),
            crate::tts::cloud::elevenlabs::ELEVENLABS_VOICES.len()
        );
        assert!(eleven.iter().any(|v| v.name == "Rachel"));

        let google = TtsEngineId::GoogleCloud.preset_voices();
        assert_eq!(google.len(), crate::tts::cloud::google::GEMINI_VOICES.len());
        assert!(google.iter().any(|v| v.id == "Kore"));
    }

    /// Every guard-comparable `Lang`, so drift checks can enumerate the full set.
    const ALL_LANGS: &[Lang] = &[
        Lang::English,
        Lang::Chinese,
        Lang::German,
        Lang::Italian,
        Lang::Portuguese,
        Lang::Spanish,
        Lang::Japanese,
        Lang::Korean,
        Lang::French,
        Lang::Russian,
        Lang::Dutch,
        Lang::Arabic,
        Lang::Hindi,
    ];

    #[test]
    fn catalog_language_view_agrees_with_guard() {
        // The "one catalog, no drift" invariant: each entry's stored language view
        // must equal the guard's `language_support()` view for the same engine.
        for cap in tts_catalog() {
            assert_eq!(
                cap.languages,
                cap.id.language_support(),
                "catalog vs guard language drift for {:?}",
                cap.id
            );
        }
    }

    #[test]
    fn qwen_langs_and_qwen_names_do_not_drift() {
        // Guard-allows must never exceed adapter-accepts: every QWEN_LANGS entry
        // maps to a valid Qwen name...
        assert!(QWEN_LANGS.iter().all(|l| lang_to_qwen_name(*l).is_some()));
        // ...and no Qwen name exists for a language outside QWEN_LANGS.
        let named = ALL_LANGS
            .iter()
            .filter(|l| lang_to_qwen_name(**l).is_some())
            .count();
        assert_eq!(named, QWEN_LANGS.len());
    }

    #[test]
    fn qwen_voice_lookup_resolves_preset() {
        let v = qwen_voice("serena").expect("known voice");
        assert_eq!(v.display_name, "Serena");
        assert_eq!(v.gender, Gender::Female);
        assert!(!v.instruct.is_empty());
        assert!(qwen_voice("nope").is_none());
    }

    #[test]
    fn serialized_catalog_resolves_per_kind_availability() {
        let entries = tts_catalog_serialized(&BTreeSet::new(), CloudTtsConsent::Granted);
        assert_eq!(entries.len(), 5);

        let orpheus = entries
            .iter()
            .find(|e| e.id == TtsEngineId::Orpheus)
            .unwrap();
        assert!(orpheus.available);
        assert_eq!(orpheus.preset_voices.len(), 8);
        assert!(!orpheus.multilingual);
        assert_eq!(orpheus.supported_languages, vec![Lang::English]);
        assert_eq!(orpheus.required_model_ids, vec!["orpheus", "snac"]);

        // With no keyed kinds, every cloud engine is unavailable.
        for id in [
            TtsEngineId::OpenAiCompatible,
            TtsEngineId::ElevenLabs,
            TtsEngineId::GoogleCloud,
        ] {
            let cloud = entries.iter().find(|e| e.id == id).unwrap();
            assert!(!cloud.available, "{id:?} without a key must be unavailable");
            assert_eq!(
                cloud.unavailable_reason.as_deref(),
                Some("Requires an API key")
            );
            assert!(cloud.multilingual);
            assert!(cloud.supported_languages.is_empty());
            assert!(cloud.required_model_ids.is_empty());
        }

        // A key for ONE provider enables only that provider's row (#40).
        let one_keyed = tts_catalog_serialized(
            &BTreeSet::from([CloudTtsKind::ElevenLabs]),
            CloudTtsConsent::Granted,
        );
        let eleven = one_keyed
            .iter()
            .find(|e| e.id == TtsEngineId::ElevenLabs)
            .unwrap();
        assert!(eleven.available);
        let google = one_keyed
            .iter()
            .find(|e| e.id == TtsEngineId::GoogleCloud)
            .unwrap();
        assert!(!google.available);

        let qwen = entries
            .iter()
            .find(|e| e.id == TtsEngineId::Qwen3Local)
            .unwrap();
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert!(qwen.available);
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            assert!(!qwen.available);
            assert_eq!(
                qwen.unavailable_reason.as_deref(),
                Some("Requires Apple Silicon")
            );
        }
    }

    #[test]
    fn consent_withheld_beats_the_key_gate_in_the_unavailable_reason() {
        let all_keyed = BTreeSet::from([
            CloudTtsKind::OpenAiCompatible,
            CloudTtsKind::ElevenLabs,
            CloudTtsKind::GoogleCloud,
        ]);
        let entries = tts_catalog_serialized(&all_keyed, CloudTtsConsent::Withheld);

        for id in [
            TtsEngineId::OpenAiCompatible,
            TtsEngineId::ElevenLabs,
            TtsEngineId::GoogleCloud,
        ] {
            let cloud = entries.iter().find(|e| e.id == id).unwrap();
            assert!(
                !cloud.available,
                "{id:?} must be unavailable without consent"
            );
            // Telling a user who already has a key to "add an API key" sends them
            // to a control that would not unblock the engine.
            assert_eq!(
                cloud.unavailable_reason.as_deref(),
                Some(CLOUD_TTS_CONSENT_REASON)
            );
        }

        // Local engines carry no key, so consent never touches their rows.
        let orpheus = entries
            .iter()
            .find(|e| e.id == TtsEngineId::Orpheus)
            .unwrap();
        assert!(orpheus.available);
        assert!(orpheus.unavailable_reason.is_none());
    }

    #[test]
    fn consent_withheld_and_no_key_still_reports_consent_first() {
        let entries = tts_catalog_serialized(&BTreeSet::new(), CloudTtsConsent::Withheld);
        let cloud = entries
            .iter()
            .find(|e| e.id == TtsEngineId::OpenAiCompatible)
            .unwrap();
        assert_eq!(
            cloud.unavailable_reason.as_deref(),
            Some(CLOUD_TTS_CONSENT_REASON)
        );
    }
}
