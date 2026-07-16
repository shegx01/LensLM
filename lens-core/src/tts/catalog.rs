//! Static per-engine TTS capability catalog + language model (#194 / 161f).
//!
//! Single source of truth for both the engine selector and the language guard.
//! Keyed by a NON-cfg-gated [`TtsEngineId`] so every engine (incl. `Qwen3Local`
//! off Apple Silicon) is enumerable on every platform; the cfg-gated
//! [`TtsBackend`] is used only for dispatch.
//!
//! The guard/validation/mapping `pub` symbols are exercised by tests and
//! re-exported for callers; they are wired into the live synth path in #28/#161.

use serde::{Deserialize, Serialize};

use crate::error::LensError;
use crate::tts::{CloudTtsKind, Gender, TtsBackend, TtsVoice};

/// A Qwen3-TTS CustomVoice preset: a fixed speaker selected by `id`, delivery
/// steered by an `instruct` string (no reference clip, no transcript).
///
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

/// Resolves a preset voice by id (used by `src-tauri` to map a `VoiceRef::Named(id)`
/// to its speaker id + instruct string).
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

/// Languages the Orpheus backend can synthesize. English only.
const ORPHEUS_LANGS: &[Lang] = &[Lang::English];

/// Languages the Qwen3Local backend can synthesize (the shipped 10).
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
/// the cfg-gated [`TtsBackend`] dispatch enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsEngineId {
    Orpheus,
    Qwen3Local,
    Cloud,
}

impl TtsEngineId {
    /// Bridge to the cfg-gated dispatch enum. PARTIAL + LOSSY, not a bijection:
    /// off Apple Silicon `TtsBackend::Qwen3Local` does not exist → `None`.
    pub fn to_backend(self) -> Option<TtsBackend> {
        match self {
            TtsEngineId::Orpheus => Some(TtsBackend::Orpheus),
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            TtsEngineId::Qwen3Local => Some(TtsBackend::Qwen3Local),
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            TtsEngineId::Qwen3Local => None,
            TtsEngineId::Cloud => Some(TtsBackend::Cloud(CloudTtsKind::default())),
        }
    }

    /// Collapse a dispatch backend to its catalog identity. LOSSY: `Cloud(kind)`
    /// drops `kind` — the catalog identity is provider-agnostic by design.
    pub fn from_backend(backend: &TtsBackend) -> TtsEngineId {
        match backend {
            TtsBackend::Orpheus => TtsEngineId::Orpheus,
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            TtsBackend::Qwen3Local => TtsEngineId::Qwen3Local,
            TtsBackend::Cloud(_) => TtsEngineId::Cloud,
        }
    }

    fn language_support(self) -> LanguageSupport {
        match self {
            TtsEngineId::Orpheus => LanguageSupport::Set(ORPHEUS_LANGS),
            TtsEngineId::Qwen3Local => LanguageSupport::Set(QWEN_LANGS),
            TtsEngineId::Cloud => LanguageSupport::Multilingual,
        }
    }

    /// Preset named voices for this engine's selector display. Derived from the
    /// canonical voice lists (no duplication): Orpheus from its adapter catalog,
    /// Qwen from [`QWEN_VOICES`]. Cloud has no local presets.
    pub fn preset_voices(self) -> Vec<TtsVoice> {
        match self {
            TtsEngineId::Orpheus => crate::tts::orpheus::ORPHEUS_VOICES
                .iter()
                .map(|&(id, name, gender)| TtsVoice::new(id, name, gender))
                .collect(),
            TtsEngineId::Qwen3Local => QWEN_VOICES
                .iter()
                .map(|v| TtsVoice::new(v.id, v.display_name, v.gender))
                .collect(),
            TtsEngineId::Cloud => Vec::new(),
        }
    }
}

/// Platform on which an engine can run.
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

// Approximate download sizes for the always-visible size label (display only).
// Orpheus = the Q4_K_M 3B GGUF (~2.3 GB; the paired SNAC decoder is small).
// Qwen3Local = the mlx-community CustomVoice weights mlx-audio fetches lazily
// (~4.5 GB, per the sidecar). Cloud downloads nothing.
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
        id: TtsEngineId::Cloud,
        platform: Platform::CrossPlatform,
        needs_key: true,
        languages: LanguageSupport::Multilingual,
        model_size_bytes: None,
        language_capability_label: "Multilingual (cloud)",
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
    fn from_capability(cap: &EngineCapability, cloud_key_present: bool) -> Self {
        let platform_available = cap.id.to_backend().is_some();
        let (available, unavailable_reason) = if !platform_available {
            (false, Some("Requires Apple Silicon".to_string()))
        } else if cap.needs_key && !cloud_key_present {
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
/// against this build (Qwen platform) and `cloud_key_present` (Cloud key gate).
pub fn tts_catalog_serialized(cloud_key_present: bool) -> Vec<EngineCatalogEntry> {
    CATALOG
        .iter()
        .map(|cap| EngineCatalogEntry::from_capability(cap, cloud_key_present))
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
    fn catalog_has_three_engines_with_expected_language_sets() {
        let catalog = tts_catalog();
        assert_eq!(catalog.len(), 3);

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

        let cloud = catalog.iter().find(|c| c.id == TtsEngineId::Cloud).unwrap();
        assert_eq!(cloud.languages, LanguageSupport::Multilingual);
        assert!(cloud.needs_key);
        assert!(cloud.model_size_bytes.is_none());
    }

    #[test]
    fn engine_id_from_backend_is_directional_and_collapses_cloud() {
        assert_eq!(
            TtsEngineId::from_backend(&TtsBackend::Orpheus),
            TtsEngineId::Orpheus
        );
        // Cloud(kind) collapses to a payload-less Cloud identity (intended, lossy).
        assert_eq!(
            TtsEngineId::from_backend(&TtsBackend::Cloud(CloudTtsKind::ElevenLabs)),
            TtsEngineId::Cloud
        );
        assert_eq!(
            TtsEngineId::from_backend(&TtsBackend::Cloud(CloudTtsKind::Deepgram)),
            TtsEngineId::Cloud
        );
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(
            TtsEngineId::from_backend(&TtsBackend::Qwen3Local),
            TtsEngineId::Qwen3Local
        );
    }

    #[test]
    fn engine_id_to_backend_is_partial() {
        assert!(TtsEngineId::Orpheus.to_backend().is_some());
        assert!(TtsEngineId::Cloud.to_backend().is_some());
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
        let v = evaluate_language_guard(TtsEngineId::Cloud, &sources);
        assert!(v.allow);
        assert!(v.offending.is_empty());
    }

    #[test]
    fn guard_null_and_unknown_are_permissive() {
        // All-unknown never blocks, for any engine (permissive default).
        let all_unknown = vec![("s1".to_string(), None), ("s2".to_string(), None)];
        assert!(evaluate_language_guard(TtsEngineId::Orpheus, &all_unknown).allow);

        // One unknown among supported does not block either.
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

        assert!(TtsEngineId::Cloud.preset_voices().is_empty());
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
    fn serialized_catalog_resolves_availability() {
        let entries = tts_catalog_serialized(false);
        assert_eq!(entries.len(), 3);

        let orpheus = entries
            .iter()
            .find(|e| e.id == TtsEngineId::Orpheus)
            .unwrap();
        assert!(orpheus.available);
        assert_eq!(orpheus.preset_voices.len(), 8);
        assert!(!orpheus.multilingual);
        assert_eq!(orpheus.supported_languages, vec![Lang::English]);
        assert_eq!(orpheus.required_model_ids, vec!["orpheus", "snac"]);

        let cloud = entries.iter().find(|e| e.id == TtsEngineId::Cloud).unwrap();
        assert!(!cloud.available, "cloud without a key is unavailable");
        assert_eq!(
            cloud.unavailable_reason.as_deref(),
            Some("Requires an API key")
        );
        assert!(cloud.multilingual);
        assert!(cloud.supported_languages.is_empty());
        assert!(cloud.required_model_ids.is_empty());

        // With a key, Cloud becomes available.
        let with_key = tts_catalog_serialized(true);
        let cloud = with_key
            .iter()
            .find(|e| e.id == TtsEngineId::Cloud)
            .unwrap();
        assert!(cloud.available);

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
}
