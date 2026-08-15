//! Pins the `AsrConfig`/`Lang` wire format (#136 Step 0). `Lang` derives
//! `Serialize`/`Deserialize` with no `rename_all`, so the wire token is
//! PascalCase ("En"), not "en" — this is the contract the frontend's
//! `AsrLang` union must match.

use lens_core::config::AsrConfig;
use lens_core::{CloudAsrProvider, Lang};

#[test]
fn language_deserializes_named_variant_as_pascal_case() {
    let cfg: AsrConfig = serde_json::from_str(r#"{ "language": "En" }"#).unwrap();
    assert_eq!(cfg.language, Some(Lang::En));
}

#[test]
fn language_deserializes_other_variant_as_tagged_object() {
    let cfg: AsrConfig = serde_json::from_str(r#"{ "language": {"Other":"ar"} }"#).unwrap();
    assert_eq!(cfg.language, Some(Lang::Other("ar".to_string())));
}

#[test]
fn language_null_and_absent_both_deserialize_to_none() {
    let cfg: AsrConfig = serde_json::from_str(r#"{ "language": null }"#).unwrap();
    assert_eq!(cfg.language, None);

    let cfg: AsrConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(cfg.language, None);
}

#[test]
fn lowercase_language_token_is_rejected() {
    let result: Result<AsrConfig, _> = serde_json::from_str(r#"{ "language": "en" }"#);
    assert!(
        result.is_err(),
        "lowercase \"en\" must not deserialize — the wire token is PascalCase"
    );
}

#[test]
fn language_serializes_named_and_other_variants() {
    let cfg = AsrConfig {
        language: Some(Lang::En),
        ..AsrConfig::default()
    };
    let json = serde_json::to_value(&cfg).unwrap();
    assert_eq!(json["language"], serde_json::json!("En"));

    let cfg = AsrConfig {
        language: Some(Lang::Other("ar".to_string())),
        ..AsrConfig::default()
    };
    let json = serde_json::to_value(&cfg).unwrap();
    assert_eq!(json["language"], serde_json::json!({"Other": "ar"}));
}

#[test]
fn asr_config_round_trips_including_apple_min_confidence() {
    let cfg = AsrConfig {
        backend: "cloud".to_string(),
        whisper_model: "small".to_string(),
        language: Some(Lang::Other("ar".to_string())),
        translate: true,
        cloud_provider: Some(CloudAsrProvider::Deepgram),
        cloud_base_url: "https://api.deepgram.com".to_string(),
        cloud_model: "nova-3".to_string(),
        cloud_api_key: "secret".to_string(),
        apple_min_confidence: 0.8,
    };

    let json = serde_json::to_string(&cfg).unwrap();
    let back: AsrConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back, cfg);
}
