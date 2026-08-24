//! Pins the `AppConfig::keymap` wire contract (#239): the dotted `ActionId`
//! strings shared with `src/lib/shortcuts/registry.ts`, the tolerant deserializer,
//! and the save/load/save stability that `normalize()` must not disturb.

use std::collections::BTreeMap;

use lens_core::{ActionId, AppConfig};
use serde_json::json;

const ALL_ACTION_IDS: [ActionId; 11] = [
    ActionId::PaletteToggle,
    ActionId::PaletteClose,
    ActionId::ChatSend,
    ActionId::ChatNewline,
    ActionId::PlayerPlayPause,
    ActionId::PlayerSeekBack,
    ActionId::PlayerSeekFwd,
    ActionId::PlayerSkipBack,
    ActionId::PlayerSkipFwd,
    ActionId::PlayerRateDown,
    ActionId::PlayerRateUp,
];

const DOTTED_IDS: [&str; 11] = [
    "palette.toggle",
    "palette.close",
    "chat.send",
    "chat.newline",
    "player.playPause",
    "player.seekBack",
    "player.seekFwd",
    "player.skipBack",
    "player.skipFwd",
    "player.rateDown",
    "player.rateUp",
];

/// Deserializes a full `AppConfig` whose `keymap` key is `keymap`. Building on the
/// serialized default keeps every non-defaulting field (`theme`, `models`, `paths`, …)
/// present, so a failure here is about the keymap and nothing else.
fn load_with_keymap(keymap: serde_json::Value) -> Result<AppConfig, serde_json::Error> {
    let mut raw = serde_json::to_value(AppConfig::default()).expect("default config serializes");
    raw["keymap"] = keymap;
    serde_json::from_value(raw)
}

#[test]
fn action_id_wire_strings_match_the_frontend_union() {
    let wire: Vec<String> = ALL_ACTION_IDS
        .iter()
        .map(|id| {
            serde_json::to_value(id)
                .expect("ActionId serializes")
                .as_str()
                .expect("ActionId serializes as a string")
                .to_string()
        })
        .collect();
    let expected: Vec<String> = DOTTED_IDS.iter().map(|s| s.to_string()).collect();

    assert_eq!(
        wire, expected,
        "the serialized ActionId set is the contract src/lib/shortcuts/registry.ts mirrors"
    );
}

#[test]
fn unknown_key_is_dropped_and_valid_key_survives() {
    let config = load_with_keymap(json!({ "bogus.action": "X", "palette.toggle": "Mod+P" }))
        .expect("an unknown keymap key must not fail the load");

    assert_eq!(
        config.keymap.get(&ActionId::PaletteToggle),
        Some(&"Mod+P".to_string())
    );
    assert_eq!(config.keymap.len(), 1, "bogus.action must be dropped");
}

#[test]
fn malformed_token_value_is_kept() {
    let config = load_with_keymap(json!({ "palette.toggle": "???" }))
        .expect("a malformed token must not fail the load");

    assert_eq!(
        config.keymap.get(&ActionId::PaletteToggle),
        Some(&"???".to_string()),
        "the engine never parses tokens, so an unparseable one round-trips"
    );
}

#[test]
fn non_string_value_is_dropped_not_fatal() {
    let config = load_with_keymap(json!({
        "palette.toggle": null,
        "player.skipBack": 3,
        "player.skipFwd": "Mod+L"
    }))
    .expect("a non-string keymap value must not fail the load");

    assert_eq!(
        config.keymap,
        BTreeMap::from([(ActionId::PlayerSkipFwd, "Mod+L".to_string())])
    );
}

#[test]
fn explicit_null_keymap_loads_as_empty() {
    let config = load_with_keymap(serde_json::Value::Null)
        .expect("null is JSON's idiomatic absent, and must never cost the user their config");

    assert!(config.keymap.is_empty());
}

#[test]
fn keymap_shapes_that_are_not_a_map_stay_fatal() {
    // Chosen, not incidental: an array or a bare string is a structural mistake with no
    // salvage, unlike `null`. Both surface as `LensError::Parse` on the real `load()` path.
    assert!(load_with_keymap(json!([])).is_err(), "array is not a map");
    assert!(
        load_with_keymap(json!("Mod+K")).is_err(),
        "a bare token is not a map"
    );
}

#[test]
fn save_load_save_leaves_keymap_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let keymap = BTreeMap::from([
        (ActionId::PaletteToggle, "Mod+Shift+P".to_string()),
        (ActionId::PlayerSkipFwd, "Alt+L".to_string()),
    ]);

    let mut config = AppConfig {
        keymap: keymap.clone(),
        ..AppConfig::default()
    };
    config.save(dir.path()).unwrap();
    let first = std::fs::read(dir.path().join("config.json")).unwrap();

    let mut reloaded = AppConfig::load(dir.path()).unwrap();
    assert_eq!(reloaded.keymap, keymap, "the keymap survives a load");

    reloaded.save(dir.path()).unwrap();
    let second = std::fs::read(dir.path().join("config.json")).unwrap();
    assert_eq!(
        first, second,
        "normalize() must leave the keymap untouched on every save"
    );
}
