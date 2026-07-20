//! Blocking config guard for Audio Overview playback (#29). Native e2e is
//! non-blocking on CI, so this cheap parse-test is the real defense against a future
//! `tauri.conf.json` edit silently dropping the `media-src` CSP directive or the asset
//! protocol — either of which breaks `<audio>` playback with no other signal.

use serde_json::Value;

fn config() -> Value {
    let raw = include_str!("../tauri.conf.json");
    serde_json::from_str(raw).expect("tauri.conf.json is valid JSON")
}

#[test]
fn csp_allows_media_from_asset_protocol() {
    let cfg = config();
    let csp = cfg["app"]["security"]["csp"]
        .as_str()
        .expect("csp is a string");

    let media_src = csp
        .split(';')
        .map(str::trim)
        .find(|d| d.starts_with("media-src"))
        .expect("csp must define a media-src directive for <audio> playback");

    assert!(
        media_src.contains("asset:") || media_src.contains("http://asset.localhost"),
        "media-src must allow the asset protocol; got {media_src:?}"
    );
}

#[test]
fn asset_protocol_enabled_and_scoped_to_notebooks() {
    let cfg = config();
    let asset = &cfg["app"]["security"]["assetProtocol"];

    assert_eq!(
        asset["enable"].as_bool(),
        Some(true),
        "assetProtocol.enable must be true or convertFileSrc URLs are blocked"
    );

    let scope = asset["scope"]
        .as_array()
        .expect("assetProtocol.scope must be an array");
    assert!(
        scope
            .iter()
            .filter_map(Value::as_str)
            .any(|s| s.contains("notebooks")),
        "assetProtocol.scope must include a notebooks-scoped entry; got {scope:?}"
    );
}
