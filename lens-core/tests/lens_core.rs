//! Integration tests for the public `lens-core` surface.

use lens_core::{LensEngine, LensError};
use rstest::rstest;

/// The `Arc<RwLock>` engine grants write then read guards without deadlock —
/// exercising the interior mutability the Tauri managed state relies on.
#[tokio::test]
async fn engine_write_then_read_roundtrip() {
    let engine = LensEngine::for_test().await;
    {
        let _write = engine.write().await;
    } // write guard dropped here
    let _read = engine.read().await;
}

/// Cloning the handle shares the same underlying state (cheap `Arc` clone),
/// so concurrent shared read guards are allowed.
#[tokio::test]
async fn cloned_handles_share_state() {
    let engine = LensEngine::for_test().await;
    let clone = engine.clone();
    let _a = engine.read().await;
    let _b = clone.read().await; // second shared read is fine
}

/// Parametrized check that the `thiserror` `Display` output is stable —
/// this is the contract surfaced to logs.
#[rstest]
#[case(LensError::Validation("bad payload".into()), "invalid input: bad payload")]
#[case(LensError::Internal("boom".into()), "internal error: boom")]
#[case(LensError::Io("disk full".into()), "io error: disk full")]
#[case(LensError::Parse("bad json".into()), "parse error: bad json")]
#[case(LensError::Model("oom".into()), "model error: oom")]
#[case(LensError::Network("timeout".into()), "network error: timeout")]
fn lens_error_display(#[case] err: LensError, #[case] expected: &str) {
    assert_eq!(err.to_string(), expected);
}

/// Snapshot the exact serde wire-format of `LensError`. This locks in the
/// `#[serde(tag = "kind", content = "message")]` contract that crosses the IPC
/// boundary — a regression here would silently break the frontend.
#[test]
fn lens_error_serialized_shape() {
    insta::assert_json_snapshot!(LensError::Validation("bad payload".into()), @r#"
    {
      "kind": "Validation",
      "message": "bad payload"
    }
    "#);
}

/// New additive variants serialize with the SAME `{kind, message}` shape as the
/// locked `Validation` snapshot above — proving extension didn't break the
/// adjacent-tagged contract.
#[test]
fn lens_error_new_variant_serialized_shape() {
    insta::assert_json_snapshot!(LensError::Io("disk full".into()), @r#"
    {
      "kind": "Io",
      "message": "disk full"
    }
    "#);
}
