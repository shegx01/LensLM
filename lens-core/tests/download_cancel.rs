//! #37: the download cancellation registry (DEC-3) and cancel propagation through
//! the two public download wrappers. Offline — a tripped token aborts before any I/O.

use std::sync::Arc;

use lens_core::{
    DownloadKey, DownloadKind, LensEngine, LensError, download_tts_model, download_whisper_model,
};
use tokio_util::sync::CancellationToken;

fn key(kind: DownloadKind, id: &str) -> DownloadKey {
    DownloadKey {
        kind,
        id: id.to_string(),
    }
}

#[tokio::test]
async fn tts_download_propagates_tripped_token_as_cancelled() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cancel = CancellationToken::new();
    cancel.cancel();

    let err = download_tts_model(dir.path(), "snac", &cancel, |_| {})
        .await
        .expect_err("a tripped token must abort the download");

    assert!(matches!(err, LensError::Cancelled(_)), "got {err:?}");
}

#[tokio::test]
async fn whisper_download_propagates_tripped_token_as_cancelled() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cancel = CancellationToken::new();
    cancel.cancel();

    let err = download_whisper_model(dir.path(), "tiny", &cancel, |_| {})
        .await
        .expect_err("a tripped token must abort the download");

    assert!(matches!(err, LensError::Cancelled(_)), "got {err:?}");
}

#[tokio::test]
async fn cancel_download_unknown_key_is_false() {
    let engine = LensEngine::for_test().await;
    assert!(!engine.cancel_download(&key(DownloadKind::Whisper, "never-registered")));
    assert!(!engine.cancel_download(&key(DownloadKind::Tts, "orpheus")));
}

#[tokio::test]
async fn second_registration_joins_the_incumbent_without_cancelling_it() {
    let engine = LensEngine::for_test().await;
    let k = key(DownloadKind::Tts, "orpheus");

    let first = engine.register_download(k.clone());
    let second = engine.register_download(k.clone());

    assert!(
        Arc::ptr_eq(&first, &second),
        "a second registration must return the incumbent token, not a new one"
    );
    assert!(
        !first.is_cancelled(),
        "registering a second caller must NOT cancel the in-flight download"
    );

    assert!(engine.cancel_download(&k));
    assert!(first.is_cancelled() && second.is_cancelled());
}

#[tokio::test]
async fn guard_drop_clears_the_key_so_a_later_download_gets_a_live_token() {
    let engine = LensEngine::for_test().await;
    let k = key(DownloadKind::Whisper, "small");

    let token = engine.register_download(k.clone());
    let guard = engine.download_cancel_guard(k.clone(), token);
    assert!(engine.cancel_download(&k));
    drop(guard);

    assert!(
        !engine.cancel_download(&k),
        "a dropped guard leaves no cancel address behind"
    );
    assert!(
        !engine.register_download(k).is_cancelled(),
        "a later download must not inherit the cancelled token"
    );
}

#[tokio::test]
async fn stale_guard_drop_does_not_evict_a_later_token() {
    let engine = LensEngine::for_test().await;
    let k = key(DownloadKind::Whisper, "base");

    // Two callers joined on one token (the collision rule), so both hold a guard.
    let joined = engine.register_download(k.clone());
    let first_guard = engine.download_cancel_guard(k.clone(), Arc::clone(&joined));
    let second_guard = engine.download_cancel_guard(k.clone(), joined);

    drop(first_guard);
    let later = engine.register_download(k.clone());
    drop(second_guard);

    assert!(
        engine.cancel_download(&k),
        "a stale guard must not evict the live token"
    );
    assert!(later.is_cancelled());
}
