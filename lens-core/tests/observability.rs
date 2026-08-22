//! Proves the log-capture helper before anything depends on it: a thread-local
//! subscriber does not reach `tokio::spawn` / `spawn_blocking`, which is where the
//! events this PR must assert are emitted from.

mod common;

use common::capture_logs;

/// `multi_thread` is load-bearing: on the default current-thread runtime
/// `tokio::spawn` stays on the test's own thread, so a thread-local subscriber
/// would still see it and this test could not discriminate.
#[tokio::test(flavor = "multi_thread")]
async fn captures_events_from_a_spawned_task() {
    let cap = capture_logs();
    tokio::spawn(async {
        tracing::info!(reason = "quiesce_probe", "waiting");
    })
    .await
    .expect("join");
    assert!(
        cap.any_with(&["quiesce_probe"]),
        "event from tokio::spawn must reach the capture sink"
    );
}

#[tokio::test]
async fn captures_events_from_spawn_blocking() {
    let cap = capture_logs();
    tokio::task::spawn_blocking(|| {
        tracing::warn!(path = "copy_probe", "skipped");
    })
    .await
    .expect("join");
    assert!(
        cap.any_with(&["copy_probe"]),
        "event from spawn_blocking must reach the capture sink"
    );
}

#[tokio::test]
async fn buffer_holds_only_this_tests_events() {
    let cap = capture_logs();
    tracing::info!(marker = "isolation_probe", "here");
    assert!(cap.any_with(&["isolation_probe"]));
    assert!(
        !cap.any_with(&["quiesce_probe"]),
        "a previous test's events must not leak into this buffer"
    );
}

/// `LevelFilter::INFO` keeps `debug!` sites off the sink's mutex, so an AC-6.1
/// event emitted at debug would be invisible.
#[tokio::test]
async fn debug_events_are_filtered_out() {
    let cap = capture_logs();
    tracing::debug!(marker = "debug_probe", "quiet");
    tracing::info!(marker = "info_probe", "loud");
    assert!(cap.any_with(&["info_probe"]));
    assert!(!cap.any_with(&["debug_probe"]));
}
