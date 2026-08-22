//! #248 AC-1.3/1.4: a data-dir relocation must not copy `lancedb/` while a
//! background entity-vector write is in flight. Every test is time-bounded so a
//! regression fails instead of hanging CI, and none of them sleep.

mod common;

use common::{capture_logs, resolution_ready_engine};
use lens_core::QuiesceGate;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

const REACHED: Duration = Duration::from_secs(10);
/// Long enough that a passing relocation would finish, short enough to keep the
/// suite quick — the assertion is `Err(Elapsed)`, so this is a blocking proof.
const BLOCKED: Duration = Duration::from_millis(750);
const COMPLETES: Duration = Duration::from_secs(30);

#[tokio::test(flavor = "multi_thread")]
async fn relocation_waits_for_an_in_flight_entity_vector_write() {
    let (_dir, engine, nb) = resolution_ready_engine().await;
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");

    let gate = Arc::new(QuiesceGate::default());
    engine
        .set_quiesce_upsert_gate_for_test(Some(gate.clone()))
        .await;

    let pass_engine = engine.clone();
    let pass_nb = nb.clone();
    let pass = tokio::spawn(async move { pass_engine.resolve_notebook_for_test(&pass_nb).await });

    // Without this handshake the assertion below could pass simply because the
    // relocation ran before the guard was ever taken.
    timeout(REACHED, gate.reached.notified())
        .await
        .expect("resolution pass must reach the in-guard seam");

    let blocked = timeout(BLOCKED, engine.relocate_data_dir(&to, &[])).await;
    assert!(
        blocked.is_err(),
        "relocation must block while a shared quiesce holder is in flight, got {:?}",
        blocked.map(|r| r.is_ok())
    );

    gate.release.notify_one();
    timeout(COMPLETES, pass)
        .await
        .expect("the released pass must finish")
        .expect("join")
        .expect("resolution pass");
    engine.set_quiesce_upsert_gate_for_test(None).await;

    timeout(COMPLETES, engine.relocate_data_dir(&to, &[]))
        .await
        .expect("relocation must not hang once the writer released")
        .expect("relocation succeeds");
}

/// AC-1.4: with nothing holding the guard, the same relocation completes — so the
/// test above is proving the guard, not an unrelated stall.
#[tokio::test(flavor = "multi_thread")]
async fn relocation_completes_when_no_writer_holds_the_guard() {
    let (_dir, engine, _nb) = resolution_ready_engine().await;
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");

    timeout(BLOCKED, engine.relocate_data_dir(&to, &[]))
        .await
        .expect("an unguarded relocation must finish well inside the blocked budget")
        .expect("relocation succeeds");
}

#[tokio::test(flavor = "multi_thread")]
async fn relocation_logs_the_quiesce_acquisition() {
    let cap = capture_logs();
    let (_dir, engine, _nb) = resolution_ready_engine().await;
    let to_parent = tempfile::tempdir().expect("to_parent");
    let to = to_parent.path().join("moved");

    timeout(COMPLETES, engine.relocate_data_dir(&to, &[]))
        .await
        .expect("relocate")
        .expect("relocate ok");

    assert!(
        cap.any_with(&["relocate"]),
        "the quiesce acquisition must name its reason"
    );
}
