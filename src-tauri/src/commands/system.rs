//! System / diagnostic commands.

use lens_core::{LensEngine, LensError};
use serde::Serialize;

#[cfg(debug_assertions)]
use tauri::ipc::Channel;

#[cfg(debug_assertions)]
use crate::stream::{StreamEvent, send_event};

/// Result of a [`health_check`]: DB reachable + applied migration count.
#[derive(Debug, Clone, Serialize)]
pub struct HealthStatus {
    /// Whether the database query succeeded.
    pub db_ok: bool,
    /// Number of migrations recorded in `_sqlx_migrations`.
    pub migration_count: i64,
}

/// Verifies the database is reachable and reports the applied migration count.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn health_check(engine: tauri::State<'_, LensEngine>) -> Result<HealthStatus, LensError> {
    let migration_count = engine.migration_count().await?;
    Ok(HealthStatus {
        db_ok: true,
        migration_count,
    })
}

/// Demonstrator that exercises the streaming primitive end to end: emits
/// `Started`, three `Progress` updates, then `Done` over the channel.
///
/// Gated behind `debug_assertions` so it never appears on the release command
/// surface — it exists only to validate the streaming plumbing during dev.
#[cfg(debug_assertions)]
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn stream_demo(channel: Channel<StreamEvent<String>>) -> Result<(), LensError> {
    let total = 3u64;
    send_event(&channel, StreamEvent::Started)?;
    for done in 1..=total {
        send_event(
            &channel,
            StreamEvent::Progress {
                done,
                total: Some(total),
            },
        )?;
    }
    send_event(&channel, StreamEvent::Done)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(debug_assertions)]
    use std::sync::{Arc, Mutex};
    use tauri::Manager;
    #[cfg(debug_assertions)]
    use tauri::ipc::Channel;

    #[tokio::test]
    async fn health_check_reports_db_ok_and_migrations() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let status = health_check(engine).await.unwrap();
        assert!(status.db_ok);
        // The single 0001_init migration is recorded.
        assert_eq!(status.migration_count, 1);
    }

    #[cfg(debug_assertions)]
    #[tokio::test]
    async fn stream_demo_emits_started_progress_done_in_order() {
        let collected = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&collected);
        let channel = Channel::new(move |body: tauri::ipc::InvokeResponseBody| {
            // The mock receives the already-serialized IPC body; deserialize it
            // back into the typed envelope to assert ordering/content.
            let event = body.deserialize::<StreamEvent<String>>().unwrap();
            sink.lock().unwrap().push(event);
            Ok(())
        });

        stream_demo(channel).await.unwrap();

        let events = collected.lock().unwrap().clone();
        assert_eq!(
            events,
            vec![
                StreamEvent::Started,
                StreamEvent::Progress {
                    done: 1,
                    total: Some(3)
                },
                StreamEvent::Progress {
                    done: 2,
                    total: Some(3)
                },
                StreamEvent::Progress {
                    done: 3,
                    total: Some(3)
                },
                StreamEvent::Done,
            ]
        );
    }
}
