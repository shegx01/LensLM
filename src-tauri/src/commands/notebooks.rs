//! Notebook commands. Thin pass-throughs to `lens-core`; full CRUD UI is M3.

use futures::StreamExt;
use lens_core::{
    AddSourceOutcome, AnswerEvent, ChatFeedback, ChatMessage, ChatState, DialoguePhase,
    DialogueScript, IngestProgress, Length, LensEngine, LensError, Notebook, NotebookId,
    NotebookSummary, Source, TrashedSource, TtsPhase,
};
use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;

use crate::stream::{StreamEvent, send_event};

/// Progress event streamed by [`set_notebook_embedding_model`] while re-embedding
/// chunks under the new coordinate. Mirrors [`IngestProgress`]'s shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReembedProgress {
    /// Chunks processed so far.
    pub done: usize,
    /// Total chunks to process.
    pub total: usize,
}

/// IPC return type for [`get_notebook_embedding_model`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelInfo {
    /// Canonical model id (e.g. `"nomic-embed-text-v1.5"`).
    pub model_id: String,
    /// Output vector dimension (e.g. `768`).
    pub dim: usize,
    /// Embedding backend serving this coordinate (`"fastembed"` | `"ollama"`).
    pub backend: String,
    /// Whether an active embedding_index row exists for this
    /// (notebook, backend, model, dim) coordinate: `"active"` when the coordinate
    /// is live, `"none"` when no index exists yet.
    pub status: String,
}

/// IPC wire form of an eval log row (#158b). The engine's `EvalReport`/`EvalOutcome`
/// deliberately do NOT derive `Serialize` (keeping the headless engine decoupled from
/// the wire format), so map them into these DTOs at the command boundary. `ran_at` is
/// carried alongside because `EvalReport` itself has no timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReportDto {
    pub graph_recall: f32,
    pub hybrid_recall: f32,
    pub delta_pp: f32,
    pub p95_ms: f32,
    pub passed: bool,
    pub sample_n: usize,
    pub dropped_n: usize,
    pub graph_enabled: bool,
    pub prompt_version: String,
    pub ran_at: String,
}

impl EvalReportDto {
    fn from_report(report: lens_core::eval::EvalReport, ran_at: String) -> Self {
        Self {
            graph_recall: report.graph_recall,
            hybrid_recall: report.hybrid_recall,
            delta_pp: report.delta_pp,
            p95_ms: report.p95_ms,
            passed: report.passed,
            sample_n: report.sample_n,
            dropped_n: report.dropped_n,
            graph_enabled: report.graph_enabled,
            prompt_version: report.prompt_version,
            ran_at,
        }
    }
}

/// IPC wire form of `EvalOutcome` (#158b). Internally tagged on `status` so the
/// frontend branches on `{ status: 'skipped' | 'ran' }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EvalOutcomeDto {
    Skipped { reason: String },
    Ran { report: EvalReportDto },
}

/// IPC wire form of `EvalPhase` (#158b), streamed as the `Chunk(T)` payload of a
/// `StreamEvent`. Two variants only, mirroring the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalPhaseDto {
    GeneratingQa,
    Done,
}

impl From<lens_core::eval::EvalPhase> for EvalPhaseDto {
    fn from(phase: lens_core::eval::EvalPhase) -> Self {
        match phase {
            lens_core::eval::EvalPhase::GeneratingQa => EvalPhaseDto::GeneratingQa,
            lens_core::eval::EvalPhase::Done => EvalPhaseDto::Done,
        }
    }
}

/// Lists live (non-trashed) notebooks with their source counts, newest first.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_notebooks(
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<NotebookSummary>, LensError> {
    engine.list_notebooks_with_counts().await
}

/// Creates a notebook with the given title and optional onboarding
/// `description`/`focus_mode`, and returns it.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn create_notebook(
    title: String,
    description: Option<String>,
    focus_mode: Option<String>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Notebook, LensError> {
    engine
        .create_notebook(&title, description.as_deref(), focus_mode.as_deref())
        .await
}

/// Inserts a file source record (no ingestion). On a path-based dedup hit (#100)
/// returns the existing live source with `wasExisting = true`.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn add_source(
    notebook_id: String,
    title: String,
    locator: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<AddSourceOutcome, LensError> {
    engine
        .add_source(&NotebookId::from(notebook_id), &title, &locator)
        .await
}

/// Lists all sources for a notebook, newest first.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_sources(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<Source>, LensError> {
    engine.list_sources(&NotebookId::from(notebook_id)).await
}

/// Inserts a managed text/markdown source. `kind` must be `"text"` or `"markdown"`.
/// On a content-dedup hit (#100) returns the existing live source with `wasExisting = true`.
#[tracing::instrument(skip(text, engine))]
#[tauri::command]
pub async fn add_text_source(
    notebook_id: String,
    title: String,
    text: String,
    kind: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<AddSourceOutcome, LensError> {
    engine
        .add_text_source(&NotebookId::from(notebook_id), &title, &text, &kind)
        .await
}

/// Inserts a URL source (`queued` row). Returns immediately — no HTTP fetch.
/// On a URL-based dedup hit (#100) returns the existing source with `wasExisting = true`.
/// `force_js_render` (#78) is `Option<bool>`: omitted by non-SPA callers → treated
/// as `false` (Tauri params cannot carry `#[serde(default)]`).
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn add_url_source(
    notebook_id: String,
    title: String,
    url: String,
    force_js_render: Option<bool>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<AddSourceOutcome, LensError> {
    engine
        .add_url_source(
            &NotebookId::from(notebook_id),
            &title,
            &url,
            force_js_render.unwrap_or(false),
        )
        .await
}

/// Inserts a managed local-file source (PDF/DOCX/text/markdown). Kind is detected from
/// extension; unsupported extensions are rejected. On a content-dedup hit (#96) returns
/// the existing source with `wasExisting = true`. Call `ingest_source` to index.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn add_file_source(
    notebook_id: String,
    path: String,
    title: Option<String>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<AddSourceOutcome, LensError> {
    engine
        .add_file_source(
            &NotebookId::from(notebook_id),
            std::path::Path::new(&path),
            title.as_deref(),
        )
        .await
}

/// Soft-deletes a source: sets `trashed_at` to now. Keeps chunks + Lance
/// vectors so the source can be restored. Errors if missing or already trashed.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn trash_source(
    source_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.trash_source(&source_id).await
}

/// Restores a trashed source: clears `trashed_at`. Errors if live or missing.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn restore_source(
    source_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.restore_source(&source_id).await
}

/// Permanently deletes a source: drops its Lance vectors then removes the
/// `sources` row. Errors if the source does not exist.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn purge_source(
    source_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.purge_source(&source_id).await
}

/// Toggles a source's `selected` flag (persisted across reloads).
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn set_source_selected(
    source_id: String,
    selected: bool,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.set_source_selected(&source_id, selected).await
}

/// Ingests a queued source end-to-end (parse → chunk → embed → index), streaming
/// `Started` → `Chunk`/`Progress` per phase → `Done` or `Failed` over `on_progress`.
#[tracing::instrument(skip(on_progress, engine))]
#[tauri::command]
pub async fn ingest_source(
    source_id: String,
    on_progress: Channel<StreamEvent<IngestProgress>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    if let Err(e) = send_event(&on_progress, StreamEvent::Started) {
        tracing::warn!("ingest_source: started event send failed: {e}");
    }

    let result = engine
        .ingest_source(&source_id, |progress| {
            let done = progress.done;
            let total = progress.total;
            if let Err(e) = send_event(&on_progress, StreamEvent::Chunk(progress)) {
                tracing::warn!("ingest_source: progress chunk send failed: {e}");
            }
            if let Err(e) = send_event(&on_progress, StreamEvent::Progress { done, total }) {
                tracing::warn!("ingest_source: progress event send failed: {e}");
            }
        })
        .await;

    match &result {
        Ok(()) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Done) {
                tracing::warn!("ingest_source: done event send failed: {e}");
            }
        }
        Err(err) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Failed(err.clone())) {
                tracing::warn!("ingest_source: failed event send failed: {e}");
            }
        }
    }

    result
}

/// Retries a failed source in place (#73), re-running the ingest pipeline on the same
/// row. Rejects non-`error` and trashed sources. Mirrors [`ingest_source`] progress stream.
#[tracing::instrument(skip(on_progress, engine))]
#[tauri::command]
pub async fn retry_ingest_source(
    source_id: String,
    on_progress: Channel<StreamEvent<IngestProgress>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    if let Err(e) = send_event(&on_progress, StreamEvent::Started) {
        tracing::warn!("retry_ingest_source: started event send failed: {e}");
    }

    let result = engine
        .retry_source(&source_id, |progress| {
            let done = progress.done;
            let total = progress.total;
            if let Err(e) = send_event(&on_progress, StreamEvent::Chunk(progress)) {
                tracing::warn!("retry_ingest_source: progress chunk send failed: {e}");
            }
            if let Err(e) = send_event(&on_progress, StreamEvent::Progress { done, total }) {
                tracing::warn!("retry_ingest_source: progress event send failed: {e}");
            }
        })
        .await;

    match &result {
        Ok(()) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Done) {
                tracing::warn!("retry_ingest_source: done event send failed: {e}");
            }
        }
        Err(err) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Failed(err.clone())) {
                tracing::warn!("retry_ingest_source: failed event send failed: {e}");
            }
        }
    }

    result
}

/// Retries every failed source in a notebook (#73). Sequential with continue-on-failure:
/// a source that fails again updates only its own `error_meta` and does not abort the
/// batch. All per-source progress streams over the shared `on_progress` channel.
#[tracing::instrument(skip(on_progress, engine))]
#[tauri::command]
pub async fn retry_all_failed_sources(
    notebook_id: String,
    on_progress: Channel<StreamEvent<IngestProgress>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    let failed: Vec<String> = engine
        .list_sources(&NotebookId::from(notebook_id))
        .await?
        .into_iter()
        .filter(|s| s.status == lens_core::notebooks::SourceStatus::Error)
        .map(|s| s.id)
        .collect();

    if let Err(e) = send_event(&on_progress, StreamEvent::Started) {
        tracing::warn!("retry_all_failed_sources: started event send failed: {e}");
    }

    for source_id in &failed {
        let result = engine
            .retry_source(source_id, |progress| {
                let done = progress.done;
                let total = progress.total;
                if let Err(e) = send_event(&on_progress, StreamEvent::Chunk(progress)) {
                    tracing::warn!("retry_all_failed_sources: progress chunk send failed: {e}");
                }
                if let Err(e) = send_event(&on_progress, StreamEvent::Progress { done, total }) {
                    tracing::warn!("retry_all_failed_sources: progress event send failed: {e}");
                }
            })
            .await;
        if let Err(err) = &result {
            tracing::warn!(
                source_id,
                "retry_all_failed_sources: source retry failed: {err}"
            );
        }
    }

    if let Err(e) = send_event(&on_progress, StreamEvent::Done) {
        tracing::warn!("retry_all_failed_sources: done event send failed: {e}");
    }

    Ok(())
}

/// Cooperatively cancels an in-flight audio ingest (#43). Returns `true` if a
/// token was found and cancelled, `false` if no ingest is in-flight for that source.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn cancel_media_ingest(
    source_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<bool, LensError> {
    Ok(engine.cancel_media_ingest(&source_id))
}

/// Answers a notebook question, streaming the grounded-answer pipeline (#173):
/// `Started` → per-event `Chunk(AnswerEvent)` → `Done`, or `Failed(err)` on any
/// ctx-gathering or in-stream error. Per-notebook single-flight: registering the
/// ask supersedes (and cancels) any prior in-flight ask for the same notebook; the
/// [`AskCancelGuard`](lens_core::AskCancelGuard) removes only its own token on drop.
#[tracing::instrument(skip(question, on_answer, engine))]
#[tauri::command]
pub async fn ask_notebook(
    notebook_id: String,
    turn_id: String,
    question: String,
    on_answer: Channel<StreamEvent<AnswerEvent>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    if let Err(e) = send_event(&on_answer, StreamEvent::Started) {
        tracing::warn!("ask_notebook: started event send failed: {e}");
    }

    // Register single-flight cancellation; the guard's Drop cleans the registry up
    // (only if still the owner) when this command returns on any path.
    let token = engine.register_ask(&notebook_id);
    let _guard = engine.ask_cancel_guard(&notebook_id, token.clone());

    // Ctx-gathering / provider-None errors surface HERE, before any stream is built.
    // `turn_id` scopes history loading so the current turn is excluded from it.
    let stream = match engine
        .answer_notebook(
            &NotebookId::from(notebook_id),
            &turn_id,
            question,
            (*token).clone(),
        )
        .await
    {
        Ok(s) => s,
        Err(err) => {
            if let Err(e) = send_event(&on_answer, StreamEvent::Failed(err.clone())) {
                tracing::warn!("ask_notebook: failed event send failed: {e}");
            }
            return Err(err);
        }
    };

    let mut stream = std::pin::pin!(stream);
    while let Some(item) = stream.next().await {
        match item {
            Ok(ev) => {
                if let Err(e) = send_event(&on_answer, StreamEvent::Chunk(ev)) {
                    tracing::warn!("ask_notebook: chunk send failed: {e}");
                }
            }
            Err(err) => {
                // In-stream failure: surface Failed with the preserved kind and stop
                // — never emit Done for a truncated/failed answer.
                if let Err(e) = send_event(&on_answer, StreamEvent::Failed(err.clone())) {
                    tracing::warn!("ask_notebook: failed event send failed: {e}");
                }
                return Err(err);
            }
        }
    }

    // The stream ends with no further items on cancellation too (engine contract),
    // so distinguish a user-cancelled run from a real completion here — otherwise a
    // stopped, truncated answer would look identical to success. Cancel → Failed
    // (kind `Cancelled`), never Done.
    if token.is_cancelled() {
        let err = LensError::Cancelled("answer generation cancelled".into());
        if let Err(e) = send_event(&on_answer, StreamEvent::Failed(err)) {
            tracing::warn!("ask_notebook: cancelled event send failed: {e}");
        }
    } else if let Err(e) = send_event(&on_answer, StreamEvent::Done) {
        tracing::warn!("ask_notebook: done event send failed: {e}");
    }
    Ok(())
}

/// Cancels the in-flight grounded answer for a notebook (#173, the stop button).
/// Returns `true` if an ask was in flight and cancelled, `false` otherwise.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn cancel_ask(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<bool, LensError> {
    Ok(engine.cancel_ask_notebook(&notebook_id))
}

/// Generates a grounded two-speaker dialogue script for a notebook (#26, first stage
/// of the M7 audio-overview pipeline). Streams honest phase markers over
/// `on_progress` (`Started` → `Chunk(Retrieving)` → `Chunk(Generating)` →
/// `Chunk(Validating)` → `Done`) and returns the validated [`DialogueScript`].
/// Per-notebook single-flight over a DEDICATED dialogue registry (never the ask
/// registry). The empty-notebook path yields `Started → Chunk(Retrieving) →
/// Failed(Validation)` with zero LLM calls; error/cancel yields `Failed(err)`.
#[tracing::instrument(skip(on_progress, engine))]
#[tauri::command]
pub async fn generate_dialogue(
    notebook_id: String,
    length: Length,
    on_progress: Channel<StreamEvent<DialoguePhase>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<DialogueScript, LensError> {
    if let Err(e) = send_event(&on_progress, StreamEvent::Started) {
        tracing::warn!("generate_dialogue: started event send failed: {e}");
    }

    let token = engine.register_dialogue(&notebook_id);
    let _guard = engine.dialogue_cancel_guard(&notebook_id, token.clone());

    let on_progress_phase = on_progress.clone();
    let result = engine
        .generate_dialogue(
            &NotebookId::from(notebook_id),
            length,
            (*token).clone(),
            move |phase| {
                if let Err(e) = send_event(&on_progress_phase, StreamEvent::Chunk(phase)) {
                    tracing::warn!("generate_dialogue: phase send failed: {e}");
                }
            },
        )
        .await;

    match result {
        Ok(script) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Done) {
                tracing::warn!("generate_dialogue: done event send failed: {e}");
            }
            Ok(script)
        }
        Err(err) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Failed(err.clone())) {
                tracing::warn!("generate_dialogue: failed event send failed: {e}");
            }
            Err(err)
        }
    }
}

/// Cancels the in-flight dialogue-script generation for a notebook (#26, stop
/// button). Returns `true` if one was in flight and cancelled, `false` otherwise.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn cancel_dialogue(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<bool, LensError> {
    Ok(engine.cancel_dialogue_generation(&notebook_id))
}

#[tracing::instrument(skip(on_progress, engine))]
#[tauri::command]
pub async fn synthesize_overview(
    notebook_id: String,
    on_progress: Channel<StreamEvent<TtsPhase>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<String, LensError> {
    if let Err(e) = send_event(&on_progress, StreamEvent::Started) {
        tracing::warn!("synthesize_overview: started event send failed: {e}");
    }

    let config = engine.config().await;

    if !engine.tts_backend_available(&config.tts).await {
        let err = LensError::Tts("no TTS backend available; install an engine".into());
        if let Err(e) = send_event(&on_progress, StreamEvent::Failed(err.clone())) {
            tracing::warn!("synthesize_overview: failed event send failed: {e}");
        }
        return Err(err);
    }

    let token = engine.register_tts(&notebook_id);
    let _guard = engine.tts_cancel_guard(&notebook_id, token.clone());

    let script = match engine
        .generate_dialogue(
            &NotebookId::from(notebook_id.clone()),
            Length::Medium,
            (*token).clone(),
            |_phase| {},
        )
        .await
    {
        Ok(script) => script,
        Err(err) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Failed(err.clone())) {
                tracing::warn!("synthesize_overview: failed event send failed: {e}");
            }
            return Err(err);
        }
    };

    let on_progress_phase = on_progress.clone();
    let result = engine
        .synthesize_overview(
            &notebook_id,
            &script,
            &config.voices,
            &config.tts,
            move |phase| {
                if let Err(e) = send_event(&on_progress_phase, StreamEvent::Chunk(phase)) {
                    tracing::warn!("synthesize_overview: phase send failed: {e}");
                }
            },
            &token,
        )
        .await;

    match result {
        Ok(path) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Done) {
                tracing::warn!("synthesize_overview: done event send failed: {e}");
            }
            Ok(path.to_string_lossy().into_owned())
        }
        Err(err) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Failed(err.clone())) {
                tracing::warn!("synthesize_overview: failed event send failed: {e}");
            }
            Err(err)
        }
    }
}

#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn cancel_synthesis(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<bool, LensError> {
    Ok(engine.cancel_synthesis(&notebook_id))
}

/// Persists a user chat message on send (#22). `turn_id` is minted by the frontend
/// and shared with the later assistant row of the same turn.
#[tracing::instrument(skip(content, engine))]
#[tauri::command]
pub async fn save_chat_user(
    notebook_id: String,
    turn_id: String,
    content: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<ChatMessage, LensError> {
    engine
        .save_chat_user(&NotebookId::from(notebook_id), &turn_id, &content)
        .await
}

/// Persists an assistant chat message on stream `Done` (#22). `citations` is the
/// typed payload; the engine serializes it to JSON.
#[tracing::instrument(skip(content, citations, engine))]
#[tauri::command]
pub async fn save_chat_assistant(
    notebook_id: String,
    turn_id: String,
    content: String,
    citations: Option<Vec<lens_core::Citation>>,
    tokens_used: u32,
    engine: tauri::State<'_, LensEngine>,
) -> Result<ChatMessage, LensError> {
    engine
        .save_chat_assistant(
            &NotebookId::from(notebook_id),
            &turn_id,
            &content,
            citations.as_deref(),
            tokens_used,
        )
        .await
}

/// Persists a terminal-state marker for a cancelled/errored turn (Plan 2 / PC-1).
/// `content` may carry the partial answer streamed so far so a reload shows it under
/// a "Stopped"/"Couldn't complete" line instead of a bare, dangling question.
#[tracing::instrument(skip(content, engine))]
#[tauri::command]
pub async fn save_chat_marker(
    notebook_id: String,
    turn_id: String,
    content: String,
    state: ChatState,
    error_kind: Option<String>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<ChatMessage, LensError> {
    engine
        .save_chat_marker(
            &NotebookId::from(notebook_id),
            &turn_id,
            &content,
            state,
            error_kind.as_deref(),
        )
        .await
}

/// Sets or clears (`null`) feedback on a chat message (#22, toggleable thumbs).
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn set_chat_feedback(
    message_id: String,
    feedback: Option<ChatFeedback>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.set_chat_feedback(&message_id, feedback).await
}

/// Lists a notebook's chat messages as flat rows in transcript order (#22).
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn list_chat_messages(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<ChatMessage>, LensError> {
    engine
        .list_chat_messages(&NotebookId::from(notebook_id))
        .await
}

/// Renames a notebook.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn rename_notebook(
    id: String,
    title: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.rename_notebook(&NotebookId::from(id), &title).await
}

/// Bumps `last_activity_at` (records an "open" for cold-launch MRU auto-open).
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn touch_notebook_activity(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine
        .touch_notebook_activity(&NotebookId::from(notebook_id))
        .await
}

/// Soft-deletes a notebook (backward-compat alias for `trash_notebook`). Recoverable.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn delete_notebook(
    id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.trash_notebook(&NotebookId::from(id)).await
}

/// Soft-deletes a notebook: sets `trashed_at`, bumps `updated_at`. Recoverable.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn trash_notebook(
    id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.trash_notebook(&NotebookId::from(id)).await
}

/// Restores a trashed notebook: clears `trashed_at`, bumps `updated_at`.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn restore_notebook(
    id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.restore_notebook(&NotebookId::from(id)).await
}

/// Lists trashed notebooks with their source counts, newest-trashed first.
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_trashed(
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<NotebookSummary>, LensError> {
    engine.list_trashed_with_counts().await
}

/// Lists individually-trashed sources whose parent notebook is still live,
/// newest-trashed first. Used by the Trash modal Sources section (issue #94).
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_trashed_sources(
    engine: tauri::State<'_, LensEngine>,
) -> Result<Vec<TrashedSource>, LensError> {
    engine.list_trashed_sources().await
}

/// Permanently deletes a notebook (child rows cascade). Used by "Delete forever".
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn purge_notebook(
    id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine.purge_notebook(&NotebookId::from(id)).await
}

/// Sets a notebook's embedding model and re-embeds all chunks, streaming
/// `Started` → `Chunk`/`Progress` per batch → `Done` or `Failed`. Unknown model
/// ids are rejected against the registry before any work begins.
#[tracing::instrument(skip(on_progress, engine))]
#[tauri::command]
pub async fn set_notebook_embedding_model(
    notebook_id: String,
    model_id: String,
    backend: String,
    on_progress: Channel<StreamEvent<ReembedProgress>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    let nb_id = NotebookId::from(notebook_id);
    let backend = lens_core::EmbeddingBackend::from_opt_str(Some(&backend));

    engine
        .set_notebook_embedding_model(&nb_id, &model_id, backend)
        .await?;

    if let Err(e) = send_event(&on_progress, StreamEvent::Started) {
        tracing::warn!("set_notebook_embedding_model: started event send failed: {e}");
    }

    let result = engine
        .reembed_notebook(&nb_id, |done, total| {
            let progress = ReembedProgress { done, total };
            if let Err(e) = send_event(&on_progress, StreamEvent::Chunk(progress)) {
                tracing::warn!("set_notebook_embedding_model: progress chunk send failed: {e}");
            }
            if let Err(e) = send_event(
                &on_progress,
                StreamEvent::Progress {
                    done: done as u64,
                    total: Some(total as u64),
                },
            ) {
                tracing::warn!("set_notebook_embedding_model: progress event send failed: {e}");
            }
        })
        .await;

    match &result {
        Ok(_outcome) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Done) {
                tracing::warn!("set_notebook_embedding_model: done event send failed: {e}");
            }
        }
        Err(err) => {
            if let Err(e) = send_event(&on_progress, StreamEvent::Failed(err.clone())) {
                tracing::warn!("set_notebook_embedding_model: failed event send failed: {e}");
            }
        }
    }

    result.map(|_| ())
}

/// Returns the notebook's current embedding model id, dimension, backend, and index
/// status (`"active"` when a live row exists for the full coordinate, `"none"` otherwise).
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn get_notebook_embedding_model(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<EmbeddingModelInfo, LensError> {
    let nb_id = NotebookId::from(notebook_id);
    let (model_id, dim, backend, status) = engine.get_notebook_embedding_info(&nb_id).await?;
    Ok(EmbeddingModelInfo {
        model_id,
        dim,
        backend: backend.as_str().to_string(),
        status,
    })
}

/// Sets the per-notebook graph-retrieval override (#158b). Always writes `Some` —
/// the UI is binary On/Off and never clears the override to `None`.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn set_notebook_graph_retrieval_enabled(
    notebook_id: String,
    enabled: bool,
    engine: tauri::State<'_, LensEngine>,
) -> Result<(), LensError> {
    engine
        .set_notebook_graph_retrieval_enabled(&NotebookId::from(notebook_id), Some(enabled))
        .await
}

/// Returns the notebook's EFFECTIVE graph-retrieval setting (#158b): the per-notebook
/// override if set, else the app-wide default. The frontend reads this rather than the
/// raw `Option<bool>` because it does not know the global default.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn get_notebook_graph_retrieval_enabled(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<bool, LensError> {
    engine
        .notebook_graph_retrieval_enabled(&NotebookId::from(notebook_id))
        .await
}

/// Returns the latest eval verdict for a notebook (#158b), or `None` if it has never
/// run. Maps the engine `LatestEval` into the wire DTO.
#[tracing::instrument(skip(engine))]
#[tauri::command]
pub async fn latest_notebook_eval(
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<Option<EvalReportDto>, LensError> {
    let latest = engine
        .latest_notebook_eval(&NotebookId::from(notebook_id))
        .await?;
    Ok(latest.map(|l| EvalReportDto::from_report(l.report, l.ran_at)))
}

/// Runs the per-notebook graph-retrieval eval on demand (#158b), streaming
/// `Started` → `Chunk(phase)` per phase → `Done`/`Failed`, and returning the outcome.
/// Mirrors [`set_notebook_embedding_model`]'s Channel exec shape. A missing or
/// unreachable provider surfaces as both a `Failed` event and the returned `Err`.
#[tracing::instrument(skip(on_event, engine))]
#[tauri::command]
pub async fn run_notebook_graph_eval(
    notebook_id: String,
    on_event: Channel<StreamEvent<EvalPhaseDto>>,
    engine: tauri::State<'_, LensEngine>,
) -> Result<EvalOutcomeDto, LensError> {
    let nb_id = NotebookId::from(notebook_id);

    if let Err(e) = send_event(&on_event, StreamEvent::Started) {
        tracing::warn!("run_notebook_graph_eval: started event send failed: {e}");
    }

    let result = engine
        .run_graph_eval(&nb_id, |phase| {
            if let Err(e) = send_event(&on_event, StreamEvent::Chunk(phase.into())) {
                tracing::warn!("run_notebook_graph_eval: phase chunk send failed: {e}");
            }
        })
        .await;

    match result {
        Ok(outcome) => {
            if let Err(e) = send_event(&on_event, StreamEvent::Done) {
                tracing::warn!("run_notebook_graph_eval: done event send failed: {e}");
            }
            match outcome {
                lens_core::eval::EvalOutcome::Skipped { reason } => {
                    Ok(EvalOutcomeDto::Skipped { reason })
                }
                lens_core::eval::EvalOutcome::Ran(report) => {
                    // `EvalReport` carries no `ran_at`; re-read the row `run_graph_eval`
                    // just wrote to stamp the DTO with the persisted timestamp.
                    let ran_at = engine
                        .latest_notebook_eval(&nb_id)
                        .await?
                        .map(|l| l.ran_at)
                        .unwrap_or_default();
                    Ok(EvalOutcomeDto::Ran {
                        report: EvalReportDto::from_report(report, ran_at),
                    })
                }
            }
        }
        Err(err) => {
            if let Err(e) = send_event(&on_event, StreamEvent::Failed(err.clone())) {
                tracing::warn!("run_notebook_graph_eval: failed event send failed: {e}");
            }
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tauri::Manager;

    #[tokio::test]
    async fn list_is_empty_then_reflects_create() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        assert!(list_notebooks(engine.clone()).await.unwrap().is_empty());

        let created = create_notebook("My Notebook".into(), None, None, engine.clone())
            .await
            .unwrap();
        assert_eq!(created.title, "My Notebook");

        let listed = list_notebooks(engine).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].notebook.id, created.id);
    }

    #[tokio::test]
    async fn create_notebook_persists_description_and_focus_mode() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let created = create_notebook(
            "Research".into(),
            Some("My notes".into()),
            Some("research".into()),
            engine.clone(),
        )
        .await
        .unwrap();
        assert_eq!(created.description.as_deref(), Some("My notes"));
        assert_eq!(created.focus_mode.as_deref(), Some("research"));

        let listed = list_notebooks(engine).await.unwrap();
        assert_eq!(listed[0].notebook.description.as_deref(), Some("My notes"));
        assert_eq!(listed[0].notebook.focus_mode.as_deref(), Some("research"));
    }

    #[tokio::test]
    async fn add_source_then_list_sources_scoped_by_notebook() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("NB".into(), None, None, engine.clone())
            .await
            .unwrap();
        let other = create_notebook("Other".into(), None, None, engine.clone())
            .await
            .unwrap();

        let src = add_source(
            nb.id.to_string(),
            "report.pdf".into(),
            "/abs/path/report.pdf".into(),
            engine.clone(),
        )
        .await
        .unwrap()
        .source;
        assert_eq!(src.kind, lens_core::parse::SourceKind::File);
        assert_eq!(src.status, lens_core::notebooks::SourceStatus::Pending);
        assert_eq!(src.locator, "/abs/path/report.pdf");
        assert_eq!(src.selected, 1);

        let sources = list_sources(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].id, src.id);

        assert!(
            list_sources(other.id.to_string(), engine)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn rename_then_delete_is_soft() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("Original".into(), None, None, engine.clone())
            .await
            .unwrap();
        rename_notebook(nb.id.to_string(), "Renamed".into(), engine.clone())
            .await
            .unwrap();
        let listed = list_notebooks(engine.clone()).await.unwrap();
        assert_eq!(listed[0].notebook.title, "Renamed");

        delete_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert!(list_notebooks(engine.clone()).await.unwrap().is_empty());
        let trashed = list_trashed(engine).await.unwrap();
        assert_eq!(trashed.len(), 1);
        assert_eq!(trashed[0].notebook.id, nb.id);
        assert!(trashed[0].notebook.trashed_at.is_some());
    }

    #[tokio::test]
    async fn list_notebooks_includes_source_count() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("NB".into(), None, None, engine.clone())
            .await
            .unwrap();

        let listed = list_notebooks(engine.clone()).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].source_count, 0);

        for i in 0..2 {
            add_source(
                nb.id.to_string(),
                format!("file{i}.pdf"),
                format!("/abs/file{i}.pdf"),
                engine.clone(),
            )
            .await
            .unwrap();
        }
        let listed = list_notebooks(engine).await.unwrap();
        assert_eq!(listed[0].source_count, 2);
    }

    #[tokio::test]
    async fn trash_restore_purge_lifecycle() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("NB".into(), None, None, engine.clone())
            .await
            .unwrap();

        trash_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert!(list_notebooks(engine.clone()).await.unwrap().is_empty());
        assert_eq!(list_trashed(engine.clone()).await.unwrap().len(), 1);

        restore_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert_eq!(list_notebooks(engine.clone()).await.unwrap().len(), 1);
        assert!(list_trashed(engine.clone()).await.unwrap().is_empty());

        trash_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        purge_notebook(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert!(list_notebooks(engine.clone()).await.unwrap().is_empty());
        assert!(list_trashed(engine).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn set_embedding_model_persists_and_get_returns_it() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("Embed Test".into(), None, None, engine.clone())
            .await
            .unwrap();

        let info = get_notebook_embedding_model(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert_eq!(info.model_id, "nomic-embed-text-v1.5");
        assert_eq!(info.dim, 768);
        assert_eq!(info.backend, "fastembed");
        assert_eq!(info.status, "none");

        engine
            .set_notebook_embedding_model(
                &lens_core::NotebookId::from(nb.id.to_string()),
                "mxbai-embed-large",
                lens_core::EmbeddingBackend::Fastembed,
            )
            .await
            .unwrap();

        let info2 = get_notebook_embedding_model(nb.id.to_string(), engine.clone())
            .await
            .unwrap();
        assert_eq!(info2.model_id, "mxbai-embed-large");
        assert_eq!(info2.dim, 1024);
        assert_eq!(info2.backend, "fastembed");
        assert_eq!(info2.status, "none");
    }

    #[tokio::test]
    async fn set_embedding_model_rejects_unknown_id() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("Reject Test".into(), None, None, engine.clone())
            .await
            .unwrap();

        let err = engine
            .set_notebook_embedding_model(
                &lens_core::NotebookId::from(nb.id.to_string()),
                "totally-unknown-model",
                lens_core::EmbeddingBackend::Fastembed,
            )
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unknown embedding model id"),
            "expected validation error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn trash_restore_purge_source_lifecycle() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("NB".into(), None, None, engine.clone())
            .await
            .unwrap();
        let src = add_source(
            nb.id.to_string(),
            "report.pdf".into(),
            "/abs/report.pdf".into(),
            engine.clone(),
        )
        .await
        .unwrap();

        assert_eq!(
            list_sources(nb.id.to_string(), engine.clone())
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(
            list_trashed_sources(engine.clone())
                .await
                .unwrap()
                .is_empty()
        );

        trash_source(src.source.id.clone(), engine.clone())
            .await
            .unwrap();
        assert!(
            list_sources(nb.id.to_string(), engine.clone())
                .await
                .unwrap()
                .is_empty()
        );
        let trashed = list_trashed_sources(engine.clone()).await.unwrap();
        assert_eq!(trashed.len(), 1);
        assert_eq!(trashed[0].source.id, src.source.id);
        assert_eq!(trashed[0].notebook_title, "NB");

        restore_source(src.source.id.clone(), engine.clone())
            .await
            .unwrap();
        assert_eq!(
            list_sources(nb.id.to_string(), engine.clone())
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(
            list_trashed_sources(engine.clone())
                .await
                .unwrap()
                .is_empty()
        );

        trash_source(src.source.id.clone(), engine.clone())
            .await
            .unwrap();
        purge_source(src.source.id.clone(), engine.clone())
            .await
            .unwrap();
        assert!(
            list_sources(nb.id.to_string(), engine.clone())
                .await
                .unwrap()
                .is_empty()
        );
        assert!(list_trashed_sources(engine).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn graph_retrieval_toggle_persists_and_survives() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("NB".into(), None, None, engine.clone())
            .await
            .unwrap();
        let id = nb.id.to_string();

        // Effective default is off (global default false, no override).
        assert!(
            !get_notebook_graph_retrieval_enabled(id.clone(), engine.clone())
                .await
                .unwrap()
        );

        set_notebook_graph_retrieval_enabled(id.clone(), true, engine.clone())
            .await
            .unwrap();
        assert!(
            get_notebook_graph_retrieval_enabled(id.clone(), engine.clone())
                .await
                .unwrap(),
            "set true is read back as effective true"
        );

        set_notebook_graph_retrieval_enabled(id.clone(), false, engine.clone())
            .await
            .unwrap();
        assert!(
            !get_notebook_graph_retrieval_enabled(id.clone(), engine.clone())
                .await
                .unwrap(),
            "set false survives"
        );
    }

    #[tokio::test]
    async fn latest_notebook_eval_none_then_row() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let nb = create_notebook("NB".into(), None, None, engine.clone())
            .await
            .unwrap();
        let id = nb.id.to_string();

        // No eval yet → None.
        assert!(
            latest_notebook_eval(id.clone(), engine.clone())
                .await
                .unwrap()
                .is_none()
        );

        // Seed one log row and read it back as an EvalReportDto.
        let pool = engine.pool().await;
        sqlx::query(
            "INSERT INTO notebook_eval_log \
             (id, notebook_id, ran_at, graph_recall, hybrid_recall, delta_pp, p95_ms, passed, \
              sample_n, dropped_n, graph_enabled, prompt_version, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(uuid::Uuid::now_v7().to_string())
        .bind(&id)
        .bind("2026-07-12T00:00:00Z")
        .bind(0.8_f64)
        .bind(0.5_f64)
        .bind(30.0_f64)
        .bind(120.0_f64)
        .bind(1_i64)
        .bind(24_i64)
        .bind(2_i64)
        .bind(1_i64)
        .bind("158a-qa-v2")
        .bind("2026-07-12T00:00:00Z")
        .execute(&pool)
        .await
        .unwrap();

        let dto = latest_notebook_eval(id, engine)
            .await
            .unwrap()
            .expect("row present");
        assert_eq!(dto.ran_at, "2026-07-12T00:00:00Z");
        assert_eq!(dto.sample_n, 24);
        assert_eq!(dto.dropped_n, 2);
        assert!(dto.passed);
        assert!(dto.graph_enabled);
        assert!((dto.delta_pp - 30.0).abs() < 1e-4);
    }
}
