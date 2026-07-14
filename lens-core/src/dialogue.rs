//! Grounded dialogue-script generation (issue #26): the first stage of the M7
//! audio-overview pipeline. Turns a notebook's selected+live sources into a
//! validated, typed two-speaker (Host/Guest) [`DialogueScript`] via one
//! `tiered_search` overview retrieval and a one-shot [`LlmProvider::generate`].
//!
//! [`generate_dialogue`] is a pure free fn over an owned [`DialogueCtx`]; the
//! fallible ctx-gathering lives in `LensEngine::generate_dialogue` (lib.rs). No
//! audio synthesis, no persistence, no UI — those are #27/#28/#161/#29.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tokenizers::Tokenizer;
use tokio_util::sync::CancellationToken;

use crate::config::{ModelConfig, RetrievalConfig, TierThresholds};
use crate::embedder::Embedder;
use crate::error::LensError;
use crate::graph::NotebookGraph;
use crate::llm::{LlmProvider, LlmRequest};
use crate::retrieval::Reranker;
use crate::retrieval::router::{ContextUnit, tiered_search};
use crate::vector_store::{Coordinate, VectorStore};

/// Higher than answer's 0.1 for conversational liveliness; determinism is unneeded
/// — output is validated, not exact-matched.
const DIALOGUE_TEMPERATURE: f32 = 0.7;

/// Synthetic query driving the overview retrieval — the dialogue covers the
/// notebook broadly, not a single user question.
const OVERVIEW_QUERY: &str = "key topics, findings, and takeaways across these sources";

/// The turn-object JSON schema, shared verbatim by the initial prompt and the
/// repair instruction so a future schema change updates one place (cf.
/// `citation::CITATION_PROMPT_INSTRUCTION`).
const TURN_SCHEMA_HINT: &str = "{\"speaker\": \"host\"|\"guest\", \"text\": string, \"emotion\": string (optional), \"source_ids\": [string]}";

/// The "emit only a bare JSON array of turn objects" instruction, shared by the
/// initial prompt and the repair instruction.
const JSON_ARRAY_ONLY_INSTRUCTION: &str = "Return ONLY a JSON array of turn objects — no prose, no markdown fences — where each object is";

/// Cancellation message shared by every cancel check / `select!` arm in
/// [`generate_dialogue`].
const CANCELLED_MSG: &str = "dialogue generation cancelled";

/// A validated two-speaker dialogue script. Serializes as `{ "turns": [...] }`; the
/// model is prompted to emit the bare `turns` array, which [`parse_dialogue`]
/// salvages into this shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DialogueScript {
    pub turns: Vec<Turn>,
}

/// One dialogue turn. `emotion` is engine-forward metadata (Kokoro ignores it;
/// Orpheus #161 renders it); an absent/unknown value deserializes to `None` and is
/// never a validation failure. `source_ids` may be empty for ungrounded turns
/// (natural transitions/backchannels).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub speaker: Speaker,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emotion: Option<Emotion>,
    #[serde(default)]
    pub source_ids: Vec<String>,
}

/// The two dialogue voices. `Host` binds to `VoiceConfig.host`, `Guest` to
/// `VoiceConfig.guest` (config.rs) — voice binding itself is #28/#161.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Speaker {
    Host,
    Guest,
}

/// Per-turn delivery hint. See [`Turn::emotion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Emotion {
    Neutral,
    Laugh,
    Sigh,
    Excited,
    Thoughtful,
}

/// Requested script length. Drives the target turn count, the hard `min_turns`
/// floor, and the `max_tokens` output budget via [`Length::preset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Length {
    Short,
    Medium,
    Long,
}

/// Honest phase markers carried over the command's progress channel. They do NOT
/// fire smoothly — the model call emits nothing for most of a Long run's wall-clock,
/// so #29 must render an indeterminate spinner during `Generating`. The empty-
/// notebook path stops at `Retrieving` (never `Generating`/`Validating`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DialoguePhase {
    Retrieving,
    Generating,
    Validating,
}

/// Resolved length parameters. `target_turns` is a soft prompt target only;
/// `min_turns` and `max_tokens` are enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthPreset {
    pub target_turns: usize,
    pub min_turns: usize,
    pub max_tokens: u32,
}

impl Length {
    pub fn preset(self) -> LengthPreset {
        match self {
            Length::Short => LengthPreset {
                target_turns: 12,
                min_turns: 8,
                max_tokens: 3_072,
            },
            Length::Medium => LengthPreset {
                target_turns: 25,
                min_turns: 16,
                max_tokens: 6_144,
            },
            Length::Long => LengthPreset {
                target_turns: 45,
                min_turns: 30,
                max_tokens: 8_192,
            },
        }
    }
}

/// Owned, `Send` bundle the pure [`generate_dialogue`] needs. Mirrors `AnswerCtx`
/// (answer.rs) — every field is owned so the orchestrator future is `Send`.
/// `selected_live_ids` is the FULL selected+live source set (not the retrieval
/// subset), the validator's grounding allow-list.
pub struct DialogueCtx {
    pub provider: Arc<dyn LlmProvider>,
    pub store: Arc<dyn VectorStore>,
    pub embedder: Arc<dyn Embedder>,
    pub reranker: Reranker,
    pub graph: Option<Arc<NotebookGraph>>,
    pub pool: SqlitePool,
    pub coord: Coordinate,
    pub model: ModelConfig,
    pub retrieval: RetrievalConfig,
    pub thresholds: TierThresholds,
    pub tokenizer: Option<Arc<Tokenizer>>,
    pub length: Length,
    pub selected_live_ids: HashSet<String>,
}

/// Builds the `(system, user)` prompt from the retrieved units. Units are numbered
/// by Vec slice position (`[i+1]`), matching the grounded-answer citation contract
/// (answer.rs). `title` falls back to the raw `source_id` when absent.
fn build_dialogue_prompt(
    units: &[ContextUnit],
    titles: &HashMap<String, String>,
    preset: &LengthPreset,
) -> (String, String) {
    let mut blocks = String::new();
    for (i, u) in units.iter().enumerate() {
        let title = titles.get(&u.source_id).unwrap_or(&u.source_id);
        blocks.push_str(&format!(
            "[{}] ({}) source_id={}: {}\n",
            i + 1,
            title,
            u.source_id,
            u.text
        ));
    }
    let system = format!(
        "You are scripting a two-speaker audio overview between a Host and a Guest, \
         grounded strictly in the numbered source units below. Produce about {turns} \
         turns that alternate between the two speakers, staying conversational and \
         natural. Cite sources by putting their exact source_id values in each turn's \
         `source_ids` array; leave `source_ids` empty for pure transitions or \
         backchannels. Where a line is naturally delivered with feeling, set \
         `emotion` to one of: neutral, laugh, sigh, excited, thoughtful.\n\n\
         {JSON_ARRAY_ONLY_INSTRUCTION} {TURN_SCHEMA_HINT}.\n\n\
         Source units:\n{blocks}",
        turns = preset.target_turns,
    );
    let user = "Write the dialogue script now as a JSON array of turns.".to_string();
    (system, user)
}

/// Deterministic trailing-comma removal before the first parse — cheap, and saves a
/// repair round for the single most common structural defect. Only rewrites when a
/// `,` immediately precedes a `]`/`}` (ignoring whitespace) OUTSIDE a string.
fn preclean(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut in_str = false;
    let mut escaped = false;
    let mut needs = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_str = true;
        } else if b == b',' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < bytes.len() && (bytes[j] == b']' || bytes[j] == b'}') {
                needs = true;
                break;
            }
        }
        i += 1;
    }
    if !needs {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut in_str = false;
    let mut escaped = false;
    let mut iter = text.char_indices().peekable();
    while let Some((_, c)) = iter.next() {
        if in_str {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        if c == '"' {
            in_str = true;
            out.push(c);
            continue;
        }
        if c == ',' {
            let mut probe = iter.clone();
            let mut next_significant = None;
            for (_, pc) in probe.by_ref() {
                if !pc.is_ascii_whitespace() {
                    next_significant = Some(pc);
                    break;
                }
            }
            if matches!(next_significant, Some(']') | Some('}')) {
                continue;
            }
        }
        out.push(c);
    }
    Cow::Owned(out)
}

/// Extracts the first balanced `[...]` JSON array from `text` so a response wrapped
/// in markdown fences or preamble still parses. Tracks `[`/`]` depth AND `{`/`}`
/// nesting AND in-string/escape state so embedded brackets inside strings or objects
/// never terminate the walk early. Returns `None` if no balanced array is found — a
/// truncated/broken array falls through to repair. A non-trivial bracket-balancing
/// extension of `enrichment::meta::extract_json_object` (which only balances braces).
fn extract_json_array(text: &str) -> Option<&str> {
    let start = text.find('[')?;
    let bytes = text.as_bytes();
    let mut bracket_depth = 0usize;
    let mut brace_depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => brace_depth += 1,
            b'}' => brace_depth = brace_depth.saturating_sub(1),
            b'[' => bracket_depth += 1,
            b']' => {
                bracket_depth = bracket_depth.saturating_sub(1);
                if bracket_depth == 0 && brace_depth == 0 {
                    return Some(&text[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Salvage-parses the model output into a [`DialogueScript`]: pre-clean trailing
/// commas → try the bare `[...]` array form → fall back to the `{"turns":[...]}`
/// object form → map serde errors to [`LensError::Parse`]. Never panics.
fn parse_dialogue(text: &str) -> Result<DialogueScript, LensError> {
    let cleaned = preclean(text);
    // The array form is tried first but must NOT short-circuit on failure: a
    // leading non-turns array (e.g. a `meta` list before `turns`) is bracket-
    // balanced and extracted, yet fails to parse as `Vec<Turn>` — that failure has
    // to fall through to the object-wrapper form rather than propagate via `?`.
    if let Some(arr) = extract_json_array(&cleaned)
        && let Ok(turns) = serde_json::from_str::<Vec<Turn>>(arr)
    {
        return Ok(DialogueScript { turns });
    }
    if let Some(obj) = crate::enrichment::meta::extract_json_object(&cleaned) {
        let script: DialogueScript = serde_json::from_str(obj)?;
        return Ok(script);
    }
    Err(LensError::Parse(
        "no JSON array or object found in model output".into(),
    ))
}

/// Validates the soft-alternation bar (AC5): `turns.len() >= min_turns`, both
/// speakers present, no speaker running >2 consecutive turns, and every PRESENT
/// `source_id` ∈ `selected_live_ids` (the full selected+live set). Empty
/// `source_ids` is permitted. Breach → [`LensError::Validation`].
fn validate_script(
    script: &DialogueScript,
    selected_live_ids: &HashSet<String>,
    min_turns: usize,
) -> Result<(), LensError> {
    if script.turns.len() < min_turns {
        return Err(LensError::Validation("dialogue script too short".into()));
    }
    let has_host = script.turns.iter().any(|t| t.speaker == Speaker::Host);
    let has_guest = script.turns.iter().any(|t| t.speaker == Speaker::Guest);
    if !has_host || !has_guest {
        return Err(LensError::Validation(
            "dialogue requires both speakers".into(),
        ));
    }
    let mut run = 0usize;
    let mut prev: Option<Speaker> = None;
    for t in &script.turns {
        if prev == Some(t.speaker) {
            run += 1;
        } else {
            run = 1;
            prev = Some(t.speaker);
        }
        if run > 2 {
            return Err(LensError::Validation(
                "speaker exceeded 2 consecutive turns".into(),
            ));
        }
    }
    for t in &script.turns {
        for id in &t.source_ids {
            if !selected_live_ids.contains(id) {
                return Err(LensError::Validation("turn cites an unknown source".into()));
            }
        }
    }
    Ok(())
}

/// The shared non-content `LlmRequest` fields for both the initial and repair
/// calls, so temperature/json/thinking can't drift between the two sites.
fn base_request(system: String, prompt: String, preset: &LengthPreset) -> LlmRequest {
    LlmRequest {
        system: Some(system),
        prompt,
        max_tokens: preset.max_tokens,
        temperature: DIALOGUE_TEMPERATURE,
        json: true,
        thinking: false,
        reasoning_effort: None,
    }
}

/// Builds the fix-oriented repair request from the prior malformed output and the
/// specific failure reason. The instruction differs for a parse failure vs a
/// validation failure; a single repair covers both.
fn build_repair_request(prior: &str, failure: &LensError, preset: &LengthPreset) -> LlmRequest {
    let reason = match failure {
        LensError::Parse(msg) => {
            format!("your previous output could not be parsed as JSON (parse error: {msg})")
        }
        LensError::Validation(msg) => format!(
            "your previous output was valid JSON but failed a content rule: {msg}. \
             Remember: at least {min} turns, both a host and a guest must speak, no \
             speaker may take more than two turns in a row, and every source_id must \
             be one copied exactly from a numbered source unit (or the array left \
             empty)",
            min = preset.min_turns,
        ),
        other => format!("your previous output failed: {other}"),
    };
    let system = format!(
        "You are repairing a malformed dialogue-script response. {reason}. Here is \
         your previous output:\n\n{prior}\n\nReturn ONLY a corrected JSON array of \
         turn objects — no prose, no markdown fences — where each object is \
         {TURN_SCHEMA_HINT}."
    );
    base_request(
        system,
        "Return the corrected JSON array of turns now.".to_string(),
        preset,
    )
}

/// The grounded dialogue-script orchestrator. Pure over the owned [`DialogueCtx`] so
/// the returned future is `Send`. Emits phase markers via `on_phase`, buffers the
/// full model text before parse/validate, and races every `generate()` against
/// `cancel` so an in-flight one-shot call is truly interruptible. Returns the
/// validated script or a terminal [`LensError`]; never a partial script.
pub async fn generate_dialogue(
    ctx: DialogueCtx,
    cancel: CancellationToken,
    on_phase: impl Fn(DialoguePhase) + Send,
) -> Result<DialogueScript, LensError> {
    let preset = ctx.length.preset();

    on_phase(DialoguePhase::Retrieving);
    if cancel.is_cancelled() {
        return Err(LensError::Cancelled(CANCELLED_MSG.into()));
    }

    // Embed the overview query fully OFF the async runtime — the fastembed
    // `std::sync::Mutex` guard must never straddle an await (mirror answer.rs).
    let embedder = ctx.embedder.clone();
    let qvec = match tokio::task::spawn_blocking(move || embedder.embed_query(OVERVIEW_QUERY)).await
    {
        Ok(Ok(v)) => v,
        Ok(Err(err)) => return Err(err),
        Err(join) => return Err(LensError::from(join)),
    };

    if cancel.is_cancelled() {
        return Err(LensError::Cancelled(CANCELLED_MSG.into()));
    }

    let out = tiered_search(
        &ctx.pool,
        &*ctx.store,
        &ctx.reranker,
        ctx.graph.as_deref(),
        &ctx.coord,
        OVERVIEW_QUERY,
        &qvec,
        &ctx.model,
        ctx.retrieval.answer_candidate_pool,
        &ctx.retrieval,
        Some(ctx.retrieval.graph_retrieval_enabled),
        &ctx.thresholds,
        ctx.tokenizer.as_deref(),
    )
    .await?;

    // Empty selected+live corpus → a zero-source dialogue is meaningless. Diverges
    // from answer.rs's refusal: fail with ZERO LLM calls (no Generating/Validating).
    if out.units.is_empty() {
        return Err(LensError::Validation(
            "notebook has no selected sources to ground on".into(),
        ));
    }

    let ids: Vec<&str> = out.units.iter().map(|u| u.source_id.as_str()).collect();
    let titles = crate::citation::source_titles(&ctx.pool, &ids).await?;

    if cancel.is_cancelled() {
        return Err(LensError::Cancelled(CANCELLED_MSG.into()));
    }

    let (system, prompt) = build_dialogue_prompt(&out.units, &titles, &preset);
    // tiered_search budgets input against the fixed RESERVED_OUTPUT=2048, so Long
    // (8192) can overcommit input on small-context models; salvage-parse + one
    // repair degrade this safely. Re-budgeting the shared router is out of #26 scope.
    let req = base_request(system, prompt, &preset);

    on_phase(DialoguePhase::Generating);

    let resp = tokio::select! {
        r = ctx.provider.generate(&req) => r?,
        _ = cancel.cancelled() => {
            return Err(LensError::Cancelled(CANCELLED_MSG.into()));
        }
    };

    let first_attempt = parse_dialogue(&resp.text).and_then(|script| {
        validate_script(&script, &ctx.selected_live_ids, preset.min_turns).map(|()| script)
    });

    let script = match first_attempt {
        Ok(script) => script,
        Err(failure) => {
            let repair = build_repair_request(&resp.text, &failure, &preset);
            let repaired = tokio::select! {
                r = ctx.provider.generate(&repair) => r?,
                _ = cancel.cancelled() => {
                    return Err(LensError::Cancelled(CANCELLED_MSG.into()));
                }
            };
            let script = parse_dialogue(&repaired.text)?;
            validate_script(&script, &ctx.selected_live_ids, preset.min_turns)?;
            script
        }
    };

    on_phase(DialoguePhase::Validating);
    Ok(script)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(speaker: Speaker, text: &str) -> Turn {
        Turn {
            speaker,
            text: text.to_string(),
            emotion: None,
            source_ids: Vec::new(),
        }
    }

    // ---- Step 1: schema + presets ----

    #[test]
    fn dialogue_script_serde_round_trip() {
        let script = DialogueScript {
            turns: vec![
                Turn {
                    speaker: Speaker::Host,
                    text: "Welcome.".into(),
                    emotion: Some(Emotion::Excited),
                    source_ids: vec!["s1".into()],
                },
                turn(Speaker::Guest, "Thanks."),
            ],
        };
        let json = serde_json::to_string(&script).unwrap();
        let back: DialogueScript = serde_json::from_str(&json).unwrap();
        assert_eq!(script, back);
    }

    #[test]
    fn enums_serialize_snake_case() {
        assert_eq!(serde_json::to_string(&Speaker::Host).unwrap(), "\"host\"");
        assert_eq!(serde_json::to_string(&Speaker::Guest).unwrap(), "\"guest\"");
        assert_eq!(
            serde_json::to_string(&Emotion::Thoughtful).unwrap(),
            "\"thoughtful\""
        );
        assert_eq!(
            serde_json::to_string(&Length::Medium).unwrap(),
            "\"medium\""
        );
        assert_eq!(
            serde_json::to_string(&DialoguePhase::Retrieving).unwrap(),
            "\"retrieving\""
        );
    }

    #[test]
    fn length_presets_table() {
        for (len, tokens) in [
            (Length::Short, 3_072u32),
            (Length::Medium, 6_144),
            (Length::Long, 8_192),
        ] {
            let p = len.preset();
            assert_eq!(p.max_tokens, tokens);
            assert!(p.min_turns <= p.target_turns);
        }
    }

    // ---- Step 2: prompt builder ----

    fn unit(source_id: &str, text: &str) -> ContextUnit {
        use crate::retrieval::HitSource;
        use crate::retrieval::router::Provenance;
        ContextUnit {
            text: text.to_string(),
            source_id: source_id.to_string(),
            chunk_id: format!("c-{source_id}"),
            parent_id: None,
            locator: None,
            order_index: 0,
            provenance: Provenance {
                source: HitSource::Dense,
                graph_confidence: None,
            },
        }
    }

    #[test]
    fn prompt_numbers_units_by_position_and_contains_target_and_json_instruction() {
        let units = vec![unit("sA", "alpha"), unit("sB", "beta")];
        let titles = HashMap::new();
        let preset = Length::Medium.preset();
        let (system, _user) = build_dialogue_prompt(&units, &titles, &preset);
        assert!(system.contains("[1] (sA) source_id=sA: alpha"));
        assert!(system.contains("[2] (sB) source_id=sB: beta"));
        assert!(system.contains(&preset.target_turns.to_string()));
        assert!(system.contains("ONLY a JSON array"));
    }

    #[test]
    fn prompt_title_falls_back_to_source_id() {
        let units = vec![unit("src-xyz", "body")];
        let mut titles = HashMap::new();
        titles.insert("src-xyz".to_string(), "My Title".to_string());
        let preset = Length::Short.preset();
        let (with_title, _) = build_dialogue_prompt(&units, &titles, &preset);
        assert!(with_title.contains("[1] (My Title) source_id=src-xyz: body"));
    }

    // ---- Step 3: preclean + extract_json_array + parse ----

    #[test]
    fn extract_json_array_handles_embedded_brackets_and_strings() {
        let input = r#"[{"text":"has ] and { chars"},{"text":"ok"}]"#;
        let extracted = extract_json_array(input).expect("balanced array");
        assert_eq!(extracted, input);
    }

    #[test]
    fn extract_json_array_terminates_at_correct_close_with_trailing_prose() {
        let input = r#"prose [{"text":"a"}] trailing ] junk"#;
        let extracted = extract_json_array(input).expect("balanced array");
        assert_eq!(extracted, r#"[{"text":"a"}]"#);
    }

    #[test]
    fn extract_json_array_broken_nested_returns_none() {
        let input = r#"[{"text":"unterminated"#;
        assert!(extract_json_array(input).is_none());
    }

    #[test]
    fn extract_json_array_stray_close_bracket_never_panics() {
        // A stray unquoted `]` while bracket_depth==0 and brace_depth>0 must not
        // underflow bracket_depth's saturating_sub (regression: previously an
        // unconditional `-= 1` panicked here with "attempt to subtract with
        // overflow"). The walk only balances brackets/braces — it is not a JSON
        // validator — so a syntactically-invalid-but-bracket-balanced input may
        // still return `Some`; the downstream `serde_json::from_str` in
        // `parse_dialogue` is what rejects it (falling to repair). The contract
        // under test here is solely: never panic.
        assert!(extract_json_array("[{]]").is_none());
        let _ = extract_json_array(r#"[{"a": ]] }]"#); // must not panic
    }

    #[test]
    fn parse_dialogue_stray_close_bracket_input_errors_without_panicking() {
        // End-to-end: a bracket-balanced-but-invalid-JSON input must surface as a
        // clean Parse error, never a panic, regardless of how extract_json_array's
        // pre-filter resolves it.
        assert!(matches!(parse_dialogue("[{]]"), Err(LensError::Parse(_))));
        assert!(matches!(
            parse_dialogue(r#"[{"a": ]] }]"#),
            Err(LensError::Parse(_))
        ));
    }

    #[test]
    fn preclean_strips_trailing_commas_outside_strings() {
        let input = r#"[{"a":1,},]"#;
        assert_eq!(preclean(input), r#"[{"a":1}]"#);
    }

    #[test]
    fn preclean_preserves_commas_inside_strings() {
        let input = r#"[{"text":"a, b, c"}]"#;
        assert_eq!(preclean(input), input);
    }

    #[test]
    fn parse_dialogue_recovers_from_fences_and_prose() {
        let text = "Here you go:\n```json\n[{\"speaker\":\"host\",\"text\":\"hi\"}]\n```\nthanks";
        let script = parse_dialogue(text).unwrap();
        assert_eq!(script.turns.len(), 1);
        assert_eq!(script.turns[0].speaker, Speaker::Host);
    }

    #[test]
    fn parse_dialogue_recovers_object_wrapper() {
        // A leading non-turns array (`meta`) means extract_json_array's bare-array
        // path grabs `[1,2]` first, which fails to parse as `Vec<Turn>` — so this
        // fixture genuinely exercises the `extract_json_object` fallback branch,
        // unlike a bare `{"turns":[...]}` (whose `[` is the turns array itself and
        // never reaches the object branch).
        let text = r#"{"meta":[1,2],"turns":[{"speaker":"host","text":"hi"},{"speaker":"guest","text":"yo"}]}"#;
        let script = parse_dialogue(text).unwrap();
        assert_eq!(script.turns.len(), 2);
        assert_eq!(script.turns[0].speaker, Speaker::Host);
        assert_eq!(script.turns[1].speaker, Speaker::Guest);
    }

    #[test]
    fn parse_dialogue_recovers_trailing_comma() {
        let text = r#"[{"speaker":"host","text":"a"},{"speaker":"guest","text":"b"},]"#;
        let script = parse_dialogue(text).unwrap();
        assert_eq!(script.turns.len(), 2);
    }

    #[test]
    fn parse_dialogue_no_json_is_parse_error() {
        let err = parse_dialogue("just some prose, no json here").unwrap_err();
        assert!(matches!(err, LensError::Parse(_)));
    }

    // ---- Emotion round-trip (C2b) ----

    #[test]
    fn emotion_round_trips_and_unknown_deserializes_to_none() {
        let script = DialogueScript {
            turns: vec![Turn {
                speaker: Speaker::Host,
                text: "haha".into(),
                emotion: Some(Emotion::Laugh),
                source_ids: Vec::new(),
            }],
        };
        let json = serde_json::to_string(&script).unwrap();
        let back: DialogueScript = serde_json::from_str(&json).unwrap();
        assert_eq!(back.turns[0].emotion, Some(Emotion::Laugh));

        // Missing/unknown emotion → None, never a hard failure. A two-speaker,
        // ungrounded script with no emotion fields still passes validation.
        let missing =
            r#"[{"speaker":"host","text":"plain"},{"speaker":"guest","text":"also plain"}]"#;
        let parsed = parse_dialogue(missing).unwrap();
        assert_eq!(parsed.turns[0].emotion, None);
        assert_eq!(parsed.turns[1].emotion, None);
        let ids = HashSet::new();
        assert!(validate_script(&parsed, &ids, 2).is_ok());
    }

    // ---- Step 4: validator ----

    fn h(t: &str) -> Turn {
        turn(Speaker::Host, t)
    }
    fn g(t: &str) -> Turn {
        turn(Speaker::Guest, t)
    }

    #[test]
    fn validate_rejects_too_short() {
        let script = DialogueScript {
            turns: vec![h("a"), g("b")],
        };
        let err = validate_script(&script, &HashSet::new(), 4).unwrap_err();
        assert!(matches!(err, LensError::Validation(_)));
    }

    #[test]
    fn validate_rejects_single_speaker() {
        let script = DialogueScript {
            turns: vec![h("a"), h("b")],
        };
        // min satisfied, but only one speaker present.
        let err = validate_script(&script, &HashSet::new(), 2).unwrap_err();
        assert!(matches!(err, LensError::Validation(m) if m.contains("both speakers")));
    }

    #[test]
    fn validate_rejects_three_consecutive_same_speaker() {
        let script = DialogueScript {
            turns: vec![h("a"), h("b"), h("c"), g("d")],
        };
        let err = validate_script(&script, &HashSet::new(), 2).unwrap_err();
        assert!(matches!(err, LensError::Validation(m) if m.contains("consecutive")));
    }

    #[test]
    fn validate_allows_exactly_two_consecutive() {
        let script = DialogueScript {
            turns: vec![h("a"), h("b"), g("c"), g("d")],
        };
        assert!(validate_script(&script, &HashSet::new(), 2).is_ok());
    }

    #[test]
    fn validate_rejects_unknown_source_id() {
        let mut t = h("cite");
        t.source_ids = vec!["ghost".into()];
        let script = DialogueScript {
            turns: vec![t, g("b")],
        };
        let err = validate_script(&script, &HashSet::new(), 2).unwrap_err();
        assert!(matches!(err, LensError::Validation(m) if m.contains("unknown source")));
    }

    #[test]
    fn validate_allows_empty_source_ids_ungrounded_turn() {
        let script = DialogueScript {
            turns: vec![h("transition"), g("reply")],
        };
        assert!(validate_script(&script, &HashSet::new(), 2).is_ok());
    }

    #[test]
    fn validate_allows_cited_selected_but_unretrieved_source() {
        // The id resolves against the FULL selected+live set, not just retrieved units.
        let mut t = h("cite");
        t.source_ids = vec!["selected-not-retrieved".into()];
        let script = DialogueScript {
            turns: vec![t, g("b")],
        };
        let mut ids = HashSet::new();
        ids.insert("selected-not-retrieved".to_string());
        assert!(validate_script(&script, &ids, 2).is_ok());
    }
}
