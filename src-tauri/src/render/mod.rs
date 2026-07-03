//! `TauriJsRenderer` — offscreen-webview implementation of `lens_core::JsRenderer`
//! (issue #78). `lens-core` cannot depend on `tauri`, so webview machinery lives
//! here and is injected via `LensEngine::set_js_renderer`.
//!
//! ## Capture model
//! SPAs render content after load, so one-shot `PageLoadEvent::Finished` captures an
//! empty shell. Instead [`INIT_JS`] installs a quiescence detector (network-idle via
//! fetch/XHR counters + DOM-idle via `MutationObserver`) and records the largest
//! `outerHTML` seen on `window.__lensBest`. The Rust side polls every [`POLL_INTERVAL`]
//! and accepts quiescence only once [`MIN_RENDER_WAIT`] has elapsed AND visible body
//! text grew past the first-poll shell baseline by [`CONTENT_GROWTH_MIN`] chars
//! (guards against accepting the idle gap right after load before hydration begins).
//! On timeout, the best capture seen is returned; `None` only if nothing was captured.
//!
//! ## Security model
//! - **No IPC bridge:** render windows (`lens-render-*`) have no Tauri ACL capability.
//! - **SSRF pre-flight:** [`lens_core::ssrf_check_url`] (blocking DNS resolve) runs
//!   before any window is built; blocked host ⇒ `Ok(None)`.
//! - **Per-navigation gate:** `on_navigation` runs non-blocking [`lens_core::ssrf_check_host`]
//!   (no DNS on event-loop thread) and also enforces the http/https scheme allowlist to
//!   block `file:`/`data:`/`blob:` navigations the host-only check would miss.
//! - **Readback provenance re-check (C1):** `webview.url()` is checked via
//!   [`lens_core::readback_host_allowed`] after capture; blocked final host ⇒ discard.
//! - **Incognito + no downloads + no popups:** ephemeral session; `on_download` → false,
//!   `on_new_window` → Deny.
//! - **Async-safe teardown (C2):** every exit arm schedules `run_on_main_thread(||
//!   webview.destroy())`. No `Drop` guard; render body is wrapped in `catch_unwind`.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::FutureExt;
use lens_core::{JsRenderer, LensError, readback_host_allowed};
use tauri::webview::{PageLoadEvent, WebviewWindowBuilder};
use tauri::{Manager, Url, WebviewUrl};

/// How the render window is hidden. Testing confirmed off-screen WKWebView runs JS
/// and captures content fine once the double-encoded-readback bug was fixed (issue #78).
#[derive(PartialEq, Eq)]
enum RenderVisibility {
    /// Visible decorated window — for watching renders during dev. Never commit as active.
    DebugVisible,
    /// Off-screen (invisible, never covers content). The confirmed production path.
    Offscreen,
    /// On-screen at origin but alpha-0 + click-through. Untested fallback if an OS is
    /// found to suspend off-screen webviews; not currently active.
    OnscreenAlphaZero,
}

/// Active render-window strategy. `Offscreen` is the confirmed production path (verified
/// on macOS). Switch to `OnscreenAlphaZero` if an OS suspends off-screen webviews.
const RENDER_VISIBILITY: RenderVisibility = RenderVisibility::Offscreen;

/// Hard cap on a single render — independent of any in-page signal. On timeout, the
/// best capture seen is returned (`None` only if nothing was ever captured). Sized for
/// SPAs that idle 2–5s after load before fetching, plus fetch+render time.
const JS_RENDER_MAX_TIMEOUT: Duration = Duration::from_secs(20);

/// Poll interval matched to the in-page `SETTLE_MS` so a quiesced page is observed
/// within roughly one interval.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Per-poll timeout for one `eval_with_callback` result. Without it, a webview that
/// accepts the eval but never runs the callback (suspended/throttled) hangs the render.
/// Repeated timeouts here indicate a suspended webview.
const EVAL_CALLBACK_TIMEOUT: Duration = Duration::from_secs(2);

/// Minimum body-text growth (chars) over the first-poll baseline before a `quiescent`
/// signal is accepted. SPA shells already contain nav text so an absolute threshold
/// would fire on the shell; requiring GROWTH guards against premature empty captures.
const CONTENT_GROWTH_MIN: usize = 800;

/// Minimum polling time before any `quiescent` accept, so the idle gap after load
/// (before the SPA starts fetching) cannot be mistaken for a settled render.
const MIN_RENDER_WAIT: Duration = Duration::from_secs(3);

/// Off-screen position (logical px) — far outside any real display so the window
/// is never seen and never covers on-screen content.
const OFFSCREEN_XY: f64 = -32000.0;
const RENDER_WIDTH: f64 = 1280.0;
const RENDER_HEIGHT: f64 = 800.0;

/// Trusted init script (runs before page JS). Installs a network-idle (fetch/XHR
/// in-flight counters) + DOM-idle (MutationObserver) quiescence detector and records
/// the largest `outerHTML` on `window.__lensBest`. Never throws; installs no IPC.
const INIT_JS: &str = r#"
(function () {
  try {
    var SETTLE_MS = 500;
    window.__lensBest = "";
    window.__lensQuiescent = false;
    var inflight = 0;
    var settleTimer = null;

    function snapshot() {
      try {
        var el = document.documentElement;
        if (el) {
          var html = el.outerHTML || "";
          // Keep only the LARGEST snapshot ever seen (best capture).
          if (html.length >= window.__lensBest.length) { window.__lensBest = html; }
        }
      } catch (e) { /* never throw from a trusted hook */ }
    }

    // A single shared settle timer gates BOTH signals: it only counts down while
    // the network is idle (inflight === 0). Any mutation or new request resets it.
    function scheduleSettle() {
      if (settleTimer) { clearTimeout(settleTimer); settleTimer = null; }
      window.__lensQuiescent = false;
      if (inflight > 0) { return; } // network busy → not settling yet
      settleTimer = setTimeout(function () {
        // Fired after SETTLE_MS with no mutations AND no in-flight requests.
        if (inflight === 0) {
          snapshot();
          window.__lensQuiescent = true;
        }
      }, SETTLE_MS);
    }

    function onNetStart() { inflight++; scheduleSettle(); }
    function onNetEnd() {
      inflight = inflight > 0 ? inflight - 1 : 0;
      scheduleSettle();
    }

    // Hook fetch (runs before page JS, so the SPA sees our wrapper).
    if (typeof window.fetch === "function") {
      var origFetch = window.fetch.bind(window);
      window.fetch = function () {
        onNetStart();
        var done = false;
        function finish() { if (!done) { done = true; onNetEnd(); } }
        try {
          return origFetch.apply(this, arguments).then(
            function (r) { finish(); return r; },
            function (e) { finish(); throw e; }
          );
        } catch (e) { finish(); throw e; }
      };
    }

    // Hook XMLHttpRequest.
    if (typeof window.XMLHttpRequest === "function") {
      var origSend = window.XMLHttpRequest.prototype.send;
      window.XMLHttpRequest.prototype.send = function () {
        var self = this;
        var counted = true;
        onNetStart();
        function finish() { if (counted) { counted = false; onNetEnd(); } }
        try {
          self.addEventListener("loadend", finish);
        } catch (e) { /* ignore */ }
        try {
          return origSend.apply(self, arguments);
        } catch (e) { finish(); throw e; }
      };
    }

    // Initial settle attempt (covers static pages with no network/mutations).
    scheduleSettle();

    // Re-snapshot + reset settle on any DOM mutation, debounced by SETTLE_MS.
    if (typeof MutationObserver !== "undefined") {
      var obs = new MutationObserver(function () { snapshot(); scheduleSettle(); });
      if (document.documentElement) {
        obs.observe(document.documentElement, {
          childList: true, subtree: true, attributes: true, characterData: true
        });
      }
    }
  } catch (e) { /* swallow: init must never throw */ }
})();
"#;

/// Poll readback script (via `eval_with_callback`). Never throws — Windows silently
/// drops thrown eval exceptions. Returns `{quiescent, best, live, textLen}` on
/// success, `{err}` on failure.
const POLL_JS: &str = r#"
(function () {
  try {
    var el = document.documentElement;
    var live = el ? (el.outerHTML || "") : "";
    var best = (typeof window.__lensBest === "string") ? window.__lensBest : "";
    var q = (window.__lensQuiescent === true);
    // Visible text length — the "has the content actually rendered?" signal. A
    // client-rendered SPA shell has little body text until its JS populates the
    // DOM, so the Rust poll loop refuses to accept quiescence until this crosses
    // a threshold (otherwise it would capture the empty shell during the idle gap
    // right after load, before hydration begins).
    var textLen = 0;
    try { textLen = (document.body && document.body.innerText) ? document.body.innerText.length : 0; } catch (e) {}
    return JSON.stringify({ quiescent: q, best: best, live: live, textLen: textLen });
  } catch (e) {
    return JSON.stringify({ err: String(e) });
  }
})()
"#;

/// The offscreen-webview renderer. Holds an `AppHandle` for `run_on_main_thread`
/// window create/destroy.
pub struct TauriJsRenderer {
    pub(crate) app: tauri::AppHandle,
}

impl TauriJsRenderer {
    pub fn new(app: tauri::AppHandle) -> Self {
        Self { app }
    }
}

/// Schedules main-thread teardown of the render window (C2). Safe on every exit arm;
/// a no-op if the window is already gone.
fn schedule_destroy(app: &tauri::AppHandle, label: String) {
    let app2 = app.clone();
    let _ = app.run_on_main_thread(move || {
        if let Some(w) = app2.get_webview_window(&label) {
            let _ = w.destroy();
        }
    });
}

/// Runs one `eval_with_callback(POLL_JS)` on the main thread and awaits its result.
/// Returns `None` if the window is gone or the eval could not be dispatched (poll
/// loop stops). POLL_JS never throws; returns `{err}` instead (Windows swallows JS).
async fn poll_once(app: &tauri::AppHandle, label: &str) -> Option<Readback> {
    let (tx, rx) = tokio::sync::oneshot::channel::<String>();
    // The `Fn` callback may (in principle) fire more than once; guard the sender.
    let tx = Arc::new(std::sync::Mutex::new(Some(tx)));
    let app2 = app.clone();
    let label_owned = label.to_string();

    let dispatched = app.run_on_main_thread(move || {
        let Some(w) = app2.get_webview_window(&label_owned) else {
            tracing::warn!(target: "lens::js_render", label = %label_owned, "poll: render window not found by label (get_webview_window → None)");
            return;
        };
        let tx = tx.clone();
        if let Err(e) = w.eval_with_callback(POLL_JS, move |json: String| {
            if let Ok(mut guard) = tx.lock()
                && let Some(sender) = guard.take()
            {
                let _ = sender.send(json);
            }
        }) {
            tracing::warn!(target: "lens::js_render", label = %label_owned, error = %e, "poll: eval_with_callback dispatch FAILED");
        }
    });

    if dispatched.is_err() {
        return None;
    }

    // Per-poll timeout: if the callback never fires the webview accepted the eval but
    // JS is not executing (suspended/throttled). Return None to retry rather than hang.
    match tokio::time::timeout(EVAL_CALLBACK_TIMEOUT, rx).await {
        Ok(Ok(json)) => {
            tracing::debug!(target: "lens::js_render", label = %label, json_len = json.len(), "poll readback");
            Some(parse_readback(&json))
        }
        // Sender dropped (window gone before the callback fired) ⇒ stop polling.
        Ok(Err(_)) => None,
        Err(_) => {
            tracing::warn!(target: "lens::js_render", label = %label, "poll: eval callback did not fire within timeout (JS not executing — webview likely suspended)");
            None
        }
    }
}

/// Per-navigation allow decision: pure, non-blocking (no DNS). Enforces http/https
/// allowlist AND host SSRF gate. Extracted for unit-testability without a live webview.
fn nav_url_allowed(u: &tauri::Url) -> bool {
    matches!(u.scheme(), "http" | "https") && lens_core::ssrf_check_host(u.host_str()).is_ok()
}

/// Parsed poll-readback payload. `best`/`live` are null-guarded to `""` when absent.
enum Readback {
    Poll {
        quiescent: bool,
        best: String,
        live: String,
        /// `body.innerText.length` — the "content rendered yet?" signal that guards
        /// against accepting quiescence on the bare SPA shell.
        text_len: usize,
    },
    Err(String),
}

impl Readback {
    fn best_html(&self) -> String {
        match self {
            Readback::Poll { best, live, .. } => {
                if live.len() >= best.len() {
                    live.clone()
                } else {
                    best.clone()
                }
            }
            Readback::Err(_) => String::new(),
        }
    }
}

fn parse_readback(json: &str) -> Readback {
    match serde_json::from_str::<serde_json::Value>(json) {
        Ok(v) => {
            // Tauri/wry `eval_with_callback` already `JSON.stringify`s the return value,
            // and POLL_JS also returns a JSON string → DOUBLE-encoded: a JSON string
            // whose contents are our object. Unwrap once more; single-encoding still works.
            let v = match &v {
                serde_json::Value::String(inner) => {
                    serde_json::from_str::<serde_json::Value>(inner).unwrap_or(v)
                }
                _ => v,
            };
            if let Some(err) = v.get("err").and_then(|e| e.as_str()) {
                return Readback::Err(err.to_string());
            }
            let has_snapshot =
                v.get("quiescent").is_some() || v.get("best").is_some() || v.get("live").is_some();
            if has_snapshot {
                let quiescent = v
                    .get("quiescent")
                    .and_then(|q| q.as_bool())
                    .unwrap_or(false);
                let best = v
                    .get("best")
                    .and_then(|b| b.as_str())
                    .unwrap_or("")
                    .to_string();
                let live = v
                    .get("live")
                    .and_then(|l| l.as_str())
                    .unwrap_or("")
                    .to_string();
                let text_len = v.get("textLen").and_then(|t| t.as_u64()).unwrap_or(0) as usize;
                Readback::Poll {
                    quiescent,
                    best,
                    live,
                    text_len,
                }
            } else {
                Readback::Err(format!("unrecognized readback payload: {json}"))
            }
        }
        Err(e) => Readback::Err(format!("readback JSON parse failed: {e}")),
    }
}

#[async_trait]
impl JsRenderer for TauriJsRenderer {
    async fn render_html(&self, url: &str) -> Result<Option<String>, LensError> {
        // SSRF pre-flight: one blocking DNS resolve. Off the event-loop thread, so
        // blocking is fine. A blocked host never gets a webview.
        if lens_core::ssrf_check_url(url).is_err() {
            tracing::warn!(target: "lens::js_render", url, "pre-flight SSRF check failed; not rendering");
            return Ok(None);
        }

        let label = format!("lens-render-{}", uuid::Uuid::now_v7());

        // C2: wrap in catch_unwind so a panic still schedules destroy. No Drop guard.
        let app = self.app.clone();
        let label_for_body = label.clone();
        let body =
            std::panic::AssertUnwindSafe(
                async move { render_inner(&app, &label_for_body, url).await },
            );

        match body.catch_unwind().await {
            Ok(result) => result,
            Err(panic) => {
                schedule_destroy(&self.app, label);
                std::panic::resume_unwind(panic);
            }
        }
    }
}

/// The render body proper. Separated so `render_html` can wrap it in `catch_unwind`.
/// Schedules teardown on every exit arm. Not automatable — drives a live Tauri webview;
/// pure testable pieces (`parse_readback`, `nav_url_allowed`) are unit-tested separately.
async fn render_inner(
    app: &tauri::AppHandle,
    label: &str,
    url: &str,
) -> Result<Option<String>, LensError> {
    let parsed = match Url::parse(url) {
        Ok(u) => u,
        Err(_) => return Ok(None),
    };

    // `WebviewWindowBuilder` must be built on the event-loop thread (on Windows,
    // building from a sync handler deadlocks). Dispatch via oneshot to learn success.
    let (build_tx, build_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
    let app_for_build = app.clone();
    let label_owned = label.to_string();

    let dispatch = app.run_on_main_thread(move || {
        let (pos_x, pos_y) = match RENDER_VISIBILITY {
            RenderVisibility::DebugVisible => (120.0_f64, 120.0_f64),
            RenderVisibility::Offscreen => (OFFSCREEN_XY, OFFSCREEN_XY),
            RenderVisibility::OnscreenAlphaZero => (0.0_f64, 0.0_f64),
        };
        let debug_visible = RENDER_VISIBILITY == RenderVisibility::DebugVisible;

        let builder =
            WebviewWindowBuilder::new(&app_for_build, &label_owned, WebviewUrl::External(parsed))
                .position(pos_x, pos_y)
                .inner_size(RENDER_WIDTH, RENDER_HEIGHT)
                .visible(RENDER_VISIBILITY != RenderVisibility::OnscreenAlphaZero)
                .decorations(debug_visible)
                .shadow(false)
                .skip_taskbar(!debug_visible)
                .focused(false)
                .resizable(false)
                .always_on_top(true)
                .title("LensLM · rendering")
                .incognito(true)
                .on_navigation(nav_url_allowed)
                .on_download(|_, _| false)
                .on_new_window(|_, _| tauri::webview::NewWindowResponse::Deny)
                .initialization_script(INIT_JS)
                // `on_page_load(Finished)` is a convenience trigger only; the poll loop
                // is authoritative for capture.
                .on_page_load(|_wv, payload| {
                    if matches!(payload.event(), PageLoadEvent::Finished) {
                        tracing::trace!(target: "lens::js_render", "page load finished (poll loop drives capture)");
                    }
                });

        match builder.build() {
            Ok(_window) => {
                match RENDER_VISIBILITY {
                    RenderVisibility::OnscreenAlphaZero => {
                        #[cfg(target_os = "macos")]
                        {
                            if let Ok(ns_ptr) = _window.ns_window() {
                                let ns = ns_ptr as *mut objc2::runtime::AnyObject;
                                if !ns.is_null() {
                                    // SAFETY: `ns_window()` returns this window's NSWindow*;
                                    // called on main thread; `setAlphaValue:` is a standard setter.
                                    unsafe {
                                        let _: () =
                                            objc2::msg_send![&*ns, setAlphaValue: 0.0_f64];
                                    }
                                }
                            }
                            let _ = _window.set_ignore_cursor_events(true);
                        }
                        let _ = _window.show();
                    }
                    RenderVisibility::DebugVisible | RenderVisibility::Offscreen => {
                        let _ = _window.show();
                    }
                }
                let found = app_for_build.get_webview_window(&label_owned).is_some();
                tracing::info!(target: "lens::js_render", label = %label_owned, found_after_build = found, "webview build() returned Ok");
                let _ = build_tx.send(Ok(()));
            }
            Err(e) => {
                let _ = build_tx.send(Err(e.to_string()));
            }
        }
    });

    if dispatch.is_err() {
        tracing::warn!(target: "lens::js_render", label, "failed to dispatch webview build to main thread");
        return Ok(None);
    }

    match build_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::warn!(target: "lens::js_render", label, error = %e, "webview creation failed");
            return Ok(None);
        }
        Err(_) => {
            // Build closure dropped without sending; schedule a defensive destroy
            // in case a partial window exists.
            schedule_destroy(app, label.to_string());
            return Ok(None);
        }
    }

    // From here every exit arm must schedule teardown.
    let candidate = tokio::time::timeout(JS_RENDER_MAX_TIMEOUT, async {
        let mut best_seen = String::new();
        let mut baseline_text: Option<usize> = None;
        let mut polls: u32 = 0;
        // A just-built webview is not immediately eval-able; None before first success
        // means "not ready yet — retry"; after first success it means "window vanished".
        let mut saw_poll = false;
        loop {
            match poll_once(app, label).await {
                Some(Readback::Poll {
                    quiescent,
                    best,
                    live,
                    text_len,
                }) => {
                    polls += 1;
                    saw_poll = true;
                    let baseline = *baseline_text.get_or_insert(text_len);
                    let this = if live.len() >= best.len() { live } else { best };
                    if this.len() > best_seen.len() {
                        best_seen = this;
                    }
                    let waited = polls as u64 * POLL_INTERVAL.as_millis() as u64
                        >= MIN_RENDER_WAIT.as_millis() as u64;
                    let content_grew = text_len >= baseline.saturating_add(CONTENT_GROWTH_MIN);
                    if quiescent && waited && content_grew {
                        tracing::debug!(
                            target: "lens::js_render", label, polls, text_len, baseline,
                            "render settled with content; accepting capture"
                        );
                        break;
                    }
                }
                Some(Readback::Err(e)) => {
                    // POLL_JS reported an error (e.g. no documentElement yet); webview IS
                    // reachable — keep polling.
                    saw_poll = true;
                    tracing::debug!(target: "lens::js_render", label, error = %e, "poll readback error; retrying");
                }
                None => {
                    if saw_poll {
                        tracing::debug!(target: "lens::js_render", label, "poll: webview unreachable after prior success; stopping");
                        break;
                    }
                    // Not ready yet (page still loading; was the bug that made SPAs
                    // capture nothing — elapsed_ms≈66, captured 0). Keep retrying.
                    tracing::trace!(target: "lens::js_render", label, "poll: webview not ready yet; retrying");
                }
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
        best_seen
    })
    .await
    .unwrap_or_else(|_| {
        tracing::warn!(target: "lens::js_render", label, timeout_s = JS_RENDER_MAX_TIMEOUT.as_secs(), "render timed out before content growth; using best capture seen");
        String::new()
    });

    // Final best-effort read: recovers the init script's `best` capture after a timeout
    // (the cancelled poll future dropped its accumulated best_seen). Bounded so a
    // pathological webview cannot stall the ingest permit past the hard timeout.
    let candidate = {
        let mut best = candidate;
        let final_read = tokio::time::timeout(POLL_INTERVAL * 4, poll_once(app, label)).await;
        if let Ok(Some(rb @ Readback::Poll { .. })) = final_read {
            let this = rb.best_html();
            if this.len() > best.len() {
                best = this;
            }
        }
        if best.is_empty() { None } else { Some(best) }
    };

    tracing::info!(
        target: "lens::js_render",
        label,
        captured_html_len = candidate.as_ref().map(|s| s.len()).unwrap_or(0),
        "render capture complete"
    );

    // C1: provenance re-check on the final committed URL (off event-loop thread).
    let out = if let Some(html) = candidate {
        let final_url = app
            .get_webview_window(label)
            .and_then(|w| w.url().ok())
            .map(|u| u.to_string());
        match final_url {
            Some(u) if readback_host_allowed(&u) => Some(html),
            Some(u) => {
                tracing::warn!(target: "lens::js_render", label, final_url = %u, "final-committed host blocked; discarding render output");
                None
            }
            None => {
                tracing::warn!(target: "lens::js_render", label, "could not read final webview URL; discarding render output");
                None
            }
        }
    } else {
        None
    };

    schedule_destroy(app, label.to_string());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_readback_handles_poll_err_and_garbage() {
        // live longer than best ⇒ best_html == live.
        match parse_readback(
            r#"{"quiescent":true,"best":"<html>a</html>","live":"<html>abc</html>"}"#,
        ) {
            Readback::Poll {
                quiescent,
                best,
                live,
                ..
            } => {
                assert!(quiescent);
                assert_eq!(best, "<html>a</html>");
                assert_eq!(live, "<html>abc</html>");
            }
            Readback::Err(e) => panic!("expected Poll, got Err({e})"),
        }
        assert_eq!(
            parse_readback(
                r#"{"quiescent":true,"best":"<html>a</html>","live":"<html>abc</html>"}"#
            )
            .best_html(),
            "<html>abc</html>"
        );
        // best longer than live ⇒ best_html falls back to best.
        assert_eq!(
            parse_readback(r#"{"quiescent":false,"best":"<html>abcd</html>","live":""}"#)
                .best_html(),
            "<html>abcd</html>"
        );

        // Missing scalars default safely (not quiescent, empty html, zero text).
        match parse_readback(r#"{"best":"x"}"#) {
            Readback::Poll {
                quiescent,
                best,
                live,
                text_len,
            } => {
                assert!(!quiescent);
                assert_eq!(best, "x");
                assert_eq!(live, "");
                assert_eq!(text_len, 0);
            }
            Readback::Err(e) => panic!("expected Poll, got Err({e})"),
        }

        // textLen is parsed when present.
        match parse_readback(r#"{"quiescent":true,"best":"","live":"<p>hi</p>","textLen":1234}"#) {
            Readback::Poll { text_len, .. } => assert_eq!(text_len, 1234),
            Readback::Err(e) => panic!("expected Poll, got Err({e})"),
        }

        // The in-page error shape ⇒ Err.
        match parse_readback(r#"{"err":"boom"}"#) {
            Readback::Err(e) => assert_eq!(e, "boom"),
            Readback::Poll { .. } => panic!("expected Err"),
        }
        assert_eq!(parse_readback(r#"{"err":"boom"}"#).best_html(), "");

        // Malformed / unrecognized ⇒ Err (fail closed).
        assert!(matches!(parse_readback("not json"), Readback::Err(_)));
        assert!(matches!(parse_readback(r#"{"other":1}"#), Readback::Err(_)));
    }

    /// `eval_with_callback` double-encodes: POLL_JS returns a JSON string and the bridge
    /// `JSON.stringify`s it again. `parse_readback` must unwrap the outer encoding.
    #[test]
    fn parse_readback_unwraps_double_encoded() {
        let inner = r#"{"quiescent":true,"best":"","live":"<html><body>hi there</body></html>","textLen":8}"#;
        // Double-encoded: a JSON string whose content IS that object.
        let double = serde_json::to_string(inner).unwrap();
        match parse_readback(&double) {
            Readback::Poll {
                quiescent,
                live,
                text_len,
                ..
            } => {
                assert!(quiescent);
                assert_eq!(live, "<html><body>hi there</body></html>");
                assert_eq!(text_len, 8);
            }
            Readback::Err(e) => panic!("double-encoded payload must parse, got Err({e})"),
        }
        assert_eq!(
            parse_readback(&double).best_html(),
            "<html><body>hi there</body></html>"
        );
    }

    /// C1 provenance gate: blocked/malformed hosts → discard; public → keep.
    #[test]
    fn readback_provenance_discards_blocked_and_malformed() {
        assert!(!readback_host_allowed(
            "http://169.254.169.254/latest/meta-data/"
        ));
        assert!(!readback_host_allowed("http://127.0.0.1/x"));
        assert!(!readback_host_allowed("http://[::1]/"));
        assert!(!readback_host_allowed("http://10.0.0.5/"));
        // `localhost` hostname must be blocked (readback re-check resolves DNS).
        assert!(!readback_host_allowed("http://localhost/whatever"));
        assert!(!readback_host_allowed("not a url"));
        assert!(readback_host_allowed("http://8.8.8.8/page"));
    }

    #[test]
    fn nav_url_allowed_enforces_scheme_and_host() {
        let allow = |s: &str| nav_url_allowed(&tauri::Url::parse(s).unwrap());
        assert!(!allow("file://localhost/etc/passwd"));
        assert!(!allow("file:///etc/passwd"));
        assert!(!allow("data:text/html,<h1>x</h1>"));
        assert!(!allow("http://169.254.169.254/latest/meta-data/"));
        assert!(!allow("http://127.0.0.1/x"));
        assert!(!allow("https://[::1]/"));
        assert!(allow("https://example.com/page"));
        assert!(allow("http://8.8.8.8/page"));
    }
}
