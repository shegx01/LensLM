//! `TauriJsRenderer` — the concrete offscreen-webview implementation of
//! `lens_core::JsRenderer` (issue #78, Layer b step 6).
//!
//! `lens-core` cannot depend on `tauri` (crate-boundary invariant), so the
//! webview machinery lives here in `src-tauri` and is injected into the engine
//! via the `Arc<RwLock<Option<Arc<dyn JsRenderer>>>>` DI seam
//! (`LensEngine::set_js_renderer`). When a URL source extracts near-empty text
//! via the static path (`needs_js`), the ingest fallback calls
//! [`TauriJsRenderer::render_html`], which loads the URL in an isolated,
//! offscreen, incognito webview, waits for the DOM to settle, and returns
//! `document.documentElement.outerHTML` to be fed through the SAME static
//! extractor.
//!
//! ## Poll-until-quiescent-or-timeout capture model
//! A client-rendered SPA may inject its real content *after* the page-load
//! event fires (and may never fire a fresh load event at all), so a single
//! one-shot readback on `PageLoadEvent::Finished` would capture an empty shell.
//! Instead the trusted [`INIT_JS`] init script (running before page JS) installs
//! a hybrid **quiescence detector** and continuously records the largest
//! `document.documentElement.outerHTML` seen on `window.__lensBest`:
//! - **Network idle:** it wraps `fetch`/`XMLHttpRequest` with in-flight
//!   counters; the network is considered idle once no request has been in flight
//!   for ~`SETTLE_MS`.
//! - **DOM idle:** a `MutationObserver` debounces mutations by ~`SETTLE_MS`.
//! - `window.__lensQuiescent` becomes `true` only once BOTH have been idle for
//!   the settle window.
//!
//! The Rust side then **polls** (every [`POLL_INTERVAL`], via `eval_with_callback`
//! dispatched on the main thread) reading `{quiescent, best, live}`, keeping the
//! largest capture across polls. Polling STOPS and resolves as soon as
//! `quiescent` is `true`, OR when [`JS_RENDER_MAX_TIMEOUT`] elapses — in which
//! case the best capture seen so far is returned (only `None` if we truly never
//! captured anything). `PageLoadEvent::Finished` is used as a convenience trigger
//! to kick the first poll but is NOT depended upon.
//!
//! ## Security model (executes UNTRUSTED page JS)
//! - **No IPC bridge:** the render window's label (`lens-render-*`) matches NO
//!   capability (`default.json` is `windows:["main"]`), so per Tauri's ACL it
//!   has zero IPC/command access. `capabilities/renderer-empty.json` is
//!   defense-in-depth.
//! - **SSRF pre-flight:** [`lens_core::ssrf_check_url`] (blocking, one DNS
//!   resolve) runs BEFORE any window is built; a blocked host ⇒ `Ok(None)`.
//! - **Per-navigation gate:** `on_navigation` runs the NON-BLOCKING
//!   [`lens_core::ssrf_check_host`] (no DNS on the event-loop thread) and
//!   cancels blocked hops.
//! - **Readback provenance re-check (C1):** after capturing the DOM we read
//!   `webview.url()` and run its host through [`lens_core::readback_host_allowed`]
//!   (off the event-loop thread); a blocked final host ⇒ discard ⇒ `Ok(None)`.
//! - **Incognito + no downloads + no popups:** ephemeral session, `on_download`
//!   returns `false`, `on_new_window` returns `Deny`.
//! - **Async-safe teardown (C2):** every exit arm — success, timeout,
//!   nav-cancel, provenance-blocked, error, AND panic — schedules an explicit
//!   `run_on_main_thread(|| webview.destroy())`. There is NO `Drop` guard; the
//!   render body is wrapped in `catch_unwind` so a panic still schedules destroy
//!   before re-propagating.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::FutureExt;
use lens_core::{JsRenderer, LensError, readback_host_allowed};
use tauri::webview::{PageLoadEvent, WebviewWindowBuilder};
use tauri::{Manager, Url, WebviewUrl};

/// Hard wall-clock cap on a single render, enforced Rust-side with
/// `tokio::time::timeout` around the poll loop — independent of any in-page
/// signal, so a hostile/broken page cannot exceed it. On timeout the best
/// capture seen so far is returned (then the caller's content oracle decides);
/// `None` only if nothing was ever captured.
const JS_RENDER_MAX_TIMEOUT: Duration = Duration::from_secs(15);

/// How often the Rust side polls the webview for `{quiescent, best, live}`
/// (via `eval_with_callback`). Matched to the in-page `SETTLE_MS` settle window
/// so a page that quiesces is observed within roughly one interval.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Offscreen position (logical px) — far outside any real display so the webview
/// renders (avoiding occlusion/timer throttling that `.visible(false)` incurs)
/// without ever appearing on screen.
const OFFSCREEN_XY: f64 = -32000.0;
const RENDER_WIDTH: f64 = 1280.0;
const RENDER_HEIGHT: f64 = 800.0;

/// Trusted init script (runs before page JS, main frame only). It installs a
/// hybrid **quiescence detector** — network-idle (fetch/XHR in-flight counters)
/// plus `MutationObserver` DOM-idle — and continuously records the largest
/// `document.documentElement.outerHTML` seen on `window.__lensBest`, so a render
/// that never fully quiesces still has a "best capture seen" available at
/// timeout. It also exposes `window.__lensQuiescent`, which flips to `true` only
/// once BOTH the network and the DOM have been idle for one `SETTLE_MS` window.
/// It NEVER throws (Windows swallows eval exceptions) and installs no IPC.
///
/// Because this script runs BEFORE any page JS, wrapping `fetch`/`XMLHttpRequest`
/// here reliably intercepts the SPA's own async data loads. The ~500 ms
/// `SETTLE_MS` window debounces both signals: any DOM mutation or in-flight
/// network request resets the shared settle timer, and quiescence is declared
/// only after `SETTLE_MS` of combined quiet.
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

/// Rust-initiated poll readback (via `eval_with_callback`). Returns a JSON STRING
/// (never throws — Windows silently drops thrown eval exceptions, research §3):
/// `{quiescent, best, live}` on success, `{err}` on failure. `best` is the init
/// script's largest recorded `outerHTML`; `live` is the current live
/// `document.documentElement.outerHTML`. The Rust poll loop keeps the larger of
/// the two across polls and stops once `quiescent` is `true`.
const POLL_JS: &str = r#"
(function () {
  try {
    var el = document.documentElement;
    var live = el ? (el.outerHTML || "") : "";
    var best = (typeof window.__lensBest === "string") ? window.__lensBest : "";
    var q = (window.__lensQuiescent === true);
    return JSON.stringify({ quiescent: q, best: best, live: live });
  } catch (e) {
    return JSON.stringify({ err: String(e) });
  }
})()
"#;

/// The offscreen-webview renderer. Holds an `AppHandle` so it can build/destroy
/// windows on the main thread and inject itself at app setup.
pub struct TauriJsRenderer {
    /// The Tauri app handle used to `run_on_main_thread` for window create/destroy.
    pub(crate) app: tauri::AppHandle,
}

impl TauriJsRenderer {
    /// Constructs a renderer bound to `app`.
    pub fn new(app: tauri::AppHandle) -> Self {
        Self { app }
    }
}

/// Schedules an explicit main-thread teardown of the render window by label
/// (C2). Fire-and-schedule: `run_on_main_thread` returns immediately and the
/// `destroy()` runs later on the event loop. Safe to call on every exit arm; a
/// no-op if the window is already gone.
fn schedule_destroy(app: &tauri::AppHandle, label: String) {
    let app2 = app.clone();
    // Ignore the dispatch error: if the event loop is already gone there is
    // nothing to destroy.
    let _ = app.run_on_main_thread(move || {
        if let Some(w) = app2.get_webview_window(&label) {
            let _ = w.destroy();
        }
    });
}

/// Runs ONE `eval_with_callback(POLL_JS)` against the render window and awaits its
/// JSON result, parsed into a [`Readback`]. Dispatches the eval on the main thread
/// (via `run_on_main_thread` + a oneshot bridge back to this async task). Returns
/// `None` if the window is gone or the eval could not be dispatched (⇒ the poll
/// loop stops). The `eval_with_callback` callback receives the serialized-JSON
/// result string; on Windows any thrown JS exception is swallowed, so POLL_JS
/// never throws and returns `{err}` instead.
async fn poll_once(app: &tauri::AppHandle, label: &str) -> Option<Readback> {
    let (tx, rx) = tokio::sync::oneshot::channel::<String>();
    // The `Fn` callback may (in principle) fire more than once; guard the sender.
    let tx = Arc::new(std::sync::Mutex::new(Some(tx)));
    let app2 = app.clone();
    let label_owned = label.to_string();

    let dispatched = app.run_on_main_thread(move || {
        let Some(w) = app2.get_webview_window(&label_owned) else {
            // Window gone: drop the sender so the awaiting `rx` resolves to Err.
            return;
        };
        let tx = tx.clone();
        let _ = w.eval_with_callback(POLL_JS, move |json: String| {
            if let Ok(mut guard) = tx.lock()
                && let Some(sender) = guard.take()
            {
                let _ = sender.send(json);
            }
        });
    });

    if dispatched.is_err() {
        return None;
    }

    match rx.await {
        Ok(json) => Some(parse_readback(&json)),
        // Sender dropped (window gone before the callback fired) ⇒ stop polling.
        Err(_) => None,
    }
}

/// Per-navigation allow decision for the render webview's `on_navigation` gate.
/// Pure + NON-BLOCKING (no DNS): enforces the http/https scheme allowlist AND the
/// host-string SSRF gate. Extracted so it is unit-testable without a live webview.
/// Rejecting `file:`/`data:`/`blob:` here stops page-JS-initiated client navigations
/// that the host-only check would miss (e.g. `file://localhost/etc/passwd`).
fn nav_url_allowed(u: &tauri::Url) -> bool {
    matches!(u.scheme(), "http" | "https") && lens_core::ssrf_check_host(u.host_str()).is_ok()
}

/// Parsed poll-readback payload. Either a snapshot of the render's current state
/// (`quiescent` flag + `best`/`live` `outerHTML`) or an error string. `best` and
/// `live` are null-guarded to `""` when absent.
enum Readback {
    /// A successful poll snapshot.
    Poll {
        /// Whether the page has quiesced (network + DOM idle for the settle window).
        quiescent: bool,
        /// The largest `outerHTML` the init script has recorded so far.
        best: String,
        /// The current live `document.documentElement.outerHTML`.
        live: String,
    },
    /// The in-page readback reported an error (or the payload was unparseable).
    Err(String),
}

impl Readback {
    /// The larger of `best`/`live` for a `Poll`; empty for an `Err`.
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
            if let Some(err) = v.get("err").and_then(|e| e.as_str()) {
                return Readback::Err(err.to_string());
            }
            // A poll payload has at least one of the snapshot fields; treat a
            // missing scalar as its safe default (not quiescent / empty html).
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
                Readback::Poll {
                    quiescent,
                    best,
                    live,
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
        // ── SSRF pre-flight (the ONE blocking DNS resolve). A blocked entry host
        // never gets a webview. Off the event-loop thread (we are on an async
        // ingest task), so blocking is fine.
        if lens_core::ssrf_check_url(url).is_err() {
            tracing::warn!(target: "lens::js_render", url, "pre-flight SSRF check failed; not rendering");
            return Ok(None);
        }

        // Unique per-render label so the `lens-render-*` capability scope holds
        // and teardown can never target the wrong window. permit=1 upstream
        // means only one render window is ever live.
        let label = format!("lens-render-{}", uuid::Uuid::now_v7());

        // Wrap the whole render body in catch_unwind so a panic in the
        // untrusted-JS path still schedules destroy before re-propagating (C2 —
        // NO Drop guard).
        let app = self.app.clone();
        let label_for_body = label.clone();
        let body =
            std::panic::AssertUnwindSafe(
                async move { render_inner(&app, &label_for_body, url).await },
            );

        match body.catch_unwind().await {
            Ok(result) => result,
            Err(panic) => {
                // Teardown was already scheduled inside render_inner on the arms
                // that created the window; schedule again defensively (idempotent
                // — destroy of a gone window is a no-op) then re-propagate.
                schedule_destroy(&self.app, label);
                std::panic::resume_unwind(panic);
            }
        }
    }
}

/// The render body proper. Separated so `render_html` can wrap it in
/// `catch_unwind`. Schedules teardown on EVERY exit arm.
async fn render_inner(
    app: &tauri::AppHandle,
    label: &str,
    url: &str,
) -> Result<Option<String>, LensError> {
    // Parse once for `WebviewUrl::External`; a parse failure here is a caller
    // bug (pre-flight already validated), but fail closed anyway.
    let parsed = match Url::parse(url) {
        Ok(u) => u,
        Err(_) => return Ok(None),
    };

    // ── Build the offscreen webview on the MAIN thread. `WebviewWindowBuilder`
    // must be constructed + built on the event-loop thread (research §5: on
    // Windows, building from a sync command/handler deadlocks). We dispatch the
    // build via a oneshot so the async task learns whether creation succeeded.
    let (build_tx, build_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
    let app_for_build = app.clone();
    let label_owned = label.to_string();

    let dispatch = app.run_on_main_thread(move || {
        let builder =
            WebviewWindowBuilder::new(&app_for_build, &label_owned, WebviewUrl::External(parsed))
                // Offscreen + visible (NOT `.visible(false)`: a truly hidden webview is
                // occlusion/timer-throttled on macOS/Windows, stalling SPA JS).
                .position(OFFSCREEN_XY, OFFSCREEN_XY)
                .inner_size(RENDER_WIDTH, RENDER_HEIGHT)
                .visible(true)
                // Ephemeral session: no shared cookies/localStorage/cache to exfiltrate
                // or bleed across renders.
                .incognito(true)
                // NON-BLOCKING per-navigation SSRF gate. Runs on the UI/event-loop
                // thread, so it MUST NOT resolve DNS — `ssrf_check_host` is host-string
                // only. Returning false CANCELS the navigation. We ALSO enforce the
                // http/https scheme allowlist here (matching pre-flight): page JS can
                // initiate a client-side navigation to `file://host/…`, `data:`, etc.,
                // which the host-only gate would not catch (e.g. `file://localhost/…`
                // has a non-IP host). Scheme check is pure string inspection — still no
                // DNS, still non-blocking.
                .on_navigation(nav_url_allowed)
                // Block all downloads.
                .on_download(|_, _| false)
                // Block window.open/popups.
                .on_new_window(|_, _| tauri::webview::NewWindowResponse::Deny)
                // Trusted quiesce detector + best-capture recorder (runs before page
                // JS): installs the network-idle + MutationObserver quiescence flag
                // and keeps `window.__lensBest`. The Rust poll loop below reads it.
                .initialization_script(INIT_JS)
                // `on_page_load(Finished)` is a convenience trigger only: an SPA may
                // never fire a fresh load event, so the poll loop — not this event —
                // is authoritative for capture. Kept as a lightweight trace hook.
                .on_page_load(|_wv, payload| {
                    if matches!(payload.event(), PageLoadEvent::Finished) {
                        tracing::trace!(target: "lens::js_render", "page load finished (poll loop drives capture)");
                    }
                });

        match builder.build() {
            Ok(_) => {
                let _ = build_tx.send(Ok(()));
            }
            Err(e) => {
                let _ = build_tx.send(Err(e.to_string()));
            }
        }
    });

    // If we could not even DISPATCH the build to the main thread, nothing was
    // created — no teardown to schedule.
    if dispatch.is_err() {
        tracing::warn!(target: "lens::js_render", label, "failed to dispatch webview build to main thread");
        return Ok(None);
    }

    // Await the build outcome. A build ERROR ⇒ creation failure ⇒ render_failed;
    // the window was never created, so NO destroy is scheduled here.
    match build_rx.await {
        Ok(Ok(())) => { /* created; continue to the poll loop */ }
        Ok(Err(e)) => {
            tracing::warn!(target: "lens::js_render", label, error = %e, "webview creation failed");
            return Ok(None);
        }
        Err(_) => {
            // The build closure was dropped without sending (should not happen);
            // fail closed and schedule a defensive destroy in case a partial
            // window exists.
            schedule_destroy(app, label.to_string());
            return Ok(None);
        }
    }

    // ── The window exists. From here EVERY exit arm must schedule teardown.
    //
    // POLL-UNTIL-QUIESCENT-OR-TIMEOUT (FIX 1). We repeatedly `eval_with_callback`
    // the POLL_JS reading `{quiescent, best, live}`, keeping the LARGEST capture
    // seen across polls. We STOP and resolve as soon as `quiescent` is true, OR
    // when the hard `JS_RENDER_MAX_TIMEOUT` elapses — returning the best capture
    // seen so far (only `None` if we truly never captured anything). This is what
    // lets a client-rendered SPA whose content arrives AFTER load be captured.
    let candidate = tokio::time::timeout(JS_RENDER_MAX_TIMEOUT, async {
        let mut best_seen = String::new();
        loop {
            match poll_once(app, label).await {
                Some(rb @ Readback::Poll { .. }) => {
                    let quiescent = matches!(rb, Readback::Poll { quiescent: true, .. });
                    // Keep the largest capture across ALL polls (the page may
                    // shrink transiently between renders).
                    let this = rb.best_html();
                    if this.len() > best_seen.len() {
                        best_seen = this;
                    }
                    if quiescent {
                        // Settled: return the best capture now.
                        break;
                    }
                }
                Some(Readback::Err(e)) => {
                    // The in-page readback erred (e.g. no documentElement yet).
                    // Keep polling; the init script may recover on the next tick.
                    tracing::debug!(target: "lens::js_render", label, error = %e, "poll readback error; retrying");
                }
                None => {
                    // The window vanished or the eval could not be dispatched.
                    // Nothing more to poll; return what we have.
                    break;
                }
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
        best_seen
    })
    .await
    .unwrap_or_else(|_| {
        // Hard timeout: the cancelled inner future dropped its accumulated
        // `best_seen`, so signal the timeout with an empty string and let the
        // trailing final read below recover the init script's best capture. We
        // return the best capture seen so far, NOT None (unless nothing was ever
        // captured) — this is what the timeout arm MUST do (FIX 1).
        tracing::warn!(target: "lens::js_render", label, timeout_s = JS_RENDER_MAX_TIMEOUT.as_secs(), "render timed out; using best capture seen");
        String::new()
    });

    // On timeout the cancelled poll future dropped its accumulated `best_seen`,
    // so make ONE final best-effort read to recover the init script's recorded
    // best capture before teardown. (On the quiescent path `candidate` is already
    // populated and this read only ever grows it.)
    let candidate = {
        let mut best = candidate;
        // Bound this recovery read so a pathological webview (dispatch succeeds but
        // the callback never fires) cannot re-introduce an unbounded await past the
        // hard `JS_RENDER_MAX_TIMEOUT` and stall the held ingest permit.
        let final_read = tokio::time::timeout(POLL_INTERVAL * 4, poll_once(app, label)).await;
        if let Ok(Some(rb @ Readback::Poll { .. })) = final_read {
            let this = rb.best_html();
            if this.len() > best.len() {
                best = this;
            }
        }
        if best.is_empty() { None } else { Some(best) }
    };

    // ── Final-committed-URL provenance re-check (C1). Read `webview.url()` and
    // run its host through the shared SSRF policy (off the event-loop thread).
    // A blocked final host ⇒ discard so internal content never reaches indexing.
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
                // Could not read the final URL — fail closed.
                tracing::warn!(target: "lens::js_render", label, "could not read final webview URL; discarding render output");
                None
            }
        }
    } else {
        None
    };

    // Teardown on the success / timeout / provenance-blocked / empty arms.
    schedule_destroy(app, label.to_string());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The poll-readback JSON parser handles the payload shapes POLL_JS emits:
    /// a `{quiescent, best, live}` snapshot (with `best_html` picking the larger
    /// of best/live), an `{err}` failure, missing scalars defaulted safely, and
    /// malformed/unrecognized input (fail closed to `Err`).
    #[test]
    fn parse_readback_handles_poll_err_and_garbage() {
        // Full snapshot: quiescent, live longer than best ⇒ best_html == live.
        match parse_readback(
            r#"{"quiescent":true,"best":"<html>a</html>","live":"<html>abc</html>"}"#,
        ) {
            Readback::Poll {
                quiescent,
                best,
                live,
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

        // Missing scalars default safely (not quiescent, empty html).
        match parse_readback(r#"{"best":"x"}"#) {
            Readback::Poll {
                quiescent,
                best,
                live,
            } => {
                assert!(!quiescent);
                assert_eq!(best, "x");
                assert_eq!(live, "");
            }
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

    /// C1 provenance decision (delegates to the lens-core helper). Blocked final
    /// host → discard; public → keep; malformed → discard (fail-closed). This is
    /// the CI-runnable form of the readback re-check — the live `webview.url()`
    /// wiring feeds this exact helper.
    #[test]
    fn readback_provenance_discards_blocked_and_malformed() {
        assert!(!readback_host_allowed(
            "http://169.254.169.254/latest/meta-data/"
        ));
        assert!(!readback_host_allowed("http://127.0.0.1/x"));
        assert!(!readback_host_allowed("http://[::1]/"));
        assert!(!readback_host_allowed("http://10.0.0.5/"));
        assert!(!readback_host_allowed("not a url"));
        assert!(readback_host_allowed("http://8.8.8.8/page"));
    }

    /// Per-navigation gate (M1): enforces the http/https scheme allowlist AND the
    /// host SSRF gate, non-blocking. Rejects page-JS-initiated navigations to
    /// non-http schemes and blocked hosts; allows public http/https.
    #[test]
    fn nav_url_allowed_enforces_scheme_and_host() {
        let allow = |s: &str| nav_url_allowed(&tauri::Url::parse(s).unwrap());
        // Non-http(s) schemes are rejected even with a benign-looking host.
        assert!(!allow("file://localhost/etc/passwd"));
        assert!(!allow("file:///etc/passwd"));
        assert!(!allow("data:text/html,<h1>x</h1>"));
        // Blocked hosts are rejected on http(s) too.
        assert!(!allow("http://169.254.169.254/latest/meta-data/"));
        assert!(!allow("http://127.0.0.1/x"));
        assert!(!allow("https://[::1]/"));
        // Public http/https is allowed.
        assert!(allow("https://example.com/page"));
        assert!(allow("http://8.8.8.8/page"));
    }
}
