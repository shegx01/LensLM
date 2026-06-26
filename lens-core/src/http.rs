//! One hardened `reqwest` client builder, shared by every outbound HTTP path
//! that must NOT follow redirects.
//!
//! Three call sites (`llm::llm_client`, `system_check::probe_client`,
//! `model_catalog::catalog_client`) previously each hand-rolled the same builder
//! and each ended in `.unwrap_or_default()` — which silently yields a client with
//! NO timeouts AND redirects FOLLOWED, dropping the SSRF guard. This module
//! collapses them onto one [`hardened_client`] whose fallback PRESERVES the
//! no-redirect policy + timeouts (it never degrades to a bare default).
//!
//! The TTS download client is deliberately NOT routed through here: it legitimately
//! needs redirects (HuggingFace `/resolve/` 302s to a CDN), so it keeps its own
//! builder.

use std::time::Duration;

/// Builds a hardened [`reqwest::Client`]: bounded `connect`/`read` timeouts plus
/// SSRF hardening (never follow a redirect — a malicious / misconfigured endpoint
/// could 30x toward an internal host).
///
/// The TLS backend is pure-Rust rustls (no system deps), so the builder can only
/// fail under pathological conditions; the fallback rebuilds with the SAME
/// hardening (timeouts + `redirect(none)`) rather than degrading to a bare
/// default — so a redirect-following, timeout-less client can never escape.
pub fn hardened_client(connect: Duration, read: Duration) -> reqwest::Client {
    let builder = || {
        reqwest::Client::builder()
            .connect_timeout(connect)
            .timeout(read)
            .redirect(reqwest::redirect::Policy::none())
    };
    // The fallback rebuilds the identical hardened client. The final `expect` is a
    // last-resort guard on an unreachable path (rustls has no system deps): we
    // never substitute a bare default that would silently follow redirects.
    builder().build().unwrap_or_else(|_| {
        builder()
            .build()
            .expect("hardened reqwest client must build (rustls has no system deps)")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardened_client_builds() {
        // Smoke test: the builder succeeds with bounded timeouts on the happy path
        // (the fallback `expect` only guards the unreachable rustls-init failure).
        let _client = hardened_client(Duration::from_secs(1), Duration::from_secs(5));
    }
}
