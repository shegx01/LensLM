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
    build_hardened(|b| b.connect_timeout(connect).timeout(read))
}

/// Like [`hardened_client`] but bounded by an idle read timeout instead of a
/// total-request deadline; see `llm::LLM_GENERATION_IDLE_TIMEOUT` for why.
pub fn hardened_client_idle(connect: Duration, idle_read: Duration) -> reqwest::Client {
    build_hardened(|b| b.connect_timeout(connect).read_timeout(idle_read))
}

/// Builds a hardened client (same timeouts + `redirect(none)` as
/// [`hardened_client`]) whose connection is PINNED to `addrs` for `host` via
/// `resolve_to_addrs`, so reqwest connects ONLY to the supplied addresses and
/// performs NO second DNS resolution at connect time.
///
/// This closes the DNS-rebinding TOCTOU on a loopback-validated base URL: the
/// host is resolved + guard-checked once (e.g. by `ingest::require_loopback`), and
/// the resulting loopback addresses are pinned here so a short-TTL / attacker
/// record cannot rebind to a non-loopback address between validation and connect.
/// `addrs` MUST be non-empty (an IP-literal host needs no pinning and should use
/// [`hardened_client`] directly).
pub fn hardened_client_pinned(
    connect: Duration,
    read: Duration,
    host: &str,
    addrs: &[std::net::SocketAddr],
) -> reqwest::Client {
    build_hardened(|b| {
        b.connect_timeout(connect)
            .timeout(read)
            .resolve_to_addrs(host, addrs)
    })
}

/// Shared hardened-builder core: applies the no-redirect policy, then runs
/// `customize` (each caller sets its own timeout model + any `resolve_to_addrs`
/// pin). The fallback rebuilds the identical hardened client. The final `expect` is
/// a last-resort guard on an unreachable path (rustls has no system deps): we never
/// substitute a bare default that would silently follow redirects.
fn build_hardened(
    customize: impl Fn(reqwest::ClientBuilder) -> reqwest::ClientBuilder,
) -> reqwest::Client {
    let builder =
        || customize(reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()));
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
