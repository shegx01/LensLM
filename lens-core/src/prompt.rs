//! Shared prompt-fencing helpers for grounded prompts (answer + dialogue). The fence
//! wraps untrusted source text in a per-request nonce marker so injected text cannot
//! forge a boundary and escape the data region.

/// A fresh per-request fence nonce (12 hex chars). Untrusted source text is authored
/// at ingest — before this exists — so it can never pre-forge the marker.
pub(crate) fn fence_nonce() -> String {
    uuid::Uuid::now_v7().simple().to_string()[..12].to_string()
}

/// Wraps one excerpt's `inner` text in the `<<SRC:nonce>> … <<END:nonce>>` fence.
/// The single source of truth for the marker format, so the prompt builders cannot
/// drift apart.
pub(crate) fn fence_excerpt(nonce: &str, inner: &str) -> String {
    format!("<<SRC:{nonce}>>\n{inner}\n<<END:{nonce}>>\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonce_is_12_hex_chars() {
        let n = fence_nonce();
        assert_eq!(n.len(), 12);
        assert!(n.bytes().all(|b| b.is_ascii_hexdigit()));
    }

    #[test]
    fn excerpt_wraps_inner_between_markers() {
        assert_eq!(
            fence_excerpt("abc", "body"),
            "<<SRC:abc>>\nbody\n<<END:abc>>\n"
        );
    }
}
