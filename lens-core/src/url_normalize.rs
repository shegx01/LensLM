//! Moderate URL normalization for content-hash dedup (issue #100).
//!
//! [`normalize_url`] produces a canonical string form of a URL so that two
//! user-entered URLs that are *equivalent* under a conservative set of rules
//! hash to the same [`raw_content_hash`](crate::notebooks::Source::raw_content_hash)
//! and therefore deduplicate. The normalization is deliberately **moderate**:
//! it only collapses differences that never change which resource is fetched
//! (scheme/host case, default ports, fragment, a single trailing slash). It does
//! NOT touch anything that *can* change the response — query-string content and
//! ordering are preserved verbatim, and path case is preserved.
//!
//! The heavy lifting (RFC 3986 parsing, IDNA/punycode host handling, default-port
//! stripping, scheme+host lowercasing) is done by the `url` crate — already a
//! dependency. Only the fragment strip and trailing-slash strip are applied on
//! top.

use crate::LensError;

/// Normalizes a URL for dedup purposes, returning its canonical string form.
///
/// Applied transforms:
/// * **scheme + host lowercased** (via the `url` crate) — `HTTPS://Example.COM`
///   → `https://example.com`.
/// * **default port stripped** (via the `url` crate) — `:80` for `http`, `:443`
///   for `https`; a non-default port is preserved.
/// * **fragment dropped** — `…/page#section` → `…/page`.
/// * **exactly one trailing slash removed from the path**, except when the path
///   is the root `/` (which is preserved). `…/page/` → `…/page`; `…/a//` → `…/a/`.
///
/// Explicitly **preserved** (never altered):
/// * path case — `/Page/To/Thing` stays as-is.
/// * query string content AND parameter order — `?b=2&a=1` is NOT reordered and
///   is distinct from `?a=1&b=2`.
///
/// Returns [`LensError::Validation`] when `input` is not a parseable absolute
/// URL (the same error shape used by the fetch path in
/// [`crate::ingest`]).
pub fn normalize_url(input: &str) -> Result<String, LensError> {
    let mut parsed = url::Url::parse(input)
        .map_err(|e| LensError::Validation(format!("invalid URL {input:?}: {e}")))?;

    // Drop the fragment — it never affects the fetched resource.
    parsed.set_fragment(None);

    // Strip exactly one trailing slash from the path, unless the path is the
    // bare root "/" (which we preserve). `set_path` re-serializes cleanly.
    let path = parsed.path().to_string();
    if path.len() > 1 && path.ends_with('/') {
        parsed.set_path(&path[..path.len() - 1]);
    }

    Ok(parsed.as_str().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercases_scheme_and_host() {
        assert_eq!(
            normalize_url("https://Example.COM/path").unwrap(),
            "https://example.com/path"
        );
    }

    #[test]
    fn strips_default_https_port() {
        assert_eq!(
            normalize_url("https://example.com:443/path").unwrap(),
            "https://example.com/path"
        );
    }

    #[test]
    fn strips_default_http_port() {
        assert_eq!(
            normalize_url("http://example.com:80/path").unwrap(),
            "http://example.com/path"
        );
    }

    #[test]
    fn preserves_non_default_port() {
        assert_eq!(
            normalize_url("http://example.com:8080/path").unwrap(),
            "http://example.com:8080/path"
        );
    }

    #[test]
    fn strips_fragment() {
        assert_eq!(
            normalize_url("https://example.com/path#section").unwrap(),
            "https://example.com/path"
        );
    }

    #[test]
    fn strips_single_trailing_slash() {
        assert_eq!(
            normalize_url("https://example.com/path/").unwrap(),
            "https://example.com/path"
        );
    }

    #[test]
    fn preserves_root_trailing_slash() {
        assert_eq!(
            normalize_url("https://example.com/").unwrap(),
            "https://example.com/"
        );
    }

    #[test]
    fn preserves_path_case() {
        assert_eq!(
            normalize_url("https://example.com/Path/To/Page").unwrap(),
            "https://example.com/Path/To/Page"
        );
    }

    #[test]
    fn preserves_query_order() {
        assert_eq!(
            normalize_url("https://example.com/path?b=2&a=1").unwrap(),
            "https://example.com/path?b=2&a=1"
        );
    }

    #[test]
    fn query_order_is_significant() {
        // Two different orderings must NOT normalize to the same string.
        let a = normalize_url("https://example.com/path?a=1&b=2").unwrap();
        let b = normalize_url("https://example.com/path?b=2&a=1").unwrap();
        assert_ne!(a, b, "query parameter order must be preserved (distinct)");
    }

    #[test]
    fn preserves_query_value_encoding() {
        assert_eq!(
            normalize_url("https://example.com/path?q=hello+world").unwrap(),
            "https://example.com/path?q=hello+world"
        );
    }

    #[test]
    fn path_case_is_significant() {
        // `/Page` and `/page` are distinct resources — must not dedup.
        let a = normalize_url("https://example.com/Page").unwrap();
        let b = normalize_url("https://example.com/page").unwrap();
        assert_ne!(a, b, "path case must be preserved (distinct)");
    }

    #[test]
    fn plan_example_composite() {
        // `Example.com:443/Page/#top` → `https://example.com/Page`
        assert_eq!(
            normalize_url("https://Example.com:443/Page/#top").unwrap(),
            "https://example.com/Page"
        );
    }

    #[test]
    fn only_one_trailing_slash_removed() {
        assert_eq!(
            normalize_url("https://example.com/a//").unwrap(),
            "https://example.com/a/"
        );
    }

    #[test]
    fn invalid_url_rejects() {
        let err = normalize_url("not a url");
        assert!(
            matches!(err, Err(LensError::Validation(_))),
            "unparseable input must return Validation, got {err:?}"
        );
    }

    #[test]
    fn relative_url_rejects() {
        let err = normalize_url("/relative/path");
        assert!(
            matches!(err, Err(LensError::Validation(_))),
            "a relative URL has no base and must return Validation, got {err:?}"
        );
    }
}
