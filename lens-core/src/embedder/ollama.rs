//! [`OllamaEmbedder`] — the [`EmbeddingBackend::Ollama`] arm of the
//! [`Embedder`] trait (M4 Phase 4b-B, Step 3).
//!
//! Computes embeddings via a LOCAL Ollama server's `POST /api/embed` endpoint
//! instead of the on-device fastembed ONNX session. The same registry model
//! (e.g. `nomic-embed-text-v1.5`/768) can be served by either backend, and the
//! two are physically distinct vector sets (different numerical embeddings) that
//! live in separate LanceDB tables — see the `Coordinate` backend axis.
//!
//! ## Loopback-only (security contract)
//!
//! The embedder is constructed with a base URL that is validated to be
//! LOOPBACK-ONLY ([`crate::ingest::require_loopback`]) — the inverse of the URL
//! source's SSRF guard, which *rejects* loopback. We POST the document/query
//! text being embedded to this server; allowing a LAN/public host would let a
//! misconfigured or malicious endpoint exfiltrate the user's documents or act as
//! an SSRF pivot. The loopback check happens in [`OllamaEmbedder::new`] BEFORE
//! any request is made, so a non-loopback URL fails construction with no
//! network traffic.
//!
//! ## Blocking contract
//!
//! The [`Embedder`] trait is synchronous (it mirrors fastembed's sync `embed`),
//! and every production caller already wraps embed calls in
//! [`tokio::task::spawn_blocking`]. The Ollama HTTP client is async, so this
//! embedder captures a [`tokio::runtime::Handle`] at construction and drives each
//! request with [`tokio::runtime::Handle::block_on`]. That is sound ONLY off a
//! runtime worker thread — which is exactly where `spawn_blocking` runs the trait
//! methods. A direct call on a Tokio worker would panic.

use std::time::Duration;

use futures_util::StreamExt;
use serde::Deserialize;

use crate::LensError;
use crate::embedder::Embedder;
use crate::embedder::registry::EmbeddingModelSpec;
use crate::http::hardened_client;

/// `connect` timeout for an Ollama embed request (the server is loopback, so a
/// connect should be near-instant; a longer connect means the server is down).
const OLLAMA_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// `read` timeout for an Ollama embed request. Embedding a batch of chunks on a
/// CPU-only Ollama server can take several seconds, so this is generous.
const OLLAMA_READ_TIMEOUT: Duration = Duration::from_secs(120);

/// Upper bound on the embed-response body we will buffer before deserializing.
///
/// Defense-in-depth: the Ollama server is loopback, but a hostile or compromised
/// local process bound to the port could stream a multi-gigabyte body and OOM
/// the app. We cap the buffered bytes and abort the stream once it is exceeded
/// (mirroring the URL-ingest streaming cap in `ingest.rs`).
///
/// Sizing: the ≤32-input bound is the CALLER's contract — `ingest.rs` batches by
/// `EMBED_BATCH` (32) and `reembed.rs` by `REEMBED_BATCH` (32); `post_embed` itself
/// accepts an unbounded slice, so this cap is sized against that caller convention,
/// not an internal limit. A single embed request thus POSTs at most 32 inputs, so
/// the response is at most 32 vectors of the largest registry dim (2560, for the
/// Ollama-only `qwen3-embedding:4b`; was 1024 before issue #80) of `f32`s. Each
/// float serializes to ~12 JSON bytes (sign/digits/`.`/exponent + a `,`
/// delimiter), so a worst-case legitimate body is `32 * 2560 * 12 ≈ 960 KiB`. We
/// keep the 8 MiB cap — still ~8x headroom for whitespace / extra fields / future
/// batch growth — well below anything that could pressure memory.
const MAX_OLLAMA_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// Response shape of Ollama's `POST /api/embed`: a list of embedding vectors,
/// one per input, in order.
#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

/// An [`Embedder`] backed by a LOCAL Ollama server's `/api/embed` endpoint.
///
/// Construction validates the base URL is loopback-only and stores the model id,
/// dimension, and prefix convention from the registry [`EmbeddingModelSpec`].
/// Embedding requests apply the spec's document/query prefixes, L2-normalize the
/// returned vectors, and verify the dimension against the spec (a wrong-dim
/// response — e.g. the wrong Ollama model tag installed — is an error, never
/// silently stored).
#[derive(Debug)]
pub struct OllamaEmbedder {
    /// Async HTTP client (hardened: bounded timeouts + no redirects).
    client: reqwest::Client,
    /// Runtime handle captured at construction so the sync trait methods can
    /// drive the async request via `block_on` from a `spawn_blocking` thread.
    handle: tokio::runtime::Handle,
    /// Loopback-validated `POST {base}/api/embed` URL.
    embed_url: String,
    /// The Ollama model tag to embed with (the registry model id).
    model: String,
    /// Stable registry model id reported by [`Embedder::model_id`].
    model_id: String,
    /// Output dimension to validate every response against.
    dim: usize,
    /// Document prefix from the spec (`""` = none).
    prefix_doc: String,
    /// Query prefix from the spec (`""` = none).
    prefix_query: String,
}

impl OllamaEmbedder {
    /// Builds an `OllamaEmbedder` targeting `base_url` for the model in `spec`.
    ///
    /// Validates `base_url` is LOOPBACK-ONLY before doing anything else (no
    /// request is made on a rejected URL). Captures the current Tokio runtime
    /// handle for the sync→async bridge.
    ///
    /// # Errors
    ///
    /// Returns [`LensError::Validation`] if `base_url` is not a loopback
    /// `http`/`https` address.
    ///
    /// # Panics
    ///
    /// Panics if called outside a Tokio runtime context (no current handle).
    /// Every production construction site (`LensEngine::embedder_for`) is async,
    /// so a runtime is always present.
    pub fn new(base_url: &str, spec: &EmbeddingModelSpec) -> Result<Self, LensError> {
        // Defense-in-depth backend guard (issue #80): a spec that does not list
        // Ollama among its backends must never be served by this embedder. The
        // primary, user-facing guard lives in `embedder_for`; this is the last
        // line of defense so a direct construction can't smuggle a fastembed-only
        // model onto the Ollama path.
        if !spec.supports(crate::embedder::EmbeddingBackend::Ollama) {
            return Err(LensError::Validation(format!(
                "model {} does not support the ollama backend",
                spec.id
            )));
        }
        // Loopback gate — a rejected URL must make NO request. On success it
        // returns the resolved loopback addrs so we can PIN reqwest to them.
        let target = crate::ingest::require_loopback(base_url)?;
        let base = base_url.trim_end_matches('/');
        let embed_url = format!("{base}/api/embed");
        // DNS-rebinding TOCTOU defense: for a HOSTNAME base URL, pin reqwest to the
        // exact loopback addresses `require_loopback` resolved + checked, so reqwest
        // does NOT run a second, unchecked DNS lookup at connect time. An IP-literal
        // host has no DNS step (empty `pinned_addrs`) → the plain hardened client.
        let client = if target.pinned_addrs.is_empty() {
            hardened_client(OLLAMA_CONNECT_TIMEOUT, OLLAMA_READ_TIMEOUT)
        } else {
            crate::http::hardened_client_pinned(
                OLLAMA_CONNECT_TIMEOUT,
                OLLAMA_READ_TIMEOUT,
                &target.host,
                &target.pinned_addrs,
            )
        };
        Ok(Self {
            client,
            handle: tokio::runtime::Handle::current(),
            embed_url,
            model: spec.id.to_string(),
            model_id: spec.id.to_string(),
            dim: spec.dim,
            prefix_doc: spec.prefix_doc.to_string(),
            prefix_query: spec.prefix_query.to_string(),
        })
    }

    /// POSTs `inputs` to `/api/embed`, returning the raw embedding vectors.
    ///
    /// Drives the async request to completion on the captured runtime handle.
    fn post_embed(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>, LensError> {
        let client = self.client.clone();
        let url = self.embed_url.clone();
        let body = serde_json::json!({ "model": self.model, "input": inputs });
        let resp: OllamaEmbedResponse = self.handle.block_on(async move {
            let r = client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| LensError::Network(format!("ollama embed request failed: {e}")))?;
            if !r.status().is_success() {
                return Err(LensError::Network(format!(
                    "ollama embed returned HTTP {}",
                    r.status()
                )));
            }
            // Short-circuit on a declared Content-Length over the cap (avoids
            // streaming a body the server already admits is oversized). Uses
            // `Option::filter` (a single `if let`, not a `let`-chain) to keep the
            // build portable while still binding `len` for the error message.
            if let Some(len) = r
                .content_length()
                .filter(|&len| len > MAX_OLLAMA_RESPONSE_BYTES as u64)
            {
                return Err(LensError::Parse(format!(
                    "ollama embed response declares {len} bytes, exceeding the \
                         {MAX_OLLAMA_RESPONSE_BYTES}-byte cap"
                )));
            }
            // Stream the body, enforcing the cap as bytes arrive so a server
            // that lies about (or omits) Content-Length cannot OOM the app.
            let mut buf: Vec<u8> = Vec::new();
            let mut stream = r.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| {
                    LensError::Network(format!("ollama embed body read failed: {e}"))
                })?;
                if buf.len() + chunk.len() > MAX_OLLAMA_RESPONSE_BYTES {
                    return Err(LensError::Parse(format!(
                        "ollama embed response exceeds the {MAX_OLLAMA_RESPONSE_BYTES}-byte cap"
                    )));
                }
                buf.extend_from_slice(&chunk);
            }
            serde_json::from_slice::<OllamaEmbedResponse>(&buf)
                .map_err(|e| LensError::Parse(format!("ollama embed response decode failed: {e}")))
        })?;
        Ok(resp.embeddings)
    }

    /// Validates each returned vector has the expected dimension, then
    /// L2-normalizes it via the shared [`crate::embedder::l2_normalize`] (Ollama
    /// does NOT normalize; fastembed does, so we normalize here to keep both
    /// backends' vectors directly comparable by cosine distance). A wrong-dim
    /// response (the wrong Ollama model tag) is an error.
    fn finalize(&self, mut vecs: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, LensError> {
        for (i, v) in vecs.iter_mut().enumerate() {
            if v.len() != self.dim {
                return Err(LensError::Model(format!(
                    "ollama returned vector {i} with dim {} (expected {})",
                    v.len(),
                    self.dim
                )));
            }
            crate::embedder::l2_normalize(v);
        }
        Ok(vecs)
    }
}

impl Embedder for OllamaEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LensError> {
        let inputs: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.prefix_doc))
            .collect();
        let vecs = self.post_embed(inputs)?;
        self.finalize(vecs)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, LensError> {
        let input = format!("{}{text}", self.prefix_query);
        let vecs = self.post_embed(vec![input])?;
        let vecs = self.finalize(vecs)?;
        vecs.into_iter()
            .next()
            .ok_or_else(|| LensError::Model("ollama returned empty batch for embed_query".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::registry::resolve;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Step 4 (issue #80): the construction-time backend guard rejects a
    /// fastembed-only spec BEFORE the loopback check, with a clear error naming the
    /// model. This is the defense-in-depth backstop under `embedder_for`'s guard.
    #[tokio::test]
    async fn rejects_fastembed_only_model() {
        // A valid loopback URL, so ONLY the backend guard can reject this.
        let err = OllamaEmbedder::new("http://127.0.0.1:11434", resolve("nomic-embed-text-v1.5"))
            .expect_err("a fastembed-only model must be rejected by the ollama backend guard");
        match err {
            LensError::Validation(msg) => {
                assert!(
                    msg.contains("nomic-embed-text-v1.5"),
                    "names the model: {msg}"
                );
                assert!(msg.contains("ollama"), "names the backend: {msg}");
            }
            other => panic!("expected LensError::Validation, got {other:?}"),
        }
    }

    // --- Loopback gate: rejected URLs make NO request (validated in `new`) ---

    #[tokio::test]
    async fn rejects_private_lan_url() {
        let err = OllamaEmbedder::new("http://10.0.0.5:11434", resolve("nomic-embed-text-v2-moe"))
            .expect_err("a private LAN URL must be rejected");
        assert!(matches!(err, LensError::Validation(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn rejects_public_hostname_url() {
        let err = OllamaEmbedder::new("http://evil.example", resolve("nomic-embed-text-v2-moe"))
            .expect_err("a public hostname must be rejected");
        // Either a Validation (resolved to non-loopback) or a Network (DNS) error,
        // but NEVER an Ok — and crucially no embed request is ever issued.
        assert!(
            matches!(err, LensError::Validation(_) | LensError::Network(_)),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn accepts_localhost_127_and_ipv6_loopback() {
        for base in [
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "http://[::1]:11434",
        ] {
            OllamaEmbedder::new(base, resolve("nomic-embed-text-v2-moe"))
                .unwrap_or_else(|e| panic!("{base} should be accepted: {e:?}"));
        }
    }

    // --- Happy path against a loopback wiremock server ---

    /// 768-dim L2-normalized vector with a marker in the first slot so we can
    /// assert the request landed.
    fn mock_vec(marker: f32, dim: usize) -> Vec<f32> {
        let mut v = vec![0.5_f32; dim];
        v[0] = marker;
        v
    }

    #[tokio::test]
    async fn embed_documents_happy_path_normalizes_and_checks_dim() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "embeddings": [mock_vec(1.0, 768), mock_vec(2.0, 768)]
        });
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let embedder =
            OllamaEmbedder::new(&server.uri(), resolve("nomic-embed-text-v2-moe")).unwrap();
        // The embed call is sync + blocks on the runtime; run it off the worker.
        let vecs = tokio::task::spawn_blocking(move || embedder.embed_documents(&["a", "b"]))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(vecs.len(), 2);
        for v in &vecs {
            assert_eq!(v.len(), 768);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "vector not normalized: {norm}");
        }
    }

    #[tokio::test]
    async fn embed_query_applies_prefix_in_request_body() {
        use wiremock::matchers::body_string_contains;
        let server = MockServer::start().await;
        let body = serde_json::json!({ "embeddings": [mock_vec(1.0, 768)] });
        // nomic's query prefix is "search_query: " — assert it is sent in the body.
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .and(body_string_contains("search_query: hello"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let embedder =
            OllamaEmbedder::new(&server.uri(), resolve("nomic-embed-text-v2-moe")).unwrap();
        let v = tokio::task::spawn_blocking(move || embedder.embed_query("hello"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(v.len(), 768);
    }

    #[tokio::test]
    async fn embed_documents_applies_doc_prefix_in_request_body() {
        use wiremock::matchers::body_string_contains;
        let server = MockServer::start().await;
        let body = serde_json::json!({ "embeddings": [mock_vec(1.0, 768)] });
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .and(body_string_contains("search_document: doc text"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let embedder =
            OllamaEmbedder::new(&server.uri(), resolve("nomic-embed-text-v2-moe")).unwrap();
        let vecs = tokio::task::spawn_blocking(move || embedder.embed_documents(&["doc text"]))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(vecs.len(), 1);
    }

    #[tokio::test]
    async fn over_cap_response_body_is_rejected_without_oom() {
        let server = MockServer::start().await;
        // A body well over MAX_OLLAMA_RESPONSE_BYTES (8 MiB). We do NOT set
        // Content-Length deliberately small — the streaming accumulator must abort
        // on byte count regardless of the declared length. Use a raw oversized
        // string body (not valid JSON shape) so a successful decode is impossible;
        // the cap must trip before any parse.
        let oversized = "x".repeat(MAX_OLLAMA_RESPONSE_BYTES + 1024);
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(oversized, "application/json"))
            .mount(&server)
            .await;

        let embedder =
            OllamaEmbedder::new(&server.uri(), resolve("nomic-embed-text-v2-moe")).unwrap();
        let err = tokio::task::spawn_blocking(move || embedder.embed_documents(&["x"]))
            .await
            .unwrap()
            .expect_err("an over-cap body must be rejected");
        // A clear Parse error mentioning the cap, NOT a panic / unbounded alloc.
        match err {
            LensError::Parse(msg) => assert!(
                msg.contains("cap"),
                "expected a cap-related Parse error, got: {msg}"
            ),
            other => panic!("expected LensError::Parse, got {other:?}"),
        }
    }

    // --- DNS-rebinding pin: the embedder builds a connection PINNED to the
    // loopback addrs `require_loopback` resolved (no second unchecked resolve) ---

    /// A HOSTNAME base URL (`localhost`) makes `require_loopback` return non-empty
    /// `pinned_addrs`, so `OllamaEmbedder::new` takes the PINNED-client branch
    /// rather than the plain hardened client. An IP-literal host returns empty
    /// (no DNS step to pin).
    #[test]
    fn require_loopback_returns_pinned_addrs_for_hostname() {
        let hostname = crate::ingest::require_loopback("http://localhost:11434")
            .expect("localhost is loopback");
        assert!(
            !hostname.pinned_addrs.is_empty(),
            "a hostname must yield resolved loopback addrs to pin"
        );
        assert!(
            hostname.pinned_addrs.iter().all(|a| a.ip().is_loopback()),
            "every pinned addr must be loopback"
        );

        let literal = crate::ingest::require_loopback("http://127.0.0.1:11434")
            .expect("127.0.0.1 is loopback");
        assert!(
            literal.pinned_addrs.is_empty(),
            "an IP-literal host has no DNS step to pin"
        );
    }

    /// Proves the embedder's pinned-client builder (`hardened_client_pinned`, the
    /// branch `OllamaEmbedder::new` takes for a hostname) connects to the PINNED
    /// address and does NOT re-resolve the host — the DNS-rebinding TOCTOU defense.
    /// A bogus host with NO DNS record is pinned to the mock's loopback addr; the
    /// request succeeds ONLY because reqwest honors the pin.
    #[tokio::test]
    async fn pinned_client_connects_only_to_pinned_addr() {
        use std::net::ToSocketAddrs;
        let server = MockServer::start().await;
        let body = serde_json::json!({ "embeddings": [mock_vec(1.0, 768)] });
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let mock_url = url::Url::parse(&server.uri()).unwrap();
        let port = mock_url.port().unwrap();
        let addrs: Vec<std::net::SocketAddr> = (mock_url.host_str().unwrap(), port)
            .to_socket_addrs()
            .unwrap()
            .collect();

        // A hostname that does NOT resolve via DNS — only the pin can reach the mock.
        let bogus_host = "ollama-pinned.invalid";
        let client = crate::http::hardened_client_pinned(
            OLLAMA_CONNECT_TIMEOUT,
            OLLAMA_READ_TIMEOUT,
            bogus_host,
            &addrs,
        );
        let url = format!("http://{bogus_host}:{port}/api/embed");
        let resp = client
            .post(&url)
            .json(&serde_json::json!({ "model": "m", "input": ["x"] }))
            .send()
            .await
            .expect("pinned client must reach the mock via the pinned addr");
        assert!(resp.status().is_success(), "got {}", resp.status());
    }

    #[tokio::test]
    async fn wrong_dim_response_is_error() {
        let server = MockServer::start().await;
        // Server returns 512-dim but the spec wants 768 → error.
        let body = serde_json::json!({ "embeddings": [mock_vec(1.0, 512)] });
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let embedder =
            OllamaEmbedder::new(&server.uri(), resolve("nomic-embed-text-v2-moe")).unwrap();
        let err = tokio::task::spawn_blocking(move || embedder.embed_documents(&["x"]))
            .await
            .unwrap()
            .expect_err("a 512-dim response for a 768 spec must error");
        assert!(matches!(err, LensError::Model(_)), "got {err:?}");
    }
}
