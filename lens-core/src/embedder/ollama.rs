//! [`OllamaEmbedder`] — the [`EmbeddingBackend::Ollama`] arm of [`Embedder`].
//!
//! Security: base URL is validated LOOPBACK-ONLY at construction time (no request
//! is made on rejection). Posting doc/query text to a non-loopback host risks
//! document exfiltration. Blocking: the async HTTP client is driven via
//! `Handle::block_on`, sound only off a `spawn_blocking` worker thread.

use std::time::Duration;

use futures_util::StreamExt;
use serde::Deserialize;

use crate::LensError;
use crate::embedder::Embedder;
use crate::embedder::registry::EmbeddingModelSpec;
use crate::http::hardened_client;

const OLLAMA_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

const OLLAMA_READ_TIMEOUT: Duration = Duration::from_secs(120);

/// Response body cap. Worst-case legitimate response: 32 inputs × 2560-dim ×
/// ~12 JSON bytes/float ≈ 960 KiB. 8 MiB gives ~8× headroom; a hostile local
/// process can't OOM the app by streaming an unbounded body.
const MAX_OLLAMA_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

/// [`Embedder`] backed by a LOCAL Ollama server's `/api/embed` endpoint.
/// Applies spec prefixes, L2-normalizes returned vectors, and validates dimension.
#[derive(Debug)]
pub struct OllamaEmbedder {
    client: reqwest::Client,
    handle: tokio::runtime::Handle,
    embed_url: String,
    model: String,
    model_id: String,
    dim: usize,
    prefix_doc: String,
    prefix_query: String,
}

impl OllamaEmbedder {
    /// Builds an `OllamaEmbedder` for `base_url` + `spec`.
    ///
    /// # Errors
    /// [`LensError::Validation`] if `base_url` is not a loopback address.
    ///
    /// # Panics
    /// Panics outside a Tokio runtime context (no current handle).
    pub fn new(base_url: &str, spec: &EmbeddingModelSpec) -> Result<Self, LensError> {
        // Defense-in-depth (issue #80): primary guard is in `embedder_for`.
        if !spec.supports(crate::embedder::EmbeddingBackend::Ollama) {
            return Err(LensError::Validation(format!(
                "model {} does not support the ollama backend",
                spec.id
            )));
        }
        let target = crate::ingest::require_loopback(base_url)?;
        let base = base_url.trim_end_matches('/');
        let embed_url = format!("{base}/api/embed");
        // DNS-rebinding TOCTOU defense: pin reqwest to the addresses
        // `require_loopback` resolved so there is no second unchecked DNS lookup.
        // IP-literal hosts have no DNS step → empty `pinned_addrs` → plain client.
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

    /// POSTs `inputs` to `/api/embed` via the captured runtime handle.
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
            // Short-circuit on a declared Content-Length over the cap.
            // Uses Option::filter (not a let-chain) for build portability.
            if let Some(len) = r
                .content_length()
                .filter(|&len| len > MAX_OLLAMA_RESPONSE_BYTES as u64)
            {
                return Err(LensError::Parse(format!(
                    "ollama embed response declares {len} bytes, exceeding the \
                         {MAX_OLLAMA_RESPONSE_BYTES}-byte cap"
                )));
            }
            // Enforce cap as bytes arrive; a server that lies about Content-Length
            // cannot OOM the app.
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

    /// Validates dimension and L2-normalizes each vector. Ollama does NOT
    /// normalize; we do to keep both backends' vectors cosine-comparable.
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
        match err {
            LensError::Parse(msg) => assert!(
                msg.contains("cap"),
                "expected a cap-related Parse error, got: {msg}"
            ),
            other => panic!("expected LensError::Parse, got {other:?}"),
        }
    }

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
