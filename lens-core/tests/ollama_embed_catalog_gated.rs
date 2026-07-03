//! Gated dim-canary for the curated Ollama embedding catalog (issue #80, Step 10).
//!
//! For each Ollama-ONLY registry spec, if a local Ollama server is reachable AND
//! the model is pulled (present in `GET /api/tags`), embed one document and assert
//! the returned vector width equals the spec's pinned `dim`. This is the runtime
//! proof that the hardcoded dim constant is correct — a wrong constant fails loudly
//! here.
//!
//! **Gated by contract:** when Ollama is unreachable OR the specific model is not
//! pulled, the test `eprintln!`s a skip line and returns SUCCESS — it NEVER fails
//! on an Ollama-absent machine (CI runs without Ollama). Run it against a live
//! daemon with the four models pulled to exercise the canary.

use lens_core::system_check::{list_ollama_models, ollama_base_url};
use lens_core::{AppConfig, Embedder, EmbeddingBackend, OllamaEmbedder, REGISTRY};

/// Whether `installed` (a `GET /api/tags` model name, possibly `id:tag`) is the
/// pulled form of registry `id`. Mirrors the exact-tag D3 rule: a colon-bearing
/// id matches exactly; an untagged id matches itself or `id:<tag>`.
fn tag_matches(installed: &str, id: &str) -> bool {
    let installed = installed.to_ascii_lowercase();
    let id = id.to_ascii_lowercase();
    if id.contains(':') {
        installed == id
    } else {
        installed == id || installed.starts_with(&format!("{id}:"))
    }
}

#[tokio::test]
async fn ollama_catalog_dim_canary_gated() {
    let base_url = ollama_base_url(&AppConfig::default());
    let installed = list_ollama_models(&base_url).await;
    if installed.is_empty() {
        eprintln!(
            "SKIP ollama_catalog_dim_canary_gated: no Ollama server reachable at {base_url} \
             (or no models pulled); this is expected on CI."
        );
        return;
    }

    let ollama_only: Vec<&'static lens_core::EmbeddingModelSpec> = REGISTRY
        .iter()
        .filter(|s| s.supports(EmbeddingBackend::Ollama))
        .collect();

    for spec in ollama_only {
        let pulled = installed.iter().any(|m| tag_matches(m, spec.id));
        if !pulled {
            eprintln!(
                "SKIP {}: model not present in Ollama /api/tags; pull it to exercise the canary.",
                spec.id
            );
            continue;
        }

        let embedder = OllamaEmbedder::new(&base_url, spec)
            .unwrap_or_else(|e| panic!("construct OllamaEmbedder for {}: {e:?}", spec.id));
        // The embed call is sync + blocks on the runtime; run it off the worker.
        let v = tokio::task::spawn_blocking(move || {
            embedder.embed_query("a short canary document for dimension verification")
        })
        .await
        .unwrap()
        .unwrap_or_else(|e| panic!("embed with {}: {e:?}", spec.id));

        assert_eq!(
            v.len(),
            spec.dim,
            "{} returned dim {} but the registry pins {}",
            spec.id,
            v.len(),
            spec.dim
        );
        eprintln!("OK {}: embed dim {} matches registry", spec.id, spec.dim);
    }
}
