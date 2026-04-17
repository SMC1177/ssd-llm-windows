//! Ollama-backed `Embedder`.
//!
//! Talks to a local Ollama daemon over HTTP. The daemon must be running
//! and the target model must already be pulled (`ollama pull nomic-embed-text`).
//! One blocking POST per `embed()` call — sufficient for the v1 use case
//! (batch indexing on a local SSD-resident corpus).

use crate::Embedder;
use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:11434/api/embeddings";
const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Embedder backed by a local Ollama HTTP daemon.
pub struct OllamaEmbedder {
    client: reqwest::blocking::Client,
    endpoint: String,
    model: String,
    dim: usize,
    tag: String,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

impl OllamaEmbedder {
    /// Generic constructor. Pass the model name as it appears in `ollama list`
    /// and the expected vector dimensionality. The dimensionality is stamped
    /// into the store's identity so an index built with one model can't be
    /// silently queried with another.
    pub fn new(model: impl Into<String>, dim: usize) -> anyhow::Result<Self> {
        let model = model.into();
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .context("build reqwest client")?;
        let tag = format!("ollama:{}", model);
        Ok(Self {
            client,
            endpoint: DEFAULT_ENDPOINT.to_string(),
            model,
            dim,
            tag,
        })
    }

    /// Convenience constructor for the canonical retrieval-tuned model used
    /// by ssd-index's v1 (nomic-embed-text, 768-dim).
    pub fn nomic_embed_text() -> anyhow::Result<Self> {
        Self::new("nomic-embed-text", 768)
    }

    /// Override the Ollama endpoint. Useful only when running Ollama on a
    /// non-default host/port (e.g., from a container or a remote dev box).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}

impl Embedder for OllamaEmbedder {
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        if text.is_empty() {
            return Err(anyhow!("cannot embed empty text"));
        }
        let body = EmbedRequest {
            model: &self.model,
            prompt: text,
        };
        let resp = self
            .client
            .post(&self.endpoint)
            .json(&body)
            .send()
            .with_context(|| format!("POST {}", self.endpoint))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(anyhow!(
                "ollama returned HTTP {}: {}",
                status,
                body.chars().take(500).collect::<String>()
            ));
        }
        let parsed: EmbedResponse = resp.json().context("parse ollama embed response")?;
        if parsed.embedding.len() != self.dim {
            return Err(anyhow!(
                "ollama returned {} dims, expected {} (model {})",
                parsed.embedding.len(),
                self.dim,
                self.model
            ));
        }
        Ok(parsed.embedding)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn tag(&self) -> &str {
        &self.tag
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chunk, Hit, Retriever, Store};

    /// Minimal Store impl for the seam test. Thread-safe (Send + Sync) via
    /// AtomicUsize rather than Cell so it satisfies the Store trait bound.
    struct FakeStore {
        chunks: Vec<(Chunk, Vec<f32>)>,
        queries: std::sync::atomic::AtomicUsize,
    }
    impl FakeStore {
        fn new() -> Self {
            Self {
                chunks: Vec::new(),
                queries: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }
    impl Store for FakeStore {
        fn put(&mut self, chunk: &Chunk, vector: &[f32]) -> anyhow::Result<()> {
            self.chunks.push((chunk.clone(), vector.to_vec()));
            Ok(())
        }
        fn query(&self, _vector: &[f32], top_k: usize) -> anyhow::Result<Vec<Hit>> {
            self.queries
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(self
                .chunks
                .iter()
                .take(top_k)
                .map(|(c, _)| Hit {
                    chunk_id: c.id.clone(),
                    score: 1.0,
                })
                .collect())
        }
        fn get_chunk(&self, id: &str) -> anyhow::Result<Option<Chunk>> {
            Ok(self
                .chunks
                .iter()
                .find(|(c, _)| c.id == id)
                .map(|(c, _)| c.clone()))
        }
        fn len(&self) -> usize {
            self.chunks.len()
        }
    }

    /// Seam test: OllamaEmbedder satisfies the Embedder trait well enough
    /// to compose into a Retriever<OllamaEmbedder, _>. Doesn't invoke embed()
    /// so Ollama doesn't have to be running.
    #[test]
    fn ollama_embedder_plugs_into_retriever() {
        let e = OllamaEmbedder::nomic_embed_text().expect("construct");
        assert_eq!(e.dim(), 768);
        assert_eq!(e.tag(), "ollama:nomic-embed-text");
        let _r: Retriever<OllamaEmbedder, FakeStore> = Retriever::new(e, FakeStore::new());
        // Compiling is the assertion — if the trait bound breaks, this fails
        // at build time, not here.
    }

    /// With_endpoint builder path is the only way to point at a non-default
    /// Ollama URL. Verify it actually overrides rather than silently being
    /// ignored.
    #[test]
    fn with_endpoint_overrides_default() {
        let e = OllamaEmbedder::new("nomic-embed-text", 768)
            .unwrap()
            .with_endpoint("http://10.0.0.5:11434/api/embeddings");
        assert_eq!(e.endpoint, "http://10.0.0.5:11434/api/embeddings");
    }

    /// Integration test against a live Ollama daemon.
    /// Requires `ollama serve` running locally AND `ollama pull nomic-embed-text`.
    /// Run explicitly: `cargo test --package ssd-index -- --ignored ollama_live_embed`.
    #[test]
    #[ignore]
    fn ollama_live_embed() {
        let e = OllamaEmbedder::nomic_embed_text().expect("construct");
        let v = e.embed("hello world").expect("embed request to local ollama");
        assert_eq!(v.len(), 768, "nomic-embed-text must return 768-dim vectors");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            norm > 0.0 && norm.is_finite(),
            "embedding must have finite positive norm, got {}",
            norm
        );
        // Determinism check: same input → same output. If this fails, the
        // Ollama backend is non-deterministic and the caller must not rely
        // on hash-equal vectors for caching.
        let v2 = e.embed("hello world").expect("second embed call");
        assert_eq!(v, v2, "nomic-embed-text should be deterministic");
    }
}
