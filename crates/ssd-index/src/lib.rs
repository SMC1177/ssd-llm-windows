//! ssd-index — file-system semantic index and retrieval.
//!
//! Provides the scaffolding traits that decouple the retrieval layer from:
//!   * which embedder turns text into vectors (Ollama, ssd-llm, etc.)
//!   * which generator answers follow-up questions or reranks
//!   * which storage backend holds the vectors on SSD
//!
//! The v1 storage target is mmap-backed `.bin` files so the index itself
//! lives on SSD with the same streaming discipline as model weights.

pub mod embed;
pub mod store;

use serde::{Deserialize, Serialize};

/// One indexable unit of text with provenance back to the source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Stable identifier (typically `<file_hash>:<byte_offset>`).
    pub id: String,
    /// The textual content that will be embedded.
    pub text: String,
    /// Absolute path to the source file.
    pub path: String,
    /// Byte offset inside the source file.
    pub offset: usize,
    /// Byte length of this chunk in the source file.
    pub len: usize,
    /// Optional language/type tag (e.g., "rust", "markdown", "typescript").
    pub lang: Option<String>,
}

/// A retrieval hit: a chunk id with its similarity score to the query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    pub chunk_id: String,
    pub score: f32,
}

/// Converts text into a dense vector. Implementations:
///   * `OllamaEmbedder` — HTTP call to a local Ollama daemon running
///     a retrieval-tuned model such as `nomic-embed-text`.
///   * (future) `SsdLlmEmbedder` — pooled hidden states from an
///     ssd-llm-hosted generative model.
pub trait Embedder: Send + Sync {
    /// Embed a single string. Returns a vector of length `self.dim()`.
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;

    /// Dimensionality of the produced vectors. MUST match whatever the
    /// backing `Store` was created with.
    fn dim(&self) -> usize;

    /// Human-readable tag (e.g., "ollama:nomic-embed-text:v1.5") used to
    /// stamp index files so a mismatched embedder can't query an index
    /// built with a different model.
    fn tag(&self) -> &str;
}

/// Persistent vector store. v1 target: mmap-backed `.bin` files on SSD.
pub trait Store: Send + Sync {
    /// Append a chunk and its vector. `vector.len()` MUST equal the
    /// dimensionality the store was opened with.
    fn put(&mut self, chunk: &Chunk, vector: &[f32]) -> anyhow::Result<()>;

    /// Return the top-K most similar chunks to `vector`.
    fn query(&self, vector: &[f32], top_k: usize) -> anyhow::Result<Vec<Hit>>;

    /// Resolve a hit back to its chunk. Required for the Retriever to
    /// return text, not just IDs.
    fn get_chunk(&self, chunk_id: &str) -> anyhow::Result<Option<Chunk>>;

    /// Number of vectors currently stored.
    fn len(&self) -> usize;

    /// Whether the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// High-level retrieval: embed the query, ask the store, return chunks.
pub struct Retriever<E: Embedder, S: Store> {
    pub embedder: E,
    pub store: S,
}

impl<E: Embedder, S: Store> Retriever<E, S> {
    pub fn new(embedder: E, store: S) -> Self {
        Self { embedder, store }
    }

    pub fn retrieve(&self, query: &str, top_k: usize) -> anyhow::Result<Vec<(Hit, Chunk)>> {
        let q_vec = self.embedder.embed(query)?;
        let hits = self.store.query(&q_vec, top_k)?;
        let mut out = Vec::with_capacity(hits.len());
        for hit in hits {
            if let Some(chunk) = self.store.get_chunk(&hit.chunk_id)? {
                out.push((hit, chunk));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seam test: Retriever must compose Embedder + Store without knowing
    /// the concrete types. Exercises the full retrieve() path with fakes.
    struct FakeEmbedder;
    impl Embedder for FakeEmbedder {
        fn embed(&self, _: &str) -> anyhow::Result<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0])
        }
        fn dim(&self) -> usize {
            3
        }
        fn tag(&self) -> &str {
            "fake:v0"
        }
    }

    struct FakeStore {
        chunks: Vec<Chunk>,
    }
    impl Store for FakeStore {
        fn put(&mut self, chunk: &Chunk, _v: &[f32]) -> anyhow::Result<()> {
            self.chunks.push(chunk.clone());
            Ok(())
        }
        fn query(&self, _v: &[f32], top_k: usize) -> anyhow::Result<Vec<Hit>> {
            Ok(self
                .chunks
                .iter()
                .take(top_k)
                .map(|c| Hit {
                    chunk_id: c.id.clone(),
                    score: 0.5,
                })
                .collect())
        }
        fn get_chunk(&self, chunk_id: &str) -> anyhow::Result<Option<Chunk>> {
            Ok(self.chunks.iter().find(|c| c.id == chunk_id).cloned())
        }
        fn len(&self) -> usize {
            self.chunks.len()
        }
    }

    #[test]
    fn retriever_composes_embedder_and_store() {
        let chunk = Chunk {
            id: "c1".into(),
            text: "hello world".into(),
            path: "a.txt".into(),
            offset: 0,
            len: 11,
            lang: None,
        };
        let mut store = FakeStore { chunks: vec![] };
        store.put(&chunk, &[1.0, 0.0, 0.0]).unwrap();
        let retr = Retriever::new(FakeEmbedder, store);
        let hits = retr.retrieve("ignored", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].1.text, "hello world");
        assert_eq!(hits[0].0.score, 0.5);
    }

    #[test]
    fn empty_store_returns_empty_hits() {
        let store = FakeStore { chunks: vec![] };
        let retr = Retriever::new(FakeEmbedder, store);
        let hits = retr.retrieve("ignored", 5).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn top_k_limits_results() {
        let mut store = FakeStore { chunks: vec![] };
        for i in 0..10 {
            store
                .put(
                    &Chunk {
                        id: format!("c{}", i),
                        text: format!("chunk {}", i),
                        path: "a.txt".into(),
                        offset: i * 10,
                        len: 10,
                        lang: None,
                    },
                    &[1.0, 0.0, 0.0],
                )
                .unwrap();
        }
        let retr = Retriever::new(FakeEmbedder, store);
        let hits = retr.retrieve("ignored", 3).unwrap();
        assert_eq!(hits.len(), 3);
    }
}
