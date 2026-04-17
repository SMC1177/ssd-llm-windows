//! Concrete `Embedder` implementations.
//!
//! Each backend is behind its own module so callers can compile out ones
//! they don't need. The trait itself lives in the crate root (`crate::Embedder`).

pub mod ollama;

pub use ollama::OllamaEmbedder;
