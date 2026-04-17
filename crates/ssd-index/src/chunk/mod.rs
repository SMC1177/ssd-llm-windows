//! Chunkers turn a file's text into a sequence of `Chunk` records that
//! the embedder + store can ingest.
//!
//! v1 has a single `TextChunker` that uses a line-window + overlap strategy
//! and treats all file types the same. Code-aware variants (tree-sitter)
//! are deferred to a separate module; they'll plug in through this trait.

use crate::Chunk;
use std::path::Path;

pub mod text;
pub use text::TextChunker;

/// Turn a file's contents into zero or more `Chunk` records.
///
/// Implementations must be deterministic: given the same `path` and
/// `content` bytes, `chunk()` must produce byte-identical `Chunk`s
/// (same IDs, same offsets, same text). This is load-bearing for
/// incremental reindexing — an unchanged file must produce the same
/// chunk IDs so the store's upsert path is a no-op.
pub trait Chunker: Send + Sync {
    fn chunk(&self, path: &Path, content: &str) -> Vec<Chunk>;
}
