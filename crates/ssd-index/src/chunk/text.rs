//! Line-window chunker with overlap. File-type agnostic.
//!
//! Slides a window of `window` lines across the file, advancing by
//! `window - stride` each step. Every chunk carries enough context
//! through the `stride` lines that overlap with its neighbors.
//!
//! Chunk IDs are content-addressed:
//!     blake3(path)[..16] : offset : blake3(content)[..8]
//!
//! * path component locates the file
//! * offset locates the chunk within the file
//! * content hash makes reindexing idempotent AND change-sensitive:
//!   unchanged content → same ID → store upsert is a no-op;
//!   changed content at the same position → different ID, old chunk
//!   is orphaned (cleaned up by a future compact() pass) and new
//!   chunk is inserted.

use crate::chunk::Chunker;
use crate::Chunk;
use anyhow::{anyhow, Result};
use std::path::Path;

/// Line-window chunker. See module docs for ID scheme.
#[derive(Debug, Clone)]
pub struct TextChunker {
    window: usize,
    stride: usize,
}

impl TextChunker {
    /// Construct with `window` lines per chunk and `stride` lines of
    /// overlap between adjacent chunks. Returns Err on degenerate
    /// configurations (zero window, stride ≥ window) so the live
    /// `chunk()` path can stay infallible.
    pub fn new(window: usize, stride: usize) -> Result<Self> {
        if window == 0 {
            return Err(anyhow!("window must be > 0"));
        }
        if stride >= window {
            return Err(anyhow!(
                "stride ({}) must be < window ({})",
                stride,
                window
            ));
        }
        Ok(Self { window, stride })
    }

    /// Default tuned for code: 40 lines per chunk, 10 lines overlap.
    pub fn default_code() -> Self {
        Self::new(40, 10).expect("valid defaults")
    }

    /// Default tuned for prose: 20 lines per chunk, 5 lines overlap.
    pub fn default_prose() -> Self {
        Self::new(20, 5).expect("valid defaults")
    }

    pub fn window(&self) -> usize {
        self.window
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
}

impl Chunker for TextChunker {
    fn chunk(&self, path: &Path, content: &str) -> Vec<Chunk> {
        if content.is_empty() {
            return Vec::new();
        }

        // Build per-line (byte_offset, byte_len_incl_newline) table once,
        // then slice into `content` for each chunk without reallocating.
        let mut line_info: Vec<(usize, usize)> = Vec::new();
        let mut pos: usize = 0;
        for line in content.split_inclusive('\n') {
            line_info.push((pos, line.len()));
            pos += line.len();
        }
        let n_lines = line_info.len();
        let lang = infer_lang(path);
        let step = self.window - self.stride; // validated > 0 in new()

        let mut chunks = Vec::new();
        let mut start = 0usize;
        loop {
            let end = (start + self.window).min(n_lines);
            let byte_start = line_info[start].0;
            let last = &line_info[end - 1];
            let byte_end = last.0 + last.1;
            let text = &content[byte_start..byte_end];
            let id = make_id(path, byte_start, text);
            chunks.push(Chunk {
                id,
                text: text.to_string(),
                path: path.to_string_lossy().into_owned(),
                offset: byte_start,
                len: byte_end - byte_start,
                lang: lang.clone(),
            });
            if end == n_lines {
                break;
            }
            start += step;
        }
        chunks
    }
}

fn make_id(path: &Path, offset: usize, content: &str) -> String {
    let path_hash = blake3::hash(path.to_string_lossy().as_bytes()).to_hex();
    let content_hash = blake3::hash(content.as_bytes()).to_hex();
    // path[..16] = 64 bits of path identity (collision-free at any real scale)
    // content[..8] = 32 bits of change-detection (plenty for local collisions
    // within one path+offset; global uniqueness comes from path+offset prefix)
    format!("{}:{}:{}", &path_hash[..16], offset, &content_hash[..8])
}

/// Best-effort language tag from file extension. Returns None for types
/// we don't explicitly recognize — the retriever can still embed and
/// query them, just without a lang filter.
fn infer_lang(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    Some(
        match ext.as_str() {
            "rs" => "rust",
            "py" => "python",
            "js" | "mjs" | "cjs" => "javascript",
            "ts" | "tsx" => "typescript",
            "jsx" => "javascript",
            "go" => "go",
            "java" => "java",
            "c" | "h" => "c",
            "cpp" | "cc" | "cxx" | "hpp" | "hh" => "cpp",
            "rb" => "ruby",
            "php" => "php",
            "swift" => "swift",
            "kt" | "kts" => "kotlin",
            "scala" => "scala",
            "md" | "markdown" => "markdown",
            "txt" => "text",
            "json" => "json",
            "toml" => "toml",
            "yaml" | "yml" => "yaml",
            "sh" | "bash" => "bash",
            "zsh" => "zsh",
            "html" | "htm" => "html",
            "css" => "css",
            "scss" | "sass" => "scss",
            "sql" => "sql",
            "lua" => "lua",
            "ex" | "exs" => "elixir",
            "hs" => "haskell",
            "zig" => "zig",
            _ => return None,
        }
        .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> &'static Path {
        // Leak a small &'static str for readable test paths. Totally fine
        // in tests — these paths never escape the test runner's lifetime.
        Path::new(Box::leak(s.to_string().into_boxed_str()))
    }

    #[test]
    fn empty_file_yields_no_chunks() {
        let c = TextChunker::new(10, 2).unwrap();
        assert!(c.chunk(p("x.txt"), "").is_empty());
    }

    #[test]
    fn single_line_no_newline_yields_one_chunk() {
        let c = TextChunker::new(10, 2).unwrap();
        let chunks = c.chunk(p("x.txt"), "hello");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello");
        assert_eq!(chunks[0].offset, 0);
        assert_eq!(chunks[0].len, 5);
    }

    #[test]
    fn content_shorter_than_window_yields_one_chunk() {
        let c = TextChunker::new(10, 2).unwrap();
        let text = "a\nb\nc\n";
        let chunks = c.chunk(p("x.txt"), text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
    }

    #[test]
    fn exact_window_yields_one_chunk() {
        let c = TextChunker::new(3, 1).unwrap();
        let chunks = c.chunk(p("x.txt"), "a\nb\nc\n");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn multiple_windows_with_overlap_share_lines() {
        let c = TextChunker::new(3, 1).unwrap(); // step = 2
        let text = "a\nb\nc\nd\ne\n"; // 5 lines
        let chunks = c.chunk(p("x.txt"), text);
        assert_eq!(chunks.len(), 2, "expected windows [0..3], [2..5]");
        assert_eq!(chunks[0].text, "a\nb\nc\n");
        assert_eq!(chunks[1].text, "c\nd\ne\n");
        assert_eq!(chunks[0].offset, 0);
        assert_eq!(chunks[1].offset, 4, "line 2 starts at byte 4");
        // Overlap: last line of chunk 0 ("c\n") must equal first line of chunk 1
        assert_eq!(&chunks[0].text[4..], &chunks[1].text[..2]);
    }

    #[test]
    fn stride_ge_window_errors_at_construct() {
        assert!(TextChunker::new(5, 5).is_err());
        assert!(TextChunker::new(5, 10).is_err());
        assert!(TextChunker::new(0, 0).is_err());
    }

    #[test]
    fn chunk_ids_are_deterministic_across_calls() {
        let c = TextChunker::new(5, 1).unwrap();
        let text = "alpha\nbeta\ngamma\n";
        let a = c.chunk(p("same.rs"), text);
        let b = c.chunk(p("same.rs"), text);
        assert_eq!(a[0].id, b[0].id);
    }

    #[test]
    fn chunk_ids_change_when_content_changes() {
        let c = TextChunker::new(5, 1).unwrap();
        let a = c.chunk(p("f.rs"), "a\nb\nc\n");
        let b = c.chunk(p("f.rs"), "a\nB\nc\n"); // middle byte flipped
        assert_ne!(
            a[0].id, b[0].id,
            "content hash component must flip when content changes"
        );
    }

    #[test]
    fn chunk_ids_change_when_path_changes() {
        let c = TextChunker::new(5, 1).unwrap();
        let a = c.chunk(p("alpha.rs"), "a\nb\nc\n");
        let b = c.chunk(p("beta.rs"), "a\nb\nc\n");
        assert_ne!(
            a[0].id, b[0].id,
            "path hash component must flip when the same content moves files"
        );
    }

    #[test]
    fn chunk_id_shape_is_parseable() {
        let c = TextChunker::new(5, 1).unwrap();
        let chunks = c.chunk(p("x.rs"), "hi\n");
        let id = &chunks[0].id;
        let parts: Vec<&str> = id.split(':').collect();
        assert_eq!(parts.len(), 3, "id must be <path_hash>:<offset>:<content_hash>");
        assert_eq!(parts[0].len(), 16, "path hash prefix length");
        assert!(parts[1].parse::<usize>().is_ok(), "offset must parse as usize");
        assert_eq!(parts[2].len(), 8, "content hash suffix length");
    }

    #[test]
    fn lang_inferred_from_extension() {
        let c = TextChunker::new(10, 2).unwrap();
        assert_eq!(
            c.chunk(p("x.rs"), "hello")[0].lang.as_deref(),
            Some("rust")
        );
        assert_eq!(
            c.chunk(p("x.md"), "hello")[0].lang.as_deref(),
            Some("markdown")
        );
        assert_eq!(
            c.chunk(p("x.TS"), "hello")[0].lang.as_deref(),
            Some("typescript"),
            "extension matching must be case-insensitive"
        );
        assert_eq!(c.chunk(p("x.weird"), "hello")[0].lang, None);
        assert_eq!(c.chunk(p("no_extension"), "hello")[0].lang, None);
    }

    #[test]
    fn defaults_construct_and_run() {
        let _ = TextChunker::default_code().chunk(p("x.rs"), "a\nb\n");
        let _ = TextChunker::default_prose().chunk(p("x.md"), "a\nb\n");
    }

    /// Seam test: TextChunker must be usable through the `Chunker` trait
    /// object. Guards against an accidental removal of the `impl Chunker
    /// for TextChunker` block.
    #[test]
    fn text_chunker_usable_as_trait_object() {
        let c: Box<dyn Chunker> = Box::new(TextChunker::new(3, 1).unwrap());
        let chunks = c.chunk(p("x.txt"), "a\nb\nc\nd\n");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn offsets_are_byte_offsets_into_original_content() {
        let c = TextChunker::new(2, 0).unwrap();
        let text = "äß\nb\nc\n"; // non-ASCII; ä=2 bytes, ß=2 bytes, \n=1 byte → line 0 = 5 bytes
        let chunks = c.chunk(p("x.txt"), text);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].offset, 0);
        // Chunk 1 starts at line 2 (zero-based). Line 0 = 5 bytes, line 1 = "b\n" = 2 bytes → offset 7.
        assert_eq!(chunks[1].offset, 7, "byte offset must account for multi-byte chars");
        // Reconstructing from the offset must give back the chunk's text.
        assert_eq!(
            &text[chunks[1].offset..chunks[1].offset + chunks[1].len],
            chunks[1].text
        );
    }
}
