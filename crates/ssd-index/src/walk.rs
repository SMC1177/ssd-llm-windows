//! Directory walker + end-to-end indexing glue.
//!
//! `Walker` traverses a root directory while respecting `.gitignore` /
//! `.ignore` / hidden-file defaults (via the `ignore` crate, same machinery
//! ripgrep uses). `index_directory()` wires Walker + Chunker + Embedder +
//! Store together so a caller can turn "this folder on disk" into "queryable
//! vectors in the store" with a single function call.

use crate::{Chunker, Embedder, Store};
use anyhow::Context;
use std::path::{Path, PathBuf};

/// Default cap for files we're willing to embed. Skips binary blobs,
/// lock files, and generated artifacts that would waste embedding time
/// without useful retrieval signal.
pub const DEFAULT_MAX_FILE_BYTES: u64 = 10 * 1024 * 1024;

/// Traversal policy. Knows *which* files to visit but nothing about what
/// to do with them.
#[derive(Debug, Clone)]
pub struct Walker {
    /// If non-empty, only files whose extension matches (case-insensitively)
    /// one of these are yielded. Empty = no extension filter, rely on
    /// `ignore`'s defaults.
    pub include_ext: Vec<String>,
    /// Files larger than this are skipped and counted in the report's
    /// `files_skipped`.
    pub max_file_bytes: u64,
    /// Whether to follow symlinks. Default false — protects against
    /// recursive loops and accidental traversal into system dirs.
    pub follow_symlinks: bool,
    /// Whether hidden files/dirs (dotfiles) are visible.
    /// Default false. Applies even when no `.gitignore` exists.
    pub show_hidden: bool,
}

impl Walker {
    /// Default for traversing a code repository. Respects .gitignore,
    /// skips hidden dirs, caps at 10 MB, no symlink traversal.
    pub fn default_code() -> Self {
        Self {
            include_ext: Vec::new(),
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            follow_symlinks: false,
            show_hidden: false,
        }
    }

    /// Only emit files with these extensions. Case-insensitive.
    pub fn with_extensions<I, S>(mut self, exts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.include_ext = exts.into_iter().map(|s| s.into().to_lowercase()).collect();
        self
    }

    fn extension_ok(&self, path: &Path) -> bool {
        if self.include_ext.is_empty() {
            return true;
        }
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => self.include_ext.contains(&ext.to_lowercase()),
            None => false,
        }
    }
}

/// Per-file outcome, aggregated into `IndexReport`. Kept out of the public
/// surface — callers see the summary counters only.
#[derive(Debug)]
enum Outcome {
    Indexed(usize),               // chunks put
    SkippedTooBig,
    SkippedWrongExt,
    SkippedUnreadable,
    Error(PathBuf, String),
}

/// Summary of an `index_directory` run. Every visited file accounts for
/// exactly one of the counters (or one entry in `errors`). Callers use
/// this to report progress and to detect silent failures.
#[derive(Debug, Default, Clone)]
pub struct IndexReport {
    pub files_visited: usize,
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub chunks_put: usize,
    pub errors: Vec<(PathBuf, String)>,
}

impl IndexReport {
    fn apply(&mut self, outcome: Outcome, path: &Path) {
        self.files_visited += 1;
        match outcome {
            Outcome::Indexed(n) => {
                self.files_indexed += 1;
                self.chunks_put += n;
            }
            Outcome::SkippedTooBig | Outcome::SkippedWrongExt | Outcome::SkippedUnreadable => {
                self.files_skipped += 1;
                // SkippedUnreadable is a soft error: we recorded it, but it's
                // also useful to see the path explicitly for the user.
                if let Outcome::SkippedUnreadable = outcome {
                    self.errors
                        .push((path.to_path_buf(), "unreadable".to_string()));
                }
            }
            Outcome::Error(p, msg) => {
                self.files_skipped += 1;
                self.errors.push((p, msg));
            }
        }
    }
}

/// Walk `root` and push every surviving file through chunker → embedder →
/// store. One error per file is captured in the report rather than
/// aborting the whole run — large corpora will always contain a few
/// unreadable files, and aborting halfway is worse than skipping them.
pub fn index_directory<C, E, S>(
    root: &Path,
    walker: &Walker,
    chunker: &C,
    embedder: &E,
    store: &mut S,
) -> anyhow::Result<IndexReport>
where
    C: Chunker,
    E: Embedder,
    S: Store,
{
    let mut report = IndexReport::default();

    let mut builder = ignore::WalkBuilder::new(root);
    builder
        .follow_links(walker.follow_symlinks)
        .hidden(!walker.show_hidden) // ignore crate: hidden(true) = HIDE them
        // Respect .gitignore, .ignore, and global gitignore — same as ripgrep.
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true);

    for entry in builder.build() {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                report
                    .errors
                    .push((PathBuf::from(root), format!("walk error: {}", e)));
                continue;
            }
        };
        // Only process regular files. Directories come through too; skip them.
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();

        if !walker.extension_ok(path) {
            report.apply(Outcome::SkippedWrongExt, path);
            continue;
        }

        let metadata = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(e) => {
                report.apply(Outcome::Error(path.to_path_buf(), e.to_string()), path);
                continue;
            }
        };
        if metadata.len() > walker.max_file_bytes {
            report.apply(Outcome::SkippedTooBig, path);
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => {
                // Not UTF-8 or otherwise unreadable as text. Skip rather
                // than error — binary files are legitimately present in
                // most directories.
                report.apply(Outcome::SkippedUnreadable, path);
                continue;
            }
        };

        let chunks = chunker.chunk(path, &content);
        let mut put_count = 0usize;
        for ch in &chunks {
            let vec = match embedder
                .embed(&ch.text)
                .with_context(|| format!("embed chunk {}", ch.id))
            {
                Ok(v) => v,
                Err(e) => {
                    report
                        .apply(Outcome::Error(path.to_path_buf(), format!("{:#}", e)), path);
                    // Stop processing this file on embed failure; keep going
                    // on the next file.
                    put_count = 0;
                    break;
                }
            };
            if let Err(e) = store.put(ch, &vec) {
                report.apply(
                    Outcome::Error(path.to_path_buf(), format!("put: {:#}", e)),
                    path,
                );
                put_count = 0;
                break;
            }
            put_count += 1;
        }

        if put_count > 0 || chunks.is_empty() {
            // Empty-chunk file still counts as "indexed" — the chunker
            // examined it and produced no chunks (e.g., empty file).
            report.apply(Outcome::Indexed(put_count), path);
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chunk, Hit};
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn tmpdir(label: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let base = std::env::temp_dir()
            .join("ssd-index-walk-tests")
            .join(format!("{}-{}-{}", label, pid, n));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        base
    }

    /// Keyword-counting embedder: for a fixed vocabulary, produces a
    /// vector whose i-th component is 1 if word_i is present in the text.
    /// Deterministic, no network, retrieval quality is testable by
    /// construction.
    struct KeywordEmbedder {
        vocab: Vec<String>,
    }
    impl KeywordEmbedder {
        fn new(vocab: &[&str]) -> Self {
            Self {
                vocab: vocab.iter().map(|s| s.to_lowercase()).collect(),
            }
        }
    }
    impl Embedder for KeywordEmbedder {
        fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            let lower = text.to_lowercase();
            Ok(self
                .vocab
                .iter()
                .map(|w| if lower.contains(w.as_str()) { 1.0 } else { 0.0 })
                .collect())
        }
        fn dim(&self) -> usize {
            self.vocab.len()
        }
        fn tag(&self) -> &str {
            "keyword:test"
        }
    }

    /// Minimal Store impl used for walker-only tests so we don't pull in
    /// FileStore's disk IO for pure walker behavior checks.
    #[derive(Default)]
    struct MemStore {
        chunks: Vec<Chunk>,
        vecs: Vec<Vec<f32>>,
    }
    impl Store for MemStore {
        fn put(&mut self, chunk: &Chunk, vector: &[f32]) -> anyhow::Result<()> {
            self.chunks.push(chunk.clone());
            self.vecs.push(vector.to_vec());
            Ok(())
        }
        fn query(&self, _v: &[f32], _k: usize) -> anyhow::Result<Vec<Hit>> {
            Ok(Vec::new())
        }
        fn get_chunk(&self, id: &str) -> anyhow::Result<Option<Chunk>> {
            Ok(self.chunks.iter().find(|c| c.id == id).cloned())
        }
        fn len(&self) -> usize {
            self.chunks.len()
        }
    }

    #[test]
    fn walker_filters_by_extension() {
        let dir = tmpdir("filter_ext");
        fs::write(dir.join("keep.rs"), "fn main() {}").unwrap();
        fs::write(dir.join("keep.md"), "# title").unwrap();
        fs::write(dir.join("skip.log"), "noise").unwrap();

        let walker = Walker::default_code().with_extensions(["rs", "md"]);
        let chunker = crate::chunk::TextChunker::new(10, 2).unwrap();
        let embedder = KeywordEmbedder::new(&["main", "title"]);
        let mut store = MemStore::default();
        let report =
            index_directory(&dir, &walker, &chunker, &embedder, &mut store).unwrap();

        assert_eq!(report.files_indexed, 2, "rs + md indexed, log skipped");
        assert!(report.files_skipped >= 1);
    }

    #[test]
    fn walker_skips_files_over_max_bytes() {
        let dir = tmpdir("too_big");
        fs::write(dir.join("ok.txt"), "short").unwrap();
        // 100 byte file capped at 10 bytes → skipped.
        fs::write(dir.join("huge.txt"), "a".repeat(100)).unwrap();

        let mut walker = Walker::default_code().with_extensions(["txt"]);
        walker.max_file_bytes = 10;
        let chunker = crate::chunk::TextChunker::new(50, 10).unwrap();
        let embedder = KeywordEmbedder::new(&["short"]);
        let mut store = MemStore::default();

        let report =
            index_directory(&dir, &walker, &chunker, &embedder, &mut store).unwrap();
        assert_eq!(
            report.files_indexed, 1,
            "only ok.txt fits under max_file_bytes"
        );
        assert_eq!(report.files_skipped, 1);
    }

    #[test]
    fn walker_respects_ignore_file() {
        // The `ignore` crate honors `.ignore` files in any directory
        // (git-independent). `.gitignore` requires an actual git repo —
        // tested separately below.
        let dir = tmpdir("ignore_file");
        fs::write(dir.join(".ignore"), "ignored.txt\n").unwrap();
        fs::write(dir.join("kept.txt"), "hello").unwrap();
        fs::write(dir.join("ignored.txt"), "shh").unwrap();

        let walker = Walker::default_code().with_extensions(["txt"]);
        let chunker = crate::chunk::TextChunker::new(10, 2).unwrap();
        let embedder = KeywordEmbedder::new(&["hello"]);
        let mut store = MemStore::default();
        let report =
            index_directory(&dir, &walker, &chunker, &embedder, &mut store).unwrap();
        assert_eq!(
            report.files_indexed, 1,
            ".ignore file must exclude ignored.txt"
        );
        assert!(store.chunks[0].path.contains("kept.txt"));
    }

    #[test]
    fn walker_respects_gitignore_inside_git_repo() {
        // `.gitignore` is only honored when the walker sees this directory
        // as a git repo. The `ignore` crate detects that via the presence
        // of a `.git` directory. We stub the minimal thing it recognizes.
        let dir = tmpdir("gitignore_in_repo");
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::write(dir.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
        fs::write(dir.join(".gitignore"), "secret.txt\n").unwrap();
        fs::write(dir.join("public.txt"), "visible").unwrap();
        fs::write(dir.join("secret.txt"), "hidden").unwrap();

        let walker = Walker::default_code().with_extensions(["txt"]);
        let chunker = crate::chunk::TextChunker::new(10, 2).unwrap();
        let embedder = KeywordEmbedder::new(&["visible"]);
        let mut store = MemStore::default();
        let report =
            index_directory(&dir, &walker, &chunker, &embedder, &mut store).unwrap();
        assert_eq!(
            report.files_indexed, 1,
            ".gitignore must exclude secret.txt inside a git repo"
        );
        assert!(store.chunks[0].path.contains("public.txt"));
    }

    #[test]
    fn walker_skips_hidden_by_default() {
        let dir = tmpdir("hidden");
        fs::write(dir.join("visible.txt"), "seen").unwrap();
        fs::write(dir.join(".hidden.txt"), "not seen").unwrap();

        let walker = Walker::default_code().with_extensions(["txt"]);
        let chunker = crate::chunk::TextChunker::new(10, 2).unwrap();
        let embedder = KeywordEmbedder::new(&["seen"]);
        let mut store = MemStore::default();
        let report =
            index_directory(&dir, &walker, &chunker, &embedder, &mut store).unwrap();
        assert_eq!(report.files_indexed, 1);
    }

    #[test]
    fn report_errors_on_embed_failure_but_continues() {
        let dir = tmpdir("embed_err");
        fs::write(dir.join("a.txt"), "alpha").unwrap();
        fs::write(dir.join("b.txt"), "beta").unwrap();

        /// Embedder that fails the first call and succeeds the second.
        /// Proves that a bad file doesn't abort the whole pipeline.
        struct FlakyEmbedder {
            calls: std::sync::atomic::AtomicUsize,
        }
        impl Embedder for FlakyEmbedder {
            fn embed(&self, _: &str) -> anyhow::Result<Vec<f32>> {
                let n = self.calls.fetch_add(1, Ordering::Relaxed);
                if n == 0 {
                    Err(anyhow::anyhow!("simulated embed failure"))
                } else {
                    Ok(vec![1.0; 3])
                }
            }
            fn dim(&self) -> usize {
                3
            }
            fn tag(&self) -> &str {
                "flaky:test"
            }
        }

        let walker = Walker::default_code().with_extensions(["txt"]);
        let chunker = crate::chunk::TextChunker::new(10, 2).unwrap();
        let embedder = FlakyEmbedder {
            calls: Default::default(),
        };
        let mut store = MemStore::default();
        let report =
            index_directory(&dir, &walker, &chunker, &embedder, &mut store).unwrap();
        assert_eq!(report.files_indexed, 1, "second file must succeed");
        assert_eq!(report.errors.len(), 1, "first file's error must be recorded");
    }

    /// End-to-end integration test: real FileStore + real chunker + a
    /// deterministic keyword embedder. No Ollama, no network, but every
    /// real component in the pipeline runs. This is the test that proves
    /// the v0.6 scaffolding actually retrieves the right chunk.
    #[test]
    fn end_to_end_retrieval_finds_matching_chunk() {
        let dir = tmpdir("e2e");
        let index_dir = dir.join("index");
        fs::write(dir.join("alpha.txt"), "apple banana cherry").unwrap();
        fs::write(dir.join("beta.txt"), "xylophone yak zebra").unwrap();
        fs::write(dir.join("gamma.md"), "# Docs\nSome text about dogs").unwrap();

        let vocab = &["apple", "banana", "cherry", "xylophone", "yak", "zebra", "dogs"];
        let embedder = KeywordEmbedder::new(vocab);
        let walker = Walker::default_code().with_extensions(["txt", "md"]);
        let chunker = crate::chunk::TextChunker::new(100, 10).unwrap();
        let mut store = crate::store::FileStore::create(
            &index_dir,
            embedder.dim(),
            embedder.tag(),
        )
        .unwrap();

        let report =
            index_directory(&dir, &walker, &chunker, &embedder, &mut store).unwrap();
        assert_eq!(report.files_indexed, 3);
        assert!(report.chunks_put >= 3);

        // Query: "banana" should bring back the alpha.txt chunk.
        let q = embedder.embed("banana").unwrap();
        let hits = store.query(&q, 3).unwrap();
        assert!(!hits.is_empty());
        let top = store.get_chunk(&hits[0].chunk_id).unwrap().unwrap();
        assert!(
            top.text.contains("banana"),
            "top hit for 'banana' must contain the word; got text: {:?}",
            top.text
        );

        // Different query: "dogs" must land in gamma.md, not the alpha or beta files.
        let q = embedder.embed("dogs").unwrap();
        let hits = store.query(&q, 3).unwrap();
        let top = store.get_chunk(&hits[0].chunk_id).unwrap().unwrap();
        assert!(
            top.path.ends_with("gamma.md"),
            "top hit for 'dogs' must come from gamma.md; got path: {}",
            top.path
        );
    }

    #[test]
    fn reindex_is_idempotent_under_upsert() {
        let dir = tmpdir("idem");
        let index_dir = dir.join("index");
        fs::write(dir.join("a.txt"), "hello world").unwrap();

        let embedder = KeywordEmbedder::new(&["hello", "world"]);
        let walker = Walker::default_code().with_extensions(["txt"]);
        let chunker = crate::chunk::TextChunker::new(100, 10).unwrap();
        let mut store = crate::store::FileStore::create(
            &index_dir,
            embedder.dim(),
            embedder.tag(),
        )
        .unwrap();

        let r1 = index_directory(&dir, &walker, &chunker, &embedder, &mut store).unwrap();
        let count_after_first = store.len();
        assert_eq!(r1.files_indexed, 1);
        assert!(count_after_first >= 1);

        // Run again without changing files.
        let r2 = index_directory(&dir, &walker, &chunker, &embedder, &mut store).unwrap();
        assert_eq!(r2.files_indexed, 1);
        assert_eq!(
            store.len(),
            count_after_first,
            "reindexing unchanged files must not grow the store"
        );
    }
}
