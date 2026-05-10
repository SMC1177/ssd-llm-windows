//! Disk-backed `Store` — vectors in `vectors.bin`, chunks in `chunks.jsonl`,
//! metadata in `manifest.json`. Single-directory layout so one index is one
//! folder on the SSD.
//!
//! v1 keeps the live working set in RAM (`Vec<f32>` of vectors, `Vec<Chunk>`
//! of chunks, a `HashMap` for id lookup) and flushes whole files to disk on
//! demand or on drop. The trait is the stable contract; when indexes get
//! large enough to need actual mmap-backed row access, the internals swap
//! without changing any caller code.

use crate::{Chunk, Embedder, Hit, Store};
use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const MANIFEST_VERSION: u32 = 1;
const MANIFEST_FILE: &str = "manifest.json";
const VECTORS_FILE: &str = "vectors.bin";
const CHUNKS_FILE: &str = "chunks.jsonl";

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    dim: usize,
    tag: String,
    count: usize,
}

/// Disk-backed vector store. One instance == one directory on the SSD.
#[derive(Debug)]
pub struct FileStore {
    dir: PathBuf,
    dim: usize,
    tag: String,
    /// Contiguous row-major vectors: chunk i lives at vectors[i*dim .. (i+1)*dim].
    /// Always L2-normalized so cosine similarity reduces to dot product.
    vectors: Vec<f32>,
    chunks: Vec<Chunk>,
    chunk_index: HashMap<String, usize>,
    dirty: bool,
}

impl FileStore {
    /// Create a new empty store at `dir` (directory is created if missing).
    /// `tag` should be the exact `Embedder::tag()` value of the embedder
    /// whose vectors will be written here. Queries using any other tag
    /// will be rejected at `open` time.
    pub fn create(dir: impl Into<PathBuf>, dim: usize, tag: impl Into<String>) -> anyhow::Result<Self> {
        let dir = dir.into();
        let tag = tag.into();
        if dim == 0 {
            return Err(anyhow!("dim must be > 0"));
        }
        fs::create_dir_all(&dir).with_context(|| format!("create_dir_all {}", dir.display()))?;
        let store = Self {
            dir,
            dim,
            tag,
            vectors: Vec::new(),
            chunks: Vec::new(),
            chunk_index: HashMap::new(),
            dirty: true,
        };
        store.write_manifest(0)?;
        Ok(store)
    }

    /// Open an existing store at `dir`. Verifies `expected_dim` and
    /// `expected_tag` against the on-disk manifest — mismatched values
    /// return an error rather than silently producing garbage hits.
    pub fn open(
        dir: impl Into<PathBuf>,
        expected_dim: usize,
        expected_tag: &str,
    ) -> anyhow::Result<Self> {
        let dir = dir.into();
        let manifest_path = dir.join(MANIFEST_FILE);
        let manifest_text = fs::read_to_string(&manifest_path)
            .with_context(|| format!("read manifest {}", manifest_path.display()))?;
        let manifest: Manifest =
            serde_json::from_str(&manifest_text).context("parse manifest.json")?;

        if manifest.version != MANIFEST_VERSION {
            return Err(anyhow!(
                "unsupported manifest version {} (expected {})",
                manifest.version,
                MANIFEST_VERSION
            ));
        }
        if manifest.dim != expected_dim {
            return Err(anyhow!(
                "dim mismatch: on-disk {} vs expected {}",
                manifest.dim,
                expected_dim
            ));
        }
        if manifest.tag != expected_tag {
            return Err(anyhow!(
                "tag mismatch: on-disk {:?} vs expected {:?}",
                manifest.tag,
                expected_tag
            ));
        }

        let vectors = read_vectors_bin(&dir.join(VECTORS_FILE), manifest.count, manifest.dim)?;
        let chunks = read_chunks_jsonl(&dir.join(CHUNKS_FILE))?;
        if chunks.len() != manifest.count {
            return Err(anyhow!(
                "chunks.jsonl has {} lines but manifest says count={}",
                chunks.len(),
                manifest.count
            ));
        }
        let mut chunk_index = HashMap::with_capacity(chunks.len());
        for (i, c) in chunks.iter().enumerate() {
            if chunk_index.insert(c.id.clone(), i).is_some() {
                return Err(anyhow!("duplicate chunk id {} in chunks.jsonl", c.id));
            }
        }

        Ok(Self {
            dir,
            dim: manifest.dim,
            tag: manifest.tag,
            vectors,
            chunks,
            chunk_index,
            dirty: false,
        })
    }

    /// Open an existing store at `dir`, or create a new empty one if the
    /// manifest is missing. The expected embedder identity must always
    /// match when the manifest does exist.
    pub fn open_or_create(
        dir: impl Into<PathBuf>,
        expected_dim: usize,
        expected_tag: &str,
    ) -> anyhow::Result<Self> {
        let dir = dir.into();
        if dir.join(MANIFEST_FILE).exists() {
            Self::open(dir, expected_dim, expected_tag)
        } else {
            Self::create(dir, expected_dim, expected_tag.to_string())
        }
    }

    /// Convenience: create/open the store that matches a given embedder.
    pub fn for_embedder<E: Embedder>(dir: impl Into<PathBuf>, e: &E) -> anyhow::Result<Self> {
        Self::open_or_create(dir, e.dim(), e.tag())
    }

    /// Remove all chunks whose `path` starts with `prefix`. Returns the
    /// number of chunks removed. Path comparison normalizes Windows
    /// backslashes to forward slashes on both sides so callers can pass
    /// either separator regardless of how the chunk was originally
    /// indexed.
    ///
    /// Safety: an empty prefix is refused (returns 0 without mutation) so
    /// a caller cannot accidentally wipe the entire store with a blank
    /// argument.
    pub fn remove_by_path_prefix(&mut self, prefix: &str) -> usize {
        if prefix.is_empty() {
            return 0;
        }
        let needle = prefix.replace('\\', "/");

        // Single pass: keep entries whose normalized path does NOT start
        // with the normalized prefix. Rebuild vectors / chunks / index in
        // lockstep so row i in vectors.bin still corresponds to line i in
        // chunks.jsonl after the rewrite.
        let dim = self.dim;
        let n = self.chunks.len();
        let mut new_vectors: Vec<f32> = Vec::with_capacity(self.vectors.len());
        let mut new_chunks: Vec<Chunk> = Vec::with_capacity(n);
        let mut new_index: HashMap<String, usize> = HashMap::with_capacity(n);
        let mut removed = 0usize;

        for i in 0..n {
            let normalized = self.chunks[i].path.replace('\\', "/");
            if normalized.starts_with(&needle) {
                removed += 1;
                continue;
            }
            let new_idx = new_chunks.len();
            new_vectors.extend_from_slice(&self.vectors[i * dim..(i + 1) * dim]);
            new_index.insert(self.chunks[i].id.clone(), new_idx);
            new_chunks.push(self.chunks[i].clone());
        }

        if removed > 0 {
            self.vectors = new_vectors;
            self.chunks = new_chunks;
            self.chunk_index = new_index;
            self.dirty = true;
        }
        removed
    }

    /// Mark in-RAM mutations as not-to-be-persisted. Clears the dirty
    /// flag so the `Drop` impl will NOT auto-flush. Used by tools that
    /// perform a destructive op in RAM (e.g. dry-run prune) and want to
    /// discard the result without writing.
    pub fn discard_pending_changes(&mut self) {
        self.dirty = false;
    }

    /// Explicitly write vectors + chunks + manifest to disk. Clears dirty.
    pub fn flush(&mut self) -> anyhow::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        write_vectors_bin(&self.dir.join(VECTORS_FILE), &self.vectors)?;
        write_chunks_jsonl(&self.dir.join(CHUNKS_FILE), &self.chunks)?;
        self.write_manifest(self.chunks.len())?;
        self.dirty = false;
        Ok(())
    }

    fn write_manifest(&self, count: usize) -> anyhow::Result<()> {
        let manifest = Manifest {
            version: MANIFEST_VERSION,
            dim: self.dim,
            tag: self.tag.clone(),
            count,
        };
        let path = self.dir.join(MANIFEST_FILE);
        let s = serde_json::to_string_pretty(&manifest).context("serialize manifest")?;
        fs::write(&path, s).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }
}

impl Drop for FileStore {
    fn drop(&mut self) {
        // Best-effort flush on drop. Errors are logged but must not panic
        // through drop (would poison program shutdown). Callers who need
        // hard persistence guarantees should call .flush() explicitly and
        // handle the Result.
        if self.dirty {
            if let Err(e) = self.flush() {
                tracing::warn!(
                    "FileStore drop: flush failed for {}: {:#}",
                    self.dir.display(),
                    e
                );
            }
        }
    }
}

impl Store for FileStore {
    /// Upsert semantics. If a chunk with the same ID already exists, the
    /// stored vector and chunk metadata are overwritten in place (no grow).
    /// If new, the chunk is appended. This makes reindexing idempotent —
    /// unchanged files produce same-id chunks, so re-putting them is a
    /// zero-net-effect operation on the store's size.
    fn put(&mut self, chunk: &Chunk, vector: &[f32]) -> anyhow::Result<()> {
        if vector.len() != self.dim {
            return Err(anyhow!(
                "put: vector dim {} != store dim {}",
                vector.len(),
                self.dim
            ));
        }
        let mut v = vector.to_vec();
        l2_normalize(&mut v);

        if let Some(&idx) = self.chunk_index.get(&chunk.id) {
            // Upsert: overwrite vector slice and chunk record in place.
            let start = idx * self.dim;
            let end = start + self.dim;
            self.vectors[start..end].copy_from_slice(&v);
            self.chunks[idx] = chunk.clone();
        } else {
            let idx = self.chunks.len();
            self.vectors.extend_from_slice(&v);
            self.chunks.push(chunk.clone());
            self.chunk_index.insert(chunk.id.clone(), idx);
        }
        self.dirty = true;
        Ok(())
    }

    fn query(&self, query: &[f32], top_k: usize) -> anyhow::Result<Vec<Hit>> {
        if query.len() != self.dim {
            return Err(anyhow!(
                "query: vector dim {} != store dim {}",
                query.len(),
                self.dim
            ));
        }
        if top_k == 0 || self.chunks.is_empty() {
            return Ok(Vec::new());
        }
        let mut q = query.to_vec();
        l2_normalize(&mut q);

        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(self.chunks.len());
        for i in 0..self.chunks.len() {
            let start = i * self.dim;
            let end = start + self.dim;
            let row = &self.vectors[start..end];
            let dot: f32 = q.iter().zip(row).map(|(a, b)| a * b).sum();
            scored.push((i, dot));
        }
        // Highest score first; NaN sinks to end rather than crashing the sort.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        Ok(scored
            .into_iter()
            .map(|(i, score)| Hit {
                chunk_id: self.chunks[i].id.clone(),
                score,
            })
            .collect())
    }

    fn get_chunk(&self, chunk_id: &str) -> anyhow::Result<Option<Chunk>> {
        Ok(self
            .chunk_index
            .get(chunk_id)
            .map(|&i| self.chunks[i].clone()))
    }

    fn len(&self) -> usize {
        self.chunks.len()
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > 0.0 && norm.is_finite() {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    // If norm is zero or non-finite we leave the vector as-is. Querying
    // with a zero vector returns 0-score hits which is semantically honest
    // (no similarity signal).
}

fn write_vectors_bin(path: &Path, vectors: &[f32]) -> anyhow::Result<()> {
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = BufWriter::new(f);
    let mut buf = [0u8; 4];
    for &v in vectors {
        buf.copy_from_slice(&v.to_le_bytes());
        w.write_all(&buf)
            .with_context(|| format!("write {}", path.display()))?;
    }
    w.flush().with_context(|| format!("flush {}", path.display()))?;
    Ok(())
}

fn read_vectors_bin(path: &Path, count: usize, dim: usize) -> anyhow::Result<Vec<f32>> {
    if count == 0 {
        if path.exists() {
            // Allow empty or missing vectors.bin when count is zero.
        }
        return Ok(Vec::new());
    }
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let expected = count
        .checked_mul(dim)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| anyhow!("count*dim*4 overflows"))?;
    if bytes.len() != expected {
        return Err(anyhow!(
            "vectors.bin size {} != count*dim*4 {} (count={}, dim={})",
            bytes.len(),
            expected,
            count,
            dim
        ));
    }
    let mut out = Vec::with_capacity(count * dim);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().expect("chunks_exact yields 4-byte slices");
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

fn write_chunks_jsonl(path: &Path, chunks: &[Chunk]) -> anyhow::Result<()> {
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = BufWriter::new(f);
    for chunk in chunks {
        let s = serde_json::to_string(chunk)
            .with_context(|| format!("serialize chunk {}", chunk.id))?;
        w.write_all(s.as_bytes())
            .with_context(|| format!("write chunk to {}", path.display()))?;
        w.write_all(b"\n")
            .with_context(|| format!("write newline to {}", path.display()))?;
    }
    w.flush().with_context(|| format!("flush {}", path.display()))?;
    Ok(())
}

fn read_chunks_jsonl(path: &Path) -> anyhow::Result<Vec<Chunk>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for (lineno, line) in r.lines().enumerate() {
        let line = line.with_context(|| format!("read line {} of {}", lineno + 1, path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let chunk: Chunk = serde_json::from_str(&line)
            .with_context(|| format!("parse line {} of {}", lineno + 1, path.display()))?;
        out.push(chunk);
    }
    Ok(out)
}

/// Ensure write_to_disk path doesn't silently drop file handles. Used by
/// `OpenOptions::append` callers in future incremental-write modes.
#[allow(dead_code)]
fn open_append(path: &Path) -> anyhow::Result<File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open append {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{embed::OllamaEmbedder, Retriever};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Process-local temp directory helper. Each test gets its own
    /// subdirectory so they can run in parallel without stepping on each
    /// other's files.
    fn tmpdir(label: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let base = std::env::temp_dir()
            .join("ssd-index-tests")
            .join(format!("{}-{}-{}", label, pid, n));
        let _ = fs::remove_dir_all(&base);
        base
    }

    fn chunk(id: &str, text: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            text: text.to_string(),
            path: "fixture.txt".to_string(),
            offset: 0,
            len: text.len(),
            lang: None,
        }
    }

    #[test]
    fn create_then_reopen_round_trips_state() {
        let dir = tmpdir("round_trip");
        let tag = "test:unit";
        {
            let mut s = FileStore::create(&dir, 3, tag).unwrap();
            s.put(&chunk("a", "alpha"), &[1.0, 0.0, 0.0]).unwrap();
            s.put(&chunk("b", "beta"), &[0.0, 1.0, 0.0]).unwrap();
            s.flush().unwrap();
        } // drop closes the store
        let reopened = FileStore::open(&dir, 3, tag).unwrap();
        assert_eq!(reopened.len(), 2);
        let got_a = reopened.get_chunk("a").unwrap().expect("a present");
        assert_eq!(got_a.text, "alpha");
        let got_b = reopened.get_chunk("b").unwrap().expect("b present");
        assert_eq!(got_b.text, "beta");
    }

    #[test]
    fn tag_mismatch_on_open_errors() {
        let dir = tmpdir("tag_mismatch");
        {
            let mut s = FileStore::create(&dir, 2, "model-A").unwrap();
            s.put(&chunk("x", "x"), &[1.0, 0.0]).unwrap();
            s.flush().unwrap();
        }
        let err = FileStore::open(&dir, 2, "model-B").unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("tag mismatch"),
            "expected tag-mismatch error, got: {}",
            msg
        );
    }

    #[test]
    fn dim_mismatch_on_open_errors() {
        let dir = tmpdir("dim_mismatch_open");
        {
            let mut s = FileStore::create(&dir, 4, "model-A").unwrap();
            s.put(&chunk("x", "x"), &[1.0, 0.0, 0.0, 0.0]).unwrap();
            s.flush().unwrap();
        }
        let err = FileStore::open(&dir, 8, "model-A").unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("dim mismatch"),
            "expected dim-mismatch error, got: {}",
            msg
        );
    }

    #[test]
    fn dim_mismatch_on_put_errors() {
        let dir = tmpdir("dim_mismatch_put");
        let mut s = FileStore::create(&dir, 3, "t").unwrap();
        let err = s.put(&chunk("x", "x"), &[1.0, 0.0]).unwrap_err();
        assert!(format!("{:#}", err).contains("vector dim"));
    }

    #[test]
    fn query_returns_top_k_sorted_desc() {
        let dir = tmpdir("topk_sorted");
        let mut s = FileStore::create(&dir, 3, "t").unwrap();
        s.put(&chunk("near", "near"), &[1.0, 0.0, 0.0]).unwrap();
        s.put(&chunk("mid", "mid"), &[0.5, 0.5, 0.0]).unwrap();
        s.put(&chunk("far", "far"), &[0.0, 1.0, 0.0]).unwrap();

        let hits = s.query(&[1.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].chunk_id, "near", "highest score first");
        assert_eq!(hits[1].chunk_id, "mid");
        assert_eq!(hits[2].chunk_id, "far");
        assert!(hits[0].score >= hits[1].score);
        assert!(hits[1].score >= hits[2].score);
    }

    #[test]
    fn cosine_orthogonality_and_identity() {
        let dir = tmpdir("cos");
        let mut s = FileStore::create(&dir, 3, "t").unwrap();
        s.put(&chunk("x", "x"), &[1.0, 0.0, 0.0]).unwrap();
        s.put(&chunk("y", "y"), &[0.0, 1.0, 0.0]).unwrap();

        let hits = s.query(&[1.0, 0.0, 0.0], 2).unwrap();
        let score_of = |id: &str| hits.iter().find(|h| h.chunk_id == id).unwrap().score;
        assert!(
            (score_of("x") - 1.0).abs() < 1e-5,
            "self-similarity should be ~1.0, got {}",
            score_of("x")
        );
        assert!(
            score_of("y").abs() < 1e-5,
            "orthogonal vectors should score ~0.0, got {}",
            score_of("y")
        );
    }

    #[test]
    fn put_same_id_twice_upserts_not_errors() {
        let dir = tmpdir("upsert_basic");
        let mut s = FileStore::create(&dir, 2, "t").unwrap();
        s.put(&chunk("same", "first"), &[1.0, 0.0]).unwrap();
        s.put(&chunk("same", "second"), &[0.0, 1.0])
            .expect("upsert must not error on id collision");
        assert_eq!(s.len(), 1, "upsert must not grow the store");
        assert_eq!(
            s.get_chunk("same").unwrap().unwrap().text,
            "second",
            "upsert replaces chunk metadata with the new value"
        );
    }

    #[test]
    fn upsert_actually_replaces_stored_vector() {
        let dir = tmpdir("upsert_vec");
        let mut s = FileStore::create(&dir, 2, "t").unwrap();
        s.put(&chunk("x", "x"), &[1.0, 0.0]).unwrap();

        // Before: x points along [1,0], so query [1,0] gives ~1.0 self-similarity.
        let before_score = s
            .query(&[1.0, 0.0], 1)
            .unwrap()
            .into_iter()
            .find(|h| h.chunk_id == "x")
            .expect("x must be present before upsert")
            .score;
        assert!(
            (before_score - 1.0).abs() < 1e-5,
            "before upsert, x should self-score ~1.0; got {}",
            before_score
        );

        // Upsert x with an orthogonal vector. Same id, different direction.
        s.put(&chunk("x", "x"), &[0.0, 1.0]).unwrap();

        let after_score = s
            .query(&[1.0, 0.0], 1)
            .unwrap()
            .into_iter()
            .find(|h| h.chunk_id == "x")
            .expect("x must still be present after upsert")
            .score;
        assert!(
            after_score.abs() < 1e-5,
            "after upsert to [0,1], query [1,0] should score ~0.0; got {}",
            after_score
        );
        assert_eq!(s.len(), 1, "upsert must not change store size");
    }

    #[test]
    fn upsert_survives_reopen() {
        let dir = tmpdir("upsert_reopen");
        {
            let mut s = FileStore::create(&dir, 2, "t").unwrap();
            s.put(&chunk("x", "original"), &[1.0, 0.0]).unwrap();
            s.put(&chunk("x", "updated"), &[0.0, 1.0]).unwrap();
            s.flush().unwrap();
        }
        let reopened = FileStore::open(&dir, 2, "t").unwrap();
        assert_eq!(reopened.len(), 1);
        assert_eq!(
            reopened.get_chunk("x").unwrap().unwrap().text,
            "updated",
            "upsert must be persisted across open/close cycles"
        );
    }

    #[test]
    fn query_on_empty_store_returns_empty() {
        let dir = tmpdir("empty_q");
        let s = FileStore::create(&dir, 3, "t").unwrap();
        let hits = s.query(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn top_k_zero_returns_empty_even_with_data() {
        let dir = tmpdir("topk_zero");
        let mut s = FileStore::create(&dir, 2, "t").unwrap();
        s.put(&chunk("a", "a"), &[1.0, 0.0]).unwrap();
        let hits = s.query(&[1.0, 0.0], 0).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn top_k_greater_than_n_returns_all() {
        let dir = tmpdir("topk_big");
        let mut s = FileStore::create(&dir, 2, "t").unwrap();
        s.put(&chunk("a", "a"), &[1.0, 0.0]).unwrap();
        s.put(&chunk("b", "b"), &[0.0, 1.0]).unwrap();
        let hits = s.query(&[1.0, 0.0], 100).unwrap();
        assert_eq!(hits.len(), 2);
    }

    /// Seam test: FileStore plugs into Retriever alongside OllamaEmbedder.
    /// Compile-time bound proof — no Ollama daemon needed.
    #[test]
    fn file_store_plugs_into_retriever_with_ollama() {
        let dir = tmpdir("retriever_seam");
        let e = OllamaEmbedder::nomic_embed_text().unwrap();
        let s = FileStore::create(&dir, e.dim(), e.tag()).unwrap();
        let _r: Retriever<OllamaEmbedder, FileStore> = Retriever::new(e, s);
        // If the generic instantiation compiles, the seam holds.
    }

    #[test]
    fn drop_flushes_dirty_state() {
        let dir = tmpdir("drop_flush");
        {
            let mut s = FileStore::create(&dir, 2, "t").unwrap();
            s.put(&chunk("a", "alpha"), &[1.0, 0.0]).unwrap();
            // NOTE: deliberately no explicit flush — rely on Drop
        }
        let reopened = FileStore::open(&dir, 2, "t").unwrap();
        assert_eq!(reopened.len(), 1);
        assert_eq!(
            reopened.get_chunk("a").unwrap().unwrap().text,
            "alpha"
        );
    }

    fn chunk_with_path(id: &str, path: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            text: format!("text-{}", id),
            path: path.to_string(),
            offset: 0,
            len: 4,
            lang: None,
        }
    }

    #[test]
    fn remove_by_path_prefix_happy_path() {
        let dir = tmpdir("remove_happy");
        {
            let mut s = FileStore::create(&dir, 2, "t").unwrap();
            s.put(&chunk_with_path("a", "src/a.rs"), &[1.0, 0.0]).unwrap();
            s.put(&chunk_with_path("b", "src/b.rs"), &[0.0, 1.0]).unwrap();
            s.put(&chunk_with_path("c", "src/c.rs"), &[1.0, 1.0]).unwrap();

            let removed = s.remove_by_path_prefix("src/b.rs");
            assert_eq!(removed, 1, "exactly one chunk should be removed");
            assert_eq!(s.len(), 2);
            assert!(s.get_chunk("a").unwrap().is_some(), "a still present");
            assert!(s.get_chunk("c").unwrap().is_some(), "c still present");
            assert!(s.get_chunk("b").unwrap().is_none(), "b is gone in RAM");
            s.flush().unwrap();
        }
        let reopened = FileStore::open(&dir, 2, "t").unwrap();
        assert_eq!(reopened.len(), 2, "deletion persisted to disk");
        assert!(reopened.get_chunk("b").unwrap().is_none(), "b gone on disk");
        assert!(reopened.get_chunk("a").unwrap().is_some());
        assert!(reopened.get_chunk("c").unwrap().is_some());
    }

    #[test]
    fn remove_by_path_prefix_empty_prefix_is_safety_noop() {
        let dir = tmpdir("remove_empty");
        let mut s = FileStore::create(&dir, 2, "t").unwrap();
        s.put(&chunk_with_path("a", "src/a.rs"), &[1.0, 0.0]).unwrap();
        s.put(&chunk_with_path("b", "src/b.rs"), &[0.0, 1.0]).unwrap();

        let removed = s.remove_by_path_prefix("");
        assert_eq!(removed, 0, "empty prefix must NOT delete everything");
        assert_eq!(s.len(), 2, "store untouched");
    }

    #[test]
    fn remove_by_path_prefix_no_match_is_noop() {
        let dir = tmpdir("remove_nomatch");
        let mut s = FileStore::create(&dir, 2, "t").unwrap();
        s.put(&chunk_with_path("a", "src/a.rs"), &[1.0, 0.0]).unwrap();
        s.put(&chunk_with_path("b", "src/b.rs"), &[0.0, 1.0]).unwrap();
        s.flush().unwrap(); // start the prune from a clean (non-dirty) baseline

        let removed = s.remove_by_path_prefix("nonexistent/");
        assert_eq!(removed, 0);
        assert_eq!(s.len(), 2);
        assert!(!s.dirty, "no-match prune must not flip dirty flag");
    }

    #[test]
    fn remove_by_path_prefix_normalizes_separators() {
        let dir = tmpdir("remove_normalize");
        let mut s = FileStore::create(&dir, 2, "t").unwrap();
        // Store has Windows-style backslash path…
        s.put(&chunk_with_path("x", "src\\foo.ts"), &[1.0, 0.0]).unwrap();
        s.put(&chunk_with_path("y", "src/bar.ts"), &[0.0, 1.0]).unwrap();

        // …caller passes forward-slash prefix.
        let removed = s.remove_by_path_prefix("src/foo.ts");
        assert_eq!(removed, 1, "backslash path should match forward-slash prefix");
        assert!(s.get_chunk("x").unwrap().is_none());
        assert!(s.get_chunk("y").unwrap().is_some());
    }

    #[test]
    fn remove_by_path_prefix_keeps_vectors_and_chunks_aligned() {
        let dir = tmpdir("remove_align");
        let dim = 4;
        let mut s = FileStore::create(&dir, dim, "t").unwrap();
        for i in 0..6 {
            let id = format!("id{}", i);
            let path = if i % 2 == 0 {
                format!("keep/{}.rs", i)
            } else {
                format!("drop/{}.rs", i)
            };
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            s.put(&chunk_with_path(&id, &path), &v).unwrap();
        }
        let removed = s.remove_by_path_prefix("drop/");
        assert_eq!(removed, 3);
        s.flush().unwrap();

        // Re-open and verify on-disk row count matches chunk count.
        let reopened = FileStore::open(&dir, dim, "t").unwrap();
        assert_eq!(reopened.len(), 3, "3 chunks remain");
        // vectors len must equal len * dim (no row drift).
        assert_eq!(
            reopened.vectors.len(),
            reopened.len() * dim,
            "vectors.bin row count must match chunks.jsonl line count"
        );
        // And every remaining chunk is from the keep/ tree.
        for c in &reopened.chunks {
            assert!(c.path.starts_with("keep/"), "stray chunk left: {}", c.path);
        }
    }

    #[test]
    fn corrupt_vectors_bin_size_rejected() {
        let dir = tmpdir("corrupt_size");
        {
            let mut s = FileStore::create(&dir, 3, "t").unwrap();
            s.put(&chunk("a", "a"), &[1.0, 0.0, 0.0]).unwrap();
            s.flush().unwrap();
        }
        // Truncate vectors.bin by 1 byte to force size mismatch.
        let vec_path = dir.join(VECTORS_FILE);
        let bytes = fs::read(&vec_path).unwrap();
        fs::write(&vec_path, &bytes[..bytes.len() - 1]).unwrap();

        let err = FileStore::open(&dir, 3, "t").unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("vectors.bin size"),
            "expected size-mismatch error, got: {}",
            msg
        );
    }
}
