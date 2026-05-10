//! ssd-index-prune — remove chunks from a FileStore by source-path prefix.
//!
//! Usage:
//!   ssd-index-prune --store-dir <DIR> --paths-file <FILE> [--dry-run]
//!                   [--i-know-this-is-live]
//!
//! Reads `<FILE>` (newline-delimited list of source-path prefixes), opens
//! the store at `<DIR>`, and removes every chunk whose path starts with
//! any of the listed prefixes. Prints a summary and flushes (unless
//! --dry-run).
//!
//! Safety guard: if `<DIR>` resolves to the live Lifeline store path,
//! `--i-know-this-is-live` is mandatory. Without it the tool refuses with
//! exit code 2.
//!
//! Args are hand-parsed to avoid pulling clap into the crate's dep graph.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ssd_index::store::FileStore;

const LIVE_STORE_HINT: &str =
    r"C:\Users\Stephen\.openclaw\workspace\workspaces\main\.ssd-index-lifeline";

fn print_usage() {
    eprintln!(
        "ssd-index-prune\n\
         \n\
         Required:\n\
           --store-dir <PATH>     Directory of the FileStore to mutate\n\
           --paths-file <PATH>    Newline-delimited file of path prefixes to remove\n\
         \n\
         Optional:\n\
           --dry-run              Report what would be removed; do not write\n\
           --i-know-this-is-live  Required when --store-dir is the live Lifeline store\n"
    );
}

#[derive(Debug)]
struct Args {
    store_dir: PathBuf,
    paths_file: PathBuf,
    dry_run: bool,
    i_know_live: bool,
}

fn parse_args(argv: Vec<String>) -> Result<Args, String> {
    let mut store_dir: Option<PathBuf> = None;
    let mut paths_file: Option<PathBuf> = None;
    let mut dry_run = false;
    let mut i_know_live = false;

    let mut iter = argv.into_iter().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--store-dir" => {
                let v = iter.next().ok_or("--store-dir requires a value")?;
                store_dir = Some(PathBuf::from(v));
            }
            "--paths-file" => {
                let v = iter.next().ok_or("--paths-file requires a value")?;
                paths_file = Some(PathBuf::from(v));
            }
            "--dry-run" => dry_run = true,
            "--i-know-this-is-live" => i_know_live = true,
            "-h" | "--help" => return Err("help".to_string()),
            other => return Err(format!("unknown argument: {}", other)),
        }
    }
    let store_dir = store_dir.ok_or("missing --store-dir")?;
    let paths_file = paths_file.ok_or("missing --paths-file")?;
    Ok(Args { store_dir, paths_file, dry_run, i_know_live })
}

/// Best-effort canonicalize for the safety check: fall back to the raw
/// path if canonicalize fails (e.g., dir doesn't exist yet).
fn canonical_string(p: &Path) -> String {
    fs::canonicalize(p)
        .map(|c| c.to_string_lossy().to_string())
        .unwrap_or_else(|_| p.to_string_lossy().to_string())
        // Normalize Windows extended-length prefix \\?\ that canonicalize adds.
        .trim_start_matches(r"\\?\")
        .to_string()
}

fn paths_match(a: &str, b: &str) -> bool {
    let na = a.replace('/', "\\").to_ascii_lowercase();
    let nb = b.replace('/', "\\").to_ascii_lowercase();
    na.trim_end_matches('\\') == nb.trim_end_matches('\\')
}

fn run() -> Result<(), (String, u8)> {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(argv) {
        Ok(a) => a,
        Err(e) => {
            if e == "help" {
                print_usage();
                return Ok(());
            }
            print_usage();
            return Err((e, 1));
        }
    };

    // Live-store guard.
    let resolved = canonical_string(&args.store_dir);
    if paths_match(&resolved, LIVE_STORE_HINT) && !args.i_know_live {
        return Err((
            format!(
                "refusing to mutate live Lifeline store {:?}; pass --i-know-this-is-live to override",
                resolved
            ),
            2,
        ));
    }

    let prefixes_text = fs::read_to_string(&args.paths_file)
        .map_err(|e| (format!("read --paths-file {:?}: {}", args.paths_file, e), 1))?;
    let prefixes: Vec<&str> = prefixes_text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    if prefixes.is_empty() {
        eprintln!("paths-file contained no non-empty entries; nothing to do");
        return Ok(());
    }

    if args.dry_run {
        // Open read-only-ish: we still construct a FileStore but never flush.
        // We need dim+tag to open; cheap workaround — read manifest first.
        let manifest_path = args.store_dir.join("manifest.json");
        let manifest_text = fs::read_to_string(&manifest_path)
            .map_err(|e| (format!("read manifest {:?}: {}", manifest_path, e), 1))?;
        let manifest: serde_json::Value = serde_json::from_str(&manifest_text)
            .map_err(|e| (format!("parse manifest: {}", e), 1))?;
        let dim = manifest["dim"].as_u64().ok_or(("manifest.dim missing".to_string(), 1))? as usize;
        let tag = manifest["tag"].as_str().ok_or(("manifest.tag missing".to_string(), 1))?.to_string();

        let mut store = FileStore::open(&args.store_dir, dim, &tag)
            .map_err(|e| (format!("open store: {:#}", e), 1))?;

        let mut total_removed = 0usize;
        let mut zero_match = 0usize;
        for p in &prefixes {
            // Use the in-RAM mutation but never call flush. dirty flag is
            // discarded when the process exits without the tool calling flush.
            // Drop-time auto-flush WOULD persist — explicitly mark clean.
            let n = store.remove_by_path_prefix(p);
            if n == 0 {
                zero_match += 1;
            }
            total_removed += n;
            println!("[dry-run] would remove {} chunks for prefix {:?}", n, p);
        }
        // Critical: prevent Drop from auto-persisting our in-RAM deletions.
        store.discard_pending_changes();
        println!(
            "[dry-run] Pruned {} chunks across {} paths ({} paths matched zero); NOT WRITTEN",
            total_removed,
            prefixes.len(),
            zero_match
        );
    } else {
        let manifest_path = args.store_dir.join("manifest.json");
        let manifest_text = fs::read_to_string(&manifest_path)
            .map_err(|e| (format!("read manifest {:?}: {}", manifest_path, e), 1))?;
        let manifest: serde_json::Value = serde_json::from_str(&manifest_text)
            .map_err(|e| (format!("parse manifest: {}", e), 1))?;
        let dim = manifest["dim"].as_u64().ok_or(("manifest.dim missing".to_string(), 1))? as usize;
        let tag = manifest["tag"].as_str().ok_or(("manifest.tag missing".to_string(), 1))?.to_string();

        let mut store = FileStore::open(&args.store_dir, dim, &tag)
            .map_err(|e| (format!("open store: {:#}", e), 1))?;

        let mut total_removed = 0usize;
        let mut zero_match = 0usize;
        for p in &prefixes {
            let n = store.remove_by_path_prefix(p);
            if n == 0 {
                zero_match += 1;
            }
            total_removed += n;
            println!("removed {} chunks for prefix {:?}", n, p);
        }
        store
            .flush()
            .map_err(|e| (format!("flush store: {:#}", e), 1))?;
        println!(
            "Pruned {} chunks across {} paths ({} paths matched zero)",
            total_removed,
            prefixes.len(),
            zero_match
        );
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err((msg, code)) => {
            eprintln!("ssd-index-prune: {}", msg);
            ExitCode::from(code)
        }
    }
}
