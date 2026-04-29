//! Filesystem retrieval tools for agentic code assistance.
//!
//! Each tool takes a `serde_json::Value` args object and returns a structured
//! `Value` — never a formatted string. Outputs are designed to be injected back
//! into the model context without post-processing.

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

// ── Tool trait ────────────────────────────────────────────────────────────────

pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn execute(&self, args: &Value) -> Result<Value>;
}

// ── Registry ──────────────────────────────────────────────────────────────────

/// Constant-time lookup by name; insertion-order iteration for determinism.
pub struct ToolRegistry {
    index: HashMap<&'static str, usize>,
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            tools: Vec::new(),
        }
    }

    /// Register the 4 default codebase retrieval tools.
    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        r.register(Box::new(GrepTool));
        r.register(Box::new(ReadChunkTool));
        r.register(Box::new(ListSymbolsTool));
        r.register(Box::new(FindDefinitionTool));
        r
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.name();
        let idx = self.tools.len();
        self.tools.push(tool);
        self.index.insert(name, idx);
    }

    /// Execute a named tool. Returns structured error JSON if tool not found or
    /// args are invalid, so callers can inject errors back into context cleanly.
    pub fn execute(&self, name: &str, args: &Value) -> Result<Value> {
        let idx = self
            .index
            .get(name)
            .ok_or_else(|| anyhow!("unknown tool: {}", name))?;
        self.tools[*idx].execute(args)
    }

    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ── Shared utilities ──────────────────────────────────────────────────────────

const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".next", "__pycache__", ".venv"];
const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024; // skip files > 4 MB

fn should_skip_dir(name: &str) -> bool {
    SKIP_DIRS.contains(&name) || name.starts_with('.')
}

/// Walk `root` recursively, calling `f(path, content)` for each readable text file.
/// Skips binary files, oversized files, and SKIP_DIRS. Errors on individual files
/// are silently ignored so a single unreadable file doesn't abort the search.
fn walk_text_files(root: &Path, mut f: impl FnMut(&Path, &str)) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !should_skip_dir(name) {
                    stack.push(path);
                }
            } else if meta.is_file() && meta.len() <= MAX_FILE_BYTES {
                if let Ok(text) = fs::read_to_string(&path) {
                    f(&path, &text);
                }
            }
        }
    }
}

/// Return the canonical extension of a path, lower-cased.
fn ext(path: &Path) -> &str {
    path.extension().and_then(|e| e.to_str()).unwrap_or("")
}

// ── GrepTool ──────────────────────────────────────────────────────────────────

/// Search files under `path` for lines containing `query` (case-sensitive substring).
///
/// Args:
///   query      string  (required) substring to search for
///   path       string  (optional, default ".") root directory to search
///   max_results u64    (optional, default 50) cap on total matches returned
///
/// Returns:
///   { "matches": [{"file": "…", "line": u32, "text": "…"}, …], "truncated": bool }
pub struct GrepTool;

impl Tool for GrepTool {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn execute(&self, args: &Value) -> Result<Value> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| anyhow!("grep: missing required arg 'query'"))?;
        let root = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;

        let root_path = PathBuf::from(root);
        let mut matches: Vec<Value> = Vec::new();
        let mut truncated = false;

        walk_text_files(&root_path, |path, text| {
            if truncated {
                return;
            }
            for (i, line) in text.lines().enumerate() {
                if line.contains(query) {
                    matches.push(json!({
                        "file": path.to_string_lossy(),
                        "line": i + 1,
                        "text": line.trim()
                    }));
                    if matches.len() >= max_results {
                        truncated = true;
                        return;
                    }
                }
            }
        });

        Ok(json!({ "matches": matches, "truncated": truncated }))
    }
}

// ── ReadChunkTool ─────────────────────────────────────────────────────────────

/// Read a contiguous range of lines from a file (1-indexed, inclusive).
///
/// Args:
///   path   string  (required) file path
///   start  u64     (required) first line to return (1-indexed)
///   end    u64     (required) last line to return (inclusive)
///
/// Returns:
///   { "path": "…", "start": u32, "end": u32, "text": "…", "total_lines": u32 }
pub struct ReadChunkTool;

impl Tool for ReadChunkTool {
    fn name(&self) -> &'static str {
        "read_chunk"
    }

    fn execute(&self, args: &Value) -> Result<Value> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| anyhow!("read_chunk: missing required arg 'path'"))?;
        let start = args["start"]
            .as_u64()
            .ok_or_else(|| anyhow!("read_chunk: missing required arg 'start'"))? as usize;
        let end = args["end"]
            .as_u64()
            .ok_or_else(|| anyhow!("read_chunk: missing required arg 'end'"))? as usize;

        if start == 0 {
            return Err(anyhow!("read_chunk: 'start' is 1-indexed (got 0)"));
        }
        if end < start {
            return Err(anyhow!(
                "read_chunk: 'end' ({}) must be >= 'start' ({})",
                end,
                start
            ));
        }

        let text = fs::read_to_string(path)
            .map_err(|e| anyhow!("read_chunk: cannot read '{}': {}", path, e))?;
        let lines: Vec<&str> = text.lines().collect();
        let total_lines = lines.len();

        let s = (start - 1).min(total_lines);
        let e = end.min(total_lines);
        let chunk = lines[s..e].join("\n");

        Ok(json!({
            "path": path,
            "start": start,
            "end": e,
            "text": chunk,
            "total_lines": total_lines
        }))
    }
}

// ── ListSymbolsTool ───────────────────────────────────────────────────────────

/// Extract top-level symbol definitions from a source file.
/// Uses regex-free line prefix matching; covers Rust, Python, JS/TS, Go.
///
/// Args:
///   path   string  (required) file path
///
/// Returns:
///   { "symbols": [{"name": "…", "kind": "fn|struct|enum|…", "line": u32}, …] }
pub struct ListSymbolsTool;

impl Tool for ListSymbolsTool {
    fn name(&self) -> &'static str {
        "list_symbols"
    }

    fn execute(&self, args: &Value) -> Result<Value> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| anyhow!("list_symbols: missing required arg 'path'"))?;

        let text = fs::read_to_string(path)
            .map_err(|e| anyhow!("list_symbols: cannot read '{}': {}", path, e))?;

        let extension = ext(Path::new(path));
        let symbols = match extension {
            "rs" => extract_rust_symbols(&text),
            "py" => extract_python_symbols(&text),
            "js" | "ts" | "jsx" | "tsx" => extract_js_symbols(&text),
            "go" => extract_go_symbols(&text),
            _ => extract_rust_symbols(&text), // best-effort fallback
        };

        Ok(json!({ "symbols": symbols }))
    }
}

fn extract_rust_symbols(text: &str) -> Vec<Value> {
    let mut symbols = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let trimmed = strip_visibility(line.trim());
        let (kind, rest) = if let Some(r) = trimmed.strip_prefix("fn ") {
            ("fn", r)
        } else if let Some(r) = trimmed.strip_prefix("async fn ") {
            ("fn", r)
        } else if let Some(r) = trimmed.strip_prefix("struct ") {
            ("struct", r)
        } else if let Some(r) = trimmed.strip_prefix("enum ") {
            ("enum", r)
        } else if let Some(r) = trimmed.strip_prefix("trait ") {
            ("trait", r)
        } else if let Some(r) = trimmed.strip_prefix("impl ") {
            ("impl", r)
        } else if let Some(r) = trimmed.strip_prefix("mod ") {
            ("mod", r)
        } else if let Some(r) = trimmed.strip_prefix("type ") {
            ("type", r)
        } else if let Some(r) = trimmed.strip_prefix("const ") {
            ("const", r)
        } else if let Some(r) = trimmed.strip_prefix("static ") {
            ("static", r)
        } else if let Some(r) = trimmed.strip_prefix("macro_rules! ") {
            ("macro", r)
        } else {
            continue;
        };
        let name = symbol_name(rest);
        if !name.is_empty() {
            symbols.push(json!({ "name": name, "kind": kind, "line": i + 1 }));
        }
    }
    symbols
}

/// Strip `pub`, `pub(crate)`, `pub(super)` visibility qualifiers.
fn strip_visibility(s: &str) -> &str {
    let s = s.strip_prefix("pub").unwrap_or(s);
    let s = s.trim_start();
    // strip (crate), (super), (in path::…)
    if s.starts_with('(') {
        if let Some(end) = s.find(')') {
            return s[end + 1..].trim_start();
        }
    }
    s
}

/// Take the first identifier-like token from a declaration tail.
fn symbol_name(s: &str) -> &str {
    let s = s.trim_start();
    // For `impl Trait for Type`, capture "Trait for Type" as the name
    let end = s
        .find(|c: char| c == '{' || c == '(' || c == ';' || c == '<' || c == '\n')
        .unwrap_or(s.len());
    s[..end].trim()
}

fn extract_python_symbols(text: &str) -> Vec<Value> {
    let mut symbols = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        let (kind, rest) = if let Some(r) = trimmed.strip_prefix("def ") {
            ("fn", r)
        } else if let Some(r) = trimmed.strip_prefix("async def ") {
            ("fn", r)
        } else if let Some(r) = trimmed.strip_prefix("class ") {
            ("struct", r)
        } else {
            continue;
        };
        let name = rest.split(|c| c == '(' || c == ':').next().unwrap_or("").trim();
        if !name.is_empty() {
            symbols.push(json!({ "name": name, "kind": kind, "line": i + 1 }));
        }
    }
    symbols
}

fn extract_js_symbols(text: &str) -> Vec<Value> {
    let mut symbols = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        let (kind, rest) = if let Some(r) = trimmed.strip_prefix("function ") {
            ("fn", r)
        } else if let Some(r) = trimmed.strip_prefix("async function ") {
            ("fn", r)
        } else if let Some(r) = trimmed.strip_prefix("class ") {
            ("struct", r)
        } else if let Some(r) = trimmed.strip_prefix("export function ") {
            ("fn", r)
        } else if let Some(r) = trimmed.strip_prefix("export default function ") {
            ("fn", r)
        } else if let Some(r) = trimmed.strip_prefix("export class ") {
            ("struct", r)
        } else {
            continue;
        };
        let name = rest.split(|c| c == '(' || c == '{' || c == ' ').next().unwrap_or("").trim();
        if !name.is_empty() {
            symbols.push(json!({ "name": name, "kind": kind, "line": i + 1 }));
        }
    }
    symbols
}

fn extract_go_symbols(text: &str) -> Vec<Value> {
    let mut symbols = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        let (kind, rest) = if let Some(r) = trimmed.strip_prefix("func ") {
            ("fn", r)
        } else if let Some(r) = trimmed.strip_prefix("type ") {
            ("type", r)
        } else {
            continue;
        };
        let name = rest.split(|c| c == '(' || c == '{' || c == ' ').next().unwrap_or("").trim();
        if !name.is_empty() {
            symbols.push(json!({ "name": name, "kind": kind, "line": i + 1 }));
        }
    }
    symbols
}

// ── FindDefinitionTool ────────────────────────────────────────────────────────

/// Search a codebase for where a named symbol is defined.
/// Matches `fn symbol`, `struct Symbol`, `class Symbol`, etc. across all text files.
///
/// Args:
///   symbol  string  (required) symbol name to find
///   root    string  (optional, default ".") directory to search
///
/// Returns:
///   { "matches": [{"file": "…", "line": u32, "snippet": "…"}, …] }
pub struct FindDefinitionTool;

impl Tool for FindDefinitionTool {
    fn name(&self) -> &'static str {
        "find_definition"
    }

    fn execute(&self, args: &Value) -> Result<Value> {
        let symbol = args["symbol"]
            .as_str()
            .ok_or_else(|| anyhow!("find_definition: missing required arg 'symbol'"))?;
        let root = args.get("root").and_then(|v| v.as_str()).unwrap_or(".");

        let root_path = PathBuf::from(root);
        let mut matches: Vec<Value> = Vec::new();

        // Patterns that indicate a definition line (checked by substring on trimmed line)
        let def_keywords: &[&str] = &[
            "fn ", "struct ", "enum ", "trait ", "impl ", "class ",
            "def ", "async def ", "func ", "type ", "macro_rules! ",
        ];

        walk_text_files(&root_path, |path, text| {
            for (i, line) in text.lines().enumerate() {
                let trimmed = line.trim();
                // Line must contain the symbol and look like a definition
                if !trimmed.contains(symbol) {
                    continue;
                }
                let visible = strip_visibility(trimmed);
                let is_def = def_keywords
                    .iter()
                    .any(|kw| visible.starts_with(kw));
                if !is_def {
                    continue;
                }
                // Quick false-positive filter: symbol must appear as a token, not just substring.
                // We check that the char before or after symbol is non-alphanumeric/non-underscore.
                if let Some(pos) = trimmed.find(symbol) {
                    let before_ok = pos == 0
                        || !trimmed.as_bytes()[pos - 1].is_ascii_alphanumeric()
                            && trimmed.as_bytes()[pos - 1] != b'_';
                    let after_pos = pos + symbol.len();
                    let after_ok = after_pos >= trimmed.len()
                        || !trimmed.as_bytes()[after_pos].is_ascii_alphanumeric()
                            && trimmed.as_bytes()[after_pos] != b'_';
                    if before_ok && after_ok {
                        matches.push(json!({
                            "file": path.to_string_lossy(),
                            "line": i + 1,
                            "snippet": trimmed
                        }));
                    }
                }
            }
        });

        Ok(json!({ "matches": matches }))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use tempfile::{tempdir, NamedTempFile};

    // Helper: write content to a temp file, return its path.
    fn tmp_file(content: &str, suffix: &str) -> NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix(suffix)
            .tempfile()
            .expect("tempfile");
        f.write_all(content.as_bytes()).expect("write");
        f
    }

    // ── GrepTool ──────────────────────────────────────────────────────────────

    #[test]
    fn grep_finds_matching_lines() {
        let f = tmp_file("hello world\nfoo bar\nhello again\n", ".txt");
        let args = json!({ "query": "hello", "path": f.path().parent().unwrap().to_str().unwrap() });
        let result = GrepTool.execute(&args).unwrap();
        let matches = result["matches"].as_array().unwrap();
        // Both "hello" lines must appear
        assert!(matches.iter().any(|m| m["text"].as_str().unwrap().contains("hello world")));
        assert!(matches.iter().any(|m| m["text"].as_str().unwrap().contains("hello again")));
    }

    #[test]
    fn grep_returns_empty_when_no_match() {
        let f = tmp_file("no match here\n", ".txt");
        let args = json!({ "query": "ZZZNOTFOUND", "path": f.path().parent().unwrap().to_str().unwrap() });
        let result = GrepTool.execute(&args).unwrap();
        assert!(result["matches"].as_array().unwrap().is_empty());
        assert!(!result["truncated"].as_bool().unwrap());
    }

    #[test]
    fn grep_respects_max_results() {
        // 10 matching lines, max_results=3 → truncated
        let content = "needle\n".repeat(10);
        let f = tmp_file(&content, ".txt");
        let args = json!({
            "query": "needle",
            "path": f.path().parent().unwrap().to_str().unwrap(),
            "max_results": 3
        });
        let result = GrepTool.execute(&args).unwrap();
        assert_eq!(result["matches"].as_array().unwrap().len(), 3);
        assert!(result["truncated"].as_bool().unwrap());
    }

    #[test]
    fn grep_missing_query_returns_error() {
        let result = GrepTool.execute(&json!({ "path": "." }));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("query"));
    }

    // ── ReadChunkTool ─────────────────────────────────────────────────────────

    #[test]
    fn read_chunk_returns_correct_lines() {
        let f = tmp_file("line1\nline2\nline3\nline4\nline5\n", ".txt");
        let args = json!({ "path": f.path().to_str().unwrap(), "start": 2, "end": 4 });
        let result = ReadChunkTool.execute(&args).unwrap();
        assert_eq!(result["start"].as_u64().unwrap(), 2);
        let text = result["text"].as_str().unwrap();
        assert!(text.contains("line2"));
        assert!(text.contains("line3"));
        assert!(text.contains("line4"));
        assert!(!text.contains("line1"), "line before range must not appear");
        assert!(!text.contains("line5"), "line after range must not appear");
    }

    #[test]
    fn read_chunk_clamps_end_to_file_length() {
        let f = tmp_file("a\nb\nc\n", ".txt");
        let args = json!({ "path": f.path().to_str().unwrap(), "start": 2, "end": 999 });
        let result = ReadChunkTool.execute(&args).unwrap();
        assert_eq!(result["total_lines"].as_u64().unwrap(), 3);
        assert!(result["text"].as_str().unwrap().contains("b"));
    }

    #[test]
    fn read_chunk_zero_start_returns_error() {
        let f = tmp_file("x\n", ".txt");
        let result = ReadChunkTool.execute(&json!({
            "path": f.path().to_str().unwrap(), "start": 0, "end": 1
        }));
        assert!(result.is_err());
    }

    #[test]
    fn read_chunk_end_before_start_returns_error() {
        let f = tmp_file("x\n", ".txt");
        let result = ReadChunkTool.execute(&json!({
            "path": f.path().to_str().unwrap(), "start": 5, "end": 3
        }));
        assert!(result.is_err());
    }

    #[test]
    fn read_chunk_missing_path_returns_error() {
        let result = ReadChunkTool.execute(&json!({ "start": 1, "end": 5 }));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("path"));
    }

    // ── ListSymbolsTool ───────────────────────────────────────────────────────

    #[test]
    fn list_symbols_extracts_rust_fn_and_struct() {
        let src = "pub fn foo() {}\nstruct Bar;\npub(crate) enum Baz {}\n";
        let f = tmp_file(src, ".rs");
        let args = json!({ "path": f.path().to_str().unwrap() });
        let result = ListSymbolsTool.execute(&args).unwrap();
        let syms = result["symbols"].as_array().unwrap();
        assert!(syms.iter().any(|s| s["name"] == "foo" && s["kind"] == "fn"));
        assert!(syms.iter().any(|s| s["name"] == "Bar" && s["kind"] == "struct"));
        assert!(syms.iter().any(|s| s["name"] == "Baz" && s["kind"] == "enum"));
    }

    #[test]
    fn list_symbols_extracts_python_def_and_class() {
        let src = "def my_func():\n    pass\nclass MyClass:\n    pass\n";
        let f = tmp_file(src, ".py");
        let args = json!({ "path": f.path().to_str().unwrap() });
        let result = ListSymbolsTool.execute(&args).unwrap();
        let syms = result["symbols"].as_array().unwrap();
        assert!(syms.iter().any(|s| s["name"] == "my_func" && s["kind"] == "fn"));
        assert!(syms.iter().any(|s| s["name"] == "MyClass" && s["kind"] == "struct"));
    }

    #[test]
    fn list_symbols_empty_file_returns_empty() {
        let f = tmp_file("", ".rs");
        let args = json!({ "path": f.path().to_str().unwrap() });
        let result = ListSymbolsTool.execute(&args).unwrap();
        assert!(result["symbols"].as_array().unwrap().is_empty());
    }

    #[test]
    fn list_symbols_captures_line_numbers() {
        let src = "// comment\nfn alpha() {}\n// another\nfn beta() {}\n";
        let f = tmp_file(src, ".rs");
        let args = json!({ "path": f.path().to_str().unwrap() });
        let result = ListSymbolsTool.execute(&args).unwrap();
        let syms = result["symbols"].as_array().unwrap();
        let alpha = syms.iter().find(|s| s["name"] == "alpha").unwrap();
        let beta = syms.iter().find(|s| s["name"] == "beta").unwrap();
        assert_eq!(alpha["line"].as_u64().unwrap(), 2);
        assert_eq!(beta["line"].as_u64().unwrap(), 4);
    }

    #[test]
    fn list_symbols_missing_path_returns_error() {
        let result = ListSymbolsTool.execute(&json!({}));
        assert!(result.is_err());
    }

    // ── FindDefinitionTool ────────────────────────────────────────────────────

    #[test]
    fn find_definition_locates_struct_in_file() {
        let src = "// comment about MyStruct\nstruct MyStruct {\n    x: u32,\n}\n";
        let f = tmp_file(src, ".rs");
        let dir = f.path().parent().unwrap();
        let args = json!({ "symbol": "MyStruct", "root": dir.to_str().unwrap() });
        let result = FindDefinitionTool.execute(&args).unwrap();
        let matches = result["matches"].as_array().unwrap();
        assert!(!matches.is_empty(), "should find MyStruct definition");
        assert!(matches.iter().any(|m| m["line"].as_u64().unwrap() == 2));
    }

    #[test]
    fn find_definition_does_not_match_substring() {
        // "Foo" should not match "FooBar" or "setFoo"
        let src = "struct FooBar;\nfn setFoo() {}\nstruct Foo;\n";
        let f = tmp_file(src, ".rs");
        let dir = f.path().parent().unwrap();
        let args = json!({ "symbol": "Foo", "root": dir.to_str().unwrap() });
        let result = FindDefinitionTool.execute(&args).unwrap();
        let matches = result["matches"].as_array().unwrap();
        // Only `struct Foo;` (line 3) should match
        for m in matches {
            let snippet = m["snippet"].as_str().unwrap();
            assert!(
                !snippet.contains("FooBar") && !snippet.contains("setFoo"),
                "false positive: {}",
                snippet
            );
        }
    }

    #[test]
    fn find_definition_returns_empty_for_unknown_symbol() {
        let dir = tempdir().unwrap();
        // Empty directory → no matches
        let args = json!({ "symbol": "XYZNOTEXIST", "root": dir.path().to_str().unwrap() });
        let result = FindDefinitionTool.execute(&args).unwrap();
        assert!(result["matches"].as_array().unwrap().is_empty());
    }

    #[test]
    fn find_definition_missing_symbol_returns_error() {
        let result = FindDefinitionTool.execute(&json!({ "root": "." }));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("symbol"));
    }

    // ── ToolRegistry ─────────────────────────────────────────────────────────

    #[test]
    fn registry_with_defaults_has_all_four_tools() {
        let reg = ToolRegistry::with_defaults();
        let names = reg.tool_names();
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"read_chunk"));
        assert!(names.contains(&"list_symbols"));
        assert!(names.contains(&"find_definition"));
        assert_eq!(names.len(), 4);
    }

    #[test]
    fn registry_execute_unknown_tool_returns_error() {
        let reg = ToolRegistry::with_defaults();
        let result = reg.execute("no_such_tool", &json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown tool"));
    }

    #[test]
    fn registry_routes_to_correct_tool() {
        let f = tmp_file("fn example() {}\n", ".rs");
        let reg = ToolRegistry::with_defaults();
        let result = reg.execute(
            "list_symbols",
            &json!({ "path": f.path().to_str().unwrap() }),
        ).unwrap();
        assert!(result.get("symbols").is_some());
    }
}
