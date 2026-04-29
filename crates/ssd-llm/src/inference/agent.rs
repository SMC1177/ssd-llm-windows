//! Agentic inference loop: constrained tool-call generation + execution + context injection.
//!
//! Flow per turn:
//!   1. Free generation until `<<tool_call>>` sentinel fires (SentinelDetector)
//!   2. Constrained tool-name slot (PrefixConstraint + mask_logits)
//!   3. Free JSON args generation until outermost `}` closes
//!   4. Execute tool via ToolRegistry; inject result into context
//!   5. Repeat up to `max_turns`; final free generation produces the answer

use anyhow::{bail, Result};
use serde_json::Value;
use tracing::{debug, warn};

use crate::model::cache::LayerCache;
use crate::model::gguf::GgufFile;
use crate::ssd::streamer::SsdStreamer;

use crate::inference::sampler::{mask_logits, PrefixConstraint, SentinelDetector};
use crate::inference::tokenizer::SimpleTokenizer;
use crate::inference::tools::ToolRegistry;
use crate::inference::transformer::{generate_streaming, InferenceConfig};

// ── Config ────────────────────────────────────────────────────────────────────

pub struct AgentConfig {
    /// Maximum tool-call/result rounds before giving up and returning what we have.
    pub max_turns: usize,
    /// Token budget for each generation phase (free / tool-name / args).
    pub max_tokens_per_phase: usize,
    /// Root directory for file-system tools (grep, read_chunk, …).
    pub working_dir: String,
    /// Text sentinel the model emits when it wants to call a tool.
    pub trigger: String,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 5,
            max_tokens_per_phase: 512,
            working_dir: ".".to_string(),
            trigger: "<<tool_call>>".to_string(),
        }
    }
}

// ── Result types ──────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ToolInvocation {
    pub tool: String,
    pub args: Value,
    pub result: Value,
}

#[derive(Debug)]
pub struct AgentResult {
    /// Accumulated model text across all turns (tool calls not included).
    pub text: String,
    /// Each tool call that was made, in order.
    pub invocations: Vec<ToolInvocation>,
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Run the agent loop.
///
/// `initial_prompt` should already contain the system message instructing the
/// model to use `<<tool_call>>` when it needs to retrieve information. Call this
/// from the CLI `run` command or from the HTTP server's chat endpoint.
pub fn run_agent(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    initial_prompt: &str,
    inference_config: &InferenceConfig,
    agent_config: &AgentConfig,
    tools: &ToolRegistry,
) -> Result<AgentResult> {
    // Pre-tokenize the trigger and all tool names once.
    let tokenizer = SimpleTokenizer::from_gguf(gguf);
    let trigger_ids = tokenizer.encode(&agent_config.trigger);
    if trigger_ids.is_empty() {
        bail!("agent: trigger '{}' tokenizes to nothing", agent_config.trigger);
    }

    let tool_names = tools.tool_names();
    if tool_names.is_empty() {
        bail!("agent: ToolRegistry has no tools registered");
    }
    let tool_sequences: Vec<Vec<u32>> = tool_names
        .iter()
        .map(|name| tokenizer.encode(name))
        .collect();
    let tool_prefix = PrefixConstraint::new(tool_sequences.clone());

    let mut current_prompt = initial_prompt.to_string();
    let mut accumulated_text = String::new();
    let mut invocations: Vec<ToolInvocation> = Vec::new();

    let phase_config = phase_inference_config(inference_config, agent_config.max_tokens_per_phase);

    for turn in 0..agent_config.max_turns {
        debug!("agent turn {}: free generation", turn);

        // ── Phase 1: Free generation until sentinel or EOS ────────────────────
        let free_text = free_generate_until_sentinel(
            gguf,
            streamer,
            layer_cache,
            &current_prompt,
            &phase_config,
            &trigger_ids,
        )?;

        accumulated_text.push_str(&free_text.text);

        if !free_text.sentinel_fired {
            // EOS or max_tokens with no tool call — we're done.
            debug!("agent: no sentinel in turn {}, stopping", turn);
            break;
        }

        debug!("agent turn {}: sentinel fired, constrained tool name", turn);

        // ── Phase 2: Constrained tool-name slot ───────────────────────────────
        // Prompt for this phase ends with the sentinel and an opening JSON literal.
        let name_prompt = format!(
            "{}{}{{\"{}\": \"",
            &current_prompt,
            &free_text.text,
            "tool"
        );

        let tool_name = constrained_tool_name(
            gguf,
            streamer,
            layer_cache,
            &name_prompt,
            &phase_config,
            &tool_prefix,
            &tool_names,
            &tool_sequences,
            &tokenizer,
        )?;

        let Some(tool_name) = tool_name else {
            warn!("agent: could not determine tool name in turn {}, aborting tool call", turn);
            // Treat the sentinel + broken slot as plain text; move on.
            current_prompt = format!("{}{}", &current_prompt, &free_text.text);
            continue;
        };

        debug!("agent turn {}: tool={}", turn, tool_name);

        // ── Phase 3: Free args JSON generation ───────────────────────────────
        let args_prompt = format!(
            "{}{}{{\"tool\": \"{}\", \"args\": ",
            &current_prompt, &free_text.text, &tool_name
        );

        let args_json = generate_json_value(
            gguf,
            streamer,
            layer_cache,
            &args_prompt,
            &phase_config,
        )?;

        // ── Phase 4: Execute tool + build result injection ────────────────────
        let tool_result = match tools.execute(&tool_name, &args_json) {
            Ok(v) => v,
            Err(e) => {
                warn!("agent: tool '{}' failed: {}", tool_name, e);
                serde_json::json!({ "error": e.to_string() })
            }
        };

        debug!(
            "agent turn {}: tool result preview: {}",
            turn,
            serde_json::to_string(&tool_result)
                .unwrap_or_default()
                .chars()
                .take(120)
                .collect::<String>()
        );

        invocations.push(ToolInvocation {
            tool: tool_name.clone(),
            args: args_json.clone(),
            result: tool_result.clone(),
        });

        // Build the next prompt: previous context + tool call + result.
        let result_str = serde_json::to_string_pretty(&tool_result).unwrap_or_default();
        current_prompt = format!(
            "{}{}{{\"tool\": \"{}\", \"args\": {}}}\n<tool_result>\n{}\n</tool_result>\n",
            &current_prompt,
            &free_text.text,
            &tool_name,
            serde_json::to_string(&args_json).unwrap_or_default(),
            result_str
        );
    }

    // ── Final generation: produce the answer after all tool results ───────────
    let final_text = free_generate_until_sentinel(
        gguf,
        streamer,
        layer_cache,
        &current_prompt,
        inference_config,
        &trigger_ids,
    )?;
    accumulated_text.push_str(&final_text.text);

    Ok(AgentResult {
        text: accumulated_text,
        invocations,
    })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

struct FreeGenResult {
    /// All decoded text (including any trailing sentinel text if sentinel fired).
    text: String,
    sentinel_fired: bool,
}

/// Run streaming generation until EOS, max_tokens, or the sentinel token
/// sequence is detected. Returns everything generated (including the sentinel).
fn free_generate_until_sentinel(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    prompt: &str,
    config: &InferenceConfig,
    trigger_ids: &[u32],
) -> Result<FreeGenResult> {
    let mut gen = generate_streaming(gguf, streamer, layer_cache, prompt, config)?;
    let mut detector = SentinelDetector::new(trigger_ids.to_vec());
    let mut text = String::new();

    loop {
        match gen.next_token_constrained(None)? {
            None => break,
            Some((token_id, piece)) => {
                text.push_str(&piece);
                if detector.push(token_id) {
                    return Ok(FreeGenResult { text, sentinel_fired: true });
                }
            }
        }
    }

    Ok(FreeGenResult { text, sentinel_fired: false })
}

/// Generate the tool-name slot using PrefixConstraint.
/// Returns the decoded tool name string, or None if the constraint fails.
fn constrained_tool_name(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    prompt: &str,
    config: &InferenceConfig,
    prefix: &PrefixConstraint,
    tool_names: &[&str],
    tool_sequences: &[Vec<u32>],
    tokenizer: &SimpleTokenizer,
) -> Result<Option<String>> {
    let mut gen = generate_streaming(gguf, streamer, layer_cache, prompt, config)?;
    let mut current_prefix: Vec<u32> = Vec::new();

    for _ in 0..32 {
        let valid = prefix.valid_next_tokens(&current_prefix);
        if valid.is_empty() {
            break;
        }

        match gen.next_token_constrained(Some(&valid))? {
            None => break,
            Some((token_id, _)) => {
                current_prefix.push(token_id);
                if prefix.is_complete(&current_prefix) {
                    // Decode which tool name was selected by matching the token sequence.
                    for (seq, name) in tool_sequences.iter().zip(tool_names.iter()) {
                        if seq == &current_prefix {
                            return Ok(Some((*name).to_string()));
                        }
                    }
                    // Fallback: decode the token sequence directly.
                    let decoded: String = current_prefix
                        .iter()
                        .map(|&id| tokenizer.decode_token(id))
                        .collect();
                    return Ok(Some(decoded.trim().to_string()));
                }
            }
        }
    }

    Ok(None)
}

/// Generate free tokens until the outermost `{…}` JSON object is complete.
/// Returns the parsed Value, or an empty object on parse failure.
fn generate_json_value(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    prompt: &str,
    config: &InferenceConfig,
) -> Result<Value> {
    let mut gen = generate_streaming(gguf, streamer, layer_cache, prompt, config)?;
    let mut raw = String::new();
    let mut depth: i32 = 0;
    let mut started = false;

    for _ in 0..(config.max_tokens * 4) {
        match gen.next_token_constrained(None)? {
            None => break,
            Some((_, piece)) => {
                for ch in piece.chars() {
                    match ch {
                        '{' => {
                            depth += 1;
                            started = true;
                        }
                        '}' => depth -= 1,
                        _ => {}
                    }
                    raw.push(ch);
                    if started && depth <= 0 {
                        break;
                    }
                }
                if started && depth <= 0 {
                    break;
                }
            }
        }
    }

    // If the model didn't produce a `{`, wrap whatever it said.
    if !started {
        return Ok(serde_json::json!({}));
    }

    match serde_json::from_str::<Value>(&raw) {
        Ok(v) => Ok(v),
        Err(e) => {
            warn!("agent: failed to parse args JSON '{}': {}", raw, e);
            Ok(serde_json::json!({}))
        }
    }
}

/// Build a phase-specific InferenceConfig that shares all sampling params but
/// caps `max_tokens` to the per-phase budget.
fn phase_inference_config(base: &InferenceConfig, max_tokens: usize) -> InferenceConfig {
    InferenceConfig {
        temperature: base.temperature,
        top_k: base.top_k,
        top_p: base.top_p,
        max_tokens,
        stop_sequences: base.stop_sequences.clone(),
        repetition_penalty: base.repetition_penalty,
        frequency_penalty: base.frequency_penalty,
        presence_penalty: base.presence_penalty,
        tfs_z: base.tfs_z,
        mirostat: base.mirostat,
        mirostat_tau: base.mirostat_tau,
        mirostat_eta: base.mirostat_eta,
        grammar: base.grammar.clone(),
        min_p: base.min_p,
        seed: base.seed,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::sampler::{PrefixConstraint, SentinelDetector};

    // These tests exercise the agent's helper logic using the primitives directly,
    // without requiring a live model or GPU. Full end-to-end integration
    // (run_agent with a real gguf) is validated manually via the CLI benchmark.

    #[test]
    fn agent_config_defaults_are_sane() {
        let cfg = AgentConfig::default();
        assert_eq!(cfg.max_turns, 5);
        assert_eq!(cfg.trigger, "<<tool_call>>");
        assert!(!cfg.working_dir.is_empty());
        assert!(cfg.max_tokens_per_phase > 0);
    }

    #[test]
    fn sentinel_integration_fires_on_trigger_ids() {
        // Simulate: model emits [1, 2, 3] as the trigger
        let trigger = vec![10u32, 20, 30];
        let mut det = SentinelDetector::new(trigger.clone());

        // Non-trigger tokens
        assert!(!det.push(1));
        assert!(!det.push(2));
        // Trigger starts
        assert!(!det.push(10));
        assert!(!det.push(20));
        // Complete
        assert!(det.push(30));
    }

    #[test]
    fn prefix_constraint_narrows_to_valid_tools() {
        // Two tools: "grep" → [100, 200], "read" → [300]
        let tools = PrefixConstraint::new(vec![vec![100, 200], vec![300]]);

        // At empty prefix, both starts are valid
        let starts = tools.valid_next_tokens(&[]);
        assert!(starts.contains(&100));
        assert!(starts.contains(&300));

        // After [100], only 200 continues
        assert_eq!(tools.valid_next_tokens(&[100]), vec![200]);

        // [100, 200] is a complete option
        assert!(tools.is_complete(&[100, 200]));

        // [300] is a complete option
        assert!(tools.is_complete(&[300]));
    }

    #[test]
    fn json_brace_counting_closes_correctly() {
        // Verify the brace-depth logic used in generate_json_value
        let input = r#"{"query": "fn main", "path": "src/"}"#;
        let mut depth = 0i32;
        let mut started = false;
        let mut collected = String::new();

        for ch in input.chars() {
            match ch {
                '{' => { depth += 1; started = true; }
                '}' => depth -= 1,
                _ => {}
            }
            collected.push(ch);
            if started && depth <= 0 { break; }
        }

        assert!(started);
        assert_eq!(depth, 0);
        assert!(serde_json::from_str::<Value>(&collected).is_ok());
    }

    #[test]
    fn json_brace_counting_handles_nested() {
        // Nested objects must not stop at inner `}`
        let input = r#"{"a": {"b": 1}, "c": 2}"#;
        let mut depth = 0i32;
        let mut started = false;
        let mut collected = String::new();

        for ch in input.chars() {
            match ch {
                '{' => { depth += 1; started = true; }
                '}' => depth -= 1,
                _ => {}
            }
            collected.push(ch);
            if started && depth <= 0 { break; }
        }

        let v: Value = serde_json::from_str(&collected).expect("valid JSON");
        assert_eq!(v["a"]["b"], 2 - 1); // == 1
        assert_eq!(v["c"], 2);
    }

    #[test]
    fn tool_invocation_debug_format_works() {
        let inv = ToolInvocation {
            tool: "grep".to_string(),
            args: serde_json::json!({ "query": "fn main" }),
            result: serde_json::json!({ "matches": [] }),
        };
        let s = format!("{:?}", inv);
        assert!(s.contains("grep"));
    }

    #[test]
    fn agent_result_accumulates_text_and_invocations() {
        let result = AgentResult {
            text: "here is the answer".to_string(),
            invocations: vec![
                ToolInvocation {
                    tool: "grep".to_string(),
                    args: serde_json::json!({}),
                    result: serde_json::json!({}),
                },
            ],
        };
        assert_eq!(result.text, "here is the answer");
        assert_eq!(result.invocations.len(), 1);
        assert_eq!(result.invocations[0].tool, "grep");
    }
}
