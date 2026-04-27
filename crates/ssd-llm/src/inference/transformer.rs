//! Transformer forward pass — layer-by-layer streaming inference with KV cache

use crate::inference::attention::{multi_head_attention_fused_rope, multi_head_attention_quantized};
use crate::inference::feed_forward::{feed_forward, feed_forward_gpu, feed_forward_quantized};
use crate::inference::kv_cache::KvCache;
use crate::inference::lora::LoraManager;
use crate::inference::moe::{self, ExpertWeights, MoeConfig};
use crate::inference::sampler::{MirostatMode, Sampler};
use crate::inference::tokenizer::SimpleTokenizer;
use crate::metal::compute::MetalCompute;
use crate::metal::gpu::MetalGpu;
use crate::model::cache::{CachedLayer, LayerCache};
use crate::model::gguf::GgufFile;
use crate::ssd::prefetch::{PrefetchStrategy, Prefetcher};
use crate::ssd::streamer::SsdStreamer;
use anyhow::{bail, Result};
use std::time::Instant;
use tracing::{debug, info, warn};

/// Process one token through a single layer using zero-copy quantized matvec.
/// No f32 weight allocations — all projections operate directly on mmap'd raw bytes.
/// Peak memory: ~n_ff * 8 bytes (two intermediate FFN vectors) instead of ~1.9 GB per layer.
fn streaming_layer_forward(
    hidden_state: &mut Vec<f32>,
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_idx: u32,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    position: usize,
    kv_cache: &mut crate::inference::kv_cache::LayerKvCache,
    theta_base: f32,
    rms_eps: f32,
) -> Result<()> {
    let n_embd = hidden_state.len();
    let layer_name = |suffix: &str| format!("blk.{}.{}", layer_idx, suffix);

    // 1. Pre-attention RMSNorm (F32, tiny — ~21 KB)
    let mut attn_input = hidden_state.clone();
    if let Ok(norm_w) = streamer.load_named_tensor_f32(gguf, &layer_name("attn_norm.weight")) {
        if layer_idx == 0 {
            info!("NORM WIRING: attn_norm[0..4]=[{:.6},{:.6},{:.6},{:.6}] len={}", norm_w[0], norm_w[1], norm_w[2], norm_w[3], norm_w.len());
        }
        crate::metal::compute::rmsnorm_f32_fast(&mut attn_input, &norm_w, rms_eps);
    }
    // GROUND-TRUTH DUMP: attn_norm output for layer 0, first token
    if layer_idx == 0 && std::env::var("SSD_DUMP_ATTN_NORM").is_ok() {
        let path = std::env::var("SSD_DUMP_ATTN_NORM").unwrap_or_else(|_| "ssd_llm_attn_norm.bin".to_string());
        let bytes: Vec<u8> = attn_input.iter().flat_map(|f| f.to_le_bytes()).collect();
        match std::fs::write(&path, &bytes) {
            Ok(()) => info!("SSD_DUMP_ATTN_NORM: wrote {} floats to {}", attn_input.len(), path),
            Err(e) => warn!("SSD_DUMP_ATTN_NORM: {}", e),
        }
    }
    // Diagnostic: check attn_input after norm (should be bounded ~[-5,5])
    if layer_idx <= 2 {
        let ni_max = attn_input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let ni_min = attn_input.iter().copied().fold(f32::INFINITY, f32::min);
        debug!("Layer {} attn_input after norm: max={:.4} min={:.4}", layer_idx, ni_max, ni_min);
    }

    // 2. Attention: QKV + output projections on raw quantized bytes
    //    Derive actual dimensions from tensor shapes (Gemma 4 has non-standard Q/K/V dims)
    let wq_tensor = gguf.find_tensor(&layer_name("attn_q.weight"));
    let wk_tensor = gguf.find_tensor(&layer_name("attn_k.weight"));
    let wv_tensor = gguf.find_tensor(&layer_name("attn_v.weight"));
    let wo_tensor = gguf.find_tensor(&layer_name("attn_output.weight"));

    // Load QK-Norm weights if present (Qwen3, etc.)
    let q_norm_w = streamer.load_named_tensor_f32(gguf, &layer_name("attn_q_norm.weight")).ok();
    let k_norm_w = streamer.load_named_tensor_f32(gguf, &layer_name("attn_k_norm.weight")).ok();

    if let (Some(tq), Some(tk), Some(tv), Some(to_)) = (wq_tensor, wk_tensor, wv_tensor, wo_tensor)
    {
        let wq_raw = streamer.raw_tensor_data(tq)?;
        let wk_raw = streamer.raw_tensor_data(tk)?;
        let wv_raw = streamer.raw_tensor_data(tv)?;
        let wo_raw = streamer.raw_tensor_data(to_)?;

        // Derive actual projection dimensions from weight tensor shapes
        let q_dim = tq.dimensions.get(1).copied().unwrap_or(n_embd as u64) as usize;
        let kv_dim = tk.dimensions.get(1).copied().unwrap_or(n_embd as u64) as usize;

        if layer_idx == 0 && tracing::enabled!(tracing::Level::DEBUG) {
            if let Ok(wq_f32) = streamer.load_tensor_f32(tq) {
                let y_normal = crate::metal::compute::matvec_f32_simd(&wq_f32, &attn_input, q_dim, n_embd);
                let mut wq_t = vec![0.0f32; q_dim * n_embd];
                for r in 0..q_dim {
                    for c in 0..n_embd {
                        wq_t[c * q_dim + r] = wq_f32[r * n_embd + c];
                    }
                }
                let y_transposed = crate::metal::compute::matvec_f32_simd(&wq_t, &attn_input, q_dim, n_embd);
                debug!(
                    "TRANSPOSE TEST attn_q layer 0: normal[0..4]=[{:.4},{:.4},{:.4},{:.4}] transposed[0..4]=[{:.4},{:.4},{:.4},{:.4}]",
                    y_normal[0], y_normal[1], y_normal[2], y_normal[3],
                    y_transposed[0], y_transposed[1], y_transposed[2], y_transposed[3],
                );
                let n_diff = y_normal.iter().zip(y_transposed.iter()).filter(|(a,b)| (*a - *b).abs() > 0.01).count();
                debug!("TRANSPOSE TEST: {} of {} elements differ by >0.01", n_diff, q_dim);
            }
        }

        let attn_output = multi_head_attention_quantized(
            &attn_input,
            wq_raw,
            tq.dtype,
            wk_raw,
            tk.dtype,
            wv_raw,
            tv.dtype,
            wo_raw,
            to_.dtype,
            n_head,
            n_head_kv,
            q_dim,
            kv_dim,
            position,
            kv_cache,
            theta_base,
            q_norm_w.as_deref(),
            k_norm_w.as_deref(),
            rms_eps,
        );

        // Diagnostic: check attention output magnitude
        if layer_idx <= 2 {
            let ao_max = attn_output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ao_min = attn_output.iter().copied().fold(f32::INFINITY, f32::min);
            debug!("Layer {} attn_output: max={:.4} min={:.4}", layer_idx, ao_max, ao_min);
        }

        // Residual connection
        for (h, a) in hidden_state.iter_mut().zip(attn_output.iter()) {
            *h += a;
        }
    }

    // Diagnostic: hidden state after attention residual
    if layer_idx <= 2 {
        let ha_max = hidden_state.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let ha_min = hidden_state.iter().copied().fold(f32::INFINITY, f32::min);
        debug!("Layer {} after attn+residual: max={:.4} min={:.4}", layer_idx, ha_max, ha_min);
    }

    // 3. Pre-FFN RMSNorm (F32, tiny)
    let mut ffn_input = hidden_state.clone();
    match streamer.load_named_tensor_f32(gguf, &layer_name("ffn_norm.weight")) {
        Ok(norm_w) => {
            if layer_idx == 0 {
                debug!("NORM WIRING: ffn_norm[0..4]=[{:.6},{:.6},{:.6},{:.6}] len={}", norm_w[0], norm_w[1], norm_w[2], norm_w[3], norm_w.len());
            }
            crate::metal::compute::rmsnorm_f32_fast(&mut ffn_input, &norm_w, rms_eps);
            if layer_idx <= 2 {
                let fi_max = ffn_input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let fi_min = ffn_input.iter().copied().fold(f32::INFINITY, f32::min);
                debug!("Layer {} ffn_input after norm: max={:.4} min={:.4} (norm_w len={})", layer_idx, fi_max, fi_min, norm_w.len());
            }
        }
        Err(e) => {
            warn!("Layer {} FFN NORM FAILED TO LOAD: {}", layer_idx, e);
        }
    }

    // 4. FFN: gate/up/down projections on raw quantized bytes
    let gate_tensor = gguf.find_tensor(&layer_name("ffn_gate.weight"));
    let up_tensor = gguf.find_tensor(&layer_name("ffn_up.weight"));
    let down_tensor = gguf.find_tensor(&layer_name("ffn_down.weight"));

    if let (Some(tg), Some(tu), Some(td)) = (gate_tensor, up_tensor, down_tensor) {
        let gate_raw = streamer.raw_tensor_data(tg)?;
        let up_raw = streamer.raw_tensor_data(tu)?;
        let down_raw = streamer.raw_tensor_data(td)?;

        let n_ff = tg.dimensions[1] as usize; // output dim of gate projection

        let ffn_output = feed_forward_quantized(
            &ffn_input, gate_raw, tg.dtype, up_raw, tu.dtype, down_raw, td.dtype, n_embd, n_ff,
        );

        // Diagnostic: check FFN output magnitude
        if layer_idx <= 2 {
            let fo_max = ffn_output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let fo_min = ffn_output.iter().copied().fold(f32::INFINITY, f32::min);
            debug!("Layer {} ffn_output: max={:.4} min={:.4}", layer_idx, fo_max, fo_min);
        }

        // Residual connection
        for (h, f) in hidden_state.iter_mut().zip(ffn_output.iter()) {
            *h += f;
        }
    }

    Ok(())
}

pub struct InferenceConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
    pub stop_sequences: Vec<String>,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// Tail-Free Sampling z parameter (0.0 = disabled, 0.95 = typical)
    pub tfs_z: f32,
    /// Mirostat mode: 0 = disabled, 1 = v1, 2 = v2
    pub mirostat: u8,
    /// Mirostat target surprise (tau), default 5.0
    pub mirostat_tau: f32,
    /// Mirostat learning rate (eta), default 0.1
    pub mirostat_eta: f32,
    /// GBNF grammar string for constrained generation (empty = disabled)
    pub grammar: String,
    /// Min-P filtering threshold (0.0 = disabled, 0.05–0.2 typical)
    pub min_p: f32,
    /// Optional seed for reproducible sampling (None = random)
    pub seed: Option<u64>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            max_tokens: 512,
            stop_sequences: Vec::new(),
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            tfs_z: 0.0,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            grammar: String::new(),
            min_p: 0.0,
            seed: None,
        }
    }
}

/// Build the appropriate sampler from inference configuration
fn build_sampler(config: &InferenceConfig) -> Sampler {
    let sampler = match config.mirostat {
        1 => Sampler::with_mirostat(
            config.temperature,
            MirostatMode::V1,
            config.mirostat_tau,
            config.mirostat_eta,
        ),
        2 => Sampler::with_mirostat(
            config.temperature,
            MirostatMode::V2,
            config.mirostat_tau,
            config.mirostat_eta,
        ),
        _ => {
            if config.tfs_z > 0.0 && config.tfs_z < 1.0 {
                Sampler::with_tfs(
                    config.temperature,
                    config.top_k,
                    config.top_p,
                    config.tfs_z,
                    config.repetition_penalty,
                    config.frequency_penalty,
                    config.presence_penalty,
                )
            } else {
                Sampler::with_min_p(
                    config.temperature,
                    config.top_k,
                    config.top_p,
                    config.min_p,
                    config.repetition_penalty,
                    config.frequency_penalty,
                    config.presence_penalty,
                )
            }
        }
    };
    if let Some(seed) = config.seed {
        sampler.with_seed(seed)
    } else {
        sampler
    }
}

pub struct GenerationResult {
    pub text: String,
    pub token_count: usize,
    pub tokens_per_sec: f64,
    pub prompt_tokens: usize,
    pub kv_cache_bytes: usize,
}

/// Batch prefill: process multiple tokens through all layers efficiently.
/// Loads each layer once and processes all tokens through it before moving to the next layer.
/// This minimizes SSD reads compared to the per-token approach.
pub fn batch_prefill(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    kv_cache: &mut KvCache,
    token_embeddings: &[Vec<f32>], // one embedding per token
    start_position: usize,
    prefetcher: &Prefetcher,
) -> Result<Vec<f32>> {
    batch_prefill_lora(
        gguf,
        streamer,
        layer_cache,
        kv_cache,
        token_embeddings,
        start_position,
        prefetcher,
        None,
    )
}

/// Batch prefill with optional LoRA adapter support.
///
/// v1.35: Uses fused Metal GPU kernels for attention (QKV+RoPE) and FFN (SwiGLU).
pub fn batch_prefill_lora(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    kv_cache: &mut KvCache,
    token_embeddings: &[Vec<f32>],
    start_position: usize,
    prefetcher: &Prefetcher,
    mut lora: Option<&mut LoraManager>,
) -> Result<Vec<f32>> {
    let n_layers = gguf.n_layers();
    let n_embd = gguf.n_embd() as usize;
    let n_head = gguf.n_head() as usize;
    let n_head_kv = gguf.n_head_kv() as usize;
    let head_dim = n_embd / n_head;
    let moe_config = detect_moe_config(gguf);
    let n_tokens = token_embeddings.len();

    if n_tokens == 0 {
        return Ok(vec![0.0f32; n_embd]);
    }

    // Initialize hidden states from embeddings
    let mut hidden_states: Vec<Vec<f32>> = token_embeddings.to_vec();

    // GROUND-TRUTH DUMP: raw embedding for first token (pre-layer-0)
    if std::env::var("SSD_DUMP_EMBED").is_ok() && !hidden_states.is_empty() {
        use std::sync::atomic::{AtomicBool, Ordering};
        static EMBED_DUMPED: AtomicBool = AtomicBool::new(false);
        if !EMBED_DUMPED.swap(true, Ordering::Relaxed) {
            let path = std::env::var("SSD_DUMP_EMBED").unwrap_or_else(|_| "ssd_llm_embed.bin".to_string());
            let emb = &hidden_states[0];
            let bytes: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();
            match std::fs::write(&path, &bytes) {
                Ok(()) => info!("SSD_DUMP_EMBED: wrote {} floats to {}", emb.len(), path),
                Err(e) => warn!("SSD_DUMP_EMBED: {}", e),
            }
        }
    }

    // Read model-specific parameters from GGUF metadata
    let theta_base = gguf.rope_theta();
    let rms_eps = gguf.rms_norm_eps();

    // DIAGNOSTIC: limit to N layers to isolate divergence point
    let max_layers = std::env::var("SSD_MAX_LAYERS").ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(n_layers);
    let effective_layers = n_layers.min(max_layers);
    if effective_layers < n_layers {
        info!("LAYER LIMIT: running {}/{} layers", effective_layers, n_layers);
    }

    for layer_idx in 0..effective_layers {
        // Prefetch next layer's mmap pages from SSD
        prefetcher.on_layer_start(layer_idx, n_layers, streamer, gguf, layer_cache);

        debug!("Streaming layer {}/{}", layer_idx, n_layers);

        // Process ALL tokens through this layer using quantized matvec on raw mmap bytes
        for (t, hidden_state) in hidden_states.iter_mut().enumerate().take(n_tokens) {
            let position = start_position + t;
            let kv_layer = kv_cache.layer_mut(layer_idx as usize);
            streaming_layer_forward(
                hidden_state,
                gguf,
                streamer,
                layer_idx,
                n_head,
                n_head_kv,
                head_dim,
                position,
                kv_layer,
                theta_base,
                rms_eps,
            )?;
            // Per-layer diagnostic for last token at key layers
            if t == n_tokens - 1 && (layer_idx == 0 || layer_idx == 1 || layer_idx == n_layers - 1) {
                let h_max = hidden_state.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let h_min = hidden_state.iter().copied().fold(f32::INFINITY, f32::min);
                let h_mean = hidden_state.iter().sum::<f32>() / hidden_state.len() as f32;
                info!(
                    "After layer {}: max={:.4} min={:.4} mean={:.4}",
                    layer_idx, h_max, h_min, h_mean
                );
            }
            // GROUND-TRUTH COMPARISON: dump layer 0 hidden state for last token
            if layer_idx == 0 && t == n_tokens - 1 && std::env::var("SSD_DUMP_LAYER0").is_ok() {
                let path = std::env::var("SSD_DUMP_LAYER0").unwrap_or_else(|_| "ssd_llm_layer0.bin".to_string());
                let bytes: Vec<u8> = hidden_state.iter().flat_map(|f| f.to_le_bytes()).collect();
                match std::fs::write(&path, &bytes) {
                    Ok(()) => info!("SSD_DUMP_LAYER0: wrote {} floats to {}", hidden_state.len(), path),
                    Err(e) => warn!("SSD_DUMP_LAYER0: failed to write {}: {}", path, e),
                }
            }
        }

        // Signal layer done for eviction
        prefetcher.on_layer_done(layer_idx, streamer, gguf);
    }

    // Return the last token's hidden state (needed for generation)
    Ok(hidden_states.pop().unwrap_or_else(|| vec![0.0f32; n_embd]))
}

/// Run transformer forward pass (public entry point for speculative decoding)
#[allow(clippy::ptr_arg)]
pub fn forward_pass_pub(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    kv_cache: &mut KvCache,
    hidden_state: &mut Vec<f32>,
    position: usize,
    prefetcher: &Prefetcher,
) -> Result<()> {
    forward_pass(
        gguf,
        streamer,
        layer_cache,
        kv_cache,
        hidden_state,
        position,
        prefetcher,
        None,
    )
}

/// Run transformer forward pass for a single token through all layers (with KV cache).
///
/// v1.35: Uses fused Metal GPU kernels when available:
/// - Fused QKV + RoPE projection (1 dispatch instead of 5)
/// - Fused SwiGLU FFN (1 dispatch instead of 4)
/// - Fused residual + RMSNorm
///   Falls back to CPU SIMD paths transparently when Metal is unavailable.
#[allow(clippy::ptr_arg)]
fn forward_pass(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    kv_cache: &mut KvCache,
    hidden_state: &mut Vec<f32>,
    position: usize,
    prefetcher: &Prefetcher,
    mut lora: Option<&mut LoraManager>,
) -> Result<()> {
    let n_layers = gguf.n_layers();
    let n_embd = gguf.n_embd() as usize;
    let n_head = gguf.n_head() as usize;
    let n_head_kv = gguf.n_head_kv() as usize;
    let head_dim = n_embd / n_head;
    let moe_config = detect_moe_config(gguf);

    // Read model-specific parameters from GGUF metadata
    let theta_base = gguf.rope_theta();
    let rms_eps = gguf.rms_norm_eps();

    let max_layers = std::env::var("SSD_MAX_LAYERS").ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(n_layers);
    let effective_layers = n_layers.min(max_layers);

    for layer_idx in 0..effective_layers {
        prefetcher.on_layer_start(layer_idx, n_layers, streamer, gguf, layer_cache);

        let kv_layer = kv_cache.layer_mut(layer_idx as usize);
        streaming_layer_forward(
            hidden_state,
            gguf,
            streamer,
            layer_idx,
            n_head,
            n_head_kv,
            head_dim,
            position,
            kv_layer,
            theta_base,
            rms_eps,
        )?;

        prefetcher.on_layer_done(layer_idx, streamer, gguf);
    }

    Ok(())
}

/// Detect MoE configuration from GGUF metadata
pub fn detect_moe_config(gguf: &GgufFile) -> Option<MoeConfig> {
    if gguf.is_moe() {
        let config = MoeConfig::new(gguf.n_experts() as usize, gguf.n_experts_used() as usize);
        if config.validate() {
            info!(
                "MoE model detected: {} experts, {} used per token",
                config.n_experts, config.n_experts_used
            );
            return Some(config);
        }
    }
    None
}

/// Run FFN or MoE-FFN with GPU acceleration when available.
/// For dense FFN: uses fused SwiGLU Metal kernel (1 dispatch vs 4 separate ops).
/// For MoE: uses GPU-accelerated gating + expert dispatch when available.
pub fn run_ffn_or_moe_gpu(
    ffn_input: &[f32],
    cached: &CachedLayer,
    layer_idx: u32,
    moe_config: Option<&MoeConfig>,
    n_embd: usize,
    gpu: Option<&MetalGpu>,
) -> Option<Vec<f32>> {
    // Check for MoE: look for expert tensors and gate
    if let Some(config) = moe_config {
        let gate_key = format!("blk.{}.ffn_gate_inp.weight", layer_idx);
        if let Some(gate_weights) = cached.tensors.get(&gate_key) {
            let mut experts: Vec<Option<ExpertWeights<'_>>> =
                (0..config.n_experts).map(|_| None).collect();
            let mut all_found = true;

            for (expert_idx, slot) in experts.iter_mut().enumerate().take(config.n_experts) {
                let eg = format!("blk.{}.ffn_gate.{}.weight", layer_idx, expert_idx);
                let eu = format!("blk.{}.ffn_up.{}.weight", layer_idx, expert_idx);
                let ed = format!("blk.{}.ffn_down.{}.weight", layer_idx, expert_idx);

                if let (Some(wg), Some(wu), Some(wd)) = (
                    cached.tensors.get(&eg),
                    cached.tensors.get(&eu),
                    cached.tensors.get(&ed),
                ) {
                    *slot = Some(ExpertWeights {
                        w_gate: wg,
                        w_up: wu,
                        w_down: wd,
                    });
                } else {
                    all_found = false;
                    break;
                }
            }

            if all_found {
                let expert_refs: Vec<ExpertWeights<'_>> =
                    experts.into_iter().map(|e| e.unwrap()).collect();
                return Some(moe::moe_forward(
                    ffn_input,
                    gate_weights,
                    &expert_refs,
                    config,
                    n_embd,
                ));
            }
        }
    }

    // Standard dense FFN with GPU acceleration
    if let (Some(w_gate), Some(w_up), Some(w_down)) = (
        find_tensor_in_layer(cached, "ffn_gate.weight", layer_idx),
        find_tensor_in_layer(cached, "ffn_up.weight", layer_idx),
        find_tensor_in_layer(cached, "ffn_down.weight", layer_idx),
    ) {
        Some(feed_forward_gpu(
            ffn_input, w_gate, w_up, w_down, n_embd, gpu,
        ))
    } else {
        None
    }
}

/// Run FFN or MoE-FFN depending on layer tensors (CPU-only legacy path).
/// For MoE: uses gating network to select top-K experts, runs only those.
/// For dense: runs standard SwiGLU FFN.
pub fn run_ffn_or_moe(
    ffn_input: &[f32],
    cached: &CachedLayer,
    layer_idx: u32,
    moe_config: Option<&MoeConfig>,
    n_embd: usize,
) -> Option<Vec<f32>> {
    // Check for MoE: look for expert tensors and gate
    if let Some(config) = moe_config {
        let gate_key = format!("blk.{}.ffn_gate_inp.weight", layer_idx);
        if let Some(gate_weights) = cached.tensors.get(&gate_key) {
            // Collect expert weights
            let mut experts: Vec<Option<ExpertWeights<'_>>> =
                (0..config.n_experts).map(|_| None).collect();
            let mut all_found = true;

            for (expert_idx, slot) in experts.iter_mut().enumerate().take(config.n_experts) {
                let eg = format!("blk.{}.ffn_gate.{}.weight", layer_idx, expert_idx);
                let eu = format!("blk.{}.ffn_up.{}.weight", layer_idx, expert_idx);
                let ed = format!("blk.{}.ffn_down.{}.weight", layer_idx, expert_idx);

                if let (Some(wg), Some(wu), Some(wd)) = (
                    cached.tensors.get(&eg),
                    cached.tensors.get(&eu),
                    cached.tensors.get(&ed),
                ) {
                    *slot = Some(ExpertWeights {
                        w_gate: wg,
                        w_up: wu,
                        w_down: wd,
                    });
                } else {
                    all_found = false;
                    break;
                }
            }

            if all_found {
                let expert_refs: Vec<ExpertWeights<'_>> =
                    experts.into_iter().map(|e| e.unwrap()).collect();
                return Some(moe::moe_forward(
                    ffn_input,
                    gate_weights,
                    &expert_refs,
                    config,
                    n_embd,
                ));
            }
        }
    }

    // Standard dense FFN fallback
    if let (Some(w_gate), Some(w_up), Some(w_down)) = (
        find_tensor_in_layer(cached, "ffn_gate.weight", layer_idx),
        find_tensor_in_layer(cached, "ffn_up.weight", layer_idx),
        find_tensor_in_layer(cached, "ffn_down.weight", layer_idx),
    ) {
        Some(feed_forward(ffn_input, w_gate, w_up, w_down, n_embd))
    } else {
        None
    }
}

/// Helper: find a tensor in a cached layer by suffix
pub fn find_tensor_in_layer<'a>(
    cached: &'a CachedLayer,
    suffix: &str,
    layer_idx: u32,
) -> Option<&'a Vec<f32>> {
    let full_name = format!("blk.{}.{}", layer_idx, suffix);
    cached.tensors.get(&full_name)
}

/// Apply all LoRA adapters to a cached layer's tensors in-place.
/// This modifies the cached layer so subsequent reads automatically include LoRA deltas.
fn apply_lora_to_layer(cached: &mut CachedLayer, lora: &mut LoraManager) {
    let tensor_names: Vec<String> = cached.tensors.keys().cloned().collect();
    for name in tensor_names {
        if lora.has_weight(&name) {
            if let Some(weight) = cached.tensors.get_mut(&name) {
                lora.apply_to_weight(&name, weight);
            }
        }
    }
}

/// RMS Normalization in-place (uses default eps=1e-5 for legacy callers)
#[allow(clippy::ptr_arg)]
pub fn rms_norm(x: &mut Vec<f32>, weight: &[f32]) {
    crate::metal::compute::rmsnorm_f32_fast(x, weight, 1e-5);
}

/// RMS Normalization in-place with explicit epsilon
#[allow(clippy::ptr_arg)]
pub fn rms_norm_eps(x: &mut Vec<f32>, weight: &[f32], eps: f32) {
    crate::metal::compute::rmsnorm_f32_fast(x, weight, eps);
}

/// Generate text token by token with KV cache
pub fn generate(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    prompt: &str,
    config: &InferenceConfig,
) -> Result<GenerationResult> {
    let n_embd = gguf.n_embd() as usize;
    let n_layers = gguf.n_layers();
    let n_head = gguf.n_head() as usize;
    let n_head_kv = gguf.n_head_kv() as usize;
    let head_dim = n_embd / n_head;
    let vocab_size = gguf.vocab_size() as usize;
    let n_ctx = gguf.n_ctx() as usize;
    let rms_eps = gguf.rms_norm_eps();

    if n_embd == 0 || n_layers == 0 {
        bail!("Invalid model: n_embd={}, n_layers={}", n_embd, n_layers);
    }

    info!(
        "Model: arch={}, layers={}, embd={}, heads={}/{}, vocab={}, ctx={}, rope_theta={}, rms_eps={}",
        gguf.architecture(),
        n_layers,
        n_embd,
        n_head,
        n_head_kv,
        vocab_size,
        n_ctx,
        gguf.rope_theta(),
        rms_eps,
    );
    // Identify Q6K tensors in forward pass
    for t in &gguf.tensors {
        if t.dtype == crate::model::gguf::GgmlType::Q6K && t.name != "output.weight" {
            info!("Q6K forward-pass tensor: {} {:?}", t.name, t.dimensions);
        }
    }

    let prefetcher = Prefetcher::new(PrefetchStrategy::LookAhead(2));
    let sampler = build_sampler(config);

    // Parse grammar if provided
    let grammar_acceptor = if !config.grammar.is_empty() {
        match crate::inference::grammar::parse_grammar(&config.grammar) {
            Ok(g) => Some(crate::inference::grammar::GrammarAcceptor::new(g)),
            Err(e) => {
                warn!("Failed to parse grammar, ignoring: {}", e);
                None
            }
        }
    } else {
        None
    };
    let mut grammar_state = grammar_acceptor;

    // Initialize KV cache
    let mut kv_cache = KvCache::new(n_layers as usize, n_head_kv, head_dim, n_ctx);

    let tokenizer = SimpleTokenizer::from_gguf(gguf);
    let tokens = tokenizer.encode(prompt);
    let prompt_len = tokens.len();
    info!("Prompt tokens: {} {:?}", prompt_len, &tokens);
    // Debug: decode each token to verify tokenization
    for (i, &tid) in tokens.iter().enumerate() {
        debug!("  token[{}] = {} -> {:?}", i, tid, tokenizer.decode_token(tid));
    }

    // TENSOR EXISTENCE CHECK: Are ALL expected tensors present?
    {
        let names = ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                      "attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                      "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "ffn_norm.weight"];
        for name in &names {
            let full = format!("blk.0.{}", name);
            let found = gguf.find_tensor(&full);
            info!("TENSOR CHECK: {} → {}", full, if found.is_some() {
                format!("FOUND {:?} {:?}", found.unwrap().dimensions, found.unwrap().dtype)
            } else {
                "MISSING!".to_string()
            });
        }
    }

    // F16 ATTN_V GROUND TRUTH CHECK: Compare F16 kernel vs scalar dequant reference
    if let Some(v_tensor) = gguf.find_tensor("blk.0.attn_v.weight") {
        if let (Ok(v_raw), Ok(v_f32)) = (streamer.raw_tensor_data(v_tensor), streamer.load_tensor_f32(v_tensor)) {
            // Use the first token's normalized embedding as input
            if let Some(emb_tensor) = gguf.find_tensor("token_embd.weight") {
                if let Ok(emb_rows) = streamer.load_tensor_rows(emb_tensor, &[tokens[0]], n_embd) {
                    let mut norm_input = emb_rows[0].clone();
                    if let Ok(norm_w) = streamer.load_named_tensor_f32(gguf, "blk.0.attn_norm.weight") {
                        crate::metal::compute::rmsnorm_f32_fast(&mut norm_input, &norm_w, rms_eps);
                    }
                    let kv_dim = 1024usize;

                    // F16 kernel result
                    let v_kernel = crate::metal::compute::matvec_quantized_cpu(
                        v_raw, &norm_input, kv_dim, n_embd, v_tensor.dtype,
                    );

                    // Scalar reference using dequantized f32 weights
                    let mut v_ref = vec![0.0f32; kv_dim];
                    for i in 0..kv_dim {
                        let mut sum = 0.0f32;
                        for j in 0..n_embd {
                            sum += v_f32[i * n_embd + j] * norm_input[j];
                        }
                        v_ref[i] = sum;
                    }

                    let max_diff: f32 = v_kernel.iter().zip(v_ref.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max);
                    let mean_diff: f32 = v_kernel.iter().zip(v_ref.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f32>() / kv_dim as f32;

                    info!("F16 V GROUND TRUTH: kernel[0..4]=[{:.6},{:.6},{:.6},{:.6}] ref[0..4]=[{:.6},{:.6},{:.6},{:.6}] max_diff={:.6} mean_diff={:.6}",
                        v_kernel[0], v_kernel[1], v_kernel[2], v_kernel[3],
                        v_ref[0], v_ref[1], v_ref[2], v_ref[3],
                        max_diff, mean_diff);

                    // Also check V head differentiation
                    info!("  V heads: h0[0..4]=[{:.4},{:.4},{:.4},{:.4}] h1[0..4]=[{:.4},{:.4},{:.4},{:.4}] h4[0..4]=[{:.4},{:.4},{:.4},{:.4}]",
                        v_kernel[0], v_kernel[1], v_kernel[2], v_kernel[3],
                        v_kernel[128], v_kernel[129], v_kernel[130], v_kernel[131],
                        v_kernel[512], v_kernel[513], v_kernel[514], v_kernel[515]);
                }
            }
        }
    }

    // LAYER DIFFERENTIATION CHECK: Are different layers loading different weights?
    {
        let l0_norm = streamer.load_named_tensor_f32(gguf, "blk.0.attn_norm.weight").ok();
        let l1_norm = streamer.load_named_tensor_f32(gguf, "blk.1.attn_norm.weight").ok();
        let l0_ffn = streamer.load_named_tensor_f32(gguf, "blk.0.ffn_norm.weight").ok();
        if let (Some(n0), Some(n1), Some(f0)) = (l0_norm.as_ref(), l1_norm.as_ref(), l0_ffn.as_ref()) {
            let same_layers = n0.iter().zip(n1.iter()).all(|(a, b)| (a - b).abs() < 1e-10);
            let same_blocks = n0.iter().zip(f0.iter()).all(|(a, b)| (a - b).abs() < 1e-10);
            info!("LAYER DIFF: blk.0.attn_norm == blk.1.attn_norm? {} | blk.0.attn_norm == blk.0.ffn_norm? {}",
                same_layers, same_blocks);
            info!("  blk.0.attn_norm[0..4] = [{:.6},{:.6},{:.6},{:.6}]", n0[0], n0[1], n0[2], n0[3]);
            info!("  blk.1.attn_norm[0..4] = [{:.6},{:.6},{:.6},{:.6}]", n1[0], n1[1], n1[2], n1[3]);
            info!("  blk.0.ffn_norm[0..4]  = [{:.6},{:.6},{:.6},{:.6}]", f0[0], f0[1], f0[2], f0[3]);
        }
        // Also check V weights across layers
        if let (Some(v0), Some(v1)) = (
            gguf.find_tensor("blk.0.attn_v.weight"),
            gguf.find_tensor("blk.1.attn_v.weight"),
        ) {
            info!("  V tensor offsets: blk.0={} blk.1={} (must differ)", v0.offset, v1.offset);
        }
    }

    // ONE-HOT IMPULSE TEST: Does Wo return row j or column j when fed e_j?
    // If y ≈ row_j → transposed (computing W^T @ x). If y ≠ row_j → correct (computing W @ x).
    if let Some(wo_tensor) = gguf.find_tensor("blk.0.attn_output.weight") {
        if let Ok(wo_raw) = streamer.raw_tensor_data(wo_tensor) {
            for j in 0..2usize {
                let mut e_j = vec![0.0f32; 2048];
                e_j[j] = 1.0;

                // matvec: should extract column j (correct) or row j (transposed)
                let y = crate::metal::compute::matvec_quantized_cpu(
                    wo_raw, &e_j, 2048, 2048, wo_tensor.dtype,
                );

                // Dequantize row j directly
                let row_j = streamer.load_tensor_rows(wo_tensor, &[j as u32], 2048)
                    .unwrap_or_default();

                if let Some(row) = row_j.first() {
                    let max_diff: f32 = y.iter().zip(row.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max);

                    info!(
                        "Wo IMPULSE TEST j={}: y[0..5]=[{:.6},{:.6},{:.6},{:.6},{:.6}] row_j[0..5]=[{:.6},{:.6},{:.6},{:.6},{:.6}] max_diff={:.6}",
                        j, y[0], y[1], y[2], y[3], y[4],
                        row[0], row[1], row[2], row[3], row[4],
                        max_diff
                    );
                    if max_diff < 1e-4 {
                        warn!("Wo IS TRANSPOSED! matvec(e_{}) = row_{}, not column_{}", j, j, j);
                    } else {
                        info!("Wo orientation OK: matvec(e_{}) ≠ row_{} (max_diff={:.4})", j, j, max_diff);
                    }
                }
            }
        }
    }

    // DIAGNOSTIC: Dump raw Q6K block 0 of output.weight to verify struct layout
    if let Some(out_t) = gguf.find_tensor("output.weight") {
        if let Ok(raw) = streamer.raw_tensor_data(out_t) {
            // Block 0: 210 bytes
            // Expected layout: ql[128] + qh[64] + scales[16] + d[2]
            let d_at_208 = half::f16::from_bits(u16::from_le_bytes([raw[208], raw[209]])).to_f32();
            let d_at_0 = half::f16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32();
            let scales_at_192: Vec<i8> = (192..208).map(|i| raw[i] as i8).collect();
            let scales_at_2: Vec<i8> = (2..18).map(|i| raw[i] as i8).collect();
            info!("Q6K BLOCK 0 raw: d@208={:.6} d@0={:.6}", d_at_208, d_at_0);
            info!("  scales@192 (expected): {:?}", scales_at_192);
            info!("  scales@2   (alt layout): {:?}", scales_at_2);
            info!("  first 4 ql bytes: [{}, {}, {}, {}]", raw[0], raw[1], raw[2], raw[3]);
            info!("  bytes 208-209: [{}, {}] (expected d)", raw[208], raw[209]);
        }
    }

    // DIAGNOSTIC: Cross-check Q6K output.weight vs Q4K token_embd.weight (should be same weights)
    if let (Some(emb_t), Some(out_t)) = (gguf.find_tensor("token_embd.weight"), gguf.find_tensor("output.weight")) {
        // Dequantize row 0 from both
        let emb_row = streamer.load_tensor_rows(emb_t, &[0], n_embd).ok();
        let out_row = streamer.load_tensor_rows(out_t, &[0], n_embd).ok();
        if let (Some(er), Some(or)) = (emb_row, out_row) {
            let emb_vals = &er[0];
            let out_vals = &or[0];
            let mut max_diff = 0.0f32;
            let mut sum_diff = 0.0f32;
            for i in 0..n_embd.min(emb_vals.len()).min(out_vals.len()) {
                let d = (emb_vals[i] - out_vals[i]).abs();
                if d > max_diff { max_diff = d; }
                sum_diff += d;
            }
            info!(
                "WEIGHT TIE CHECK row 0: max_diff={:.6} mean_diff={:.6}",
                max_diff, sum_diff / n_embd as f32,
            );
            // Print first 16 values to spot permutation pattern
            for i in 0..16 {
                info!("  [{:>3}] emb={:>10.6} out={:>10.6} diff={:.6}", i, emb_vals[i], out_vals[i], (emb_vals[i]-out_vals[i]).abs());
            }
        }
    }

    // DIAGNOSTIC: Embedding→output projection sanity check (no forward pass)
    // If output weights match embedding weights (tied), token should rank highly for itself
    if let Some(emb_tensor) = gguf.find_tensor("token_embd.weight") {
        if let Ok(emb_rows) = streamer.load_tensor_rows(emb_tensor, &[tokens[0]], n_embd) {
            let emb = &emb_rows[0];
            // Project embedding directly through output weight (Q6K)
            if let Some(out_tensor) = gguf.find_tensor("output.weight") {
                if let Ok(out_raw) = streamer.raw_tensor_data(out_tensor) {
                    // Normal orientation
                    let logits_normal = crate::metal::compute::matvec_quantized_cpu(
                        out_raw, emb, vocab_size, n_embd, out_tensor.dtype,
                    );
                    // Find rank of token[0] in the logits
                    let self_logit = logits_normal[tokens[0] as usize];
                    let mut sorted: Vec<(usize, f32)> = logits_normal.iter().enumerate().map(|(i,&v)| (i,v)).collect();
                    sorted.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
                    let self_rank = sorted.iter().position(|(i,_)| *i == tokens[0] as usize).unwrap_or(999999);
                    info!(
                        "EMBED→OUTPUT test (normal): token {} '{}' self_logit={:.4} rank={} | top3: {}='{}' {}='{}' {}='{}'",
                        tokens[0], tokenizer.decode_token(tokens[0]),
                        self_logit, self_rank,
                        sorted[0].0, tokenizer.decode_token(sorted[0].0 as u32),
                        sorted[1].0, tokenizer.decode_token(sorted[1].0 as u32),
                        sorted[2].0, tokenizer.decode_token(sorted[2].0 as u32),
                    );
                }
            }
            // Also try through token_embd (tied weights, Q4K)
            if let Ok(emb_raw) = streamer.raw_tensor_data(emb_tensor) {
                let logits_tied = crate::metal::compute::matvec_quantized_cpu(
                    emb_raw, emb, vocab_size, n_embd, emb_tensor.dtype,
                );
                let self_logit = logits_tied[tokens[0] as usize];
                let mut sorted: Vec<(usize, f32)> = logits_tied.iter().enumerate().map(|(i,&v)| (i,v)).collect();
                sorted.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
                let self_rank = sorted.iter().position(|(i,_)| *i == tokens[0] as usize).unwrap_or(999999);
                info!(
                    "EMBED→OUTPUT test (tied Q4K): token {} '{}' self_logit={:.4} rank={} | top3: {}='{}' {}='{}' {}='{}'",
                    tokens[0], tokenizer.decode_token(tokens[0]),
                    self_logit, self_rank,
                    sorted[0].0, tokenizer.decode_token(sorted[0].0 as u32),
                    sorted[1].0, tokenizer.decode_token(sorted[1].0 as u32),
                    sorted[2].0, tokenizer.decode_token(sorted[2].0 as u32),
                );
            }
        }
    }

    let mut generated_tokens = Vec::new();
    let mut hidden_state = vec![0.0f32; n_embd];

    let start = Instant::now();

    // Batch prefill: load only needed embedding rows (not the full 5+ GB tensor)
    {
        let token_embeddings: Vec<Vec<f32>> = if let Some(emb_tensor) =
            gguf.find_tensor("token_embd.weight")
        {
            let embs = streamer
                .load_tensor_rows(emb_tensor, &tokens, n_embd)
                .unwrap_or_else(|_| vec![vec![0.0f32; n_embd]; prompt_len]);
            // Diagnostic: check embedding magnitudes
            if let Some(first_emb) = embs.first() {
                let emb_max = first_emb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let emb_min = first_emb.iter().copied().fold(f32::INFINITY, f32::min);
                let emb_mean = first_emb.iter().copied().sum::<f32>() / first_emb.len() as f32;
                let emb_nonzero = first_emb.iter().filter(|&&v| v != 0.0).count();
                info!(
                    "Embedding[0] (token {}): max={:.4} min={:.4} mean={:.6} nonzero={}/{}",
                    tokens[0], emb_max, emb_min, emb_mean, emb_nonzero, first_emb.len()
                );
            }
            embs
        } else {
            vec![vec![0.0f32; n_embd]; prompt_len]
        };

        hidden_state = batch_prefill(
            gguf,
            streamer,
            layer_cache,
            &mut kv_cache,
            &token_embeddings,
            0,
            &prefetcher,
        )?;
    }

    let prefill_time = start.elapsed();
    info!(
        "Prefill (batch): {} tokens in {:.2}ms ({:.1} tok/s)",
        prompt_len,
        prefill_time.as_millis(),
        prompt_len as f64 / prefill_time.as_secs_f64()
    );

    // Generation phase (decode)
    let decode_start = Instant::now();
    let mut pos = tokens.len();
    for _ in 0..config.max_tokens {
        if kv_cache.is_full() {
            info!("KV cache full at seq_len={}, stopping", pos);
            break;
        }

        // Fused RMSNorm + output projection (v1.34: single GPU dispatch)
        let mut logits = {
            let norm_w = streamer
                .load_named_tensor_f32(gguf, "output_norm.weight")
                .ok();
            // AOT BYPASS: Skip Q6K output.weight, use Q4K tied path (known-correct)
            let out_w: Option<Vec<f32>> = None;

            if let (Some(nw), Some(ow)) = (norm_w.as_ref(), out_w.as_ref()) {
                // Dedicated output.weight exists — use f32 path
                let mc = crate::metal::compute::MetalCompute::new();
                if let Some(ref compute) = mc {
                    compute.fused_rmsnorm_linear_f32(&hidden_state, nw, ow, vocab_size, n_embd)
                } else {
                    crate::metal::compute::fused_rmsnorm_linear_f32_cpu_eps(
                        &hidden_state,
                        nw,
                        ow,
                        vocab_size,
                        n_embd,
                        rms_eps,
                    )
                }
            } else if let (Some(nw), Some(emb_tensor)) =
                (norm_w.as_ref(), gguf.find_tensor("token_embd.weight"))
            {
                // Tied weights — use quantized matvec directly on raw bytes (no 5+ GB alloc)
                if let Ok(raw) = streamer.raw_tensor_data(emb_tensor) {
                    crate::metal::compute::fused_rmsnorm_linear_quantized_cpu_eps(
                        &hidden_state,
                        nw,
                        raw,
                        emb_tensor.dtype.clone(),
                        vocab_size,
                        n_embd,
                        rms_eps,
                    )
                } else {
                    vec![0.0f32; vocab_size]
                }
            } else {
                vec![0.0f32; vocab_size]
            }
        };

        // Apply grammar constraints to logits
        if let Some(ref acceptor) = grammar_state {
            let token_strings: Vec<String> = (0..vocab_size)
                .map(|id| tokenizer.decode(&[id as u32]))
                .collect();
            crate::inference::grammar::apply_grammar_mask(&mut logits, acceptor, &token_strings);
        }

        // Diagnostic: show top logits for first 3 tokens
        if generated_tokens.len() < 3 {
            let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top5: Vec<_> = indexed.iter().take(5).collect();
            let logit_stats = (
                logits.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                logits.iter().copied().fold(f32::INFINITY, f32::min),
                logits.iter().copied().sum::<f32>() / logits.len() as f32,
            );
            info!(
                "Logits step {}: max={:.4} min={:.4} mean={:.6} | top5: {:?}",
                generated_tokens.len(),
                logit_stats.0,
                logit_stats.1,
                logit_stats.2,
                top5.iter().map(|(i, v)| (i, format!("{:.3}", v), tokenizer.decode_token(*i as u32))).collect::<Vec<_>>()
            );
            // Also show hidden state stats
            let h_max = hidden_state.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let h_min = hidden_state.iter().copied().fold(f32::INFINITY, f32::min);
            let h_mean = hidden_state.iter().copied().sum::<f32>() / hidden_state.len() as f32;
            info!("Hidden state: max={:.4} min={:.4} mean={:.6}", h_max, h_min, h_mean);
        }

        let token_id = sampler.sample(&logits);

        if token_id == tokenizer.eos_id || token_id == 0 {
            break;
        }

        generated_tokens.push(token_id);

        // Advance grammar state with the generated token
        if let Some(ref mut acceptor) = grammar_state {
            let token_str = tokenizer.decode(&[token_id]);
            acceptor.accept_str(&token_str);
        }

        // Embed new token and forward
        // Embed new token — row-slice, not full tensor load
        if let Some(emb_tensor) = gguf.find_tensor("token_embd.weight") {
            if let Ok(rows) = streamer.load_tensor_rows(emb_tensor, &[token_id], n_embd) {
                if let Some(emb) = rows.into_iter().next() {
                    hidden_state = emb;
                }
            }
        }

        forward_pass(
            gguf,
            streamer,
            layer_cache,
            &mut kv_cache,
            &mut hidden_state,
            pos,
            &prefetcher,
            None,
        )?;
        pos += 1;
    }

    let decode_time = decode_start.elapsed();
    let token_count = generated_tokens.len();
    let tokens_per_sec = if decode_time.as_secs_f64() > 0.0 {
        token_count as f64 / decode_time.as_secs_f64()
    } else {
        0.0
    };

    info!(
        "Decode: {} tokens in {:.2}ms ({:.1} tok/s), KV cache: {:.2} MB",
        token_count,
        decode_time.as_millis(),
        tokens_per_sec,
        kv_cache.size_bytes() as f64 / (1024.0 * 1024.0)
    );

    let text = tokenizer.decode(&generated_tokens);

    Ok(GenerationResult {
        text,
        token_count,
        tokens_per_sec,
        prompt_tokens: prompt_len,
        kv_cache_bytes: kv_cache.size_bytes(),
    })
}

/// Matrix-vector multiplication: hidden_state (n_embd) × weight (vocab_size × n_embd) → logits (vocab_size)
fn matmul_1d(x: &[f32], weight: &[f32], output_dim: usize) -> Vec<f32> {
    crate::metal::compute::matvec_f32_simd(weight, x, output_dim, x.len())
}

/// Streaming generator — yields one token at a time
pub struct StreamingGenerator<'a> {
    gguf: &'a GgufFile,
    streamer: &'a SsdStreamer,
    layer_cache: &'a mut LayerCache,
    kv_cache: KvCache,
    hidden_state: Vec<f32>,
    tokenizer: SimpleTokenizer,
    sampler: Sampler,
    prefetcher: Prefetcher,
    position: usize,
    remaining: usize,
    done: bool,
}

impl<'a> StreamingGenerator<'a> {
    /// Get the next generated token as a string, or None if done
    pub fn next_token(&mut self) -> Result<Option<String>> {
        if self.done || self.remaining == 0 {
            return Ok(None);
        }

        if self.kv_cache.is_full() {
            self.done = true;
            return Ok(None);
        }

        let n_embd = self.gguf.n_embd() as usize;
        let vocab_size = self.gguf.vocab_size() as usize;

        // Fused RMSNorm + output projection (v1.34)
        let logits = {
            let norm_w = self
                .streamer
                .load_named_tensor_f32(self.gguf, "output_norm.weight")
                .ok();
            let out_w = self
                .streamer
                .load_named_tensor_f32(self.gguf, "output.weight")
                .ok()
                .or_else(|| {
                    self.streamer
                        .load_named_tensor_f32(self.gguf, "token_embd.weight")
                        .ok()
                });

            match (norm_w.as_ref(), out_w.as_ref()) {
                (Some(nw), Some(ow)) => {
                    let mc = crate::metal::compute::MetalCompute::new();
                    if let Some(ref compute) = mc {
                        compute.fused_rmsnorm_linear_f32(
                            &self.hidden_state,
                            nw,
                            ow,
                            vocab_size,
                            n_embd,
                        )
                    } else {
                        crate::metal::compute::fused_rmsnorm_linear_f32_cpu(
                            &self.hidden_state,
                            nw,
                            ow,
                            vocab_size,
                            n_embd,
                        )
                    }
                }
                _ => {
                    let mut final_hidden = self.hidden_state.clone();
                    if let Some(nw) = norm_w.as_ref() {
                        rms_norm(&mut final_hidden, nw);
                    }
                    if let Some(ow) = out_w.as_ref() {
                        matmul_1d(&final_hidden, ow, vocab_size)
                    } else {
                        vec![0.0f32; vocab_size]
                    }
                }
            }
        };

        let token_id = self.sampler.sample(&logits);

        if token_id == self.tokenizer.eos_id || token_id == 0 {
            self.done = true;
            return Ok(None);
        }

        let token_text = self.tokenizer.decode_token(token_id);

        // Embed new token and forward
        if let Ok(embeddings) = self
            .streamer
            .load_named_tensor_f32(self.gguf, "token_embd.weight")
        {
            let embd_start = token_id as usize * n_embd;
            let embd_end = embd_start + n_embd;
            if embd_end <= embeddings.len() {
                self.hidden_state = embeddings[embd_start..embd_end].to_vec();
            }
        }

        forward_pass(
            self.gguf,
            self.streamer,
            self.layer_cache,
            &mut self.kv_cache,
            &mut self.hidden_state,
            self.position,
            &self.prefetcher,
            None,
        )?;

        self.position += 1;
        self.remaining -= 1;

        Ok(Some(token_text))
    }
}

/// Compute embeddings for input text.
///
/// Runs the full forward pass through all transformer layers, then applies
/// final RMS norm. Returns the last-token hidden state as the embedding vector,
/// optionally L2-normalized.
pub fn embed(
    gguf: &GgufFile,
    streamer: &SsdStreamer,
    layer_cache: &mut LayerCache,
    text: &str,
    normalize: bool,
) -> Result<EmbeddingResult> {
    let n_embd = gguf.n_embd() as usize;
    let n_layers = gguf.n_layers();
    let n_head = gguf.n_head() as usize;
    let n_head_kv = gguf.n_head_kv() as usize;
    let head_dim = n_embd / n_head;
    let n_ctx = gguf.n_ctx() as usize;

    if n_embd == 0 || n_layers == 0 {
        bail!("Invalid model: n_embd={}, n_layers={}", n_embd, n_layers);
    }

    let prefetcher = Prefetcher::new(PrefetchStrategy::LookAhead(2));
    let tokenizer = SimpleTokenizer::from_gguf(gguf);
    let tokens = tokenizer.encode(text);
    let prompt_tokens = tokens.len();

    let mut kv_cache = KvCache::new(n_layers as usize, n_head_kv, head_dim, n_ctx);

    // Batch prefill
    let embeddings_data = streamer
        .load_named_tensor_f32(gguf, "token_embd.weight")
        .ok();
    let mut token_embeddings: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
    for &token_id in &tokens {
        let mut emb = vec![0.0f32; n_embd];
        if let Some(ref emb_data) = embeddings_data {
            let s = token_id as usize * n_embd;
            let e = s + n_embd;
            if e <= emb_data.len() {
                emb = emb_data[s..e].to_vec();
            }
        }
        token_embeddings.push(emb);
    }

    let mut hidden_state = batch_prefill(
        gguf,
        streamer,
        layer_cache,
        &mut kv_cache,
        &token_embeddings,
        0,
        &prefetcher,
    )?;

    // Apply final RMS norm
    if let Ok(norm_w) = streamer.load_named_tensor_f32(gguf, "output_norm.weight") {
        rms_norm(&mut hidden_state, &norm_w);
    }

    // Optionally L2-normalize
    if normalize {
        let norm = hidden_state.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in &mut hidden_state {
                *x /= norm;
            }
        }
    }

    Ok(EmbeddingResult {
        embedding: hidden_state,
        prompt_tokens,
    })
}

/// Result of embedding extraction
pub struct EmbeddingResult {
    pub embedding: Vec<f32>,
    pub prompt_tokens: usize,
}

/// Create a streaming generator that yields tokens one at a time
pub fn generate_streaming<'a>(
    gguf: &'a GgufFile,
    streamer: &'a SsdStreamer,
    layer_cache: &'a mut LayerCache,
    prompt: &str,
    config: &InferenceConfig,
) -> Result<StreamingGenerator<'a>> {
    let n_embd = gguf.n_embd() as usize;
    let n_layers = gguf.n_layers();
    let n_head = gguf.n_head() as usize;
    let n_head_kv = gguf.n_head_kv() as usize;
    let head_dim = n_embd / n_head;
    let n_ctx = gguf.n_ctx() as usize;

    if n_embd == 0 || n_layers == 0 {
        anyhow::bail!("Invalid model: n_embd={}, n_layers={}", n_embd, n_layers);
    }

    let prefetcher = Prefetcher::new(PrefetchStrategy::LookAhead(2));
    let sampler = build_sampler(config);
    let tokenizer = SimpleTokenizer::from_gguf(gguf);
    let tokens = tokenizer.encode(prompt);

    let mut kv_cache = KvCache::new(n_layers as usize, n_head_kv, head_dim, n_ctx);
    let mut hidden_state = vec![0.0f32; n_embd];

    // Batch prefill — load embedding tensor once, layer-major traversal
    {
        let embeddings = streamer
            .load_named_tensor_f32(gguf, "token_embd.weight")
            .ok();
        let mut token_embeddings: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
        for &token_id in &tokens {
            let mut emb = vec![0.0f32; n_embd];
            if let Some(ref emb_data) = embeddings {
                let s = token_id as usize * n_embd;
                let e = s + n_embd;
                if e <= emb_data.len() {
                    emb = emb_data[s..e].to_vec();
                }
            }
            token_embeddings.push(emb);
        }

        hidden_state = batch_prefill(
            gguf,
            streamer,
            layer_cache,
            &mut kv_cache,
            &token_embeddings,
            0,
            &prefetcher,
        )?;
    }

    Ok(StreamingGenerator {
        gguf,
        streamer,
        layer_cache,
        kv_cache,
        hidden_state,
        tokenizer,
        sampler,
        prefetcher,
        position: tokens.len(),
        remaining: config.max_tokens,
        done: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert!((config.temperature - 0.7).abs() < 1e-6);
        assert_eq!(config.top_k, 40);
        assert!((config.top_p - 0.9).abs() < 1e-6);
        assert_eq!(config.max_tokens, 512);
        assert!((config.min_p - 0.0).abs() < 1e-6);
        assert!(config.seed.is_none());
        assert!((config.repetition_penalty - 1.0).abs() < 1e-6);
        assert_eq!(config.mirostat, 0);
    }

    #[test]
    fn test_build_sampler_with_min_p() {
        let config = InferenceConfig {
            min_p: 0.1,
            ..Default::default()
        };
        // Should not panic
        let sampler = build_sampler(&config);
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let token = sampler.sample(&logits);
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn test_build_sampler_with_seed_deterministic() {
        let config = InferenceConfig {
            seed: Some(42),
            temperature: 1.0,
            ..Default::default()
        };
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let s1 = build_sampler(&config);
        let s2 = build_sampler(&config);
        let t1: Vec<u32> = (0..10).map(|_| s1.sample(&logits)).collect();
        let t2: Vec<u32> = (0..10).map(|_| s2.sample(&logits)).collect();
        assert_eq!(t1, t2, "Seeded build_sampler should be deterministic");
    }

    #[test]
    fn test_build_sampler_mirostat_mode() {
        let config = InferenceConfig {
            mirostat: 2,
            ..Default::default()
        };
        let sampler = build_sampler(&config);
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let token = sampler.sample(&logits);
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn test_build_sampler_tfs() {
        let config = InferenceConfig {
            tfs_z: 0.95,
            ..Default::default()
        };
        let sampler = build_sampler(&config);
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let token = sampler.sample(&logits);
        assert!((token as usize) < logits.len());
    }
}
