//! Multi-Head Attention with KV Cache support
//!
//! v0.3: Full KV-cached attention for efficient autoregressive generation.
//! Supports Grouped-Query Attention (GQA) where n_head > n_head_kv.

use crate::inference::kv_cache::LayerKvCache;
use crate::metal::compute::{
    matvec_f32_simd, matvec_quantized_cpu, softmax_f32_fast, MetalCompute,
};
use crate::model::gguf::GgmlType;

/// Compute multi-head attention with KV cache
///
/// For each new token position:
///   1. Project input to Q, K, V
///   2. Apply RoPE to Q and K
///   3. Append K, V to the cache
///   4. Compute attention scores against all cached K
///   5. Apply softmax and weight cached V
///   6. Output projection
pub fn multi_head_attention_cached(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    position: usize,
    kv_cache: &mut LayerKvCache,
) -> Vec<f32> {
    let n_embd = x.len();
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;

    // Project: Q = x @ Wq, K = x @ Wk, V = x @ Wv
    let mut q = matvec_f32_simd(wq, x, q_dim, n_embd);
    let mut k = matvec_f32_simd(wk, x, kv_dim, n_embd);
    let v = matvec_f32_simd(wv, x, kv_dim, n_embd);

    // Apply RoPE to Q and K
    apply_rope_inplace(&mut q, head_dim, n_head, position);
    apply_rope_inplace(&mut k, head_dim, n_head_kv, position);

    // Append K, V to cache
    kv_cache.append(k, v);

    let seq_len = kv_cache.seq_len();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_group_size = (n_head / n_head_kv).max(1);

    let mut attn_output = vec![0.0f32; n_embd];

    // Process each query head
    for h in 0..n_head {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        // Compute attention scores against all cached keys
        let mut scores = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let k_head = kv_cache.key_at(pos, kv_h);
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_head[d];
            }
            scores.push(dot * scale);
        }

        // Apply causal mask (not needed since we only have past + current positions)
        // Apply softmax
        softmax_f32_fast(&mut scores);

        // Weighted sum of values
        let out_offset = h * head_dim;
        for d in 0..head_dim {
            let mut weighted = 0.0f32;
            for (pos, &score) in scores.iter().enumerate().take(seq_len) {
                let v_head = kv_cache.value_at(pos, kv_h);
                weighted += score * v_head[d];
            }
            if out_offset + d < attn_output.len() {
                attn_output[out_offset + d] = weighted;
            }
        }
    }

    // Output projection: attn_output @ Wo
    matvec_f32_simd(wo, &attn_output, n_embd, n_embd)
}

/// GPU-accelerated multi-head attention with fused QKV projection.
/// Uses Metal GPU to compute Q, K, V in a single dispatch when available.
/// Falls back to CPU when Metal is not present.
pub fn multi_head_attention_gpu(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    position: usize,
    kv_cache: &mut LayerKvCache,
    metal: Option<&MetalCompute>,
) -> Vec<f32> {
    let n_embd = x.len();
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;

    // Fused QKV projection: 1 GPU dispatch instead of 3
    let (mut q, mut k, v) = match metal {
        Some(mc) => mc.fused_qkv_f32(wq, wk, wv, x, q_dim, kv_dim, n_embd),
        None => {
            let q = matvec_f32_simd(wq, x, q_dim, n_embd);
            let k = matvec_f32_simd(wk, x, kv_dim, n_embd);
            let v = matvec_f32_simd(wv, x, kv_dim, n_embd);
            (q, k, v)
        }
    };

    // Apply RoPE to Q and K
    apply_rope_inplace(&mut q, head_dim, n_head, position);
    apply_rope_inplace(&mut k, head_dim, n_head_kv, position);

    // Append K, V to cache
    kv_cache.append(k, v);

    let seq_len = kv_cache.seq_len();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_group_size = (n_head / n_head_kv).max(1);

    let mut attn_output = vec![0.0f32; n_embd];

    for h in 0..n_head {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        let mut scores = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let k_head = kv_cache.key_at(pos, kv_h);
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_head[d];
            }
            scores.push(dot * scale);
        }

        softmax_f32_fast(&mut scores);

        let out_offset = h * head_dim;
        for d in 0..head_dim {
            let mut weighted = 0.0f32;
            for (pos, &score) in scores.iter().enumerate().take(seq_len) {
                let v_head = kv_cache.value_at(pos, kv_h);
                weighted += score * v_head[d];
            }
            if out_offset + d < attn_output.len() {
                attn_output[out_offset + d] = weighted;
            }
        }
    }

    matvec_f32_simd(wo, &attn_output, n_embd, n_embd)
}

/// GPU-accelerated multi-head attention with fused QKV + RoPE.
/// Computes Q, K, V projections AND applies RoPE in a single GPU dispatch,
/// eliminating 2 separate RoPE dispatches per layer per token.
/// Falls back to CPU when Metal is not present.
pub fn multi_head_attention_fused_rope(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    position: usize,
    kv_cache: &mut LayerKvCache,
    metal: Option<&MetalCompute>,
    theta_base: f32,
) -> Vec<f32> {
    let n_embd = x.len();
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;

    // Fused QKV + RoPE: 1 GPU dispatch instead of 3 (QKV) + 2 (RoPE Q, K)
    let (q, k, v) = match metal {
        Some(mc) => mc.fused_qkv_rope_f32(
            wq, wk, wv, x, q_dim, kv_dim, n_embd, head_dim, position, theta_base,
        ),
        None => crate::metal::compute::fused_qkv_rope_f32_cpu(
            wq, wk, wv, x, q_dim, kv_dim, n_embd, head_dim, position, theta_base,
        ),
    };

    // Append K, V to cache (already RoPE'd)
    kv_cache.append(k, v);

    let seq_len = kv_cache.seq_len();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_group_size = (n_head / n_head_kv).max(1);

    let mut attn_output = vec![0.0f32; n_embd];

    for h in 0..n_head {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        let mut scores = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let k_head = kv_cache.key_at(pos, kv_h);
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_head[d];
            }
            scores.push(dot * scale);
        }

        softmax_f32_fast(&mut scores);

        let out_offset = h * head_dim;
        for d in 0..head_dim {
            let mut weighted = 0.0f32;
            for (pos, &score) in scores.iter().enumerate().take(seq_len) {
                let v_head = kv_cache.value_at(pos, kv_h);
                weighted += score * v_head[d];
            }
            if out_offset + d < attn_output.len() {
                attn_output[out_offset + d] = weighted;
            }
        }
    }

    matvec_f32_simd(wo, &attn_output, n_embd, n_embd)
}

/// Zero-copy quantized attention: QKV + output projections run directly on raw mmap bytes.
/// No f32 weight allocations — only small intermediate vectors (Q, K, V, attention output).
/// `q_dim`, `kv_dim` are derived from actual tensor shapes (not n_head * head_dim).
pub fn multi_head_attention_quantized(
    x: &[f32],
    wq_raw: &[u8],
    wq_dtype: GgmlType,
    wk_raw: &[u8],
    wk_dtype: GgmlType,
    wv_raw: &[u8],
    wv_dtype: GgmlType,
    wo_raw: &[u8],
    wo_dtype: GgmlType,
    n_head: usize,
    n_head_kv: usize,
    q_dim: usize,
    kv_dim: usize,
    position: usize,
    kv_cache: &mut LayerKvCache,
    theta_base: f32,
    q_norm_weight: Option<&[f32]>,
    k_norm_weight: Option<&[f32]>,
    rms_eps: f32,
) -> Vec<f32> {
    let n_embd = x.len();
    let head_dim = q_dim / n_head;
    let kv_head_dim = if n_head_kv > 0 {
        kv_dim / n_head_kv
    } else {
        head_dim
    };

    // QKV projections via quantized matvec (zero-copy from mmap)
    let mut q = matvec_quantized_cpu(wq_raw, x, q_dim, n_embd, wq_dtype);
    let mut k = matvec_quantized_cpu(wk_raw, x, kv_dim, n_embd, wk_dtype);
    let v = matvec_quantized_cpu(wv_raw, x, kv_dim, n_embd, wv_dtype);

    // GROUND-TRUTH DUMP: raw Q/K/V (pre QK-norm, pre-RoPE) — first call only
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static QKV_DUMPED: AtomicBool = AtomicBool::new(false);
        if std::env::var("SSD_DUMP_QKV").is_ok() && !QKV_DUMPED.swap(true, Ordering::Relaxed) {
            let prefix = std::env::var("SSD_DUMP_QKV").unwrap_or_else(|_| "ssd_llm".to_string());
            let write_bin = |name: &str, data: &[f32]| {
                let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                let path = format!("{}_{}.bin", prefix, name);
                match std::fs::write(&path, &bytes) {
                    Ok(()) => tracing::info!("SSD_DUMP_QKV: wrote {} floats to {}", data.len(), path),
                    Err(e) => tracing::warn!("SSD_DUMP_QKV {}: {}", path, e),
                }
            };
            write_bin("Qraw", &q);
            write_bin("Kraw", &k);
            write_bin("Vraw", &v);
            write_bin("attn_input", x);
        }
    }

    // Apply QK-Norm if present (Qwen3: RMSNorm per-head before RoPE)
    if let Some(qnw) = q_norm_weight {
        for h in 0..n_head {
            let start = h * head_dim;
            let end = start + head_dim;
            if end <= q.len() {
                let slice = &mut q[start..end];
                let n = slice.len();
                let sum_sq: f32 = slice.iter().map(|v| v * v).sum();
                let inv_rms = 1.0 / (sum_sq / n as f32 + rms_eps).sqrt();
                for (i, val) in slice.iter_mut().enumerate() {
                    *val *= inv_rms * qnw.get(i).copied().unwrap_or(1.0);
                }
            }
        }
    }
    if let Some(knw) = k_norm_weight {
        for h in 0..n_head_kv {
            let start = h * kv_head_dim;
            let end = start + kv_head_dim;
            if end <= k.len() {
                let slice = &mut k[start..end];
                let n = slice.len();
                let sum_sq: f32 = slice.iter().map(|v| v * v).sum();
                let inv_rms = 1.0 / (sum_sq / n as f32 + rms_eps).sqrt();
                for (i, val) in slice.iter_mut().enumerate() {
                    *val *= inv_rms * knw.get(i).copied().unwrap_or(1.0);
                }
            }
        }
    }

    // Apply RoPE (use min of head_dim and kv_head_dim for RoPE dimension)
    let rope_dim = head_dim.min(kv_head_dim);
    crate::metal::compute::rope_f32_multi_head(&mut q, rope_dim, n_head, position, theta_base);
    crate::metal::compute::rope_f32_multi_head(
        &mut k,
        rope_dim,
        n_head_kv,
        position,
        theta_base,
    );

    // Derive actual n_head_kv from tensor dims (Gemma 4 varies per layer)
    let actual_n_head_kv = if head_dim > 0 { kv_dim / head_dim } else { n_head_kv };

    // Append K, V to cache
    kv_cache.append(k, v);

    let seq_len = kv_cache.seq_len();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_group_size = (n_head / actual_n_head_kv).max(1);

    let mut attn_output = vec![0.0f32; q_dim];

    // DIAGNOSTIC: dump attention internals for debugging
    static DIAG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let diag = DIAG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2; // first 2 calls only

    if diag {
        tracing::info!(
            "ATTN DIAG: n_head={} n_head_kv={} head_dim={} kv_head_dim={} q_dim={} kv_dim={} kv_group={} seq_len={} scale={:.6} rope_dim={}",
            n_head, n_head_kv, head_dim, kv_head_dim, q_dim, kv_dim, kv_group_size, seq_len, scale, head_dim.min(kv_head_dim),
        );
        // Q head 0 post-RoPE (first 8 values)
        tracing::info!("  Q[head0][0..8] = [{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}]",
            q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
        // K head 0 from cache (first 8 values)
        if seq_len > 0 {
            let k0 = kv_cache.key_at(seq_len - 1, 0);
            tracing::info!("  K[head0][0..8] = [{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}] (len={})",
                k0[0], k0[1], k0[2], k0[3], k0[4], k0[5], k0[6], k0[7], k0.len());
            // V head 0
            let v0 = kv_cache.value_at(seq_len - 1, 0);
            tracing::info!("  V[head0][0..8] = [{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}] (len={})",
                v0[0], v0[1], v0[2], v0[3], v0[4], v0[5], v0[6], v0[7], v0.len());
        }
        // GQA mapping
        let gqa_map: Vec<usize> = (0..n_head).map(|h| h / kv_group_size).collect();
        tracing::info!("  GQA map: {:?}", gqa_map);
    }

    for h in 0..n_head {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        let mut scores = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let k_head = kv_cache.key_at(pos, kv_h);
            assert_eq!(k_head.len(), head_dim, "K head dim mismatch: {} vs {}", k_head.len(), head_dim);
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_head[d];
            }
            scores.push(dot * scale);
        }

        // DIAGNOSTIC: pre-softmax scores for head 0
        if diag && h == 0 {
            tracing::info!("  Pre-softmax scores head 0: {:?}", &scores);
        }

        softmax_f32_fast(&mut scores);

        if diag && h == 0 {
            tracing::info!("  Post-softmax weights head 0: {:?}", &scores);
        }

        let out_offset = h * head_dim;
        for d in 0..head_dim {
            let mut weighted = 0.0f32;
            for (pos, &score) in scores.iter().enumerate().take(seq_len) {
                let v_head = kv_cache.value_at(pos, kv_h);
                weighted += score * v_head[d];
            }
            attn_output[out_offset + d] = weighted;
        }
    }

    if diag {
        // Attention output before Wo projection (first 8 values = head 0 output)
        tracing::info!("  attn_output pre-Wo[0..8] = [{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}]",
            attn_output[0], attn_output[1], attn_output[2], attn_output[3],
            attn_output[4], attn_output[5], attn_output[6], attn_output[7]);
    }

    // Output projection: maps from q_dim back to n_embd
    let result = matvec_quantized_cpu(wo_raw, &attn_output, n_embd, q_dim, wo_dtype);

    if diag {
        tracing::info!("  attn_output post-Wo[0..8] = [{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}]",
            result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]);
    }

    result
}

/// Legacy single-token attention (no cache, kept for compatibility)
pub fn multi_head_attention(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    position: usize,
) -> Vec<f32> {
    let n_embd = x.len();
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;

    let mut q = matvec_f32_simd(wq, x, q_dim, n_embd);
    let mut k = matvec_f32_simd(wk, x, kv_dim, n_embd);
    let v = matvec_f32_simd(wv, x, kv_dim, n_embd);

    apply_rope_inplace(&mut q, head_dim, n_head, position);
    apply_rope_inplace(&mut k, head_dim, n_head_kv, position);

    let _scale = 1.0 / (head_dim as f32).sqrt();
    let kv_group_size = (n_head / n_head_kv).max(1);

    let mut attn_output = vec![0.0f32; n_embd];

    for h in 0..n_head {
        let kv_h = h / kv_group_size;
        let _q_start = h * head_dim;
        let _k_start = kv_h * head_dim;
        let v_start = kv_h * head_dim;

        // Single-token: softmax of one value = 1.0 → output = V
        for d in 0..head_dim {
            let out_idx = h * head_dim + d;
            if out_idx < attn_output.len() {
                attn_output[out_idx] = v.get(v_start + d).copied().unwrap_or(0.0);
            }
        }
    }

    matvec_f32_simd(wo, &attn_output, n_embd, n_embd)
}

/// Apply RoPE in-place across all heads
/// Apply RoPE in-place across all heads (standard non-NeoX layout)
fn apply_rope_inplace(x: &mut [f32], head_dim: usize, n_heads: usize, position: usize) {
    let theta_base = 10000.0f32;
    let half_dim = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half_dim {
            let freq = 1.0 / theta_base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let (sin_val, cos_val) = angle.sin_cos();

            let idx0 = base + i;
            let idx1 = base + i + half_dim;
            if idx1 < x.len() {
                let x0 = x[idx0];
                let x1 = x[idx1];
                x[idx0] = x0 * cos_val - x1 * sin_val;
                x[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::LayerKvCache;

    #[test]
    fn test_rope_deterministic() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let mut b = a.clone();
        apply_rope_inplace(&mut a, 4, 1, 5);
        apply_rope_inplace(&mut b, 4, 1, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn test_rope_position_zero_identity() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let orig = x.clone();
        apply_rope_inplace(&mut x, 4, 1, 0);
        // At position 0, angle = 0, so cos=1, sin=0 → identity
        for i in 0..4 {
            assert!(
                (x[i] - orig[i]).abs() < 1e-6,
                "RoPE at pos 0 should be identity"
            );
        }
    }

    #[test]
    fn test_gpu_attention_matches_cpu() {
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let x = vec![1.0f32; n_embd];
        let w = vec![0.01f32; n_embd * n_embd];

        // CPU path
        let mut kv_cpu = LayerKvCache::new(n_head_kv, head_dim);
        let out_cpu = multi_head_attention_cached(
            &x,
            &w,
            &w,
            &w,
            &w,
            n_head,
            n_head_kv,
            head_dim,
            0,
            &mut kv_cpu,
        );

        // GPU path (with None metal = CPU fallback, same result)
        let mut kv_gpu = LayerKvCache::new(n_head_kv, head_dim);
        let out_gpu = multi_head_attention_gpu(
            &x,
            &w,
            &w,
            &w,
            &w,
            n_head,
            n_head_kv,
            head_dim,
            0,
            &mut kv_gpu,
            None,
        );

        for (i, (a, b)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "Mismatch at {i}: cpu={a}, gpu={b}");
        }
    }

    #[test]
    fn test_gpu_attention_with_metal() {
        let mc = MetalCompute::new();
        if mc.is_none() {
            return; // skip on non-macOS
        }
        let mc = mc.unwrap();

        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let x = vec![0.5f32; n_embd];
        let w = vec![0.02f32; n_embd * n_embd];

        let mut kv_cpu = LayerKvCache::new(n_head_kv, head_dim);
        let out_cpu = multi_head_attention_cached(
            &x,
            &w,
            &w,
            &w,
            &w,
            n_head,
            n_head_kv,
            head_dim,
            0,
            &mut kv_cpu,
        );

        let mut kv_gpu = LayerKvCache::new(n_head_kv, head_dim);
        let out_gpu = multi_head_attention_gpu(
            &x,
            &w,
            &w,
            &w,
            &w,
            n_head,
            n_head_kv,
            head_dim,
            0,
            &mut kv_gpu,
            Some(&mc),
        );

        for (i, (a, b)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "Metal mismatch at {i}: cpu={a}, gpu={b}"
            );
        }
    }

    #[test]
    fn test_cached_attention_grows() {
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let mut kv = LayerKvCache::new(n_head_kv, head_dim);

        let x = vec![1.0f32; n_embd];
        let w = vec![0.01f32; n_embd * n_embd]; // simple weight

        let _ = multi_head_attention_cached(
            &x, &w, &w, &w, &w, n_head, n_head_kv, head_dim, 0, &mut kv,
        );
        assert_eq!(kv.seq_len(), 1);

        let _ = multi_head_attention_cached(
            &x, &w, &w, &w, &w, n_head, n_head_kv, head_dim, 1, &mut kv,
        );
        assert_eq!(kv.seq_len(), 2);
    }

    #[test]
    fn test_fused_rope_matches_separate() {
        // Fused QKV+RoPE should produce the same result as separate QKV + RoPE
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let position = 5;
        let theta_base = 10000.0f32;

        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let wq: Vec<f32> = (0..n_embd * n_embd)
            .map(|i| ((i * 7 + 3) as f32 % 11.0 - 5.0) * 0.01)
            .collect();
        let wk: Vec<f32> = (0..n_embd * n_embd)
            .map(|i| ((i * 13 + 7) as f32 % 11.0 - 5.0) * 0.01)
            .collect();
        let wv: Vec<f32> = (0..n_embd * n_embd)
            .map(|i| ((i * 17 + 11) as f32 % 11.0 - 5.0) * 0.01)
            .collect();

        let q_dim = n_head * head_dim;
        let kv_dim = n_head_kv * head_dim;

        // Separate path
        let mut q_sep = crate::metal::compute::matvec_f32_simd(&wq, &x, q_dim, n_embd);
        let mut k_sep = crate::metal::compute::matvec_f32_simd(&wk, &x, kv_dim, n_embd);
        let v_sep = crate::metal::compute::matvec_f32_simd(&wv, &x, kv_dim, n_embd);
        crate::metal::compute::rope_f32_multi_head(
            &mut q_sep, head_dim, n_head, position, theta_base,
        );
        crate::metal::compute::rope_f32_multi_head(
            &mut k_sep, head_dim, n_head_kv, position, theta_base,
        );

        // Fused path
        let (q_fused, k_fused, v_fused) = crate::metal::compute::fused_qkv_rope_f32_cpu(
            &wq, &wk, &wv, &x, q_dim, kv_dim, n_embd, head_dim, position, theta_base,
        );

        for i in 0..q_dim {
            assert!(
                (q_sep[i] - q_fused[i]).abs() < 1e-5,
                "Q mismatch at {}: {} vs {}",
                i,
                q_sep[i],
                q_fused[i]
            );
        }
        for i in 0..kv_dim {
            assert!(
                (k_sep[i] - k_fused[i]).abs() < 1e-5,
                "K mismatch at {}: {} vs {}",
                i,
                k_sep[i],
                k_fused[i]
            );
        }
        for i in 0..kv_dim {
            assert!(
                (v_sep[i] - v_fused[i]).abs() < 1e-5,
                "V should be unchanged: {} vs {}",
                v_sep[i],
                v_fused[i]
            );
        }
    }

    #[test]
    fn test_fused_rope_position_zero_identity() {
        // At position 0, RoPE should be identity → fused result = plain QKV
        let n_embd = 8;
        let head_dim = 4;
        let q_dim = 8;
        let kv_dim = 8;

        let x: Vec<f32> = (0..n_embd).map(|i| i as f32 * 0.1 + 0.5).collect();
        let w: Vec<f32> = (0..n_embd * n_embd)
            .map(|i| ((i * 3 + 1) as f32 % 7.0 - 3.0) * 0.01)
            .collect();

        let (q_plain, k_plain, v_plain) =
            crate::metal::compute::fused_qkv_f32_cpu(&w, &w, &w, &x, q_dim, kv_dim, n_embd);
        let (q_rope, k_rope, v_rope) = crate::metal::compute::fused_qkv_rope_f32_cpu(
            &w, &w, &w, &x, q_dim, kv_dim, n_embd, head_dim, 0, 10000.0,
        );

        for i in 0..q_dim {
            assert!(
                (q_plain[i] - q_rope[i]).abs() < 1e-6,
                "Q at pos 0 should match plain: {} vs {}",
                q_plain[i],
                q_rope[i]
            );
        }
        for i in 0..kv_dim {
            assert!((k_plain[i] - k_rope[i]).abs() < 1e-6);
            assert!((v_plain[i] - v_rope[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fused_rope_attention_matches_gpu() {
        // multi_head_attention_fused_rope with no Metal should match separate path
        let n_embd = 8;
        let n_head = 2;
        let n_head_kv = 2;
        let head_dim = 4;
        let position = 3;

        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let wq: Vec<f32> = (0..n_embd * n_embd)
            .map(|i| ((i * 7 + 3) as f32 % 11.0 - 5.0) * 0.01)
            .collect();
        let wk: Vec<f32> = (0..n_embd * n_embd)
            .map(|i| ((i * 13 + 7) as f32 % 11.0 - 5.0) * 0.01)
            .collect();
        let wv: Vec<f32> = (0..n_embd * n_embd)
            .map(|i| ((i * 17 + 11) as f32 % 11.0 - 5.0) * 0.01)
            .collect();
        let wo: Vec<f32> = (0..n_embd * n_embd)
            .map(|i| ((i * 19 + 5) as f32 % 11.0 - 5.0) * 0.01)
            .collect();

        let mut kv1 = LayerKvCache::new(n_head_kv, head_dim);
        let mut kv2 = LayerKvCache::new(n_head_kv, head_dim);

        let out_gpu = multi_head_attention_gpu(
            &x, &wq, &wk, &wv, &wo, n_head, n_head_kv, head_dim, position, &mut kv1, None,
        );
        let out_fused = multi_head_attention_fused_rope(
            &x, &wq, &wk, &wv, &wo, n_head, n_head_kv, head_dim, position, &mut kv2, None, 10000.0,
        );

        // Both should produce same results since both use theta=10000 RoPE
        for i in 0..n_embd {
            assert!(
                (out_gpu[i] - out_fused[i]).abs() < 1e-4,
                "Attention output mismatch at {}: {} vs {}",
                i,
                out_gpu[i],
                out_fused[i]
            );
        }
    }

    #[test]
    fn test_fused_rope_gpu_matches_cpu() {
        let mc = MetalCompute::new();
        if mc.is_none() || !mc.as_ref().unwrap().has_gpu() {
            return; // skip if no GPU
        }
        let mc = mc.unwrap();

        let n_embd = 128;
        let n_head = 4;
        let n_head_kv = 4;
        let head_dim = 32;
        let q_dim = n_head * head_dim;
        let kv_dim = n_head_kv * head_dim;
        let position = 42;
        let theta_base = 10000.0f32;

        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.01).sin()).collect();
        let wq: Vec<f32> = (0..q_dim * n_embd)
            .map(|i| ((i * 7 + 3) as f32 % 13.0 - 6.0) * 0.01)
            .collect();
        let wk: Vec<f32> = (0..kv_dim * n_embd)
            .map(|i| ((i * 11 + 5) as f32 % 13.0 - 6.0) * 0.01)
            .collect();
        let wv: Vec<f32> = (0..kv_dim * n_embd)
            .map(|i| ((i * 13 + 7) as f32 % 13.0 - 6.0) * 0.01)
            .collect();

        let (q_cpu, k_cpu, v_cpu) = crate::metal::compute::fused_qkv_rope_f32_cpu(
            &wq, &wk, &wv, &x, q_dim, kv_dim, n_embd, head_dim, position, theta_base,
        );
        let (q_gpu, k_gpu, v_gpu) = mc.fused_qkv_rope_f32(
            &wq, &wk, &wv, &x, q_dim, kv_dim, n_embd, head_dim, position, theta_base,
        );

        for i in 0..q_dim {
            assert!(
                (q_cpu[i] - q_gpu[i]).abs() < 0.02,
                "Q GPU mismatch at {}: cpu={} gpu={}",
                i,
                q_cpu[i],
                q_gpu[i]
            );
        }
        for i in 0..kv_dim {
            assert!(
                (k_cpu[i] - k_gpu[i]).abs() < 0.02,
                "K GPU mismatch at {}: cpu={} gpu={}",
                i,
                k_cpu[i],
                k_gpu[i]
            );
        }
        for i in 0..kv_dim {
            assert!(
                (v_cpu[i] - v_gpu[i]).abs() < 0.02,
                "V GPU mismatch at {}: cpu={} gpu={}",
                i,
                v_cpu[i],
                v_gpu[i]
            );
        }
    }
}
