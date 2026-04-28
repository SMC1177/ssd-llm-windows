/// CUDA GPU backend for ssd-llm on Windows.
///
/// v1.4: Basic CUDA dispatch with per-call alloc/free.
/// v1.5: Pre-allocated device/host buffers (keyed by dimension) eliminate
///       cuMemAlloc/cuMemFree overhead. sgemv_gate_up uploads x once for
///       both gate and up projections instead of twice.
/// v1.9: ffn_fused — on-device silu+hadamard kernel eliminates 3 PCIe
///       round-trips per FFN layer (gate dtoh + up dtoh + inter htod).
///
/// On non-Windows platforms this compiles to a no-op stub.

#[cfg(target_os = "windows")]
mod windows_impl {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::Arc;

    use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
    use cudarc::cublas::{CudaBlas, Gemv, GemvConfig};
    use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::compile_ptx;
    use tracing::info;

    // NVRTC kernel: silu(gate) * up → out, all on device.
    // Eliminates download of gate/up and re-upload of intermediate per FFN layer.
    const SILU_HAD_SRC: &str = r#"
extern "C" __global__ void silu_hadamard(float* gate, const float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + __expf(-g))) * up[i];
    }
}
"#;

    // RMS norm: out[i] = x[i] / sqrt(mean(x²)+eps) * w[i]
    // Grid: (1,1,1), Block: (256,1,1), Shared: 256×4 bytes
    const RMS_NORM_SRC: &str = r#"
extern "C" __global__ void rms_norm_kernel(
    float* out, const float* x, const float* w, int n, float eps
) {
    extern __shared__ float s[];
    int t = threadIdx.x, B = blockDim.x;
    float ss = 0.0f;
    for (int i = t; i < n; i += B) { float v = x[i]; ss += v * v; }
    s[t] = ss; __syncthreads();
    for (int stride = B >> 1; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride]; __syncthreads();
    }
    float scale = rsqrtf(s[0] / (float)n + eps);
    for (int i = t; i < n; i += B) out[i] = x[i] * scale * w[i];
}
"#;

    // Elementwise add in-place: x[i] += y[i]
    // Grid: (ceil(n/256),1,1), Block: (256,1,1)
    const ADD_INPLACE_SRC: &str = r#"
extern "C" __global__ void add_inplace_kernel(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
"#;

    // Q4K fused-dequant SGEMV: y[out_dim] = W[Q4K, out×in] · x[in_dim]
    // grid=(out_dim,1,1), block=(256,1,1), shared=1024B (parallel reduction)
    // Q4K block layout (144B / 256 values):
    //   [0..1] d(f16LE)  [2..3] dmin(f16LE)  [4..15] scales[12](u8)  [16..143] qs[128](u8)
    // 8 sub-blocks of 32 values; thread tid → sb=tid/32, l=tid%32.
    // Requires in_dim % 256 == 0.
    // grid=(out_dim,1,1), block=(32,1,1) — 1 warp per output row.
    // Warp-shuffle reduction: no shared memory, no syncthreads, 6× fewer scheduling waves.
    // Each thread handles 8 values per Q4K block (256 values / 32 lanes).
    // Requires in_dim % 256 == 0.
    const SGEMV_Q4K_SRC: &str = r#"
__device__ __forceinline__ float f16_le_to_f32(const unsigned char* p) {
    unsigned int h = (unsigned int)p[0] | ((unsigned int)p[1] << 8u);
    unsigned int sign = (h & 0x8000u) << 16;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int bits;
    if      (exp == 0u)  bits = sign;
    else if (exp == 31u) bits = sign | 0x7F800000u | (mant << 13);
    else                 bits = sign | ((exp + 112u) << 23) | (mant << 13);
    float f; asm("mov.b32 %0, %1;" : "=f"(f) : "r"(bits)); return f;
}
// grid=(out_dim/32,1,1), block=(1024,1,1), shared=in_dim*4 bytes.
// 32 rows per block (1 warp per row). x loaded into shared memory once per block,
// amortizing DRAM reads across 32 rows — reduces x bandwidth from 49MB to 1.5MB
// per SGEMV vs the 1-warp-per-block design.
extern "C" __global__ void sgemv_q4k_kernel(
    float* __restrict__ y,
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    int in_dim
) {
    extern __shared__ float xs[];   // in_dim floats, shared by all 32 rows in block

    int warp_id = (int)(threadIdx.x >> 5);  // 0..31 = row within block
    int lane    = (int)(threadIdx.x & 31);  // 0..31 = lane within warp
    int row     = (int)blockIdx.x * 32 + warp_id;
    int bpr     = in_dim >> 8;              // in_dim / 256

    // Cooperatively load x into shared memory (all 1024 threads participate).
    // Must happen before row-bounds check so __syncthreads() is reached by all.
    for (int i = (int)threadIdx.x; i < in_dim; i += 1024) {
        xs[i] = x[i];
    }
    __syncthreads();

    const unsigned char* row_w = w + (long long)row * bpr * 144;
    float partial = 0.0f;

    for (int b = 0; b < bpr; b++) {
        const unsigned char* blk = row_w + b * 144;
        float d    = f16_le_to_f32(blk);
        float dmin = f16_le_to_f32(blk + 2);
        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;
        for (int j = 0; j < 8; j++) {
            int elem = lane + j * 32;   // 0..255
            int sb   = elem >> 5;       // sub-block 0..7
            int l    = elem & 31;       // position within sub-block
            unsigned int sc_low, m_low;
            if (sb < 4) {
                sc_low = sc[sb] & 0x3Fu;
                m_low  = sc[sb + 4] & 0x3Fu;
            } else {
                int si4 = sb - 4;
                sc_low = (sc[sb + 4] & 0x0Fu) | (((unsigned int)sc[si4] >> 6) << 4);
                m_low  = ((unsigned int)sc[sb + 4] >> 4) | (((unsigned int)sc[sb] >> 6) << 4);
            }
            float scale   = d * (float)sc_low;
            float min_val = dmin * (float)m_low;
            int qs_off = (sb >> 1) * 32;
            unsigned int byte_val = qs[qs_off + l];
            unsigned int nibble   = (sb & 1) ? (byte_val >> 4) : (byte_val & 0x0Fu);
            partial += (scale * (float)nibble - min_val) * xs[b * 256 + elem];
        }
    }
    // Warp-shuffle reduction
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 16);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 8);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 4);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 2);
    partial += __shfl_down_sync(0xFFFFFFFF, partial, 1);
    if (lane == 0) y[row] = partial;
}
"#;

    // Fused attention kernel: one launch replaces qk_norm(Q+K) + rope(Q+K) + kv_append + attn_compute.
    // Saves 4 extra WDDM launches vs 5-separate-kernels (each ~50µs on Windows WDDM).
    //
    // Grid: (n_head, 1, 1), block: (head_dim, 1, 1)
    // Derivations inside kernel (reduces param count for cudarc tuple limit):
    //   head_dim = blockDim.x
    //   n_head_kv = gridDim.x / kv_group_size
    //   scale = rsqrtf((float)head_dim)
    //   position = seq_pos  (true during single-token decode; both advance together)
    //   do_qk_norm = (rms_eps > 0.0f)
    //
    // Shared layout: [q_store: head_dim][scratch: head_dim][scores: seq_pos+1]
    // GQA: blocks kv_group_size apart share kv_h; both write identical K/V — idempotent race.
    const ATTN_FUSED_SRC: &str = r#"
extern "C" __global__ void attn_fused_kernel(
    float* out,
    float* kv_k, float* kv_v,
    const float* q_dev, const float* k_dev, const float* v_dev,
    const float* q_norm_w, const float* k_norm_w,
    int seq_pos,
    float rms_eps,
    float theta_base,
    int kv_group_size
) {
    int head_dim    = (int)blockDim.x;
    int n_head_kv   = (int)gridDim.x / kv_group_size;
    float scale     = rsqrtf((float)head_dim);
    int do_qk_norm  = (rms_eps > 0.0f) ? 1 : 0;

    extern __shared__ float sdata[];
    float* q_store = sdata;
    float* scratch = sdata + head_dim;
    float* scores  = sdata + 2 * head_dim;

    int h    = (int)blockIdx.x;
    int d    = (int)threadIdx.x;
    int kv_h = h / kv_group_size;
    int half = head_dim / 2;
    int new_seq_len = seq_pos + 1;

    // ── Phase 1: Q  qk_norm + rope ───────────────────────────────────────────
    float qv = q_dev[h * head_dim + d];
    if (do_qk_norm) {
        scratch[d] = qv * qv;
        __syncthreads();
        for (int s = head_dim >> 1; s > 0; s >>= 1) {
            if (d < s) scratch[d] += scratch[d + s];
            __syncthreads();
        }
        qv *= rsqrtf(scratch[0] / (float)head_dim + rms_eps) * q_norm_w[d];
        __syncthreads();
    }
    scratch[d] = qv;
    __syncthreads();
    if (d < half) {
        float freq  = powf(theta_base, -2.0f * (float)d / (float)head_dim);
        float angle = (float)seq_pos * freq;
        float c = cosf(angle), s = sinf(angle);
        float x0 = scratch[d], x1 = scratch[d + half];
        scratch[d]        = x0 * c - x1 * s;
        scratch[d + half] = x0 * s + x1 * c;
    }
    __syncthreads();
    q_store[d] = scratch[d];
    __syncthreads();

    // ── Phase 2: K  qk_norm + rope + kv_append ───────────────────────────────
    float kv = k_dev[kv_h * head_dim + d];
    if (do_qk_norm) {
        scratch[d] = kv * kv;
        __syncthreads();
        for (int s = head_dim >> 1; s > 0; s >>= 1) {
            if (d < s) scratch[d] += scratch[d + s];
            __syncthreads();
        }
        kv *= rsqrtf(scratch[0] / (float)head_dim + rms_eps) * k_norm_w[d];
        __syncthreads();
    }
    scratch[d] = kv;
    __syncthreads();
    if (d < half) {
        float freq  = powf(theta_base, -2.0f * (float)d / (float)head_dim);
        float angle = (float)seq_pos * freq;
        float c = cosf(angle), s = sinf(angle);
        float x0 = scratch[d], x1 = scratch[d + half];
        scratch[d]        = x0 * c - x1 * s;
        scratch[d + half] = x0 * s + x1 * c;
    }
    __syncthreads();
    // KV cache layout: [seq, n_head_kv, head_dim]; GQA idempotent writes are safe.
    kv_k[(seq_pos * n_head_kv + kv_h) * head_dim + d] = scratch[d];
    kv_v[(seq_pos * n_head_kv + kv_h) * head_dim + d] = v_dev[kv_h * head_dim + d];
    __threadfence();

    // ── Phase 3: attention scores (thread 0, serial over seq) ────────────────
    if (d == 0) {
        float max_s = -1e38f;
        for (int pos = 0; pos < new_seq_len; pos++) {
            const float* kh = kv_k + (pos * n_head_kv + kv_h) * head_dim;
            float dot = 0.0f;
            for (int j = 0; j < head_dim; j++) dot += q_store[j] * kh[j];
            scores[pos] = dot * scale;
            if (scores[pos] > max_s) max_s = scores[pos];
        }
        float sum = 0.0f;
        for (int pos = 0; pos < new_seq_len; pos++) {
            scores[pos] = __expf(scores[pos] - max_s);
            sum += scores[pos];
        }
        float inv = 1.0f / (sum + 1e-12f);
        for (int pos = 0; pos < new_seq_len; pos++) scores[pos] *= inv;
    }
    __syncthreads();

    // ── Phase 4: weighted-V (all threads parallel across head_dim) ───────────
    float w = 0.0f;
    for (int pos = 0; pos < new_seq_len; pos++) {
        w += scores[pos] * kv_v[(pos * n_head_kv + kv_h) * head_dim + d];
    }
    out[h * head_dim + d] = w;
}
"#;

    struct GpuTensor {
        slice: CudaSlice<f32>,
        out_dim: usize,
        in_dim: usize,
    }

    pub struct CudaGpu {
        device: Arc<CudaDevice>,
        blas: CudaBlas,
        tensors: HashMap<String, GpuTensor>,
        vram_used: usize,
        vram_budget: usize,
        /// Pre-allocated device input buffers keyed by in_dim — avoid cuMemAlloc per call.
        x_bufs: RefCell<HashMap<usize, CudaSlice<f32>>>,
        /// Pre-allocated device output buffers keyed by out_dim.
        y_bufs: RefCell<HashMap<usize, CudaSlice<f32>>>,
        /// Pre-allocated host output staging keyed by out_dim.
        h_bufs: RefCell<HashMap<usize, Vec<f32>>>,
        /// On-device buffer for UP projection result (ffn_fused).
        up_bufs: RefCell<HashMap<usize, CudaSlice<f32>>>,
        /// Compiled silu+hadamard NVRTC kernel (None if compilation failed).
        silu_had_fn: Option<CudaFunction>,
        /// On-device K buffer (K SGEMV output, before KV cache append).
        k_bufs: RefCell<HashMap<usize, CudaSlice<f32>>>,
        /// On-device KV cache K per layer: [max_seq * n_head_kv * head_dim].
        kv_k_bufs: RefCell<HashMap<usize, CudaSlice<f32>>>,
        /// On-device KV cache V per layer.
        kv_v_bufs: RefCell<HashMap<usize, CudaSlice<f32>>>,
        /// CPU-side sequence length per KV cache layer.
        kv_seq_lens: RefCell<HashMap<usize, usize>>,
        /// Maximum sequence length for on-device KV cache.
        kv_max_seq: usize,
        /// Small norm weight tensors on device (QK-norm weights, keyed by tensor name).
        norm_bufs: HashMap<String, CudaSlice<f32>>,
        /// NVRTC fused attention kernel (qk_norm+rope+kv_append+attn in one launch).
        attn_fused_kernel_fn: Option<CudaFunction>,
        /// NVRTC RMS-norm kernel (for GPU-resident forward pass).
        rms_norm_fn: Option<CudaFunction>,
        /// NVRTC elementwise add-inplace kernel.
        add_fn: Option<CudaFunction>,
        /// Persistent on-device hidden state for GPU-resident forward pass.
        hidden_dev: RefCell<Option<CudaSlice<f32>>>,
        /// NVRTC Q4K fused-dequant SGEMV kernel.
        sgemv_q4k_fn: Option<CudaFunction>,
        /// Q4K weight tensors on device (raw bytes). Key: tensor name. Value: (bytes, out_dim, in_dim).
        q4k_tensors: HashMap<String, (CudaSlice<u8>, usize, usize)>,
    }

    impl CudaGpu {
        pub fn new(vram_budget_bytes: usize) -> Option<Self> {
            let device = CudaDevice::new(0).ok()?;
            let blas = CudaBlas::new(device.clone()).ok()?;

            // Compile silu+hadamard kernel via NVRTC.
            let silu_had_fn = (|| -> Option<CudaFunction> {
                let ptx = compile_ptx(SILU_HAD_SRC).ok()?;
                device.load_ptx(ptx, "silu_had", &["silu_hadamard"]).ok()?;
                device.get_func("silu_had", "silu_hadamard")
            })();

            if silu_had_fn.is_some() {
                info!("CUDA: silu+hadamard kernel compiled (on-device FFN fusion enabled)");
            } else {
                tracing::warn!("CUDA: silu+hadamard kernel failed to compile — FFN fusion disabled");
            }

            // Compile fused attention NVRTC kernel (1 launch replaces 5 separate ops).
            let attn_fused_kernel_fn = (|| -> Option<CudaFunction> {
                let ptx = compile_ptx(ATTN_FUSED_SRC).ok()?;
                device.load_ptx(ptx, "attn_fused_k", &["attn_fused_kernel"]).ok()?;
                device.get_func("attn_fused_k", "attn_fused_kernel")
            })();
            if attn_fused_kernel_fn.is_some() {
                info!("CUDA: fused attention kernel compiled (GPU attention enabled)");
            } else {
                tracing::warn!("CUDA: fused attention kernel failed to compile — GPU attention disabled");
            }

            let rms_norm_fn = (|| -> Option<CudaFunction> {
                let ptx = compile_ptx(RMS_NORM_SRC).ok()?;
                device.load_ptx(ptx, "rms_norm_k", &["rms_norm_kernel"]).ok()?;
                device.get_func("rms_norm_k", "rms_norm_kernel")
            })();
            let add_fn = (|| -> Option<CudaFunction> {
                let ptx = compile_ptx(ADD_INPLACE_SRC).ok()?;
                device.load_ptx(ptx, "add_k", &["add_inplace_kernel"]).ok()?;
                device.get_func("add_k", "add_inplace_kernel")
            })();
            if rms_norm_fn.is_some() && add_fn.is_some() {
                info!("CUDA: rms_norm + add_inplace kernels compiled (GPU-resident forward pass enabled)");
            }

            let sgemv_q4k_fn = (|| -> Option<CudaFunction> {
                let ptx = compile_ptx(SGEMV_Q4K_SRC).ok()?;
                device.load_ptx(ptx, "sgemv_q4k", &["sgemv_q4k_kernel"]).ok()?;
                device.get_func("sgemv_q4k", "sgemv_q4k_kernel")
            })();
            if sgemv_q4k_fn.is_some() {
                info!("CUDA: Q4K SGEMV kernel compiled (on-device dequant enabled)");
            } else {
                tracing::warn!("CUDA: Q4K SGEMV kernel failed to compile");
            }

            info!(
                "CUDA GPU ready: device 0, VRAM budget {:.1} GB",
                vram_budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            Some(Self {
                device,
                blas,
                tensors: HashMap::new(),
                vram_used: 0,
                vram_budget: vram_budget_bytes,
                x_bufs: RefCell::new(HashMap::new()),
                y_bufs: RefCell::new(HashMap::new()),
                h_bufs: RefCell::new(HashMap::new()),
                up_bufs: RefCell::new(HashMap::new()),
                silu_had_fn,
                k_bufs: RefCell::new(HashMap::new()),
                kv_k_bufs: RefCell::new(HashMap::new()),
                kv_v_bufs: RefCell::new(HashMap::new()),
                kv_seq_lens: RefCell::new(HashMap::new()),
                kv_max_seq: 512,
                norm_bufs: HashMap::new(),
                attn_fused_kernel_fn,
                rms_norm_fn,
                add_fn,
                hidden_dev: RefCell::new(None),
                sgemv_q4k_fn,
                q4k_tensors: HashMap::new(),
            })
        }

        pub fn vram_remaining(&self) -> usize {
            self.vram_budget.saturating_sub(self.vram_used)
        }

        pub fn upload(
            &mut self,
            name: &str,
            data: &[f32],
            out_dim: usize,
            in_dim: usize,
        ) -> bool {
            let bytes = data.len() * 4;
            if self.vram_used + bytes > self.vram_budget {
                return false;
            }
            match self.device.htod_sync_copy(data) {
                Ok(slice) => {
                    self.vram_used += bytes;
                    self.tensors
                        .insert(name.to_string(), GpuTensor { slice, out_dim, in_dim });
                    true
                }
                Err(e) => {
                    tracing::warn!("GPU upload failed for {}: {}", name, e);
                    false
                }
            }
        }

        pub fn has(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }

        /// Allocate device/host buffers for a given (in_dim, out_dim) if not yet present.
        /// Returns false on allocation failure.
        fn ensure_bufs(&self, in_dim: usize, out_dim: usize) -> bool {
            {
                let mut xb = self.x_bufs.borrow_mut();
                if !xb.contains_key(&in_dim) {
                    match self.device.alloc_zeros::<f32>(in_dim) {
                        Ok(buf) => { xb.insert(in_dim, buf); }
                        Err(e) => {
                            tracing::warn!("GPU x-buf alloc failed (dim {}): {}", in_dim, e);
                            return false;
                        }
                    }
                }
            }
            {
                let mut yb = self.y_bufs.borrow_mut();
                if !yb.contains_key(&out_dim) {
                    match self.device.alloc_zeros::<f32>(out_dim) {
                        Ok(buf) => { yb.insert(out_dim, buf); }
                        Err(e) => {
                            tracing::warn!("GPU y-buf alloc failed (dim {}): {}", out_dim, e);
                            return false;
                        }
                    }
                }
            }
            {
                let mut hb = self.h_bufs.borrow_mut();
                if !hb.contains_key(&out_dim) {
                    hb.insert(out_dim, vec![0.0f32; out_dim]);
                }
            }
            true
        }

        /// Allocate on-device UP buffer of size `dim` (used by ffn_fused).
        fn ensure_up_buf(&self, dim: usize) -> bool {
            let mut ub = self.up_bufs.borrow_mut();
            if !ub.contains_key(&dim) {
                match self.device.alloc_zeros::<f32>(dim) {
                    Ok(buf) => { ub.insert(dim, buf); }
                    Err(e) => {
                        tracing::warn!("GPU up-buf alloc failed (dim {}): {}", dim, e);
                        return false;
                    }
                }
            }
            true
        }

        /// Allocate on-device K temp buffer (K SGEMV output before KV cache append).
        fn ensure_k_buf(&self, dim: usize) -> bool {
            let mut kb = self.k_bufs.borrow_mut();
            if !kb.contains_key(&dim) {
                match self.device.alloc_zeros::<f32>(dim) {
                    Ok(buf) => { kb.insert(dim, buf); }
                    Err(e) => {
                        tracing::warn!("GPU k-buf alloc failed (dim {}): {}", dim, e);
                        return false;
                    }
                }
            }
            true
        }

        /// Allocate on-device KV cache for a layer (lazy, reset happens via CudaGpu::new()).
        fn ensure_kv_cache(&self, layer_idx: usize, n_head_kv: usize, head_dim: usize) -> bool {
            let cache_size = self.kv_max_seq * n_head_kv * head_dim;
            {
                let mut kb = self.kv_k_bufs.borrow_mut();
                if !kb.contains_key(&layer_idx) {
                    match self.device.alloc_zeros::<f32>(cache_size) {
                        Ok(buf) => { kb.insert(layer_idx, buf); }
                        Err(e) => {
                            tracing::warn!("GPU kv_k alloc failed (layer {}): {}", layer_idx, e);
                            return false;
                        }
                    }
                }
            }
            {
                let mut vb = self.kv_v_bufs.borrow_mut();
                if !vb.contains_key(&layer_idx) {
                    match self.device.alloc_zeros::<f32>(cache_size) {
                        Ok(buf) => { vb.insert(layer_idx, buf); }
                        Err(e) => {
                            tracing::warn!("GPU kv_v alloc failed (layer {}): {}", layer_idx, e);
                            return false;
                        }
                    }
                }
            }
            self.kv_seq_lens.borrow_mut().entry(layer_idx).or_insert(0);
            true
        }

        /// Upload a small norm weight tensor to device (not counted toward VRAM budget).
        pub fn upload_norm(&mut self, name: &str, data: &[f32]) -> bool {
            match self.device.htod_sync_copy(data) {
                Ok(slice) => {
                    self.norm_bufs.insert(name.to_string(), slice);
                    true
                }
                Err(e) => {
                    tracing::warn!("GPU norm upload failed for {}: {}", name, e);
                    false
                }
            }
        }

        pub fn has_norm(&self, name: &str) -> bool {
            self.norm_bufs.contains_key(name)
        }

        /// Upload a weight tensor as raw Q4K bytes (144 bytes per 256 values).
        /// Uses ~14× less VRAM than f32; requires in_dim % 256 == 0.
        pub fn upload_q4k(&mut self, name: &str, data: &[u8], out_dim: usize, in_dim: usize) -> bool {
            debug_assert!(in_dim % 256 == 0, "Q4K requires in_dim % 256 == 0");
            let bytes = data.len();
            if self.vram_used + bytes > self.vram_budget { return false; }
            match self.device.htod_sync_copy(data) {
                Ok(slice) => {
                    self.vram_used += bytes;
                    self.q4k_tensors.insert(name.to_string(), (slice, out_dim, in_dim));
                    true
                }
                Err(e) => {
                    tracing::warn!("GPU Q4K upload failed for {}: {}", name, e);
                    false
                }
            }
        }

        /// True if the tensor was uploaded as Q4K bytes.
        pub fn has_q4k(&self, name: &str) -> bool {
            self.q4k_tensors.contains_key(name)
        }

        /// True if the tensor is on GPU as either f32 or Q4K.
        pub fn has_weight(&self, name: &str) -> bool {
            self.tensors.contains_key(name) || self.q4k_tensors.contains_key(name)
        }

        /// SGEMV using Q4K weights. x is a CPU slice (htod on entry); returns CPU logits.
        /// Falls back to None if the Q4K kernel is unavailable or tensor not found.
        pub fn sgemv_q4k(&self, name: &str, x: &[f32]) -> Option<Vec<f32>> {
            let fn_ = self.sgemv_q4k_fn.as_ref()?;
            let (_, out_dim, in_dim) = self.q4k_tensors.get(name)?;
            let (out_dim, in_dim) = (*out_dim, *in_dim);
            if !self.ensure_bufs(in_dim, out_dim) { return None; }
            self.device.htod_sync_copy_into(
                x, self.x_bufs.borrow_mut().get_mut(&in_dim).unwrap()
            ).ok()?;
            let cfg = LaunchConfig { grid_dim: (out_dim as u32 / 32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: in_dim as u32 * 4 };
            {
                let (w_q4k, _, _) = self.q4k_tensors.get(name).unwrap();
                let mut yb = self.y_bufs.borrow_mut();
                let y_dev = yb.get_mut(&out_dim).unwrap();
                let xb = self.x_bufs.borrow();
                let x_dev = xb.get(&in_dim).unwrap();
                unsafe { fn_.clone().launch(cfg, (y_dev, w_q4k, x_dev, in_dim as i32)).ok()? };
            }
            self.device.dtoh_sync_copy_into(
                self.y_bufs.borrow().get(&out_dim).unwrap(),
                self.h_bufs.borrow_mut().get_mut(&out_dim).unwrap()
            ).ok()?;
            Some(self.h_bufs.borrow().get(&out_dim).unwrap().clone())
        }

        /// Run SGEMV: y = W × x, [out_dim, in_dim] row-major.
        /// Uses pre-allocated device/host buffers — no cuMemAlloc per call.
        pub fn sgemv(&self, name: &str, x: &[f32]) -> Option<Vec<f32>> {
            let (in_dim, out_dim) = {
                let t = self.tensors.get(name)?;
                (t.in_dim, t.out_dim)
            };
            if !self.ensure_bufs(in_dim, out_dim) { return None; }

            // Copy x into pre-allocated device buffer (no new CUDA allocation)
            self.device.htod_sync_copy_into(
                x,
                self.x_bufs.borrow_mut().get_mut(&in_dim).unwrap(),
            ).ok()?;

            // SGEMV: x_bufs and y_bufs are separate RefCells — simultaneous borrows OK
            let t = self.tensors.get(name)?;
            let cfg = Self::gemv_cfg(in_dim, out_dim);
            unsafe {
                self.blas.gemv(
                    cfg,
                    &t.slice,
                    self.x_bufs.borrow().get(&in_dim).unwrap(),
                    self.y_bufs.borrow_mut().get_mut(&out_dim).unwrap(),
                ).ok()?
            };

            // Download result into pre-allocated host buffer
            self.device.dtoh_sync_copy_into(
                self.y_bufs.borrow().get(&out_dim).unwrap(),
                self.h_bufs.borrow_mut().get_mut(&out_dim).unwrap(),
            ).ok()?;

            Some(self.h_bufs.borrow().get(&out_dim).unwrap().clone())
        }

        /// Run SGEMV for gate and up projections with a single x upload.
        /// Saves one PCIe host→device round-trip per layer vs two sgemv calls.
        pub fn sgemv_gate_up(
            &self,
            gate_name: &str,
            up_name: &str,
            x: &[f32],
        ) -> Option<(Vec<f32>, Vec<f32>)> {
            let (in_dim, out_dim) = {
                let tg = self.tensors.get(gate_name)?;
                (tg.in_dim, tg.out_dim)
            };
            if !self.ensure_bufs(in_dim, out_dim) { return None; }

            // Upload x ONCE for both projections
            self.device.htod_sync_copy_into(
                x,
                self.x_bufs.borrow_mut().get_mut(&in_dim).unwrap(),
            ).ok()?;

            let cfg = Self::gemv_cfg(in_dim, out_dim);

            // Gate SGEMV
            {
                let tg = self.tensors.get(gate_name)?;
                unsafe {
                    self.blas.gemv(
                        cfg,
                        &tg.slice,
                        self.x_bufs.borrow().get(&in_dim).unwrap(),
                        self.y_bufs.borrow_mut().get_mut(&out_dim).unwrap(),
                    ).ok()?
                };
                self.device.dtoh_sync_copy_into(
                    self.y_bufs.borrow().get(&out_dim).unwrap(),
                    self.h_bufs.borrow_mut().get_mut(&out_dim).unwrap(),
                ).ok()?;
            }
            let gate_result = self.h_bufs.borrow().get(&out_dim).unwrap().clone();

            // Up SGEMV — x_bufs already loaded, y and h reused
            {
                let tu = self.tensors.get(up_name)?;
                unsafe {
                    self.blas.gemv(
                        cfg,
                        &tu.slice,
                        self.x_bufs.borrow().get(&in_dim).unwrap(),
                        self.y_bufs.borrow_mut().get_mut(&out_dim).unwrap(),
                    ).ok()?
                };
                self.device.dtoh_sync_copy_into(
                    self.y_bufs.borrow().get(&out_dim).unwrap(),
                    self.h_bufs.borrow_mut().get_mut(&out_dim).unwrap(),
                ).ok()?;
            }
            let up_result = self.h_bufs.borrow().get(&out_dim).unwrap().clone();

            Some((gate_result, up_result))
        }

        /// Run SGEMV for Q, K, V projections with a single x upload.
        /// Q has out_dim=q_out; K and V share out_dim=kv_out (same dims).
        /// Saves 2 PCIe host→device uploads per attn layer vs three sgemv calls.
        pub fn sgemv_qkv(
            &self,
            q_name: &str,
            k_name: &str,
            v_name: &str,
            x: &[f32],
        ) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
            let (in_dim, q_out) = {
                let tq = self.tensors.get(q_name)?;
                (tq.in_dim, tq.out_dim)
            };
            let kv_out = {
                let tk = self.tensors.get(k_name)?;
                tk.out_dim
            };
            if !self.ensure_bufs(in_dim, q_out) { return None; }
            if !self.ensure_bufs(in_dim, kv_out) { return None; }

            // Upload x ONCE for Q, K, and V
            self.device.htod_sync_copy_into(
                x,
                self.x_bufs.borrow_mut().get_mut(&in_dim).unwrap(),
            ).ok()?;

            // Q SGEMV
            {
                let tq = self.tensors.get(q_name)?;
                let cfg = Self::gemv_cfg(in_dim, q_out);
                unsafe {
                    self.blas.gemv(
                        cfg,
                        &tq.slice,
                        self.x_bufs.borrow().get(&in_dim).unwrap(),
                        self.y_bufs.borrow_mut().get_mut(&q_out).unwrap(),
                    ).ok()?
                };
                self.device.dtoh_sync_copy_into(
                    self.y_bufs.borrow().get(&q_out).unwrap(),
                    self.h_bufs.borrow_mut().get_mut(&q_out).unwrap(),
                ).ok()?;
            }
            let q_result = self.h_bufs.borrow().get(&q_out).unwrap().clone();

            // K SGEMV — x already loaded
            {
                let tk = self.tensors.get(k_name)?;
                let cfg = Self::gemv_cfg(in_dim, kv_out);
                unsafe {
                    self.blas.gemv(
                        cfg,
                        &tk.slice,
                        self.x_bufs.borrow().get(&in_dim).unwrap(),
                        self.y_bufs.borrow_mut().get_mut(&kv_out).unwrap(),
                    ).ok()?
                };
                self.device.dtoh_sync_copy_into(
                    self.y_bufs.borrow().get(&kv_out).unwrap(),
                    self.h_bufs.borrow_mut().get_mut(&kv_out).unwrap(),
                ).ok()?;
            }
            let k_result = self.h_bufs.borrow().get(&kv_out).unwrap().clone();

            // V SGEMV — x already loaded, y_buf[kv_out] reused
            {
                let tv = self.tensors.get(v_name)?;
                let cfg = Self::gemv_cfg(in_dim, kv_out);
                unsafe {
                    self.blas.gemv(
                        cfg,
                        &tv.slice,
                        self.x_bufs.borrow().get(&in_dim).unwrap(),
                        self.y_bufs.borrow_mut().get_mut(&kv_out).unwrap(),
                    ).ok()?
                };
                self.device.dtoh_sync_copy_into(
                    self.y_bufs.borrow().get(&kv_out).unwrap(),
                    self.h_bufs.borrow_mut().get_mut(&kv_out).unwrap(),
                ).ok()?;
            }
            let v_result = self.h_bufs.borrow().get(&kv_out).unwrap().clone();

            Some((q_result, k_result, v_result))
        }

        /// Fully fused FFN: gate+up SGEMV + on-device silu*hadamard + down SGEMV.
        ///
        /// vs the old sgemv_gate_up + CPU silu+had + sgemv(down) path:
        ///   Old: 2 htod + 3 dtoh = 5 PCIe transfers per layer
        ///   New: 1 htod + 1 dtoh = 2 PCIe transfers per layer
        ///   Saves: 3 transfers × 28 layers × ~0.19ms = ~16ms/token
        ///
        /// Returns None if the NVRTC kernel is unavailable (caller falls back to old path).
        pub fn ffn_fused(
            &self,
            gate_name: &str,
            up_name: &str,
            down_name: &str,
            x: &[f32],
        ) -> Option<Vec<f32>> {
            let silu_had = self.silu_had_fn.as_ref()?;

            let (in_dim, n_ff) = {
                let tg = self.tensors.get(gate_name)?;
                (tg.in_dim, tg.out_dim)
            };
            let out_dim = {
                let td = self.tensors.get(down_name)?;
                td.out_dim
            };

            // Ensure all device/host buffers are allocated.
            // x_bufs[in_dim], y_bufs[n_ff]: for gate/up input+output
            // up_bufs[n_ff]:               for up SGEMV result (separate from gate)
            // x_bufs[n_ff], y_bufs[out_dim]: for down input+output
            if !self.ensure_bufs(in_dim, n_ff) { return None; }
            if !self.ensure_up_buf(n_ff) { return None; }
            if !self.ensure_bufs(n_ff, out_dim) { return None; }

            // Upload x ONCE — reused for both gate and up projections.
            self.device.htod_sync_copy_into(
                x,
                self.x_bufs.borrow_mut().get_mut(&in_dim).unwrap(),
            ).ok()?;

            let cfg_gu = Self::gemv_cfg(in_dim, n_ff);

            // Gate SGEMV: x_bufs[in_dim] → y_bufs[n_ff]  (no dtoh)
            {
                let tg = self.tensors.get(gate_name)?;
                unsafe {
                    self.blas.gemv(
                        cfg_gu,
                        &tg.slice,
                        self.x_bufs.borrow().get(&in_dim).unwrap(),
                        self.y_bufs.borrow_mut().get_mut(&n_ff).unwrap(),
                    ).ok()?
                };
            }

            // Up SGEMV: x_bufs[in_dim] → up_bufs[n_ff]  (no dtoh)
            {
                let tu = self.tensors.get(up_name)?;
                unsafe {
                    self.blas.gemv(
                        cfg_gu,
                        &tu.slice,
                        self.x_bufs.borrow().get(&in_dim).unwrap(),
                        self.up_bufs.borrow_mut().get_mut(&n_ff).unwrap(),
                    ).ok()?
                };
            }

            // On-device silu(gate) * up → x_bufs[n_ff]
            // x_bufs[n_ff] becomes the input to the down SGEMV — no htod needed.
            // cuBLAS and kernel share device.stream, so ordering is guaranteed.
            {
                let cfg_k = LaunchConfig::for_num_elems(n_ff as u32);
                let mut yb = self.y_bufs.borrow_mut();
                let gate_dev = yb.get_mut(&n_ff).unwrap();
                let ub = self.up_bufs.borrow();
                let up_dev = ub.get(&n_ff).unwrap();
                let mut xb = self.x_bufs.borrow_mut();
                let out_dev = xb.get_mut(&n_ff).unwrap();
                unsafe {
                    silu_had.clone().launch(cfg_k, (gate_dev, up_dev, out_dev, n_ff as i32)).ok()?
                };
            }

            // Down SGEMV: x_bufs[n_ff] (kernel output) → y_bufs[out_dim]
            // No htod — the intermediate is already on device from the kernel above.
            let cfg_d = Self::gemv_cfg(n_ff, out_dim);
            {
                let td = self.tensors.get(down_name)?;
                unsafe {
                    self.blas.gemv(
                        cfg_d,
                        &td.slice,
                        self.x_bufs.borrow().get(&n_ff).unwrap(),
                        self.y_bufs.borrow_mut().get_mut(&out_dim).unwrap(),
                    ).ok()?
                };
            }

            // Single download — syncs the stream, capturing all queued GPU work.
            self.device.dtoh_sync_copy_into(
                self.y_bufs.borrow().get(&out_dim).unwrap(),
                self.h_bufs.borrow_mut().get_mut(&out_dim).unwrap(),
            ).ok()?;

            Some(self.h_bufs.borrow().get(&out_dim).unwrap().clone())
        }

        /// Fully fused attention: QKV SGEMV + single fused kernel (qk_norm+rope+kv_append+attn) + Wo SGEMV.
        ///
        /// Old path (v1.9): 1 htod + 3 dtoh (QKV) + 1 htod (ctx) + 1 dtoh (Wo) = 6 PCIe per layer.
        /// New path (v1.10): 1 htod + 1 dtoh = 2 PCIe per layer.
        /// Each PCIe sync and each kernel launch costs ~50µs on Windows WDDM.
        /// Savings: 4 PCIe syncs eliminated, 1 extra kernel launch added = net 3 × 50µs × 28 = 4.2ms/token.
        pub fn attn_fused(
            &self,
            layer_idx: usize,
            q_name: &str,
            k_name: &str,
            v_name: &str,
            wo_name: &str,
            q_norm_name: Option<&str>,
            k_norm_name: Option<&str>,
            x: &[f32],
            n_head: usize,
            n_head_kv: usize,
            head_dim: usize,
            position: usize,
            rms_eps: f32,
            theta_base: f32,
        ) -> Option<Vec<f32>> {
            let fused_fn = self.attn_fused_kernel_fn.as_ref()?;
            if head_dim == 0 || (head_dim & (head_dim - 1)) != 0 { return None; }

            let n_embd = x.len();
            let q_out = n_head * head_dim;
            let kv_out = n_head_kv * head_dim;
            let kv_group_size = (n_head / n_head_kv).max(1) as i32;

            let (wo_in_dim, wo_out_dim) = {
                let t = self.tensors.get(wo_name)?;
                (t.in_dim, t.out_dim)
            };
            if !self.tensors.contains_key(q_name) || !self.tensors.contains_key(k_name)
                || !self.tensors.contains_key(v_name) { return None; }

            if !self.ensure_bufs(n_embd, q_out)         { return None; }
            if !self.ensure_k_buf(kv_out)                { return None; }
            if !self.ensure_bufs(n_embd, kv_out)        { return None; }
            if !self.ensure_bufs(wo_in_dim, wo_out_dim) { return None; }
            if !self.ensure_kv_cache(layer_idx, n_head_kv, head_dim) { return None; }

            let seq_pos = *self.kv_seq_lens.borrow().get(&layer_idx).unwrap_or(&0);
            if seq_pos >= self.kv_max_seq { return None; }
            let new_seq_len = seq_pos + 1;

            // 1. htod: x → x_bufs[n_embd]
            self.device.htod_sync_copy_into(
                x,
                self.x_bufs.borrow_mut().get_mut(&n_embd).unwrap(),
            ).ok()?;

            // 2-4. QKV SGEMVs: all read same x_bufs[n_embd], no dtoh
            {
                let tq = self.tensors.get(q_name)?;
                unsafe {
                    self.blas.gemv(Self::gemv_cfg(n_embd, q_out), &tq.slice,
                        self.x_bufs.borrow().get(&n_embd).unwrap(),
                        self.y_bufs.borrow_mut().get_mut(&q_out).unwrap(),
                    ).ok()?
                };
            }
            {
                let tk = self.tensors.get(k_name)?;
                unsafe {
                    self.blas.gemv(Self::gemv_cfg(n_embd, kv_out), &tk.slice,
                        self.x_bufs.borrow().get(&n_embd).unwrap(),
                        self.k_bufs.borrow_mut().get_mut(&kv_out).unwrap(),
                    ).ok()?
                };
            }
            {
                let tv = self.tensors.get(v_name)?;
                unsafe {
                    self.blas.gemv(Self::gemv_cfg(n_embd, kv_out), &tv.slice,
                        self.x_bufs.borrow().get(&n_embd).unwrap(),
                        self.y_bufs.borrow_mut().get_mut(&kv_out).unwrap(),
                    ).ok()?
                };
            }

            // 5. Single fused kernel: qk_norm+rope(Q) + qk_norm+rope+append(K) + attn_compute
            //    1 kernel launch × 50µs WDDM vs old 5 separate launches.
            //    Shared layout: [q_store: head_dim] [scratch: head_dim] [scores: new_seq_len]
            let do_qk_norm = if q_norm_name.is_some() { 1i32 } else { 0i32 };
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let shared_bytes = (2 * head_dim + new_seq_len) as u32 * 4;
            let cfg = LaunchConfig {
                grid_dim: (n_head as u32, 1, 1),
                block_dim: (head_dim as u32, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            {
                let mut xb   = self.x_bufs.borrow_mut();
                let ctx_out   = xb.get_mut(&wo_in_dim).unwrap();
                let mut kkb  = self.kv_k_bufs.borrow_mut();
                let kk        = kkb.get_mut(&layer_idx).unwrap();
                let mut kvb  = self.kv_v_bufs.borrow_mut();
                let kv_v      = kvb.get_mut(&layer_idx).unwrap();
                let yb        = self.y_bufs.borrow();
                let q_dev     = yb.get(&q_out).unwrap();
                let v_dev     = yb.get(&kv_out).unwrap();
                let kb        = self.k_bufs.borrow();
                let k_dev     = kb.get(&kv_out).unwrap();
                // When rms_eps==0 kernel skips qk_norm; pass q/k as dummy norm weight pointers.
                let eps_arg = if do_qk_norm != 0 { rms_eps } else { 0.0f32 };
                let q_norm_w = q_norm_name.and_then(|n| self.norm_bufs.get(n)).unwrap_or(q_dev);
                let k_norm_w = k_norm_name.and_then(|n| self.norm_bufs.get(n)).unwrap_or(k_dev);
                unsafe {
                    fused_fn.clone().launch(cfg, (
                        ctx_out, kk, kv_v,
                        q_dev, k_dev, v_dev,
                        q_norm_w, k_norm_w,
                        seq_pos as i32,
                        eps_arg, theta_base,
                        kv_group_size,
                    )).ok()?
                };
            }
            *self.kv_seq_lens.borrow_mut().entry(layer_idx).or_insert(0) = new_seq_len;

            // 6. Wo SGEMV: x_bufs[wo_in_dim] → y_bufs[wo_out_dim]  (no htod)
            {
                let two = self.tensors.get(wo_name)?;
                unsafe {
                    self.blas.gemv(Self::gemv_cfg(wo_in_dim, wo_out_dim), &two.slice,
                        self.x_bufs.borrow().get(&wo_in_dim).unwrap(),
                        self.y_bufs.borrow_mut().get_mut(&wo_out_dim).unwrap(),
                    ).ok()?
                };
            }

            // 7. Single dtoh — syncs the entire stream (1 PCIe round-trip for whole layer)
            self.device.dtoh_sync_copy_into(
                self.y_bufs.borrow().get(&wo_out_dim).unwrap(),
                self.h_bufs.borrow_mut().get_mut(&wo_out_dim).unwrap(),
            ).ok()?;

            Some(self.h_bufs.borrow().get(&wo_out_dim).unwrap().clone())
        }

        // ── GPU-resident forward pass helpers ────────────────────────────────────

        /// Returns true if rms_norm + add kernels are available (GPU-resident capable).
        pub fn supports_resident(&self) -> bool {
            self.rms_norm_fn.is_some() && self.add_fn.is_some() && self.attn_fused_kernel_fn.is_some() && self.silu_had_fn.is_some()
        }

        /// Allocate hidden_dev if not already allocated.
        pub fn ensure_hidden(&self, n: usize) -> bool {
            let mut hd = self.hidden_dev.borrow_mut();
            if hd.is_none() {
                match self.device.alloc_zeros::<f32>(n) {
                    Ok(buf) => { *hd = Some(buf); true }
                    Err(_) => false,
                }
            } else {
                true
            }
        }

        /// Upload hidden state to device (1 htod for the entire forward pass).
        pub fn upload_hidden(&self, data: &[f32]) -> bool {
            let mut hd = self.hidden_dev.borrow_mut();
            if let Some(ref mut buf) = *hd {
                self.device.htod_sync_copy_into(data, buf).is_ok()
            } else {
                false
            }
        }

        /// Download hidden state from device (1 dtoh for the entire forward pass).
        pub fn download_hidden(&self) -> Option<Vec<f32>> {
            let hd = self.hidden_dev.borrow();
            self.device.dtoh_sync_copy(hd.as_ref()?).ok()
        }

        /// GPU RMS norm: reads hidden_dev, writes normed result to x_bufs[n] (SGEMV input).
        pub fn rms_norm_on_hidden(&self, norm_name: &str, n: usize, eps: f32) -> bool {
            (|| -> Option<()> {
                let fn_ = self.rms_norm_fn.as_ref()?;
                let norm_w = self.norm_bufs.get(norm_name)?;
                if !self.ensure_bufs(n, n) { return None; }
                let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
                let hd = self.hidden_dev.borrow();
                let hidden = hd.as_ref()?;
                let mut xb = self.x_bufs.borrow_mut();
                let out = xb.get_mut(&n).unwrap();
                unsafe { fn_.clone().launch(cfg, (out, hidden, norm_w, n as i32, eps)).ok()? };
                Some(())
            })().is_some()
        }

        /// GPU residual add: hidden_dev += y_bufs[n]  (result of last SGEMV).
        pub fn add_hidden_from_y(&self, n: usize) -> bool {
            (|| -> Option<()> {
                let fn_ = self.add_fn.as_ref()?;
                let blocks = ((n + 255) / 256) as u32;
                let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                let mut hd = self.hidden_dev.borrow_mut();
                let hidden = hd.as_mut()?;
                let yb = self.y_bufs.borrow();
                let delta = yb.get(&n)?;
                unsafe { fn_.clone().launch(cfg, (hidden, delta, n as i32)).ok()? };
                Some(())
            })().is_some()
        }

        /// Like attn_fused but reads x from x_bufs[n_embd] (set by rms_norm_on_hidden) and
        /// leaves Wo output in y_bufs[n_embd] — no htod, no dtoh.
        pub fn attn_fused_resident(
            &self,
            layer_idx: usize,
            q_name: &str, k_name: &str, v_name: &str, wo_name: &str,
            q_norm_name: Option<&str>, k_norm_name: Option<&str>,
            n_embd: usize, n_head: usize, n_head_kv: usize, head_dim: usize,
            position: usize, rms_eps: f32, theta_base: f32,
        ) -> bool {
            (|| -> Option<()> {
                let fused_fn = self.attn_fused_kernel_fn.as_ref()?;
                if head_dim == 0 || (head_dim & (head_dim - 1)) != 0 { return None; }
                let q_out = n_head * head_dim;
                let kv_out = n_head_kv * head_dim;
                let kv_group_size = (n_head / n_head_kv).max(1) as i32;
                let (wo_in_dim, wo_out_dim) = if let Some((_, od, id)) = self.q4k_tensors.get(wo_name) {
                    (*id, *od)
                } else {
                    let t = self.tensors.get(wo_name)?; (t.in_dim, t.out_dim)
                };
                if !self.has_weight(q_name) || !self.has_weight(k_name) || !self.has_weight(v_name) { return None; }
                // x_bufs[n_embd] already populated by rms_norm_on_hidden — skip htod
                if !self.ensure_bufs(n_embd, q_out) { return None; }
                if !self.ensure_k_buf(kv_out) { return None; }
                if !self.ensure_bufs(n_embd, kv_out) { return None; }
                if !self.ensure_bufs(wo_in_dim, wo_out_dim) { return None; }
                if !self.ensure_kv_cache(layer_idx, n_head_kv, head_dim) { return None; }
                let seq_pos = *self.kv_seq_lens.borrow().get(&layer_idx).unwrap_or(&0);
                if seq_pos >= self.kv_max_seq { return None; }
                // Q SGEMV (x already on device) → y_bufs[q_out]
                if let Some((wq, _, _)) = self.q4k_tensors.get(q_name) {
                    let fn_ = self.sgemv_q4k_fn.as_ref()?;
                    let cfg = LaunchConfig { grid_dim: (q_out as u32 / 32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: n_embd as u32 * 4 };
                    let mut yb = self.y_bufs.borrow_mut(); let y = yb.get_mut(&q_out)?;
                    let xb = self.x_bufs.borrow(); let xd = xb.get(&n_embd)?;
                    unsafe { fn_.clone().launch(cfg, (y, wq, xd, n_embd as i32)).ok()? };
                } else {
                    let tq = self.tensors.get(q_name)?;
                    unsafe { self.blas.gemv(Self::gemv_cfg(n_embd, q_out), &tq.slice, self.x_bufs.borrow().get(&n_embd).unwrap(), self.y_bufs.borrow_mut().get_mut(&q_out).unwrap()).ok()? };
                }
                // K SGEMV → k_bufs[kv_out]
                if let Some((wk, _, _)) = self.q4k_tensors.get(k_name) {
                    let fn_ = self.sgemv_q4k_fn.as_ref()?;
                    let cfg = LaunchConfig { grid_dim: (kv_out as u32 / 32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: n_embd as u32 * 4 };
                    let mut kb = self.k_bufs.borrow_mut(); let y = kb.get_mut(&kv_out)?;
                    let xb = self.x_bufs.borrow(); let xd = xb.get(&n_embd)?;
                    unsafe { fn_.clone().launch(cfg, (y, wk, xd, n_embd as i32)).ok()? };
                } else {
                    let tk = self.tensors.get(k_name)?;
                    unsafe { self.blas.gemv(Self::gemv_cfg(n_embd, kv_out), &tk.slice, self.x_bufs.borrow().get(&n_embd).unwrap(), self.k_bufs.borrow_mut().get_mut(&kv_out).unwrap()).ok()? };
                }
                // V SGEMV → y_bufs[kv_out]
                if let Some((wv, _, _)) = self.q4k_tensors.get(v_name) {
                    let fn_ = self.sgemv_q4k_fn.as_ref()?;
                    let cfg = LaunchConfig { grid_dim: (kv_out as u32 / 32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: n_embd as u32 * 4 };
                    let mut yb = self.y_bufs.borrow_mut(); let y = yb.get_mut(&kv_out)?;
                    let xb = self.x_bufs.borrow(); let xd = xb.get(&n_embd)?;
                    unsafe { fn_.clone().launch(cfg, (y, wv, xd, n_embd as i32)).ok()? };
                } else {
                    let tv = self.tensors.get(v_name)?;
                    unsafe { self.blas.gemv(Self::gemv_cfg(n_embd, kv_out), &tv.slice, self.x_bufs.borrow().get(&n_embd).unwrap(), self.y_bufs.borrow_mut().get_mut(&kv_out).unwrap()).ok()? };
                }
                // Single fused kernel
                let eps_arg = if q_norm_name.is_some() { rms_eps } else { 0.0f32 };
                let new_seq_len = seq_pos + 1;
                let shared_bytes = (2 * head_dim + new_seq_len) as u32 * 4;
                let cfg = LaunchConfig { grid_dim: (n_head as u32, 1, 1), block_dim: (head_dim as u32, 1, 1), shared_mem_bytes: shared_bytes };
                {
                    let mut xb = self.x_bufs.borrow_mut();
                    let ctx_out = xb.get_mut(&wo_in_dim).unwrap();
                    let mut kkb = self.kv_k_bufs.borrow_mut();
                    let kk = kkb.get_mut(&layer_idx).unwrap();
                    let mut kvb = self.kv_v_bufs.borrow_mut();
                    let kv_v = kvb.get_mut(&layer_idx).unwrap();
                    let yb = self.y_bufs.borrow();
                    let q_dev = yb.get(&q_out).unwrap();
                    let v_dev = yb.get(&kv_out).unwrap();
                    let kb = self.k_bufs.borrow();
                    let k_dev = kb.get(&kv_out).unwrap();
                    let q_norm_w = q_norm_name.and_then(|n| self.norm_bufs.get(n)).unwrap_or(q_dev);
                    let k_norm_w = k_norm_name.and_then(|n| self.norm_bufs.get(n)).unwrap_or(k_dev);
                    unsafe { fused_fn.clone().launch(cfg, (ctx_out, kk, kv_v, q_dev, k_dev, v_dev, q_norm_w, k_norm_w, seq_pos as i32, eps_arg, theta_base, kv_group_size)).ok()? };
                }
                *self.kv_seq_lens.borrow_mut().entry(layer_idx).or_insert(0) = new_seq_len;
                // Wo SGEMV → y_bufs[wo_out_dim] (no dtoh; caller reads via add_hidden_from_y)
                if let Some((wwo, _, _)) = self.q4k_tensors.get(wo_name) {
                    let fn_ = self.sgemv_q4k_fn.as_ref()?;
                    let cfg = LaunchConfig { grid_dim: (wo_out_dim as u32 / 32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: wo_in_dim as u32 * 4 };
                    let mut yb = self.y_bufs.borrow_mut(); let y = yb.get_mut(&wo_out_dim)?;
                    let xb = self.x_bufs.borrow(); let xd = xb.get(&wo_in_dim)?;
                    unsafe { fn_.clone().launch(cfg, (y, wwo, xd, wo_in_dim as i32)).ok()? };
                } else {
                    let two = self.tensors.get(wo_name)?;
                    unsafe { self.blas.gemv(Self::gemv_cfg(wo_in_dim, wo_out_dim), &two.slice, self.x_bufs.borrow().get(&wo_in_dim).unwrap(), self.y_bufs.borrow_mut().get_mut(&wo_out_dim).unwrap()).ok()? };
                }
                Some(())
            })().is_some()
        }

        /// Like ffn_fused but reads x from x_bufs[n_embd] (set by rms_norm_on_hidden) and
        /// leaves down output in y_bufs[n_embd] — no htod, no dtoh.
        pub fn ffn_fused_resident(&self, gate_name: &str, up_name: &str, down_name: &str, n_embd: usize) -> bool {
            (|| -> Option<()> {
                let silu_had = self.silu_had_fn.as_ref()?;
                let (in_dim, n_ff) = if let Some((_, od, id)) = self.q4k_tensors.get(gate_name) {
                    (*id, *od)
                } else {
                    let tg = self.tensors.get(gate_name)?; (tg.in_dim, tg.out_dim)
                };
                let out_dim = if let Some((_, od, _)) = self.q4k_tensors.get(down_name) {
                    *od
                } else {
                    self.tensors.get(down_name)?.out_dim
                };
                // x_bufs[n_embd] already populated — skip htod
                if !self.ensure_bufs(in_dim, n_ff) { return None; }
                if !self.ensure_up_buf(n_ff) { return None; }
                if !self.ensure_bufs(n_ff, out_dim) { return None; }
                // Gate SGEMV → y_bufs[n_ff]
                if let Some((wg, _, _)) = self.q4k_tensors.get(gate_name) {
                    let fn_ = self.sgemv_q4k_fn.as_ref()?;
                    let cfg = LaunchConfig { grid_dim: (n_ff as u32 / 32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: in_dim as u32 * 4 };
                    let mut yb = self.y_bufs.borrow_mut(); let y = yb.get_mut(&n_ff)?;
                    let xb = self.x_bufs.borrow(); let xd = xb.get(&in_dim)?;
                    unsafe { fn_.clone().launch(cfg, (y, wg, xd, in_dim as i32)).ok()? };
                } else {
                    let tg = self.tensors.get(gate_name)?;
                    unsafe { self.blas.gemv(Self::gemv_cfg(in_dim, n_ff), &tg.slice, self.x_bufs.borrow().get(&in_dim).unwrap(), self.y_bufs.borrow_mut().get_mut(&n_ff).unwrap()).ok()? };
                }
                // Up SGEMV → up_bufs[n_ff]
                if let Some((wu, _, _)) = self.q4k_tensors.get(up_name) {
                    let fn_ = self.sgemv_q4k_fn.as_ref()?;
                    let cfg = LaunchConfig { grid_dim: (n_ff as u32 / 32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: in_dim as u32 * 4 };
                    let mut ub = self.up_bufs.borrow_mut(); let y = ub.get_mut(&n_ff)?;
                    let xb = self.x_bufs.borrow(); let xd = xb.get(&in_dim)?;
                    unsafe { fn_.clone().launch(cfg, (y, wu, xd, in_dim as i32)).ok()? };
                } else {
                    let tu = self.tensors.get(up_name)?;
                    unsafe { self.blas.gemv(Self::gemv_cfg(in_dim, n_ff), &tu.slice, self.x_bufs.borrow().get(&in_dim).unwrap(), self.up_bufs.borrow_mut().get_mut(&n_ff).unwrap()).ok()? };
                }
                // silu(gate) * up → x_bufs[n_ff]  (input for down SGEMV)
                { let cfg_k = LaunchConfig::for_num_elems(n_ff as u32); let mut yb = self.y_bufs.borrow_mut(); let gate_dev = yb.get_mut(&n_ff).unwrap(); let ub = self.up_bufs.borrow(); let up_dev = ub.get(&n_ff).unwrap(); let mut xb = self.x_bufs.borrow_mut(); let out_dev = xb.get_mut(&n_ff).unwrap(); unsafe { silu_had.clone().launch(cfg_k, (gate_dev, up_dev, out_dev, n_ff as i32)).ok()? }; }
                // Down SGEMV: x_bufs[n_ff] → y_bufs[out_dim]
                if let Some((wd, _, _)) = self.q4k_tensors.get(down_name) {
                    let fn_ = self.sgemv_q4k_fn.as_ref()?;
                    let cfg = LaunchConfig { grid_dim: (out_dim as u32 / 32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: n_ff as u32 * 4 };
                    let mut yb = self.y_bufs.borrow_mut(); let y = yb.get_mut(&out_dim)?;
                    let xb = self.x_bufs.borrow(); let xd = xb.get(&n_ff)?;
                    unsafe { fn_.clone().launch(cfg, (y, wd, xd, n_ff as i32)).ok()? };
                } else {
                    let td = self.tensors.get(down_name)?;
                    unsafe { self.blas.gemv(Self::gemv_cfg(n_ff, out_dim), &td.slice, self.x_bufs.borrow().get(&n_ff).unwrap(), self.y_bufs.borrow_mut().get_mut(&out_dim).unwrap()).ok()? };
                }
                let _ = n_embd;
                Some(())
            })().is_some()
        }

        pub fn vram_used_mb(&self) -> f64 {
            self.vram_used as f64 / (1024.0 * 1024.0)
        }

        #[inline]
        fn gemv_cfg(in_dim: usize, out_dim: usize) -> GemvConfig<f32> {
            GemvConfig::<f32> {
                trans: CUBLAS_OP_T,
                m: in_dim as i32,
                n: out_dim as i32,
                alpha: 1.0,
                lda: in_dim as i32,
                incx: 1,
                beta: 0.0,
                incy: 1,
            }
        }
    }
}

#[cfg(target_os = "windows")]
pub use windows_impl::CudaGpu;

// ── Non-Windows no-op stub ────────────────────────────────────────────────────

#[cfg(not(target_os = "windows"))]
pub struct CudaGpu;

#[cfg(not(target_os = "windows"))]
impl CudaGpu {
    pub fn new(_vram_budget_bytes: usize) -> Option<Self> { None }
    pub fn vram_remaining(&self) -> usize { 0 }
    pub fn upload(&mut self, _: &str, _: &[f32], _: usize, _: usize) -> bool { false }
    pub fn upload_norm(&mut self, _: &str, _: &[f32]) -> bool { false }
    pub fn upload_q4k(&mut self, _: &str, _: &[u8], _: usize, _: usize) -> bool { false }
    pub fn has(&self, _: &str) -> bool { false }
    pub fn has_norm(&self, _: &str) -> bool { false }
    pub fn has_q4k(&self, _: &str) -> bool { false }
    pub fn has_weight(&self, _: &str) -> bool { false }
    pub fn sgemv(&self, _: &str, _: &[f32]) -> Option<Vec<f32>> { None }
    pub fn sgemv_q4k(&self, _: &str, _: &[f32]) -> Option<Vec<f32>> { None }
    pub fn sgemv_gate_up(&self, _: &str, _: &str, _: &[f32]) -> Option<(Vec<f32>, Vec<f32>)> { None }
    pub fn sgemv_qkv(&self, _: &str, _: &str, _: &str, _: &[f32]) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> { None }
    pub fn ffn_fused(&self, _: &str, _: &str, _: &str, _: &[f32]) -> Option<Vec<f32>> { None }
    #[allow(clippy::too_many_arguments)]
    pub fn attn_fused(&self, _: usize, _: &str, _: &str, _: &str, _: &str, _: Option<&str>, _: Option<&str>, _: &[f32], _: usize, _: usize, _: usize, _: usize, _: f32, _: f32) -> Option<Vec<f32>> { None }
    pub fn supports_resident(&self) -> bool { false }
    pub fn ensure_hidden(&self, _: usize) -> bool { false }
    pub fn upload_hidden(&self, _: &[f32]) -> bool { false }
    pub fn download_hidden(&self) -> Option<Vec<f32>> { None }
    pub fn rms_norm_on_hidden(&self, _: &str, _: usize, _: f32) -> bool { false }
    pub fn add_hidden_from_y(&self, _: usize) -> bool { false }
    #[allow(clippy::too_many_arguments)]
    pub fn attn_fused_resident(&self, _: usize, _: &str, _: &str, _: &str, _: &str, _: Option<&str>, _: Option<&str>, _: usize, _: usize, _: usize, _: usize, _: usize, _: f32, _: f32) -> bool { false }
    pub fn ffn_fused_resident(&self, _: &str, _: &str, _: &str, _: usize) -> bool { false }
    pub fn vram_used_mb(&self) -> f64 { 0.0 }
}
