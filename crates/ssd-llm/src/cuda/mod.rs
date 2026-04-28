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
    pub fn has(&self, _: &str) -> bool { false }
    pub fn sgemv(&self, _: &str, _: &[f32]) -> Option<Vec<f32>> { None }
    pub fn sgemv_gate_up(&self, _: &str, _: &str, _: &[f32]) -> Option<(Vec<f32>, Vec<f32>)> { None }
    pub fn sgemv_qkv(&self, _: &str, _: &str, _: &str, _: &[f32]) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> { None }
    pub fn ffn_fused(&self, _: &str, _: &str, _: &str, _: &[f32]) -> Option<Vec<f32>> { None }
    pub fn vram_used_mb(&self) -> f64 { 0.0 }
}
