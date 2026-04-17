//! Async SSD → RAM streaming engine with prefetching

use crate::model::cache::CachedLayer;
use crate::model::gguf::{GgmlType, GgufFile, TensorInfo};
use crate::model::loader::MmapLoader;
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

use tracing::{debug, info};

/// SSD streaming engine that loads tensor data on-demand via mmap
pub struct SsdStreamer {
    loader: MmapLoader,
    budget: usize,
}

impl SsdStreamer {
    pub fn new(path: &Path, budget: usize) -> Result<Self> {
        // We need to parse the GGUF header to get data_offset
        let gguf = GgufFile::open(path)?;
        let loader = MmapLoader::new(path, gguf.data_offset)?;

        info!(
            "SSD Streamer initialized: file={:.2}GB, budget={:.2}GB",
            loader.file_size() as f64 / (1024.0 * 1024.0 * 1024.0),
            budget as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        Ok(Self { loader, budget })
    }

    /// Load a specific tensor's data and dequantize to f32
    pub fn load_tensor_f32(&self, tensor: &TensorInfo) -> Result<Vec<f32>> {
        let data = self
            .loader
            .get_tensor_data(tensor.offset, tensor.size_bytes)?;
        let n_elements: u64 = tensor.dimensions.iter().product();
        dequantize_to_f32(data, &tensor.dtype, n_elements as usize)
    }

    /// Load all tensors for a given layer
    pub fn load_layer(&self, gguf: &GgufFile, layer_idx: u32) -> Result<CachedLayer> {
        let tensors_info = gguf.layer_tensors(layer_idx);
        let mut tensors = HashMap::new();
        let mut total_size = 0usize;

        for ti in &tensors_info {
            let data = self.load_tensor_f32(ti)?;
            total_size += data.len() * std::mem::size_of::<f32>();
            tensors.insert(ti.name.clone(), data);
        }

        debug!(
            "Loaded layer {} ({} tensors, {:.2} MB)",
            layer_idx,
            tensors.len(),
            total_size as f64 / (1024.0 * 1024.0)
        );

        Ok(CachedLayer {
            layer_idx,
            tensors,
            size_bytes: total_size,
        })
    }

    /// Prefetch a layer — issue madvise WILLNEED for all its tensors
    pub fn prefetch_layer(&self, gguf: &GgufFile, layer_idx: u32) {
        for tensor in gguf.layer_tensors(layer_idx) {
            self.loader.prefetch(tensor.offset, tensor.size_bytes);
        }
        debug!("Prefetched layer {} from SSD", layer_idx);
    }

    /// Evict a layer's pages from OS cache
    pub fn evict_layer(&self, gguf: &GgufFile, layer_idx: u32) {
        for tensor in gguf.layer_tensors(layer_idx) {
            self.loader.evict(tensor.offset, tensor.size_bytes);
        }
    }

    /// Load a named tensor directly
    pub fn load_named_tensor_f32(&self, gguf: &GgufFile, name: &str) -> Result<Vec<f32>> {
        let tensor = gguf
            .find_tensor(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found: {}", name))?;
        self.load_tensor_f32(tensor)
    }

    /// Load only specific rows from a 2D tensor (e.g., embedding lookup by token ID).
    /// Avoids dequantizing the entire tensor — only reads and dequantizes the requested rows.
    /// `row_dim` is the number of elements per row (e.g., n_embd).
    pub fn load_tensor_rows(
        &self,
        tensor: &TensorInfo,
        row_ids: &[u32],
        row_dim: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let block_size = tensor.dtype.block_size();
        let block_bytes = tensor.dtype.type_size();

        if row_dim % block_size != 0 {
            // Fallback: can't row-slice if row_dim isn't aligned to block boundaries
            let full = self.load_tensor_f32(tensor)?;
            return Ok(row_ids
                .iter()
                .map(|&id| {
                    let s = id as usize * row_dim;
                    let e = s + row_dim;
                    if e <= full.len() {
                        full[s..e].to_vec()
                    } else {
                        vec![0.0f32; row_dim]
                    }
                })
                .collect());
        }

        let blocks_per_row = row_dim / block_size;
        let row_bytes = blocks_per_row * block_bytes;

        let full_data = self
            .loader
            .get_tensor_data(tensor.offset, tensor.size_bytes)?;

        let mut result = Vec::with_capacity(row_ids.len());
        for &token_id in row_ids {
            let byte_offset = token_id as usize * row_bytes;
            if byte_offset + row_bytes > full_data.len() {
                result.push(vec![0.0f32; row_dim]);
                continue;
            }
            let row_data = &full_data[byte_offset..byte_offset + row_bytes];
            result.push(dequantize_to_f32(row_data, &tensor.dtype, row_dim)?);
        }
        Ok(result)
    }

    /// Get raw (quantized) tensor bytes without dequantization.
    /// Use with `matvec_quantized_cpu` for zero-allocation matmul on large tensors.
    pub fn raw_tensor_data<'a>(&'a self, tensor: &TensorInfo) -> Result<&'a [u8]> {
        self.loader
            .get_tensor_data(tensor.offset, tensor.size_bytes)
    }
}

/// Dequantize raw bytes to f32
fn dequantize_to_f32(data: &[u8], dtype: &GgmlType, n_elements: usize) -> Result<Vec<f32>> {
    match dtype {
        GgmlType::F32 => {
            let floats: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_elements) };
            Ok(floats.to_vec())
        }
        GgmlType::F16 => {
            let halfs: &[u16] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, n_elements) };
            Ok(halfs
                .iter()
                .map(|&h| half::f16::from_bits(h).to_f32())
                .collect())
        }
        GgmlType::BF16 => {
            let halfs: &[u16] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, n_elements) };
            Ok(halfs
                .iter()
                .map(|&h| f32::from_bits((h as u32) << 16))
                .collect())
        }
        GgmlType::Q8_0 => dequantize_q8_0(data, n_elements),
        GgmlType::Q4_0 => dequantize_q4_0(data, n_elements),
        GgmlType::Q6K => dequantize_q6_k(data, n_elements),
        GgmlType::Q4K => dequantize_q4_k(data, n_elements),
        _ => {
            // For unsupported quantization types, return zeros with a warning
            tracing::warn!("Unsupported quantization type {:?}, returning zeros", dtype);
            Ok(vec![0.0f32; n_elements])
        }
    }
}

/// Dequantize Q8_0: block_size=32, layout: f16 scale + 32 x i8
fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size = 32;
    let block_bytes = 34; // 2 (f16 scale) + 32 (int8 values)
    let n_blocks = n_elements / block_size;
    let mut output = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block_start = i * block_bytes;
        if block_start + block_bytes > data.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        for j in 0..block_size {
            let val = data[block_start + 2 + j] as i8;
            output.push(val as f32 * scale);
        }
    }

    output.resize(n_elements, 0.0);
    Ok(output)
}

/// Dequantize Q4_0: block_size=32, layout: f16 scale + 16 bytes (32 nibbles)
fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size = 32;
    let block_bytes = 18; // 2 (f16 scale) + 16 (packed nibbles)
    let n_blocks = n_elements / block_size;
    let mut output = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block_start = i * block_bytes;
        if block_start + block_bytes > data.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        for j in 0..16 {
            let byte = data[block_start + 2 + j];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            output.push(lo as f32 * scale);
            output.push(hi as f32 * scale);
        }
    }

    output.resize(n_elements, 0.0);
    Ok(output)
}

/// Dequantize Q4_K: block_size=256, block_bytes=144
/// Layout per block: f16 d (2B) + f16 dmin (2B) + scales (12B) + qs (128B)
fn dequantize_q4_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size: usize = 256;
    let block_bytes: usize = 144; // 2 + 2 + 12 + 128
    let n_blocks = n_elements / block_size;
    let mut output = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let boff = i * block_bytes;
        if boff + block_bytes > data.len() {
            break;
        }
        let d = half::f16::from_bits(u16::from_le_bytes([data[boff], data[boff + 1]])).to_f32();
        let dmin =
            half::f16::from_bits(u16::from_le_bytes([data[boff + 2], data[boff + 3]])).to_f32();
        let sc = &data[boff + 4..boff + 16];
        let qs = &data[boff + 16..boff + 144];

        // Canonical Q4K layout (matches llama.cpp dequantize_row_q4_K):
        // For each pair of sub-blocks (2k, 2k+1), 32 qs bytes are shared —
        // sub-block 2k consumes low nibbles, sub-block 2k+1 consumes high nibbles.
        // Output ordering: all 32 values of sub-block sb fill output[sb*32 .. sb*32+32].
        for sb in 0..8u8 {
            // Canonical llama.cpp get_scale_min_k4:
            //   if j<4: d = sc[j] & 0x3F ; m = sc[j+4] & 0x3F
            //   else:   d = (sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4)
            //           m = (sc[j+4] >> 4) | ((sc[j]   >> 6) << 4)
            let (sc_low, m_low) = if sb < 4 {
                (sc[sb as usize] & 0x3F, sc[sb as usize + 4] & 0x3F)
            } else {
                let si4 = (sb - 4) as usize;
                (
                    (sc[sb as usize + 4] & 0x0F) | ((sc[si4] >> 6) << 4),
                    (sc[sb as usize + 4] >> 4)   | ((sc[sb as usize] >> 6) << 4),
                )
            };
            let scale = d * sc_low as f32;
            let min_val = dmin * m_low as f32;

            let byte_chunk = (sb / 2) as usize;
            let half_high = (sb % 2) == 1;
            let qs_off = byte_chunk * 32;
            for l in 0..32usize {
                let byte_val = qs[qs_off + l];
                let nibble = if half_high { byte_val >> 4 } else { byte_val & 0x0F };
                output.push(scale * nibble as f32 - min_val);
            }
        }
    }

    output.resize(n_elements, 0.0);
    Ok(output)
}

/// Dequantize Q6_K: block_size=256
/// Layout per block: ql[128] + qh[64] + scales[16] + f16 d
/// Uses stride-32 interleaved layout matching llama.cpp's dequantize_row_q6_K.
fn dequantize_q6_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size: usize = 256;
    let block_bytes: usize = 210; // 128 + 64 + 16 + 2
    let n_blocks = n_elements / block_size;
    let mut output = vec![0.0f32; n_elements];

    for i in 0..n_blocks {
        let bs = i * block_bytes;
        if bs + block_bytes > data.len() {
            break;
        }
        let ql = &data[bs..bs + 128];
        let qh = &data[bs + 128..bs + 192];
        let scales = &data[bs + 192..bs + 208];
        let d = half::f16::from_bits(u16::from_le_bytes([data[bs + 208], data[bs + 209]])).to_f32();
        let out_base = i * block_size;

        // Two halves of 128 elements each
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        let mut out_off = 0usize;

        for _ in 0..2u32 {
            for l in 0..32usize {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0x0F) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

                let s0 = scales[sc_off + is] as i8 as f32;
                let s2 = scales[sc_off + is + 2] as i8 as f32;
                let s4 = scales[sc_off + is + 4] as i8 as f32;
                let s6 = scales[sc_off + is + 6] as i8 as f32;

                output[out_base + out_off + l] = d * s0 * q1 as f32;
                output[out_base + out_off + l + 32] = d * s2 * q2 as f32;
                output[out_base + out_off + l + 64] = d * s4 * q3 as f32;
                output[out_base + out_off + l + 96] = d * s6 * q4 as f32;
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
            out_off += 128;
        }
    }

    Ok(output)
}
