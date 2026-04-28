//! Predictive prefetcher for layer-by-layer streaming

use crate::model::cache::LayerCache;
use crate::model::gguf::GgufFile;
use crate::ssd::streamer::SsdStreamer;
use tracing::debug;

/// Prefetch strategy
#[derive(Debug, Clone)]
pub enum PrefetchStrategy {
    /// Prefetch the next N layers (synchronous — used on Linux/Mac via madvise)
    LookAhead(usize),
    /// No prefetching
    None,
    /// Prefetch the next N layers from a Rayon background thread.
    ///
    /// On Windows, `PrefetchVirtualMemory` blocks the calling thread for ~7ms/layer
    /// when pages are not yet resident. This variant moves that work off the main thread,
    /// allowing GPU compute and SSD I/O to overlap.
    ///
    /// Safety: the prefetch task holds raw pointers into the mmap region. These pointers
    /// are valid for the lifetime of the model file (never unmapped during inference).
    /// Rayon tasks are waited on at program exit, so no use-after-free is possible.
    BackgroundTouch(usize),
}

impl Default for PrefetchStrategy {
    fn default() -> Self {
        Self::LookAhead(1)
    }
}

/// Predictive prefetcher that issues madvise/touch hints ahead of computation
pub struct Prefetcher {
    strategy: PrefetchStrategy,
}

impl Prefetcher {
    pub fn new(strategy: PrefetchStrategy) -> Self {
        Self { strategy }
    }

    /// Called before processing layer `current_layer`.
    /// Issues prefetch for upcoming layers.
    pub fn on_layer_start(
        &self,
        current_layer: u32,
        total_layers: u32,
        streamer: &SsdStreamer,
        gguf: &GgufFile,
        cache: &LayerCache,
    ) {
        match &self.strategy {
            PrefetchStrategy::LookAhead(n) => {
                for i in 1..=(*n as u32) {
                    let next = current_layer + i;
                    if next < total_layers && !cache.contains(next) {
                        streamer.prefetch_layer(gguf, next);
                        debug!("Prefetcher: issued WILLNEED for layer {}", next);
                    }
                }
            }
            PrefetchStrategy::BackgroundTouch(n) => {
                // Newtype so the raw pointer can cross the rayon::spawn boundary.
                // SAFETY invariants:
                //   1. The mmap region is never written after construction.
                //   2. The mmap lives until CudaGpu is dropped (end of inference session).
                //   3. The spawned task only reads pages — no data races.
                struct MmapSlice(*const u8, usize);
                unsafe impl Send for MmapSlice {}

                for i in 1..=(*n as u32) {
                    let next = current_layer + i;
                    if next < total_layers && !cache.contains(next) {
                        let ranges: Vec<MmapSlice> = gguf
                            .layer_tensors(next)
                            .iter()
                            .filter_map(|t| streamer.tensor_raw_slice(t).ok())
                            .map(|s: &[u8]| MmapSlice(s.as_ptr(), s.len()))
                            .collect();
                        if !ranges.is_empty() {
                            rayon::spawn(move || {
                                let mut sink: u8 = 0;
                                for MmapSlice(ptr, len) in &ranges {
                                    let slice = unsafe { std::slice::from_raw_parts(*ptr, *len) };
                                    for page in slice.chunks(4096) { sink ^= page[0]; }
                                }
                                std::hint::black_box(sink);
                                debug!("BackgroundTouch: prefetched layer {}", next);
                            });
                        }
                    }
                }
            }
            PrefetchStrategy::None => {}
        }
    }

    /// Called after processing a layer — can evict old layers
    pub fn on_layer_done(&self, completed_layer: u32, streamer: &SsdStreamer, gguf: &GgufFile) {
        match &self.strategy {
            PrefetchStrategy::None => {}
            _ => {
                if completed_layer >= 2 {
                    streamer.evict_layer(gguf, completed_layer - 2);
                    debug!("Prefetcher: evicted layer {} from OS cache", completed_layer - 2);
                }
            }
        }
    }
}
