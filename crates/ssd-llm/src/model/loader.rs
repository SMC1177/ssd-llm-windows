//! Lazy mmap-based model loader with memory-mapped file access
//!
//! Platform-specific memory hints:
//! - **Unix/macOS**: `madvise(MADV_SEQUENTIAL|WILLNEED|DONTNEED)`
//! - **Windows**: `PrefetchVirtualMemory` for prefetch, `VirtualUnlock` for eviction hint

use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// A memory-mapped model file providing zero-copy access to tensor data
pub struct MmapLoader {
    mmap: Mmap,
    data_offset: u64,
}

impl MmapLoader {
    pub fn new(path: &Path, data_offset: u64) -> Result<Self> {
        let file = File::open(path).context("Failed to open model file")?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Advise the OS that we'll access this sequentially initially
        #[cfg(unix)]
        {
            unsafe {
                libc::madvise(
                    mmap.as_ptr() as *mut libc::c_void,
                    mmap.len(),
                    libc::MADV_SEQUENTIAL,
                );
            }
        }

        // On Windows, no direct MADV_SEQUENTIAL equivalent is needed —
        // the OS read-ahead heuristic handles sequential mmap access well.

        Ok(Self { mmap, data_offset })
    }

    /// Get a slice of tensor data at the given offset and size
    pub fn get_tensor_data(&self, offset: u64, size: u64) -> Result<&[u8]> {
        let abs_offset = self.data_offset + offset;
        let end = abs_offset + size;
        if end as usize > self.mmap.len() {
            anyhow::bail!(
                "Tensor data out of bounds: offset={}, size={}, file_len={}",
                abs_offset,
                size,
                self.mmap.len()
            );
        }
        Ok(&self.mmap[abs_offset as usize..end as usize])
    }

    /// Prefetch tensor data into memory
    ///
    /// - Unix: `madvise(MADV_WILLNEED)` — tells the kernel to start reading pages
    /// - Windows: `PrefetchVirtualMemory` — same effect on Win8+
    pub fn prefetch(&self, offset: u64, size: u64) {
        let abs_offset = self.data_offset + offset;

        #[cfg(unix)]
        {
            let ptr = unsafe { self.mmap.as_ptr().add(abs_offset as usize) };
            unsafe {
                libc::madvise(ptr as *mut libc::c_void, size as usize, libc::MADV_WILLNEED);
            }
        }

        #[cfg(target_os = "windows")]
        {
            use windows_sys::Win32::System::Memory::PrefetchVirtualMemory;
            use windows_sys::Win32::System::Threading::GetCurrentProcess;

            let ptr = unsafe { self.mmap.as_ptr().add(abs_offset as usize) };

            // WIN32_MEMORY_RANGE_ENTRY
            #[repr(C)]
            struct MemoryRangeEntry {
                virtual_address: *const u8,
                number_of_bytes: usize,
            }

            let entry = MemoryRangeEntry {
                virtual_address: ptr,
                number_of_bytes: size as usize,
            };

            unsafe {
                // PrefetchVirtualMemory(process, count, entries, flags)
                // Returns 0 on failure, non-zero on success
                let _ = PrefetchVirtualMemory(
                    GetCurrentProcess(),
                    1,
                    &entry as *const MemoryRangeEntry as *const _,
                    0,
                );
            }
        }
    }

    /// Hint the OS to release (evict) tensor data from the page cache
    ///
    /// - Unix: `madvise(MADV_DONTNEED)` — pages can be reclaimed immediately
    /// - Windows: `VirtualUnlock` — hints that pages are no longer needed in the
    ///   working set. The OS may still keep them cached but can reclaim them under
    ///   memory pressure. This is the closest Windows equivalent for mmap'd files.
    pub fn evict(&self, offset: u64, size: u64) {
        let abs_offset = self.data_offset + offset;

        #[cfg(unix)]
        {
            let ptr = unsafe { self.mmap.as_ptr().add(abs_offset as usize) };
            unsafe {
                libc::madvise(ptr as *mut libc::c_void, size as usize, libc::MADV_DONTNEED);
            }
        }

        #[cfg(target_os = "windows")]
        {
            use windows_sys::Win32::System::Memory::VirtualUnlock;

            let ptr = unsafe { self.mmap.as_ptr().add(abs_offset as usize) };
            unsafe {
                // VirtualUnlock hints that these pages can leave the working set.
                // It may fail if pages weren't locked — that's fine, we ignore the result.
                let _ = VirtualUnlock(ptr as *mut _, size as usize);
            }
        }
    }

    /// Total file size
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }
}
