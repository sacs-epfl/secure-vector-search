//! Background NVML peak-VRAM sampler.
//!
//! Constructed by the eval-harness (one instance per run) alongside the
//! [`crate::ClusterStore`]. Spawns a background thread that polls
//! `nvmlDeviceGetMemoryInfo` on a fixed cadence and updates an atomic
//! peak-bytes counter. The eval-harness reads the counter at run end
//! and writes `[gpu].peak-vram-bytes` into `run-metadata.toml`.
//!
//! Why a separate sampler instead of summing `DeviceCluster::device_bytes()`?
//!
//! - cuVS BfIndex's internal allocations (RMM pool pages, scratch,
//!   normalisation tables) aren't visible to the LRU's bookkeeping.
//!   Scorer-internal counting under-reports.
//! - cudarc's `CudaSlice` allocations are visible to us, but RMM
//!   pool growth on the cuVS side isn't — NVML sees both.
//! - One uniform mechanism across cuVS and cudarc scorers; one less
//!   axis of divergence between paths.
//!
//! Cost: one NVML handle + one sampler thread + sub-ms NVML calls at
//! the configured cadence. The default 50 ms cadence is well below
//! the per-query latency of every measurement in scope (worst case
//! BN-IVF at 8.8 M nprobe=2967 is multi-second per query), so the
//! sampler can't miss a peak that lasts longer than ~50 ms.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::budget::init_nvml;

/// Failure modes for [`PeakVramSampler::start`].
#[derive(Debug)]
pub enum SamplerError {
    NvmlUnavailable(nvml_wrapper::error::NvmlError),
    NoDevice(u32),
}

impl std::fmt::Display for SamplerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamplerError::NvmlUnavailable(e) => {
                write!(f, "NVML unavailable for peak sampler: {e}")
            }
            SamplerError::NoDevice(idx) => {
                write!(f, "NVML reports no device at index {idx}")
            }
        }
    }
}

impl std::error::Error for SamplerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SamplerError::NvmlUnavailable(e) => Some(e),
            SamplerError::NoDevice(_) => None,
        }
    }
}

/// Default sampling cadence — 50 ms. Pinned as a constant so the plan
/// doc and the runtime agree without a magic-number drift risk.
pub const DEFAULT_TICK: Duration = Duration::from_millis(50);

/// Background sampler. Construct via [`PeakVramSampler::start`]. The
/// sampler thread runs until `Drop`. Reading [`PeakVramSampler::peak_bytes`]
/// is lock-free.
pub struct PeakVramSampler {
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl PeakVramSampler {
    /// Convenience wrapper around [`Self::start`] with the
    /// [`DEFAULT_TICK`] cadence — what the eval-harness uses.
    pub fn start_default(device_index: u32) -> Result<Self, SamplerError> {
        Self::start(device_index, DEFAULT_TICK)
    }

    /// Start a sampler on the NVML device at `device_index`. Returns
    /// immediately; the sampler thread spawns and begins ticking
    /// every `tick` (default [`DEFAULT_TICK`]).
    ///
    /// On NVML init / device lookup failure, returns the error
    /// without spawning a thread — the eval-harness uses this to
    /// fall back to omitting `peak-vram-bytes` from the TOML rather
    /// than crashing a long sweep on an unavailable NVML.
    pub fn start(device_index: u32, tick: Duration) -> Result<Self, SamplerError> {
        let nvml = init_nvml().map_err(SamplerError::NvmlUnavailable)?;
        // Surface device-by-index errors up front; deferring them
        // into the sampler thread would silently leave peak=0 if
        // NVML didn't enumerate the requested card.
        let _ = nvml
            .device_by_index(device_index)
            .map_err(|_| SamplerError::NoDevice(device_index))?;
        drop(nvml);

        let peak = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let peak_t = Arc::clone(&peak);
        let stop_t = Arc::clone(&stop);

        let thread = thread::Builder::new()
            .name("scorer-gpu-common-nvml-sampler".to_string())
            .spawn(move || {
                // Re-init NVML inside the thread so the handle is
                // local to it — NVML handles aren't Send in nvml-wrapper.
                // Use the same `libnvidia-ml.so` / `.so.1` fallback as
                // the up-front init.
                let nvml = match init_nvml() {
                    Ok(n) => n,
                    Err(_) => return, // peak stays at 0; visible at run end.
                };
                let device = match nvml.device_by_index(device_index) {
                    Ok(d) => d,
                    Err(_) => return,
                };
                while !stop_t.load(Ordering::Relaxed) {
                    if let Ok(mem) = device.memory_info() {
                        // `used` includes everything resident on the
                        // device (this process + others sharing the
                        // GPU). For the headline sweep one process
                        // per card dominates, but record the
                        // pessimistic figure either way — the
                        // budget margin already includes everyone's
                        // overhead.
                        let used = mem.used;
                        // Atomic fetch-max via CAS loop. Lock-free
                        // and contention-free in practice (the
                        // background thread is the only writer,
                        // readers only observe).
                        let mut cur = peak_t.load(Ordering::Relaxed);
                        while used > cur {
                            match peak_t.compare_exchange_weak(
                                cur,
                                used,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => break,
                                Err(observed) => cur = observed,
                            }
                        }
                    }
                    thread::sleep(tick);
                }
            })
            .expect("spawn NVML sampler thread");

        Ok(PeakVramSampler {
            peak,
            stop,
            thread: Some(thread),
        })
    }

    /// Read the current peak. Lock-free.
    pub fn peak_bytes(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }
}

impl Drop for PeakVramSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            // Best-effort join. If the thread panicked the peak
            // we already recorded is still readable; not panicking
            // here keeps the Drop semantics clean for the harness
            // which builds + drops the sampler around every run.
            let _ = t.join();
        }
    }
}
