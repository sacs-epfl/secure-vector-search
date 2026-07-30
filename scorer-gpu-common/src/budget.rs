//! VRAM-budget resolution.
//!
//! The [`crate::ClusterStore`] takes a hard byte budget; this module
//! exposes the helper the scorer crates use to compute that budget,
//! either verbatim from a user-supplied value or auto-detected as
//! 80 % of free VRAM at handle build via NVML.
//!
//! The 80 % factor leaves headroom for cuVS / cudarc internal
//! allocations that the LRU's `DeviceCluster::device_bytes()` reports
//! can't see — RMM pool fragmentation, scratch buffers, kernel arg
//! stacks. NVML peak sampling (see [`crate::PeakVramSampler`]) is the
//! cross-check: if peak rises substantially above
//! `device_bytes_sum + 20 %`, the budget margin is too tight for the
//! scheme on this host and the operator should pin a smaller
//! explicit value.

use std::fmt;

use nvml_wrapper::Nvml;

/// Try `Nvml::init()` first (looks for `libnvidia-ml.so`), then
/// fall back to the conventional versioned name `libnvidia-ml.so.1`
/// that most Linux distros ship without a versionless symlink.
/// Returns the first successful init; otherwise propagates the
/// second error so the caller sees the more useful "couldn't find
/// libnvidia-ml.so.1" message.
pub(crate) fn init_nvml() -> Result<Nvml, nvml_wrapper::error::NvmlError> {
    match Nvml::init() {
        Ok(n) => Ok(n),
        Err(_) => Nvml::builder()
            .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
            .init(),
    }
}

/// Translate a CUDA-runtime device index (the ordinal cudarc /
/// cuvs see, which respects `CUDA_VISIBLE_DEVICES`) to the
/// physical NVML index (which does not). Required because
/// `nvml_wrapper::Device::by_index` queries the underlying GPU
/// regardless of the CUDA mask — without the translation, a
/// `CUDA_VISIBLE_DEVICES=1` run that does its work on physical
/// card 1 would have its NVML peak sampler watch the idle card 0.
///
/// Algorithm:
/// - If `CUDA_VISIBLE_DEVICES` is unset / empty / unparseable,
///   return `cuda_idx` verbatim (no mask, indices match).
/// - Otherwise split the env var on commas and take the entry at
///   position `cuda_idx`. That entry is either a physical index
///   (`"0"` / `"1"` / …) or a UUID; only numeric indices are
///   handled here. UUID-masked runs return `cuda_idx` and the
///   sampler / budget land on a best-guess physical card —
///   document the gap rather than try to enumerate by UUID, which
///   would require pulling more of NVML's surface in.
pub fn nvml_index_for_cuda_device(cuda_idx: u32) -> u32 {
    let Ok(env) = std::env::var("CUDA_VISIBLE_DEVICES") else {
        return cuda_idx;
    };
    let trimmed = env.trim();
    if trimmed.is_empty() {
        return cuda_idx;
    }
    let Some(entry) = trimmed.split(',').nth(cuda_idx as usize) else {
        return cuda_idx;
    };
    entry.trim().parse::<u32>().unwrap_or(cuda_idx)
}

/// 80 % default safety margin. Documented in the plan doc and in the
/// `--gpu-vram-budget-bytes` CLI help; pinned as a constant here so
/// downstream consumers can reference it for sanity checks.
pub const DEFAULT_HEADROOM_FACTOR: f64 = 0.8;

/// Failure modes for [`resolve_budget`].
#[derive(Debug)]
pub enum BudgetResolverError {
    /// NVML initialised but no device at the requested index.
    NoDevice(u32),
    /// NVML failed to initialise — driver mismatch or NVML absent
    /// (container without `/dev/nvidiactl` mapped through, non-NVIDIA
    /// host, ...). The auto-detect path can't proceed; callers can
    /// always supply an explicit budget instead.
    NvmlUnavailable(nvml_wrapper::error::NvmlError),
    /// `nvmlDeviceGetMemoryInfo` failed on a known-good device.
    /// Usually a permissions / cgroup issue.
    MemoryInfo(nvml_wrapper::error::NvmlError),
}

impl fmt::Display for BudgetResolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BudgetResolverError::NoDevice(idx) => {
                write!(f, "NVML reports no device at index {idx}")
            }
            BudgetResolverError::NvmlUnavailable(e) => {
                write!(
                    f,
                    "NVML unavailable (pass --gpu-vram-budget-bytes to skip auto-detect): {e}"
                )
            }
            BudgetResolverError::MemoryInfo(e) => {
                write!(f, "nvmlDeviceGetMemoryInfo failed: {e}")
            }
        }
    }
}

impl std::error::Error for BudgetResolverError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BudgetResolverError::NvmlUnavailable(e) | BudgetResolverError::MemoryInfo(e) => Some(e),
            BudgetResolverError::NoDevice(_) => None,
        }
    }
}

/// Resolve the VRAM budget for a streaming cluster store.
///
/// - `Some(b)` returns `b` verbatim. The operator pinned a hard byte
///   cap and we honour it.
/// - `None` returns `floor(free_bytes × DEFAULT_HEADROOM_FACTOR)`
///   sampled from NVML on the device at `device_index`. NVML
///   initialisation fails surface as [`BudgetResolverError::NvmlUnavailable`]
///   so the caller can either fall back to an explicit budget or
///   error out cleanly.
///
/// `device_index` is the NVML ordinal, which under
/// `CUDA_VISIBLE_DEVICES=0` is always 0 even when the physical card
/// is index 1 on the host (CUDA renumbers but NVML doesn't —
/// `CUDA_VISIBLE_DEVICES` is a CUDA-runtime mask, not an NVML one).
/// Callers should pass `0` unless they explicitly want a different
/// physical card.
pub fn resolve_budget(
    explicit: Option<u64>,
    device_index: u32,
) -> Result<u64, BudgetResolverError> {
    if let Some(b) = explicit {
        return Ok(b);
    }
    let nvml = init_nvml().map_err(BudgetResolverError::NvmlUnavailable)?;
    let device = nvml
        .device_by_index(device_index)
        .map_err(|_| BudgetResolverError::NoDevice(device_index))?;
    let mem = device
        .memory_info()
        .map_err(BudgetResolverError::MemoryInfo)?;
    // Floor on conversion is fine — the factor is a soft margin, not
    // a precision-sensitive computation.
    let budget = ((mem.free as f64) * DEFAULT_HEADROOM_FACTOR) as u64;
    Ok(budget)
}
