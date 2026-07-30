//! Error type for [`ClusterStore::get_or_upload`]. Wraps caller-side
//! upload-closure errors and surfaces CUDA / NVML errors uniformly so
//! every scorer crate can pull the same `ClusterStoreError<E>` shape
//! up into its own scheme-specific GPU error variant.

use std::error::Error as StdError;
use std::fmt;

use cudarc::driver::DriverError;

/// Failure modes for [`crate::ClusterStore::get_or_upload`]. Generic
/// over the caller's upload-closure error so each scorer can keep its
/// own per-scheme error variants on the inside of the closure.
#[derive(Debug)]
pub enum ClusterStoreError<E> {
    /// The upload closure returned an error. Surfaces every scheme-
    /// specific failure (cuVS Status, cudarc DriverError, host-side
    /// arithmetic, ...) without forcing them to share a single type.
    Upload(E),
    /// Even after evicting every other resident cluster, the requested
    /// payload's `device_bytes()` exceeds the store's configured
    /// budget. Indicates the budget itself is too small for the
    /// per-cluster shape — almost always a config-side bug, not a
    /// runtime condition. Returns the resolved budget and the
    /// requesting payload's size so the operator can size up.
    BudgetTooSmall {
        budget_bytes: u64,
        requested_bytes: u64,
    },
    /// A post-eviction context synchronize failed. The store syncs the
    /// CUDA context after evicting resident clusters so the device
    /// mempool actually reclaims the cross-stream async frees (without
    /// it, reserved VRAM climbs per query until OOM — see
    /// `get_or_upload`). A failure here means the driver/context is in a
    /// bad state, not a budgeting condition.
    Sync(DriverError),
}

impl<E: fmt::Display> fmt::Display for ClusterStoreError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClusterStoreError::Upload(e) => write!(f, "cluster upload closure failed: {e}"),
            ClusterStoreError::BudgetTooSmall {
                budget_bytes,
                requested_bytes,
            } => write!(
                f,
                "cluster payload exceeds VRAM budget even with full eviction: \
                 requested {requested_bytes} bytes, budget {budget_bytes} bytes"
            ),
            ClusterStoreError::Sync(e) => {
                write!(f, "post-eviction context synchronize failed: {e}")
            }
        }
    }
}

impl<E: StdError + 'static> StdError for ClusterStoreError<E> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            ClusterStoreError::Upload(e) => Some(e),
            ClusterStoreError::BudgetTooSmall { .. } => None,
            ClusterStoreError::Sync(e) => Some(e),
        }
    }
}
