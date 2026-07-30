//! Shared GPU infrastructure for the streaming cluster-upload pattern.
//!
//! Today every `scorer-*` crate's `gpu` module uploads every IVF
//! cluster's device state at `upload_cluster` time and holds the
//! resulting per-cluster handles device-resident for the lifetime of
//! the `ClusterHandle`. Peak VRAM is therefore
//! `n_centroids × per_cluster_device_state` — fine at the 100 k smoke
//! scale, OOMs on every scheme at 8.8 M on a 32 GB card.
//!
//! This crate offers the alternative: a [`ClusterStore`] that resolves
//! a probed cluster id to a device-resident handle lazily, uploads on
//! miss under a dedicated upload `CudaStream`, evicts an LRU tail when
//! a configurable VRAM budget is hit, and exposes an NVML-sampled
//! peak-VRAM counter that the eval-harness persists into
//! `run-metadata.toml [gpu].peak-vram-bytes`.
//!
//! The store does not own the host-side cluster payload. The scorer's
//! `ClusterHandle` keeps that around so a post-eviction re-upload is
//! a pure memcpy + (for cuVS) BfIndex rebuild.
//!
//! ## Streams (v1)
//!
//! Two streams per store: one upload, one compute. Cluster N+1's H2D
//! copy overlaps cluster N's compute via stream events. N-stream
//! pools are a v2 follow-up.
//!
//! ## LRU semantics (v1)
//!
//! Warm across query batches: a cluster scored under query Q stays
//! resident for query Q+1's lookup. Eviction is alloc-fail-driven —
//! the closure passed to [`ClusterStore::get_or_upload`] runs; on
//! cudarc / cuvs OOM the LRU tail is popped (which runs the
//! `DeviceCluster`'s `Drop`) and the closure retries.

#![forbid(unsafe_op_in_unsafe_fn)]

// The entire crate is GPU infrastructure (cudarc + NVML). It is gated on
// the `cuda` feature so a CPU-only `cargo test --workspace` — which builds
// every workspace member — does not require `nvcc`/CUDA on the host. The
// crate is only ever reached through a downstream scorer's `gpu` feature,
// which turns `cuda` on. Without the feature this lib is intentionally empty.
#[cfg(feature = "cuda")]
mod budget;
#[cfg(feature = "cuda")]
mod error;
#[cfg(feature = "cuda")]
mod sampler;
#[cfg(feature = "cuda")]
mod store;

#[cfg(feature = "cuda")]
pub use budget::{BudgetResolverError, nvml_index_for_cuda_device, resolve_budget};
#[cfg(feature = "cuda")]
pub use error::ClusterStoreError;
#[cfg(feature = "cuda")]
pub use sampler::{PeakVramSampler, SamplerError};
#[cfg(feature = "cuda")]
pub use store::{ClusterStore, ClusterStoreStats, DeviceCluster, LruCell, global_stats};
