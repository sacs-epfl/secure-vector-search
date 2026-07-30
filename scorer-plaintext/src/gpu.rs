//! cuVS-backed GPU path for `scorer-plaintext`, built on the streaming
//! cluster store from `scorer-gpu-common`.
//!
//! Compiled only when the `gpu` Cargo feature is enabled. Linking
//! requires a Conda-resident `libcuvs` with `CMAKE_PREFIX_PATH` pointing
//! at it.
//!
//! ## Streaming shape
//!
//! Building and holding all `n_centroids` per-cluster cuVS BfIndex
//! objects device-resident at once has peak VRAM
//! `n_centroids × m_avg × dim × 4` (tens of GB at large corpora),
//! which overflows a single card. Instead, `GpuState` keeps the
//! host-side fp32 cluster payloads on the host and lazily builds
//! BfIndex objects on miss. The streaming [`ClusterStore`] holds the
//! device-resident BfIndex + Resources tuple under an LRU bounded by a
//! VRAM budget (default NVML 80 %); on miss-then-evict it drops the LRU
//! tail's Resources + BfIndex, which returns device memory to RMM. A
//! subsequent re-hit on the evicted cluster pays the BfIndex rebuild
//! cost.
//!
//! The cuVS path runs compute on cuVS's internal stream — the upload
//! stream the store hands out is wired through the API surface but
//! cuVS doesn't honour it (it constructs its own internal stream
//! under `Resources::new()`). The store's API is uniform across
//! cuVS + cudarc paths so future polish can route cuVS through the
//! explicit upload/compute streams when cuvs-rs exposes the necessary
//! hooks.
//!
//! ## Routing parity invariant
//!
//! Cluster centroids stay in lock-step with the host CPU IVF index, so
//! the probed cluster set is identical across substrates. Routing runs
//! on the host (`ivf::probe_route`); per-probed-cluster scoring runs on
//! device; results merge on host.

use cudarc::driver::CudaContext;
use cuvs::ManagedTensor;
use cuvs::Resources;
use cuvs::brute_force::Index as BfIndex;
use cuvs::distance_type::DistanceType;
use ivf_index::ivf;
use ndarray::Array2;
use scorer_core::{Hit, Vector};
use scorer_gpu_common::{
    ClusterStore, ClusterStoreError, DeviceCluster, PeakVramSampler, nvml_index_for_cuda_device,
    resolve_budget,
};

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("cuVS error: {0}")]
    Cuvs(String),
    #[error("VRAM budget resolution failed: {0}")]
    Budget(String),
    #[error("NVML peak-VRAM sampler failed: {0}")]
    Sampler(String),
    #[error("CUDA driver error: {0}")]
    Driver(String),
    #[error("cluster payload {requested_bytes} bytes exceeds budget {budget_bytes} bytes")]
    BudgetTooSmall {
        budget_bytes: u64,
        requested_bytes: u64,
    },
}

impl From<cuvs::Error> for GpuError {
    fn from(e: cuvs::Error) -> Self {
        GpuError::Cuvs(format!("{e:?}"))
    }
}

impl From<scorer_gpu_common::BudgetResolverError> for GpuError {
    fn from(e: scorer_gpu_common::BudgetResolverError) -> Self {
        GpuError::Budget(format!("{e}"))
    }
}

impl From<scorer_gpu_common::SamplerError> for GpuError {
    fn from(e: scorer_gpu_common::SamplerError) -> Self {
        GpuError::Sampler(format!("{e}"))
    }
}

impl From<cudarc::driver::DriverError> for GpuError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        GpuError::Driver(format!("{e:?}"))
    }
}

impl From<ClusterStoreError<cuvs::Error>> for GpuError {
    fn from(e: ClusterStoreError<cuvs::Error>) -> Self {
        match e {
            ClusterStoreError::Upload(c) => GpuError::Cuvs(format!("{c:?}")),
            ClusterStoreError::BudgetTooSmall {
                budget_bytes,
                requested_bytes,
            } => GpuError::BudgetTooSmall {
                budget_bytes,
                requested_bytes,
            },
            ClusterStoreError::Sync(c) => GpuError::Cuvs(format!("{c:?}")),
        }
    }
}

/// Per-cluster host-side payload. Held in `GpuState` for the
/// `ClusterHandle` lifetime; fed to the upload closure on miss.
struct HostCluster {
    /// Flat fp32 vectors, n × dim, row-major.
    vectors: Vec<f32>,
    n: usize,
    dim: usize,
    /// Row index within the cluster → global vector id. Same mapping
    /// the pre-streaming path used; cuVS row results land here for
    /// translation back to global ids.
    row_to_global: Vec<u32>,
}

/// Per-cluster device-resident payload. Owns the `Resources` (one
/// per cluster — cuvs-rs's stream / RMM-resource pair) and the
/// BfIndex built over the dataset. Dropping a `CuvsCluster` runs the
/// cuVS destructors, returning device memory to the RMM pool.
///
/// `n × dim × 4` is reported as `device_bytes()` — the deterministic
/// payload size, matching what the LRU's pre-evict hint expected.
/// cuVS's internal allocations (scratch, normalisation) are not
/// tracked here; the NVML peak sampler is the source of truth for
/// realised VRAM.
struct CuvsCluster {
    // SAFETY note: a cuVS dataset-lifetime workaround is preserved
    // below — when we call `BfIndex::build(&res, ..., None, dataset_dev)`
    // the C++ index holds a non-owning view of `dataset_dev`'s pointer.
    // The DLPack `deleter` is suppressed before `build` so the device
    // allocation outlives the local `ManagedTensor`. Under the streaming
    // pattern we then own the dataset device pointer + byte count and
    // free it on Drop via cuvsRMMFree (see `dataset_dev_ptr` /
    // `dataset_bytes` below) so each LRU eviction actually returns
    // memory to the RMM pool. Without that, every cuvs cluster upload
    // leaks `m × dim × 4` bytes and the LRU-bounded budget grows
    // unbounded across a large sweep.
    //
    // Resources must live alongside the cluster — `cuvsRMMFree`
    // takes a `cuvsResources_t` and the same one used by
    // `cuvsRMMAlloc` in `ManagedTensor::to_device` is most
    // straightforward. cuvs-rs's `Resources::new` builds a fresh
    // one each time; the underlying RMM pool is process-wide so
    // any Resources handle works for free as well as alloc.
    res: Resources,
    index: BfIndex,
    n: usize,
    dim: usize,
    dataset_dev_ptr: *mut std::ffi::c_void,
    dataset_bytes: usize,
}

// SAFETY: cuvs's `Resources` and `Index` aren't declared `Send`
// upstream — they wrap opaque CUDA handles. The ClusterStore wraps
// each `CuvsCluster` in a `Mutex` and accesses them under a single
// query at a time on a `tokio::task::spawn_blocking` worker. Drop
// these `unsafe`s if cuvs-rs ever exposes `Send`/`Sync` directly.
unsafe impl Send for CuvsCluster {}

impl DeviceCluster for CuvsCluster {
    fn device_bytes(&self) -> u64 {
        // fp32 corpus residency. cuVS's internal scratch / RMM pages
        // aren't tracked here; the budget's 0.8 NVML headroom factor
        // is what keeps room for them.
        (self.n as u64) * (self.dim as u64) * 4
    }
}

impl Drop for CuvsCluster {
    fn drop(&mut self) {
        // Drop order: Rust drops fields in declaration order. `res`
        // is declared before `index`, so `res` drops first — which
        // would invalidate cuvsResourcesDestroy before the BfIndex's
        // destructor runs. To keep the order safe (index destructor
        // → free buffer → drop res), we manually free here under
        // the still-live res. The implicit field-order drops then
        // run BfIndex (no device deref) and Resources (no buffer
        // deref) in turn.
        //
        // SAFETY: `dataset_dev_ptr` was allocated by `cuvsRMMAlloc`
        // (see cuvs-rs `ManagedTensor::to_device`). The corresponding
        // free is `cuvsRMMFree`. We pass `dataset_bytes` (the
        // original allocation size) because cuvs's RMM bindings
        // require the size — RMM pool free is size-tracked, unlike
        // raw `cudaFree`.
        unsafe {
            let _ = cuvs_sys::cuvsRMMFree(self.res.0, self.dataset_dev_ptr, self.dataset_bytes);
        }
    }
}

/// Streaming GPU state for `PlaintextScorer`. Carries the host-side
/// per-cluster payloads, a `ClusterStore` LRU over device-resident
/// BfIndex objects, and an NVML peak-VRAM sampler.
pub struct GpuState {
    host: Vec<Option<HostCluster>>,
    store: ClusterStore<CuvsCluster>,
    sampler: Option<PeakVramSampler>,
}

impl GpuState {
    /// Build the streaming state. Constructs the host-side cluster
    /// payloads from `host_index`, resolves the VRAM budget (via
    /// NVML auto-detect when `vram_budget_bytes` is `None`), spins
    /// up two cudarc streams (upload + compute — both pinned to
    /// device 0 under the active `CUDA_VISIBLE_DEVICES` mask), and
    /// kicks off the NVML peak sampler. No device memory is
    /// allocated for cluster payloads here — the first miss-path
    /// query is what triggers the first upload.
    pub fn build(
        host_index: &ivf::IvfIndex,
        vram_budget_bytes: Option<u64>,
    ) -> Result<Self, GpuError> {
        let dim = host_index.centroids.first().map(|c| c.len()).unwrap_or(0);
        let mut host = Vec::with_capacity(host_index.clusters.len());
        for cluster in &host_index.clusters {
            if cluster.is_empty() {
                host.push(None);
                continue;
            }
            let n = cluster.len();
            let mut vectors: Vec<f32> = Vec::with_capacity(n * dim);
            let mut row_to_global: Vec<u32> = Vec::with_capacity(n);
            for (id, v) in cluster {
                vectors.extend_from_slice(v);
                row_to_global.push(*id);
            }
            host.push(Some(HostCluster {
                vectors,
                n,
                dim,
                row_to_global,
            }));
        }

        let ctx = CudaContext::new(0)?;
        let upload_stream = ctx.new_stream()?;
        let compute_stream = ctx.new_stream()?;
        // Translate cuda visible-device 0 to its physical NVML index
        // so the budget resolver + peak sampler watch the same card
        // cudarc / cuvs run on. `CUDA_VISIBLE_DEVICES=1` (our
        // two-card eval pattern) puts cuvs on physical card 1; the
        // sampler needs to follow.
        let nvml_idx = nvml_index_for_cuda_device(0);
        let budget = resolve_budget(vram_budget_bytes, nvml_idx)?;
        let store = ClusterStore::<CuvsCluster>::new(ctx, budget, upload_stream, compute_stream);
        let sampler = PeakVramSampler::start_default(nvml_idx).ok();

        Ok(GpuState {
            host,
            store,
            sampler,
        })
    }

    /// NVML-sampled peak VRAM since handle build. `None` if NVML was
    /// unavailable at `build()` time. Read by eval-harness once at
    /// run end, persisted into `run-metadata.toml [gpu].peak-vram-bytes`.
    pub fn peak_vram_bytes(&self) -> Option<u64> {
        self.sampler.as_ref().map(|s| s.peak_bytes())
    }

    /// Resolved VRAM budget for this handle in bytes.
    pub fn vram_budget_bytes(&self) -> u64 {
        self.store.budget_bytes()
    }

    /// IVF probe on GPU. Routes on host (same centroids as CPU),
    /// then resolves each probed cluster against the streaming
    /// store (upload on miss, evict LRU tail under budget pressure).
    /// cuVS row indices translate back to global IDs via the per-
    /// cluster `row_to_global`; results merge top-k by score.
    pub fn score(
        &self,
        host_index: &ivf::IvfIndex,
        query: &Vector,
        nprobe: usize,
        k: usize,
    ) -> Result<Vec<Hit>, GpuError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let dim = query.0.len();
        let probe_set = ivf::probe_route(host_index, &query.0, nprobe);
        let q = Array2::from_shape_vec((1, dim), query.0.clone())
            .expect("query shape (1 × dim) is valid");

        let mut all_hits: Vec<Hit> = Vec::new();
        for &ci in &probe_set {
            let Some(host_cluster) = self.host.get(ci).and_then(|c| c.as_ref()) else {
                continue;
            };
            let n = host_cluster.n;
            let dim = host_cluster.dim;
            let payload_hint = (n as u64) * (dim as u64) * 4;

            // Bind a local borrow that the upload closure can take
            // by reference. cuvs::ManagedTensor::from(&array) needs
            // a stable f32 slice — we keep the host vectors arena-
            // alive for the whole handle lifetime, so the
            // dataset.from(&host_cluster.vectors) reference stays
            // valid until BfIndex::build copies it to device.
            let cell =
                self.store
                    .get_or_upload::<_, cuvs::Error>(ci, payload_hint, |_upload_stream| {
                        build_cuvs_cluster(host_cluster)
                    })?;
            let cluster = cell.lock();
            let k_capped = k.min(cluster.n);
            if k_capped == 0 {
                continue;
            }

            let mut neighbors_host = Array2::<i64>::zeros((1, k_capped));
            let mut distances_host = Array2::<f32>::zeros((1, k_capped));
            let q_dev = ManagedTensor::from(&q).to_device(&cluster.res)?;
            let n_dev = ManagedTensor::from(&neighbors_host).to_device(&cluster.res)?;
            let d_dev = ManagedTensor::from(&distances_host).to_device(&cluster.res)?;
            cluster.index.search(&cluster.res, &q_dev, &n_dev, &d_dev)?;
            d_dev.to_host(&cluster.res, &mut distances_host)?;
            n_dev.to_host(&cluster.res, &mut neighbors_host)?;

            let row_n = neighbors_host.row(0);
            let row_d = distances_host.row(0);
            for i in 0..k_capped {
                let row = row_n[i] as usize;
                let global_id = host_cluster.row_to_global[row];
                all_hits.push(Hit {
                    id: global_id,
                    score: -row_d[i],
                });
            }
        }

        all_hits.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        all_hits.truncate(k);
        Ok(all_hits)
    }
}

/// Build a cuVS BfIndex over a host cluster's fp32 vectors. The
/// dataset-lifetime workaround (suppressing the DLPack deleter so the
/// device allocation survives `ManagedTensor` drop) is needed because
/// the cuVS C++ index stores a non-owning device pointer.
fn build_cuvs_cluster(host: &HostCluster) -> Result<CuvsCluster, cuvs::Error> {
    let dataset = Array2::from_shape_vec((host.n, host.dim), host.vectors.clone())
        .expect("cluster shape (n × dim) is valid");
    let res = Resources::new()?;
    let dataset_dev = ManagedTensor::from(&dataset).to_device(&res)?;

    // Capture the device pointer + byte count from the freshly-
    // allocated DLManagedTensor BEFORE we suppress the deleter.
    // These flow into `CuvsCluster` so `CuvsCluster::Drop` can
    // call `cuvsRMMFree` later and return the buffer to the RMM
    // pool on LRU eviction or handle teardown.
    let dataset_dev_ptr = unsafe { (*dataset_dev.as_ptr()).dl_tensor.data };
    let dataset_bytes = host.n * host.dim * std::mem::size_of::<f32>();

    // SAFETY: cuvs's C++ `brute_force::index` keeps a non-owning
    // view of `dataset_dev`; `BfIndex::build` consumes the
    // `ManagedTensor` by value and the wrapper's `Drop` would run
    // the DLPack deleter (`rmm_free_tensor` → `cuvsRMMFree`),
    // freeing the device buffer the C++ index still points at.
    // Suppressing the deleter keeps the buffer alive past the
    // build call; we own its lifetime via `dataset_dev_ptr` +
    // `dataset_bytes` and free it from `CuvsCluster::Drop`.
    unsafe {
        (*dataset_dev.as_ptr()).deleter = None;
    }

    let index = BfIndex::build(&res, DistanceType::L2Expanded, None, dataset_dev)?;
    Ok(CuvsCluster {
        res,
        index,
        n: host.n,
        dim: host.dim,
        dataset_dev_ptr,
        dataset_bytes,
    })
}
