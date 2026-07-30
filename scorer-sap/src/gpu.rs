//! cuVS-backed GPU paths for `scorer-sap`, built on the streaming
//! cluster store from `scorer-gpu-common`.
//!
//! Compiled only when the `gpu` Cargo feature is enabled. Linking
//! requires a Conda-resident `libcuvs` with `CMAKE_PREFIX_PATH` pointing
//! at it.
//!
//! ## What the GPU path actually does
//!
//! SAP encrypts each corpus vector as `c_i = s · m_i + λ_i` (a perturbed
//! point in the same d-dim Euclidean space — see `crypto.rs`). So the
//! per-query distance work is just L2 over fp64-then-downcast-to-fp32
//! ciphertexts. cuVS doesn't care that the entries are encrypted — it
//! consumes the same `Array2<f32>` shape it would for plaintext. The
//! crypto stays on CPU; only the linear-algebra GEMV moves to device.
//!
//! ## Two variants — flat and IVF
//!
//! - [`FlatGpuState`] backs `SapScorer` (the flat, single-cluster
//!   variant). One `cuvs::brute_force::Index` over the whole encrypted
//!   corpus. No LRU — there's only one cluster — but we still wrap it in
//!   a `ClusterStore` so the upload path, leak fix (manual `cuvsRMMFree`
//!   on Drop), and peak-VRAM sampling stay uniform with the IVF variant.
//!   The single cluster uploads lazily on first query.
//!
//! - [`IvfGpuState`] backs `SapIvfScorer`. Per-cluster device payload
//!   is uploaded on miss; LRU evicts when budget pressure rises.
//!   Same streaming pattern as `scorer-plaintext::gpu::GpuState`, and
//!   GPU and CPU runs share centroids and cluster assignments. Routing
//!   on host (`ivf::probe_route`); per-probed-cluster scoring on device;
//!   results merge on host.
//!
//! ## Dataset-lifetime workaround + leak fix
//!
//! cuvs's C++ `brute_force::index` stores a non-owning view of the
//! dataset; `BfIndex::build` consumes the wrapping `ManagedTensor` by
//! value so its DLPack deleter would otherwise fire and free the
//! device buffer before any query runs. Suppressing the deleter keeps
//! the buffer alive, but on its own leaks per-cluster dataset bytes on
//! every LRU eviction. Closed by capturing the device pointer + byte
//! count before suppressing the deleter and calling
//! `cuvs_sys::cuvsRMMFree` explicitly from `CuvsCluster::Drop`.

use cudarc::driver::CudaContext;
use cuvs::ManagedTensor;
use cuvs::Resources;
use cuvs::brute_force::Index as BfIndex;
use cuvs::distance_type::DistanceType;
use ivf_index::ivf;
use ndarray::Array2;
use scorer_core::Hit;
use scorer_gpu_common::{
    ClusterStore, ClusterStoreError, DeviceCluster, PeakVramSampler, nvml_index_for_cuda_device,
    resolve_budget,
};

use crate::crypto::Ciphertext;

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

// ---------------------------------------------------------------------------
// Shared CuvsCluster — leak-free cuVS BfIndex wrapper. Identical shape
// to `scorer-plaintext::gpu::CuvsCluster`; duplicated here rather than
// re-exported because cuvs-rs's `Resources` + `BfIndex` aren't declared
// `Send` upstream and the unsafe Send impl needs to live in the crate
// that constructs the value.
// ---------------------------------------------------------------------------

struct CuvsCluster {
    res: Resources,
    index: BfIndex,
    n: usize,
    dim: usize,
    dataset_dev_ptr: *mut std::ffi::c_void,
    dataset_bytes: usize,
}

unsafe impl Send for CuvsCluster {}

impl DeviceCluster for CuvsCluster {
    fn device_bytes(&self) -> u64 {
        // Encrypted fp32 corpus residency (post f64→f32 downcast).
        // cuVS internal scratch / RMM pool growth isn't tracked here;
        // the budget's 0.8 NVML headroom factor and the realised-peak
        // sampler cover those.
        (self.n as u64) * (self.dim as u64) * 4
    }
}

impl Drop for CuvsCluster {
    fn drop(&mut self) {
        // SAFETY: `dataset_dev_ptr` was allocated by `cuvsRMMAlloc`
        // (in `ManagedTensor::to_device`) and the DLPack deleter was
        // suppressed before `BfIndex::build` consumed the wrapping
        // `ManagedTensor`. We own the buffer's lifetime via this Drop;
        // `cuvsRMMFree` returns the page to the RMM pool. Rust
        // field-order drop runs `res` after `index`, so the BfIndex
        // destructor doesn't touch a freed buffer.
        unsafe {
            let _ = cuvs_sys::cuvsRMMFree(self.res.0, self.dataset_dev_ptr, self.dataset_bytes);
        }
    }
}

fn build_cuvs_cluster(
    host_vectors: &[f32],
    n: usize,
    dim: usize,
) -> Result<CuvsCluster, cuvs::Error> {
    let dataset = Array2::from_shape_vec((n, dim), host_vectors.to_vec())
        .expect("cluster shape (n × dim) is valid");
    let res = Resources::new()?;
    let dataset_dev = ManagedTensor::from(&dataset).to_device(&res)?;

    // Capture the device pointer before suppressing the deleter so
    // Drop can free the buffer later.
    let dataset_dev_ptr = unsafe { (*dataset_dev.as_ptr()).dl_tensor.data };
    let dataset_bytes = n * dim * std::mem::size_of::<f32>();
    unsafe {
        (*dataset_dev.as_ptr()).deleter = None;
    }

    let index = BfIndex::build(&res, DistanceType::L2Expanded, None, dataset_dev)?;
    Ok(CuvsCluster {
        res,
        index,
        n,
        dim,
        dataset_dev_ptr,
        dataset_bytes,
    })
}

// ---------------------------------------------------------------------------
// Flat: single-cluster streaming. The LRU is degenerate (one cluster)
// but the surface stays uniform with the IVF variant + the leak fix
// lands here too.
// ---------------------------------------------------------------------------

/// Flat GPU state for `SapScorer`. One device-resident cuVS index
/// over the entire encrypted corpus, held in a single-entry
/// `ClusterStore` so the surface is uniform with `IvfGpuState`.
pub struct FlatGpuState {
    /// Host-side encrypted-then-downcast-to-fp32 corpus, kept around
    /// so the upload closure can rebuild on (degenerate) LRU eviction
    /// and the NVML peak sampler runs over the whole run.
    host_vectors: Vec<f32>,
    n: usize,
    dim: usize,
    store: ClusterStore<CuvsCluster>,
    sampler: Option<PeakVramSampler>,
}

impl FlatGpuState {
    /// Construct the streaming state. The cuVS BfIndex isn't built
    /// yet — first `score()` call triggers the upload via
    /// `ClusterStore::get_or_upload`. That keeps the surface uniform
    /// with IVF (which also defers upload to first-touch) and lets
    /// the sampler start before any allocation.
    pub fn build(entries: &[Ciphertext], vram_budget_bytes: Option<u64>) -> Result<Self, GpuError> {
        let n = entries.len();
        let dim = entries.first().map(|c| c.c.len()).unwrap_or(0);
        let mut host_vectors: Vec<f32> = Vec::with_capacity(n * dim);
        for ct in entries {
            for &x in &ct.c {
                host_vectors.push(x as f32);
            }
        }

        let ctx = CudaContext::new(0)?;
        let upload_stream = ctx.new_stream()?;
        let compute_stream = ctx.new_stream()?;
        // Translate cuda visible-device 0 to its physical NVML index
        // so the budget resolver + peak sampler watch the same card
        // cudarc / cuvs run on. `CUDA_VISIBLE_DEVICES=1` puts cuvs on
        // physical card 1; NVML doesn't respect that mask, so we
        // resolve the physical index manually.
        let nvml_idx = nvml_index_for_cuda_device(0);
        let budget = resolve_budget(vram_budget_bytes, nvml_idx)?;
        let store = ClusterStore::<CuvsCluster>::new(ctx, budget, upload_stream, compute_stream);
        let sampler = PeakVramSampler::start_default(nvml_idx).ok();

        Ok(FlatGpuState {
            host_vectors,
            n,
            dim,
            store,
            sampler,
        })
    }

    pub fn peak_vram_bytes(&self) -> Option<u64> {
        self.sampler.as_ref().map(|s| s.peak_bytes())
    }

    pub fn vram_budget_bytes(&self) -> u64 {
        self.store.budget_bytes()
    }

    /// Score one encrypted query against the flat GPU index.
    pub fn score(&self, query_ct: &Ciphertext, k: usize) -> Result<Vec<Hit>, GpuError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let dim = query_ct.c.len();
        let q_f32: Vec<f32> = query_ct.c.iter().map(|&x| x as f32).collect();
        let q = Array2::from_shape_vec((1, dim), q_f32)
            .expect("flat-SAP query shape (1 × dim) is valid");

        let payload_hint = (self.n as u64) * (self.dim as u64) * 4;
        let host_vectors = &self.host_vectors;
        let n = self.n;
        let host_dim = self.dim;
        let cell = self.store.get_or_upload::<_, cuvs::Error>(
            // Single-cluster store: any stable id works; pick 0.
            0,
            payload_hint,
            |_upload_stream| build_cuvs_cluster(host_vectors, n, host_dim),
        )?;
        let cluster = cell.lock();
        let k_capped = k.min(cluster.n);
        if k_capped == 0 {
            return Ok(Vec::new());
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
        let mut hits = Vec::with_capacity(k_capped);
        for i in 0..k_capped {
            // Flat-SAP: row index == global corpus position.
            let row = row_n[i] as usize;
            hits.push(Hit {
                id: row as u32,
                score: -row_d[i],
            });
        }
        Ok(hits)
    }
}

// ---------------------------------------------------------------------------
// IVF: per-cluster streaming. Same shape as
// `scorer-plaintext::gpu::GpuState`.
// ---------------------------------------------------------------------------

struct HostCluster {
    vectors: Vec<f32>,
    n: usize,
    dim: usize,
    row_to_global: Vec<u32>,
}

/// Streaming GPU state for `SapIvfScorer`. Carries host-side per-
/// cluster encrypted-then-downcast-to-fp32 payloads, a
/// `ClusterStore` LRU over device-resident BfIndex objects, and an
/// NVML peak-VRAM sampler.
pub struct IvfGpuState {
    host: Vec<Option<HostCluster>>,
    store: ClusterStore<CuvsCluster>,
    sampler: Option<PeakVramSampler>,
}

impl IvfGpuState {
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
        // cudarc / cuvs run on. `CUDA_VISIBLE_DEVICES=1` puts cuvs on
        // physical card 1; NVML doesn't respect that mask, so we
        // resolve the physical index manually.
        let nvml_idx = nvml_index_for_cuda_device(0);
        let budget = resolve_budget(vram_budget_bytes, nvml_idx)?;
        let store = ClusterStore::<CuvsCluster>::new(ctx, budget, upload_stream, compute_stream);
        let sampler = PeakVramSampler::start_default(nvml_idx).ok();

        Ok(IvfGpuState {
            host,
            store,
            sampler,
        })
    }

    pub fn peak_vram_bytes(&self) -> Option<u64> {
        self.sampler.as_ref().map(|s| s.peak_bytes())
    }

    pub fn vram_budget_bytes(&self) -> u64 {
        self.store.budget_bytes()
    }

    /// IVF probe on GPU. Routing uses `routing_query_f32` (plaintext
    /// query against plaintext centroids); per-cluster cuVS scoring uses
    /// `scoring_query_f32` (encrypted-then-downcast query against
    /// encrypted-then-downcast cluster contents). Both queries must
    /// share `dim`.
    pub fn score(
        &self,
        host_index: &ivf::IvfIndex,
        routing_query_f32: &[f32],
        scoring_query_f32: &[f32],
        nprobe: usize,
        k: usize,
    ) -> Result<Vec<Hit>, GpuError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        debug_assert_eq!(
            routing_query_f32.len(),
            scoring_query_f32.len(),
            "routing and scoring queries must share dim"
        );
        let dim = scoring_query_f32.len();
        let probe_set = ivf::probe_route(host_index, routing_query_f32, nprobe);
        let q = Array2::from_shape_vec((1, dim), scoring_query_f32.to_vec())
            .expect("query shape (1 × dim) is valid");

        let mut all_hits: Vec<Hit> = Vec::new();
        for &ci in &probe_set {
            let Some(host_cluster) = self.host.get(ci).and_then(|c| c.as_ref()) else {
                continue;
            };
            let n = host_cluster.n;
            let host_dim = host_cluster.dim;
            let payload_hint = (n as u64) * (host_dim as u64) * 4;

            let cell =
                self.store
                    .get_or_upload::<_, cuvs::Error>(ci, payload_hint, |_upload_stream| {
                        build_cuvs_cluster(&host_cluster.vectors, n, host_dim)
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
