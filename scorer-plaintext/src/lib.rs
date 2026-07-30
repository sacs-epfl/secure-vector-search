use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use ivf_index::{ivf, kmeans};
use scorer_core::{
    BuildOutcome, CommunicationCost, Device, Hit, ProgressReporter, Scorer, Vector, cache,
};

#[cfg(feature = "gpu")]
mod gpu;

pub struct PlaintextConfig {
    /// Number of IVF centroids. Rule of thumb: `ceil(sqrt(N))`.
    pub n_centroids: usize,
    /// Number of clusters probed per query.
    pub nprobe: usize,
    /// Seed for k-means++ initialisation (reproducibility).
    pub train_seed: u64,
    /// Maximum Lloyd's iterations.
    pub max_iter: usize,
    /// Optional progress reporter for k-means init/iter and build-complete
    /// milestones. `on_encrypt` is unused (this scorer has no encryption).
    pub progress: Option<Arc<dyn ProgressReporter>>,
    /// Compute substrate. `Cpu` (default) runs the IVF probe on host.
    /// `Gpu` requires the `gpu` Cargo feature; without it
    /// `upload_cluster` returns [`PlaintextError::GpuFeatureNotEnabled`].
    pub device: Device,
    /// VRAM budget for the streaming cluster store on GPU runs. `None`
    /// (default) auto-detects 80 % of free VRAM via NVML at handle
    /// build. `Some(b)` pins a hard byte cap. Ignored on CPU runs.
    pub vram_budget_bytes: Option<u64>,
}

impl Default for PlaintextConfig {
    fn default() -> Self {
        PlaintextConfig {
            n_centroids: 16,
            nprobe: 4,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        }
    }
}

pub struct PlaintextHandle {
    /// Host-side IVF index. Both CPU and GPU handles hold this — the GPU
    /// path uses it for client-side routing (`ivf::probe_route`) so the
    /// same k-means centroids drive both substrates and the probed
    /// cluster set is identical across them.
    index: Arc<ivf::IvfIndex>,
    nprobe: usize,
    dim: usize,
    inner: HandleInner,
}

enum HandleInner {
    Cpu,
    /// Cluster-aware probe on GPU. The state holds one cuVS brute-force
    /// index per IVF cluster (sharing the host's centroids, so the
    /// probed cluster set is identical across substrates). Routing runs
    /// on host; per-probed-cluster scoring runs on device; results merge
    /// on host. See `gpu.rs`'s module doc-comment for the launch-latency
    /// caveat.
    #[cfg(feature = "gpu")]
    Gpu(gpu::GpuState),
}

impl fmt::Debug for PlaintextHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device = match &self.inner {
            HandleInner::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(_) => "gpu",
        };
        f.debug_struct("PlaintextHandle")
            .field("device", &device)
            .field("dim", &self.dim)
            .field("n_centroids", &self.index.centroids.len())
            .field("nprobe", &self.nprobe)
            .field(
                "cluster_sizes",
                &self
                    .index
                    .clusters
                    .iter()
                    .map(|c| c.len())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl PlaintextHandle {
    /// NVML-sampled peak VRAM for this handle's lifetime. `None` on CPU
    /// handles or when NVML was unavailable at GPU handle build.
    pub fn peak_vram_bytes(&self) -> Option<u64> {
        match &self.inner {
            HandleInner::Cpu => None,
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => state.peak_vram_bytes(),
        }
    }

    /// Resolved VRAM budget for this handle's streaming cluster store.
    /// `None` on CPU handles.
    pub fn vram_budget_bytes(&self) -> Option<u64> {
        match &self.inner {
            HandleInner::Cpu => None,
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => Some(state.vram_budget_bytes()),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PlaintextError {
    #[error("cluster must contain at least one vector")]
    EmptyCluster,
    #[error("n_centroids {0} exceeds cluster size")]
    TooManyCentroids(usize),
    #[error("query dimension {query_dim} does not match index dimension {index_dim}")]
    DimensionMismatch { query_dim: usize, index_dim: usize },
    #[error("spawn_blocking task panicked")]
    TaskPanic,
    /// `Device::Gpu` was requested but the crate was built without the
    /// `gpu` Cargo feature, so no cuVS code path is compiled in.
    /// Surfaced as a clean error rather than silently falling back to
    /// CPU — silent fallback would mislabel rows as GPU-measured when
    /// they ran on CPU.
    #[error("Device::Gpu requested but the `gpu` Cargo feature is not enabled in this build")]
    GpuFeatureNotEnabled,
    /// GPU runtime error from cuVS (build / search / device alloc).
    /// Boxed to keep the variant small when the feature is off.
    #[cfg(feature = "gpu")]
    #[error("cuVS GPU error: {0}")]
    Gpu(#[from] gpu::GpuError),
}

// (n_centroids, train_seed, max_iter, corpus_len) — enough to detect a same-corpus nprobe sweep.
type CacheKey = (usize, u64, usize, usize);

struct IndexCache {
    key: CacheKey,
    index: Arc<ivf::IvfIndex>,
}

/// Unit scorer. Uses a single-entry in-memory cache so that sweeping `nprobe` over the
/// same corpus builds k-means only once. If `cache_dir` is set, the built index is also
/// persisted to disk and reloaded on future runs, skipping k-means entirely.
pub struct PlaintextScorer {
    cache: Mutex<Option<IndexCache>>,
    cache_dir: Option<PathBuf>,
}

impl PlaintextScorer {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(None),
            cache_dir: None,
        }
    }

    pub fn with_cache_dir(dir: impl Into<PathBuf>) -> Self {
        Self {
            cache: Mutex::new(None),
            cache_dir: Some(dir.into()),
        }
    }
}

impl Default for PlaintextScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Scorer for PlaintextScorer {
    type Config = PlaintextConfig;
    type ClusterHandle = PlaintextHandle;
    type Error = PlaintextError;

    async fn upload_cluster(
        &self,
        config: &PlaintextConfig,
        vectors: &[Vector],
    ) -> Result<(PlaintextHandle, BuildOutcome), PlaintextError> {
        let build_start = Instant::now();
        if vectors.is_empty() {
            return Err(PlaintextError::EmptyCluster);
        }
        if config.n_centroids > vectors.len() {
            return Err(PlaintextError::TooManyCentroids(config.n_centroids));
        }

        let dim = vectors[0].0.len();
        let nprobe = config.nprobe;
        let cache_key: CacheKey = (
            config.n_centroids,
            config.train_seed,
            config.max_iter,
            vectors.len(),
        );

        // Check cache before doing the expensive k-means training.
        let cached = {
            let guard = self.cache.lock().unwrap();
            guard.as_ref().and_then(|c| {
                if c.key == cache_key {
                    Some(Arc::clone(&c.index))
                } else {
                    None
                }
            })
        };

        let in_memory_hit = cached.is_some();
        let mut disk_hit = false;
        let index = match cached {
            Some(idx) => idx,
            None => {
                let n_centroids = config.n_centroids;
                let train_seed = config.train_seed;
                let max_iter = config.max_iter;

                let cache_parts: [Vec<u8>; 5] = [
                    b"plaintext-ivf".to_vec(),
                    (cache_key.0 as u64).to_le_bytes().to_vec(),
                    cache_key.1.to_le_bytes().to_vec(),
                    (cache_key.2 as u64).to_le_bytes().to_vec(),
                    (cache_key.3 as u64).to_le_bytes().to_vec(),
                ];
                let parts_refs: Vec<&[u8]> = cache_parts.iter().map(|v| v.as_slice()).collect();
                let fp = cache::fingerprint(&parts_refs);
                let disk_path = self
                    .cache_dir
                    .as_deref()
                    .map(|dir| dir.join(format!(".ivf-cache-{fp}.bin")));

                // Try loading from disk before running k-means.
                let from_disk = if let Some(ref path) = disk_path {
                    if path.exists() {
                        let p = path.clone();
                        eprintln!("Loading cached IVF index from {p:?}…");
                        let parts_owned = cache_parts.clone();
                        tokio::task::spawn_blocking(move || {
                            let refs: Vec<&[u8]> =
                                parts_owned.iter().map(|v| v.as_slice()).collect();
                            ivf::load_index_with_header(&p, &refs).ok().flatten()
                        })
                        .await
                        .ok()
                        .flatten()
                    } else {
                        None
                    }
                } else {
                    None
                };

                let progress = config.progress.clone();

                let arc = if let Some(loaded) = from_disk {
                    disk_hit = true;
                    if let Some(ref p) = progress {
                        p.on_build_complete();
                    }
                    Arc::new(loaded)
                } else {
                    let data: Vec<Vec<f32>> = vectors.iter().map(|v| v.0.clone()).collect();
                    let save_path = disk_path.clone();
                    let save_parts = cache_parts.clone();

                    let built = tokio::task::spawn_blocking(move || {
                        let centroids = kmeans::train(
                            &data,
                            n_centroids,
                            train_seed,
                            max_iter,
                            |ci| {
                                if let Some(ref p) = progress {
                                    p.on_init_centroid(ci);
                                }
                            },
                            |i| {
                                if let Some(ref p) = progress {
                                    p.on_kmeans_iter(i);
                                }
                            },
                        );
                        let index = ivf::build_index(&data, centroids, dim);
                        if let Some(ref p) = progress {
                            p.on_build_complete();
                        }
                        if let Some(ref path) = save_path {
                            let refs: Vec<&[u8]> =
                                save_parts.iter().map(|v| v.as_slice()).collect();
                            if let Err(e) = ivf::save_index_with_header(&index, path, &refs) {
                                eprintln!("Warning: could not save IVF cache to {path:?}: {e}");
                            }
                        }
                        index
                    })
                    .await
                    .map_err(|_| PlaintextError::TaskPanic)?;

                    Arc::new(built)
                };

                let mut guard = self.cache.lock().unwrap();
                *guard = Some(IndexCache {
                    key: cache_key,
                    index: Arc::clone(&arc),
                });
                arc
            }
        };

        // Branch on substrate. The host-side IVF index is built either
        // way (so centroids and cluster assignments match across
        // CPU/GPU runs); GPU additionally uploads the corpus to device
        // and constructs a cuVS brute-force index over it.
        let inner = match config.device {
            Device::Cpu => HandleInner::Cpu,
            Device::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    // GPU build is non-trivial work (one cuVS index
                    // per cluster) — do it in spawn_blocking so the
                    // async runtime doesn't stall on cuVS's CMake-
                    // bound calls. Cloning the host index Arc is O(1);
                    // the cuvs build reads through the inner clusters.
                    let host_index = Arc::clone(&index);
                    let vram_budget = config.vram_budget_bytes;
                    let state = tokio::task::spawn_blocking(move || {
                        gpu::GpuState::build(&host_index, vram_budget)
                    })
                    .await
                    .map_err(|_| PlaintextError::TaskPanic)??;
                    HandleInner::Gpu(state)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(PlaintextError::GpuFeatureNotEnabled);
                }
            }
        };

        let outcome = BuildOutcome {
            cache_hit: in_memory_hit || disk_hit,
            build_duration: build_start.elapsed(),
        };
        Ok((
            PlaintextHandle {
                index,
                nprobe,
                dim,
                inner,
            },
            outcome,
        ))
    }

    async fn score(
        &self,
        handle: &PlaintextHandle,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<Hit>, PlaintextError> {
        if query.0.len() != handle.dim {
            return Err(PlaintextError::DimensionMismatch {
                query_dim: query.0.len(),
                index_dim: handle.dim,
            });
        }

        match &handle.inner {
            HandleInner::Cpu => {
                let index = Arc::clone(&handle.index);
                let query_vec = query.0.clone();
                let nprobe = handle.nprobe;
                let hits =
                    tokio::task::spawn_blocking(move || ivf::probe(&index, &query_vec, nprobe, k))
                        .await
                        .map_err(|_| PlaintextError::TaskPanic)?;
                Ok(hits)
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                // Cluster-aware probe on GPU honours nprobe. The
                // host-side IvfIndex drives routing; per-probed-cluster
                // scoring runs on device.
                state
                    .score(&handle.index, query, handle.nprobe, k)
                    .map_err(Into::into)
            }
        }
    }

    fn communication_cost(&self, handle: &PlaintextHandle, k: usize) -> CommunicationCost {
        // IVF accounting: each probed cluster ships up to min(k, m_i)
        // hits back, the client merges. This analytical estimate uses
        // the mean cluster size m_avg = N / n_centroids; the realised
        // per-query path lives in `score_with_realised_cost`.
        let n_clusters = handle.index.clusters.len();
        let m_total: usize = handle.index.clusters.iter().map(|c| c.len()).sum();
        let m_avg = m_total.checked_div(n_clusters).unwrap_or(0);
        let cluster_response_bytes = (k.min(m_avg) as u64) * 8;
        let response_bytes = cluster_response_bytes * handle.nprobe as u64;
        // GPU bandwidth proxy = fp32 vectors streamed across probed
        // clusters per query. Analytical estimate uses nprobe × m_avg.
        let effective_bytes_per_query = handle.nprobe as u64 * m_avg as u64 * handle.dim as u64 * 4;
        CommunicationCost {
            query_bytes: handle.dim as u64 * 4,
            response_bytes,
            cluster_response_bytes,
            setup_bytes: 0,
            pre_query_offline_up_bytes: 0,
            pre_query_offline_down_bytes: 0,
            effective_bytes_per_query,
        }
    }

    async fn score_with_realised_cost(
        &self,
        handle: &PlaintextHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, CommunicationCost), PlaintextError> {
        if query.0.len() != handle.dim {
            return Err(PlaintextError::DimensionMismatch {
                query_dim: query.0.len(),
                index_dim: handle.dim,
            });
        }

        match &handle.inner {
            HandleInner::Cpu => {
                let index = Arc::clone(&handle.index);
                let query_vec = query.0.clone();
                let nprobe = handle.nprobe;

                let (hits, probed_sizes) = tokio::task::spawn_blocking(move || {
                    let (hits, probe_set) = ivf::probe_with_route(&index, &query_vec, nprobe, k);
                    let sizes: Vec<usize> = probe_set
                        .iter()
                        .map(|&ci| index.clusters[ci].len())
                        .collect();
                    (hits, sizes)
                })
                .await
                .map_err(|_| PlaintextError::TaskPanic)?;

                let mut cost = self.communication_cost(handle, k);
                let response_bytes: u64 = probed_sizes
                    .iter()
                    .map(|&m_i| (k.min(m_i) as u64) * 8)
                    .sum();
                cost.response_bytes = response_bytes;
                cost.cluster_response_bytes = if probed_sizes.is_empty() {
                    0
                } else {
                    response_bytes / probed_sizes.len() as u64
                };
                let probed_total: u64 = probed_sizes.iter().map(|&m| m as u64).sum();
                cost.effective_bytes_per_query = probed_total * handle.dim as u64 * 4;
                Ok((hits, cost))
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                // Same realised-cost shape as the CPU IVF branch —
                // `response_bytes` is Σ_probed min(k, m_i)×8 and
                // `effective_bytes_per_query` is Σ_probed m_i×dim×4.
                // Routing happens twice this turn — once here for
                // accounting and once inside `state.score`. The
                // double-route is acceptable: this method is only called
                // off the throughput hot path, and `probe_route` is
                // O(n_centroids · dim) — negligible compared to the cuVS
                // launches.
                let probe_set = ivf_index::ivf::probe_route(&handle.index, &query.0, handle.nprobe);
                let probed_sizes: Vec<usize> = probe_set
                    .iter()
                    .map(|&ci| handle.index.clusters[ci].len())
                    .collect();
                let hits = state.score(&handle.index, query, handle.nprobe, k)?;

                let mut cost = self.communication_cost(handle, k);
                let response_bytes: u64 = probed_sizes
                    .iter()
                    .map(|&m_i| (k.min(m_i) as u64) * 8)
                    .sum();
                cost.response_bytes = response_bytes;
                cost.cluster_response_bytes = if probed_sizes.is_empty() {
                    0
                } else {
                    response_bytes / probed_sizes.len() as u64
                };
                let probed_total: u64 = probed_sizes.iter().map(|&m| m as u64).sum();
                cost.effective_bytes_per_query = probed_total * handle.dim as u64 * 4;
                Ok((hits, cost))
            }
        }
    }

    /// Batched override. Loop-transposes the per-query `probe_distance`
    /// scan so each probed cluster's vectors are read once per
    /// `score_batch` call and contribute distances to every query that
    /// probed it, instead of once per query. Per-pair `l2_f32` and
    /// per-query `probe_route` / `probe_merge` are unchanged, so each
    /// `out[i]` is bit-equal to `score(handle, &queries[i], k)`. GPU
    /// handles fall back to the inline sequential loop.
    async fn score_batch(
        &self,
        handle: &PlaintextHandle,
        queries: &[Vector],
        k: usize,
    ) -> Result<Vec<Vec<Hit>>, PlaintextError> {
        for q in queries {
            if q.0.len() != handle.dim {
                return Err(PlaintextError::DimensionMismatch {
                    query_dim: q.0.len(),
                    index_dim: handle.dim,
                });
            }
        }

        match &handle.inner {
            HandleInner::Cpu => {
                let index = Arc::clone(&handle.index);
                let queries_owned: Vec<Vec<f32>> = queries.iter().map(|q| q.0.clone()).collect();
                let nprobe = handle.nprobe;
                let hits = tokio::task::spawn_blocking(move || {
                    let refs: Vec<&[f32]> = queries_owned.iter().map(|q| q.as_slice()).collect();
                    ivf::probe_batch(&index, &refs, nprobe, k)
                })
                .await
                .map_err(|_| PlaintextError::TaskPanic)?;
                Ok(hits)
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(_) => {
                let mut out = Vec::with_capacity(queries.len());
                for q in queries {
                    out.push(self.score(handle, q, k).await?);
                }
                Ok(out)
            }
        }
    }
}

/// Per-query substep timings reported by [`PlaintextScorer::score_with_breakdown`].
/// Covers the substeps that apply to plaintext (route → distance →
/// merge); `encode`, `verify`, `decompress`, and `decode` do not apply
/// and are emitted as zeros at the harness layer.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlaintextTiming {
    pub route_us: u64,
    pub distance_us: u64,
    pub merge_us: u64,
}

impl PlaintextScorer {
    /// Off-hot-path equivalent of [`Scorer::score`] that reports per-
    /// substep timings. Per-call `Instant::now()` pairs add
    /// nanosecond-scale overhead per substep — do not use for
    /// throughput benchmarks.
    pub async fn score_with_breakdown(
        &self,
        handle: &PlaintextHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, PlaintextTiming), PlaintextError> {
        if query.0.len() != handle.dim {
            return Err(PlaintextError::DimensionMismatch {
                query_dim: query.0.len(),
                index_dim: handle.dim,
            });
        }

        let index = Arc::clone(&handle.index);
        let query_vec = query.0.clone();
        let nprobe = handle.nprobe;

        // Dispatch on inner. The CPU branch uses
        // `ivf::probe_with_breakdown` which times route / distance /
        // merge inside the rayon pass. The GPU branch times: (a)
        // host-side route under `route_us`, (b) the per-cluster cuVS
        // launches + transfers under `distance_us` (a black-box
        // measurement around `state.score`; cuVS doesn't expose
        // event-stream probes that would split the launch / kernel /
        // download triplet), (c) host-side top-k merge under `merge_us`.
        match &handle.inner {
            HandleInner::Cpu => {
                let (hits, probe_timing) = tokio::task::spawn_blocking(move || {
                    ivf::probe_with_breakdown(&index, &query_vec, nprobe, k)
                })
                .await
                .map_err(|_| PlaintextError::TaskPanic)?;

                Ok((
                    hits,
                    PlaintextTiming {
                        route_us: probe_timing.route_us,
                        distance_us: probe_timing.distance_us,
                        merge_us: probe_timing.merge_us,
                    },
                ))
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                // Route on host (the GPU path internally also routes
                // on host inside `state.score`; we time it once
                // outside so route_us is captured). The redundant
                // route inside `state.score` is acceptable —
                // `probe_route` is O(n_centroids · dim), microsecond-
                // scale, dwarfed by the cuVS launches `state.score`
                // dispatches per probed cluster.
                let route_t = Instant::now();
                let _probe_set = ivf::probe_route(&handle.index, &query.0, handle.nprobe);
                let route_us = route_t.elapsed().as_micros() as u64;

                let server_t = Instant::now();
                let hits = state
                    .score(&handle.index, query, handle.nprobe, k)
                    .map_err(PlaintextError::from)?;
                let distance_us = server_t.elapsed().as_micros() as u64;

                // `state.score` already merges top-k internally; we
                // can't separate merge from distance without
                // refactoring `gpu::GpuState::score` to expose the
                // pre-merge per-cluster hit list. Reporting merge_us
                // = 0 with a comment is honest — the merge cost is
                // folded into distance_us.
                Ok((
                    hits,
                    PlaintextTiming {
                        route_us,
                        distance_us,
                        merge_us: 0,
                    },
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use scorer_core::Scorer;

    fn random_vectors(n: usize, d: usize, seed: u64) -> Vec<Vector> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                Vector(
                    (0..d)
                        .map(|_| rng.random_range(-1.0_f32..1.0_f32))
                        .collect(),
                )
            })
            .collect()
    }

    fn brute_force_top_k(vectors: &[Vector], query: &Vector, k: usize) -> Vec<u32> {
        let mut dists: Vec<(u32, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, ivf_index::distance::l2_f32(&v.0, &query.0)))
            .collect();
        dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        dists[..k.min(dists.len())]
            .iter()
            .map(|(i, _)| *i)
            .collect()
    }

    fn cfg(n_centroids: usize, nprobe: usize) -> PlaintextConfig {
        PlaintextConfig {
            n_centroids,
            nprobe,
            train_seed: 42,
            max_iter: 25,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn upload_and_score_roundtrip() {
        let scorer = PlaintextScorer::new();
        let vectors = random_vectors(20, 8, 1);
        let (handle, _build) = scorer.upload_cluster(&cfg(4, 2), &vectors).await.unwrap();
        let query = random_vectors(1, 8, 99).remove(0);
        let hits = scorer.score(&handle, &query, 5).await.unwrap();
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|h| h.id < 20));
        assert!(hits.iter().all(|h| h.score.is_finite()));
    }

    #[tokio::test]
    async fn scores_are_descending() {
        let scorer = PlaintextScorer::new();
        let vectors = random_vectors(30, 8, 2);
        let (handle, _build) = scorer.upload_cluster(&cfg(5, 3), &vectors).await.unwrap();
        let query = random_vectors(1, 8, 100).remove(0);
        let hits = scorer.score(&handle, &query, 10).await.unwrap();
        for w in hits.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "scores not descending: {:?}",
                hits
            );
        }
    }

    #[tokio::test]
    async fn exact_recall_when_all_clusters_probed() {
        let scorer = PlaintextScorer::new();
        let vectors = random_vectors(50, 8, 3);
        let k = 5;
        let n_centroids = 8;
        // nprobe = n_centroids → probe every cluster → result equals brute force
        let (handle, _build) = scorer
            .upload_cluster(&cfg(n_centroids, n_centroids), &vectors)
            .await
            .unwrap();
        let query = random_vectors(1, 8, 101).remove(0);
        let hits = scorer.score(&handle, &query, k).await.unwrap();
        let got: Vec<u32> = hits.iter().map(|h| h.id).collect();
        let expected = brute_force_top_k(&vectors, &query, k);
        assert_eq!(got, expected, "full probe should match brute force");
    }

    #[tokio::test]
    async fn k_capped_at_cluster_size() {
        let scorer = PlaintextScorer::new();
        let vectors = random_vectors(3, 4, 4);
        let (handle, _build) = scorer.upload_cluster(&cfg(1, 1), &vectors).await.unwrap();
        let query = random_vectors(1, 4, 102).remove(0);
        let hits = scorer.score(&handle, &query, 100).await.unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[tokio::test]
    async fn error_on_empty_cluster() {
        let scorer = PlaintextScorer::new();
        let err = scorer.upload_cluster(&cfg(1, 1), &[]).await.unwrap_err();
        assert!(matches!(err, PlaintextError::EmptyCluster));
    }

    #[tokio::test]
    async fn error_on_too_many_centroids() {
        let scorer = PlaintextScorer::new();
        let vectors = random_vectors(3, 4, 5);
        let err = scorer
            .upload_cluster(&cfg(10, 1), &vectors)
            .await
            .unwrap_err();
        assert!(matches!(err, PlaintextError::TooManyCentroids(_)));
    }

    #[tokio::test]
    async fn error_on_dimension_mismatch() {
        let scorer = PlaintextScorer::new();
        let vectors = random_vectors(10, 4, 6);
        let (handle, _build) = scorer.upload_cluster(&cfg(2, 2), &vectors).await.unwrap();
        let wrong = random_vectors(1, 8, 103).remove(0);
        let err = scorer.score(&handle, &wrong, 3).await.unwrap_err();
        assert!(matches!(err, PlaintextError::DimensionMismatch { .. }));
    }

    #[tokio::test]
    async fn communication_cost_formula() {
        let scorer = PlaintextScorer::new();
        let vectors = random_vectors(10, 16, 7);
        let (handle, _build) = scorer.upload_cluster(&cfg(2, 2), &vectors).await.unwrap();
        let cost = scorer.communication_cost(&handle, 5);
        assert_eq!(cost.query_bytes, 16 * 4);
        // 10 vectors / 2 centroids → m_avg = 5; nprobe = 2; k = 5.
        // cluster_response_bytes = min(k, m_avg) × 8 = 5 × 8 = 40.
        // response_bytes = nprobe × cluster_response_bytes = 2 × 40 = 80.
        assert_eq!(cost.response_bytes, 2 * 5 * 8);
        assert_eq!(cost.cluster_response_bytes, 5 * 8);
        assert_eq!(cost.setup_bytes, 0);
        assert_eq!(cost.pre_query_offline_up_bytes, 0);
        assert_eq!(cost.pre_query_offline_down_bytes, 0);
        // GPU bandwidth proxy: nprobe × m_avg × dim × 4 = 2 × 5 × 16 × 4 = 640.
        assert_eq!(cost.effective_bytes_per_query, 2 * 5 * 16 * 4);
    }

    /// 1D corpus designed to force uneven cluster sizes [60, 20, 15, 5]
    /// after k-means converges. Used by the realised-cost tests as a
    /// small synthetic stand-in for a non-uniform cluster distribution.
    fn skewed_corpus_4_clusters() -> Vec<Vector> {
        let mut vectors = Vec::new();
        for _ in 0..60 {
            vectors.push(Vector(vec![0.0]));
        }
        for _ in 0..20 {
            vectors.push(Vector(vec![10.0]));
        }
        for _ in 0..15 {
            vectors.push(Vector(vec![20.0]));
        }
        for _ in 0..5 {
            vectors.push(Vector(vec![30.0]));
        }
        vectors
    }

    #[tokio::test]
    async fn realised_cost_uneven_clusters() {
        let scorer = PlaintextScorer::new();
        let vectors = skewed_corpus_4_clusters();
        let (handle, _) = scorer.upload_cluster(&cfg(4, 2), &vectors).await.unwrap();

        // Query A near 0.0: nearest two centroids are the 60- and
        // 20-clusters → response = (min(10,60) + min(10,20)) × 8 = 160.
        let (_, cost_a) = scorer
            .score_with_realised_cost(&handle, &Vector(vec![0.0]), 10)
            .await
            .unwrap();
        // Query B near 30.0: nearest two are the 5- and 15-clusters
        // → response = (min(10,5) + min(10,15)) × 8 = 120.
        let (_, cost_b) = scorer
            .score_with_realised_cost(&handle, &Vector(vec![30.0]), 10)
            .await
            .unwrap();

        assert_eq!(cost_a.response_bytes, 160);
        assert_eq!(cost_b.response_bytes, 120);
        assert_ne!(cost_a.response_bytes, cost_b.response_bytes);
    }

    #[tokio::test]
    async fn realised_cost_full_probe_recovers_total() {
        let scorer = PlaintextScorer::new();
        let vectors = skewed_corpus_4_clusters();
        let (handle, _) = scorer.upload_cluster(&cfg(4, 4), &vectors).await.unwrap();

        // nprobe = n_centroids → every cluster is probed; realised
        // response is Σ min(10, m_i) × 8 over all 4 clusters
        // = (10 + 10 + 10 + 5) × 8 = 280, regardless of query.
        let (_, cost) = scorer
            .score_with_realised_cost(&handle, &Vector(vec![15.0]), 10)
            .await
            .unwrap();
        assert_eq!(cost.response_bytes, 35 * 8);
    }

    #[tokio::test]
    async fn realised_cost_matches_communication_cost_when_uniform() {
        // 12 vectors, 4 centroids, k-means converges on cluster sizes
        // [3, 3, 3, 3]. With uniform sizes the realised number equals
        // the analytical to-the-byte for every query.
        let mut vectors = Vec::new();
        for _ in 0..3 {
            vectors.push(Vector(vec![0.0]));
        }
        for _ in 0..3 {
            vectors.push(Vector(vec![10.0]));
        }
        for _ in 0..3 {
            vectors.push(Vector(vec![20.0]));
        }
        for _ in 0..3 {
            vectors.push(Vector(vec![30.0]));
        }

        let scorer = PlaintextScorer::new();
        let (handle, _) = scorer.upload_cluster(&cfg(4, 2), &vectors).await.unwrap();

        let analytical = scorer.communication_cost(&handle, 5);
        let (_, realised) = scorer
            .score_with_realised_cost(&handle, &Vector(vec![5.0]), 5)
            .await
            .unwrap();

        assert_eq!(realised.response_bytes, analytical.response_bytes);
        assert_eq!(
            realised.cluster_response_bytes,
            analytical.cluster_response_bytes
        );
    }

    #[tokio::test]
    async fn score_with_breakdown_matches_score() {
        let scorer = PlaintextScorer::new();
        let vectors = random_vectors(40, 8, 30);
        let (handle, _build) = scorer.upload_cluster(&cfg(4, 2), &vectors).await.unwrap();
        let query = random_vectors(1, 8, 31).remove(0);

        let baseline = scorer.score(&handle, &query, 5).await.unwrap();
        let (with_breakdown, timing) = scorer
            .score_with_breakdown(&handle, &query, 5)
            .await
            .unwrap();
        assert_eq!(baseline, with_breakdown);
        // Substeps cover real work; nominally non-zero but tolerate
        // sub-microsecond on fast machines.
        let _ = timing.route_us + timing.distance_us + timing.merge_us;
    }

    /// Per-query result from the batched override is equivalent
    /// (ranking-equal + score within 1e-6 rel tol) to a sequential
    /// `score()` call on the same input. The implementation preserves
    /// summation order, so this is effectively bit-equal today; the
    /// tolerance assertion leaves headroom for a future SIMD lift.
    #[tokio::test]
    async fn plaintext_score_batch_matches_score() {
        let scorer = PlaintextScorer::new();
        let vectors = random_vectors(128, 16, 7);
        let (handle, _build) = scorer.upload_cluster(&cfg(8, 3), &vectors).await.unwrap();

        let queries = random_vectors(64, 16, 8);
        let k = 10;

        let mut from_score: Vec<Vec<Hit>> = Vec::with_capacity(queries.len());
        for q in &queries {
            from_score.push(scorer.score(&handle, q, k).await.unwrap());
        }

        for &b in &[1usize, 8, 64] {
            let chunk = &queries[..b];
            let from_batch = scorer.score_batch(&handle, chunk, k).await.unwrap();
            assert_eq!(from_batch.len(), b, "B={b}: result count mismatch");
            for (q_idx, (a, c)) in from_score[..b].iter().zip(from_batch.iter()).enumerate() {
                assert_eq!(a.len(), c.len(), "B={b} q={q_idx}: hit count mismatch");
                for (rank, (h_a, h_c)) in a.iter().zip(c).enumerate() {
                    assert_eq!(
                        h_a.id, h_c.id,
                        "B={b} q={q_idx} rank={rank}: id mismatch ({} vs {})",
                        h_a.id, h_c.id
                    );
                    let denom = h_a.score.abs().max(h_c.score.abs()).max(1.0);
                    let diff = (h_a.score - h_c.score).abs();
                    assert!(
                        diff <= 1e-6_f32 * denom,
                        "B={b} q={q_idx} rank={rank}: score diff {diff} > 1e-6 * {denom}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn build_outcome_cache_hit_smoke() {
        let scorer = PlaintextScorer::new();
        let vectors = random_vectors(40, 8, 50);
        let (_h1, b1) = scorer.upload_cluster(&cfg(4, 2), &vectors).await.unwrap();
        assert!(!b1.cache_hit, "first build must be a cold build");
        let (_h2, b2) = scorer.upload_cluster(&cfg(4, 2), &vectors).await.unwrap();
        assert!(b2.cache_hit, "second build must hit the in-memory cache");
        assert!(
            b2.build_duration < std::time::Duration::from_millis(100),
            "warm build was {:?}, expected < 100 ms",
            b2.build_duration
        );
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_full_probe_matches_brute_force() {
        // Correctness gate, full-probe regime. `nprobe = n_centroids`
        // means every cluster is scored, so the result must
        // approximately equal CPU brute force — same invariant as
        // `exact_recall_when_all_clusters_probed`, just running on GPU.
        // Requires a CUDA build environment with cuVS available.
        //
        // The threshold is set-overlap ≥ 8/10 rather than exact
        // equality: cuVS computes L2 as L2Expanded
        // (`||x||² + ||y||² - 2·x·y`), which admits catastrophic
        // cancellation when the two vectors are close, flipping ties
        // or near-ties at the top-k boundary even on identical fp32
        // inputs.
        let vectors = random_vectors(256, 64, 909);
        let query = random_vectors(1, 64, 910).remove(0);
        let k = 10;
        let n_centroids = 8;

        let scorer = PlaintextScorer::new();
        let cfg_gpu = PlaintextConfig {
            n_centroids,
            nprobe: n_centroids,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Gpu,
            vram_budget_bytes: None,
        };
        let (handle, _build) = scorer.upload_cluster(&cfg_gpu, &vectors).await.unwrap();
        let hits = scorer.score(&handle, &query, k).await.unwrap();
        assert_eq!(hits.len(), k);

        let got: std::collections::HashSet<u32> = hits.iter().map(|h| h.id).collect();
        let expected: std::collections::HashSet<u32> =
            brute_force_top_k(&vectors, &query, k).into_iter().collect();
        let overlap = got.intersection(&expected).count();
        assert!(
            overlap >= 8,
            "GPU full-probe overlap with CPU brute force was {}/10; expected ≥ 8/10. \
             got={got:?} expected={expected:?}",
            overlap
        );

        for w in hits.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "GPU scores not descending: {hits:?}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_partial_probe_returns_subset_of_probed_clusters() {
        // Structural invariant: with `nprobe < n_centroids` every
        // returned hit must come from one of the probed clusters —
        // returned IDs are constrained to the union of the probed
        // clusters' member IDs.
        //
        // Recall vs CPU brute force is *not* asserted here — it's
        // expected to be < 1.0 at small nprobe (that's the whole
        // point of IVF). The harness measures recall; the unit test
        // just checks the ID-set membership invariant.
        let vectors = random_vectors(256, 64, 911);
        let query = random_vectors(1, 64, 912).remove(0);
        let k = 10;
        let n_centroids = 8;
        let nprobe = 2;

        let scorer = PlaintextScorer::new();
        let cfg = PlaintextConfig {
            n_centroids,
            nprobe,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Gpu,
            vram_budget_bytes: None,
        };
        let (handle, _build) = scorer.upload_cluster(&cfg, &vectors).await.unwrap();
        let hits = scorer.score(&handle, &query, k).await.unwrap();

        // Re-derive the host probe set from the same routing the GPU
        // path uses internally.
        let probe_set = ivf_index::ivf::probe_route(&handle.index, &query.0, nprobe);
        let allowed: std::collections::HashSet<u32> = probe_set
            .iter()
            .flat_map(|&ci| handle.index.clusters[ci].iter().map(|(id, _)| *id))
            .collect();
        for h in &hits {
            assert!(
                allowed.contains(&h.id),
                "GPU returned hit id={} that is not a member of any probed cluster (probe_set={probe_set:?})",
                h.id
            );
        }
        assert!(hits.len() <= k);
    }

    /// GPU breakdown agrees with GPU score on hits. Validates that
    /// `score_with_breakdown` accepts a GPU handle (no longer rejects
    /// with `GpuFeatureNotEnabled`) and produces the same top-k IDs as
    /// `score()` on the same handle and query. `distance_us` should be
    /// > 0 (the cuVS launches dominate), `route_us` should be > 0,
    /// `merge_us` is folded into distance_us (cuVS returns sorted hits)
    /// so we don't assert it.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_score_with_breakdown_matches_score() {
        let vectors = random_vectors(256, 64, 915);
        let query = random_vectors(1, 64, 916).remove(0);
        let k = 10;
        let n_centroids = 8;

        let scorer = PlaintextScorer::new();
        let cfg = PlaintextConfig {
            n_centroids,
            nprobe: 4,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Gpu,
            vram_budget_bytes: None,
        };
        let (handle, _build) = scorer.upload_cluster(&cfg, &vectors).await.unwrap();

        let baseline = scorer.score(&handle, &query, k).await.unwrap();
        let (with_breakdown, timing) = scorer
            .score_with_breakdown(&handle, &query, k)
            .await
            .unwrap();

        let baseline_ids: std::collections::HashSet<u32> = baseline.iter().map(|h| h.id).collect();
        let breakdown_ids: std::collections::HashSet<u32> =
            with_breakdown.iter().map(|h| h.id).collect();
        assert_eq!(
            baseline_ids, breakdown_ids,
            "GPU score() and score_with_breakdown() must agree on top-{k} ids"
        );
        // Substep timing is dominated by the cuVS launches; both
        // route and distance should land non-zero on a real GPU run.
        // merge_us = 0 by design (folded into distance_us).
        assert!(
            timing.distance_us > 0,
            "distance_us should cover cuVS launches: {timing:?}"
        );
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_realised_cost_matches_probed_clusters() {
        // Realised-cost gate: with `nprobe < n_centroids`,
        // `effective_bytes_per_query` and `response_bytes` must reflect
        // the *probed* cluster set, not the full corpus. This is what
        // lets per-nprobe GPU bandwidth numbers stay honest.
        let vectors = random_vectors(256, 64, 913);
        let query = random_vectors(1, 64, 914).remove(0);
        let k = 10;
        let n_centroids = 8;
        let nprobe = 2;

        let scorer = PlaintextScorer::new();
        let cfg = PlaintextConfig {
            n_centroids,
            nprobe,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Gpu,
            vram_budget_bytes: None,
        };
        let (handle, _build) = scorer.upload_cluster(&cfg, &vectors).await.unwrap();
        let (_hits, cost) = scorer
            .score_with_realised_cost(&handle, &query, k)
            .await
            .unwrap();

        let probe_set = ivf_index::ivf::probe_route(&handle.index, &query.0, nprobe);
        let probed_total: u64 = probe_set
            .iter()
            .map(|&ci| handle.index.clusters[ci].len() as u64)
            .sum();
        assert_eq!(
            cost.effective_bytes_per_query,
            probed_total * handle.dim as u64 * 4,
            "effective bytes must equal Σ_probed m_i × dim × 4"
        );
        let n_total: u64 = handle.index.clusters.iter().map(|c| c.len() as u64).sum();
        assert!(
            cost.effective_bytes_per_query < n_total * handle.dim as u64 * 4,
            "with nprobe < n_centroids, effective bytes must be < full-corpus bytes"
        );
    }
}
