use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use ivf_index::{ivf, kmeans};
use scorer_core::{
    BuildOutcome, CommunicationCost, Device, Hit, ProgressReporter, Scorer, Vector, cache,
};

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::crypto::{SapKey, encrypt_with_nonce};
#[cfg(feature = "gpu")]
use crate::gpu;

pub struct SapIvfConfig {
    pub key: SapKey,
    /// Perturbation parameter β ≥ 0. β = 0 gives no perturbation (ciphertexts = s·m).
    pub beta: f64,
    /// Seed for parallel per-vector encryption. Each vector i gets seed `upload_seed ^ i`.
    pub upload_seed: u64,
    /// Number of IVF centroids (k-means). Rule of thumb: `ceil(sqrt(N))`.
    pub n_centroids: usize,
    /// Number of clusters probed per query.
    pub nprobe: usize,
    /// Seed for k-means++ initialisation.
    pub train_seed: u64,
    /// Maximum Lloyd's iterations.
    pub max_iter: usize,
    /// Optional progress reporter. `on_encrypt(i)` fires per encrypted vector
    /// (per-row); `on_init_centroid` and `on_kmeans_iter` fire during k-means.
    pub progress: Option<Arc<dyn ProgressReporter>>,
    /// Compute substrate. `Cpu` runs the IVF probe with plaintext
    /// routing + encrypted-cluster scoring on host. `Gpu` requires the
    /// `gpu` Cargo feature; without it `upload_cluster` returns
    /// [`SapIvfError::GpuFeatureNotEnabled`]. SAP encryption / k-means
    /// stay on CPU; only the per-cluster L2 scan moves to device. GPU
    /// and CPU runs share centroids and cluster assignments.
    pub device: Device,
    /// VRAM budget for the streaming cluster store on GPU runs. `None`
    /// (default) auto-detects 80 % of free VRAM via NVML at handle
    /// build. `Some(b)` pins a hard byte cap. Ignored on CPU runs.
    pub vram_budget_bytes: Option<u64>,
}

pub struct SapIvfHandle {
    /// Host-side IVF index — plaintext centroids (k-means trained on
    /// the plaintext corpus) with encrypted-then-downcast-to-f32 cluster
    /// contents. Both CPU and GPU handles hold this. Routing
    /// (`ivf::probe_route`) runs on the plaintext query against
    /// plaintext centroids, so the probed-cluster set is identical
    /// across substrates and across crypto backends; per-cluster scoring
    /// runs on the encrypted query against encrypted cluster contents.
    index: Arc<ivf::IvfIndex>,
    key_s: f64,
    prf_key: [u8; 32],
    beta: f64,
    nprobe: usize,
    dim: usize,
    inner: HandleInner,
}

enum HandleInner {
    Cpu,
    /// Per-cluster cuVS BfIndex over the encrypted corpus, sharing the
    /// host's IVF centroids so the probed-cluster set is identical
    /// across substrates. Boxed because the streaming state carries
    /// per-cluster host payloads + LRU + sampler; clippy's
    /// `large_enum_variant` lint trips otherwise.
    #[cfg(feature = "gpu")]
    Gpu(Box<gpu::IvfGpuState>),
}

impl fmt::Debug for SapIvfHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device = match &self.inner {
            HandleInner::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(_) => "gpu",
        };
        f.debug_struct("SapIvfHandle")
            .field("device", &device)
            .field("dim", &self.dim)
            .field("n_centroids", &self.index.centroids.len())
            .field("nprobe", &self.nprobe)
            .field("beta", &self.beta)
            .field("key_s", &self.key_s)
            .field("prf_key", &"[redacted]")
            .finish()
    }
}

impl SapIvfHandle {
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
pub enum SapIvfError {
    #[error("cluster must contain at least one vector")]
    EmptyCluster,
    #[error("n_centroids {0} exceeds cluster size")]
    TooManyCentroids(usize),
    #[error("invalid beta {0}: must be >= 0.0")]
    InvalidBeta(f64),
    #[error("query dimension {query_dim} does not match index dimension {index_dim}")]
    DimensionMismatch { query_dim: usize, index_dim: usize },
    #[error("spawn_blocking task panicked")]
    TaskPanic,
    /// `Device::Gpu` was requested but the crate was built without the
    /// `gpu` Cargo feature.
    #[error("Device::Gpu requested but the `gpu` Cargo feature is not enabled in this build")]
    GpuFeatureNotEnabled,
    /// GPU runtime error from cuVS (build / search / device alloc).
    #[cfg(feature = "gpu")]
    #[error("cuVS GPU error: {0}")]
    Gpu(#[from] gpu::GpuError),
}

// (n_centroids, train_seed, max_iter, corpus_len, beta_bits, upload_seed)
type CacheKey = (usize, u64, usize, usize, u64, u64);

/// SAP scorer with IVF index. k-means + IVF are trained on the
/// plaintext corpus, and each cluster's vectors are encrypted in-place
/// after assignment. Centroids stay plaintext so cluster geometry is
/// bit-identical across crypto backends; only the per-cluster scoring
/// step touches ciphertexts.
///
/// The in-memory cache holds one index per distinct cache key, so a β sweep builds
/// each index once and reuses it across nprobe values. If `cache_dir` is set the
/// built index is also persisted to disk and reloaded on future runs, skipping
/// encryption and k-means entirely. Cache files are named
/// `.sap-ivf-cache-{n_centroids}-{train_seed}-{max_iter}-{N}-{beta_bits}-{upload_seed}-{key_fp}.bin`
/// where `key_fp` is the first 8 hex digits of an FNV-1a hash over `key_s || prf_key`,
/// preventing collisions when the same corpus is indexed under different keys.
/// The in-memory cache is single-entry: a β sweep iterates configs that
/// change the cache key, and holding both the previous index and the
/// next-building one resident (each tens of GB at large corpora) can
/// exceed physical RAM and OOM the build. Single-entry forces eviction
/// on replace; the disk cache still avoids redundant encryption +
/// k-means work across re-visits.
pub struct SapIvfScorer {
    cache: Mutex<Option<(CacheKey, Arc<ivf::IvfIndex>)>>,
    cache_dir: Option<PathBuf>,
}

impl SapIvfScorer {
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

impl Default for SapIvfScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Scorer for SapIvfScorer {
    type Config = SapIvfConfig;
    type ClusterHandle = SapIvfHandle;
    type Error = SapIvfError;

    async fn upload_cluster(
        &self,
        config: &SapIvfConfig,
        vectors: &[Vector],
    ) -> Result<(SapIvfHandle, BuildOutcome), SapIvfError> {
        let build_start = Instant::now();
        if vectors.is_empty() {
            return Err(SapIvfError::EmptyCluster);
        }
        if config.n_centroids > vectors.len() {
            return Err(SapIvfError::TooManyCentroids(config.n_centroids));
        }
        if config.beta < 0.0 {
            return Err(SapIvfError::InvalidBeta(config.beta));
        }

        let dim = vectors[0].0.len();
        let key_s = config.key.s;
        let prf_key = config.key.prf_key;
        let beta = config.beta;
        let upload_seed = config.upload_seed;
        let n_centroids = config.n_centroids;
        let nprobe = config.nprobe;
        let train_seed = config.train_seed;
        let max_iter = config.max_iter;
        let progress = config.progress.clone();

        let cache_key: CacheKey = (
            n_centroids,
            train_seed,
            max_iter,
            vectors.len(),
            beta.to_bits(),
            upload_seed,
        );

        // Cache tag `sap-ivf-v2`: older `.sap-ivf-cache-*.bin` files
        // (from when k-means trained on ciphertexts, giving confounded
        // IVF geometry) re-fingerprint to a different name and fall
        // through to a clean rebuild. The plaintext-trained partition
        // lives under the v2 tag.
        let cache_parts: [Vec<u8>; 8] = [
            b"sap-ivf-v2".to_vec(),
            (n_centroids as u64).to_le_bytes().to_vec(),
            train_seed.to_le_bytes().to_vec(),
            (max_iter as u64).to_le_bytes().to_vec(),
            (vectors.len() as u64).to_le_bytes().to_vec(),
            beta.to_bits().to_le_bytes().to_vec(),
            upload_seed.to_le_bytes().to_vec(),
            {
                // Hash key material into the fingerprint so different keys can't collide.
                let mut k = key_s.to_le_bytes().to_vec();
                k.extend_from_slice(&prf_key);
                k
            },
        ];
        let disk_path = self.cache_dir.as_deref().map(|dir| {
            let parts_refs: Vec<&[u8]> = cache_parts.iter().map(|v| v.as_slice()).collect();
            let fp = cache::fingerprint(&parts_refs);
            dir.join(format!(".sap-ivf-cache-{fp}.bin"))
        });

        let cached = {
            let mut guard = self.cache.lock().unwrap();
            match guard.as_ref() {
                Some((k, idx)) if k == &cache_key => Some(Arc::clone(idx)),
                Some(_) => {
                    // Cached entry exists but doesn't match — evict it
                    // NOW (before we start the new build) rather than at
                    // insert-time. Evicting post-build would keep both
                    // indices alive concurrently and can blow past
                    // physical RAM. With the previous handle already
                    // dropped by the caller, our Arc is the last owner;
                    // clearing here frees the old IvfIndex before the
                    // new build allocates its data clone + ciphertexts.
                    *guard = None;
                    None
                }
                None => None,
            }
        };

        let in_memory_hit = cached.is_some();
        let mut disk_hit = false;
        let index = match cached {
            Some(idx) => {
                if let Some(ref p) = progress {
                    p.on_build_complete();
                }
                idx
            }
            None => {
                // Try disk cache before running expensive encryption + k-means.
                let from_disk = if let Some(ref path) = disk_path {
                    if path.exists() {
                        let p = path.clone();
                        eprintln!("Loading cached SAP+IVF index from {p:?}…");
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

                let arc = if let Some(loaded) = from_disk {
                    disk_hit = true;
                    if let Some(ref p) = progress {
                        p.on_build_complete();
                    }
                    Arc::new(loaded)
                } else {
                    // k-means + IVF train over the plaintext corpus.
                    // Per-cluster encryption runs after assignment,
                    // in-place, replacing each row's f32 contents with
                    // the f32-downcast ciphertext. Centroids stay
                    // plaintext, so routing at score() time runs on the
                    // plaintext query against plaintext centroids and
                    // probes a cluster set identical to the plaintext
                    // scorer's. Per-row nonce is derived from
                    // (upload_seed XOR original vector id), keeping
                    // ciphertexts position-independent across cluster
                    // ordering changes.
                    let data: Vec<Vector> = vectors.to_vec();
                    let save_path = disk_path.clone();
                    let save_parts = cache_parts.clone();

                    let built = tokio::task::spawn_blocking(move || {
                        // Materialise the plaintext f32 corpus by
                        // moving each Vector's payload out — peaks at
                        // one corpus copy resident, not two.
                        let data_f32: Vec<Vec<f32>> = data.into_iter().map(|v| v.0).collect();
                        let centroids = kmeans::train(
                            &data_f32,
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
                        let mut index = ivf::build_index(&data_f32, centroids, dim);
                        // `build_index` cloned each row into
                        // `index.clusters`; the second copy is now
                        // surplus.
                        drop(data_f32);

                        // Per-row in-place encryption. `Vec::par_iter_mut`
                        // over clusters + sequential within each lets
                        // the f64 scratch buffer reuse across rows
                        // (per-row `Vec<f64>` allocation), bounded to
                        // `dim × 8 = 6 kB` per thread; rayon's worker
                        // count caps total scratch at a few MB.
                        let p_for_rows = progress.clone();
                        let encrypted_count = std::sync::atomic::AtomicUsize::new(0);
                        index.clusters.par_iter_mut().for_each(|cluster| {
                            for (id, vec) in cluster.iter_mut() {
                                let mut prng =
                                    ChaCha20Rng::seed_from_u64(upload_seed ^ (*id as u64));
                                let nonce: [u8; 16] = std::array::from_fn(|_| prng.random());
                                let m_f64: Vec<f64> = vec.iter().map(|&x| x as f64).collect();
                                let ct = encrypt_with_nonce(key_s, &prf_key, &m_f64, beta, nonce);
                                for (slot, ct_val) in vec.iter_mut().zip(ct.c.iter()) {
                                    *slot = *ct_val as f32;
                                }
                                let i = encrypted_count
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if let Some(ref p) = p_for_rows {
                                    p.on_encrypt(i);
                                }
                            }
                        });

                        if let Some(ref p) = progress {
                            p.on_build_complete();
                        }
                        if let Some(ref path) = save_path {
                            let refs: Vec<&[u8]> =
                                save_parts.iter().map(|v| v.as_slice()).collect();
                            if let Err(e) = ivf::save_index_with_header(&index, path, &refs) {
                                eprintln!("Warning: could not save SAP+IVF cache to {path:?}: {e}");
                            }
                        }
                        index
                    })
                    .await
                    .map_err(|_| SapIvfError::TaskPanic)?;

                    Arc::new(built)
                };

                // Single-entry replace; drops the previous Arc so its
                // IVF payload is freed before the next config's build
                // starts. This is what keeps a β sweep within RAM.
                *self.cache.lock().unwrap() = Some((cache_key, Arc::clone(&arc)));
                arc
            }
        };

        // Branch on substrate. The host-side IVF index is built either
        // way (so centroids and cluster assignments match across
        // CPU/GPU runs); GPU additionally constructs one cuVS BfIndex
        // per cluster from the encrypted f32 corpus. SAP encryption +
        // k-means stay on host either way.
        let inner = match config.device {
            Device::Cpu => HandleInner::Cpu,
            Device::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    let host_index = Arc::clone(&index);
                    let vram_budget = config.vram_budget_bytes;
                    let state = tokio::task::spawn_blocking(move || {
                        gpu::IvfGpuState::build(&host_index, vram_budget)
                    })
                    .await
                    .map_err(|_| SapIvfError::TaskPanic)??;
                    HandleInner::Gpu(Box::new(state))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(SapIvfError::GpuFeatureNotEnabled);
                }
            }
        };

        let outcome = BuildOutcome {
            cache_hit: in_memory_hit || disk_hit,
            build_duration: build_start.elapsed(),
        };
        Ok((
            SapIvfHandle {
                index,
                key_s,
                prf_key,
                beta,
                nprobe,
                dim,
                inner,
            },
            outcome,
        ))
    }

    async fn score(
        &self,
        handle: &SapIvfHandle,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<Hit>, SapIvfError> {
        if query.0.len() != handle.dim {
            return Err(SapIvfError::DimensionMismatch {
                query_dim: query.0.len(),
                index_dim: handle.dim,
            });
        }

        let query_f64: Vec<f64> = query.0.iter().map(|&x| x as f64).collect();
        let key_s = handle.key_s;
        let prf_key = handle.prf_key;
        let beta = handle.beta;
        let nprobe = handle.nprobe;

        // ThreadRng is !Send; generate nonce before spawn_blocking, move [u8; 16] in.
        let nonce: [u8; 16] = {
            let mut rng = rand::rng();
            std::array::from_fn(|_| rng.random())
        };

        let plaintext_query_f32: Vec<f32> = query.0.clone();
        match &handle.inner {
            HandleInner::Cpu => {
                let index = Arc::clone(&handle.index);
                let hits = tokio::task::spawn_blocking(move || {
                    // Route on the plaintext query against plaintext
                    // centroids; score with the encrypted query against
                    // encrypted per-cluster contents.
                    let probe_set = ivf::probe_route(&index, &plaintext_query_f32, nprobe);
                    let query_ct = encrypt_with_nonce(key_s, &prf_key, &query_f64, beta, nonce);
                    let query_f32: Vec<f32> = query_ct.c.iter().map(|&x| x as f32).collect();
                    let dists = ivf::probe_distance(&index, &query_f32, &probe_set);
                    ivf::probe_merge(dists, k)
                })
                .await
                .map_err(|_| SapIvfError::TaskPanic)?;
                Ok(hits)
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                // Routing query (plaintext) and scoring query
                // (encrypted, downcast) are passed separately to
                // `state.score`.
                let query_ct = encrypt_with_nonce(key_s, &prf_key, &query_f64, beta, nonce);
                let query_f32: Vec<f32> = query_ct.c.iter().map(|&x| x as f32).collect();
                state
                    .score(&handle.index, &plaintext_query_f32, &query_f32, nprobe, k)
                    .map_err(Into::into)
            }
        }
    }

    fn communication_cost(&self, handle: &SapIvfHandle, k: usize) -> CommunicationCost {
        // IVF accounting: each probed cluster ships up to min(k, m_i)
        // hits back; the client merges. This analytical estimate uses
        // mean cluster size m_avg = N / n_centroids; per-query realised
        // numbers live in `score_with_realised_cost`.
        let n_clusters = handle.index.clusters.len();
        let m_total: usize = handle.index.clusters.iter().map(|c| c.len()).sum();
        let m_avg = m_total.checked_div(n_clusters).unwrap_or(0);
        let cluster_response_bytes = (k.min(m_avg) as u64) * 8;
        let response_bytes = cluster_response_bytes * handle.nprobe as u64;
        // GPU bandwidth proxy = perturbed-fp32 vectors streamed across
        // probed clusters. Analytical estimate = nprobe × m_avg.
        let effective_bytes_per_query = handle.nprobe as u64 * m_avg as u64 * handle.dim as u64 * 4;
        CommunicationCost {
            query_bytes: handle.dim as u64 * 8,
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
        handle: &SapIvfHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, CommunicationCost), SapIvfError> {
        if query.0.len() != handle.dim {
            return Err(SapIvfError::DimensionMismatch {
                query_dim: query.0.len(),
                index_dim: handle.dim,
            });
        }

        let query_f64: Vec<f64> = query.0.iter().map(|&x| x as f64).collect();
        let key_s = handle.key_s;
        let prf_key = handle.prf_key;
        let beta = handle.beta;
        let nprobe = handle.nprobe;

        let nonce: [u8; 16] = {
            let mut rng = rand::rng();
            std::array::from_fn(|_| rng.random())
        };

        let plaintext_query_f32: Vec<f32> = query.0.clone();
        let (hits, probed_sizes) = match &handle.inner {
            HandleInner::Cpu => {
                let index = Arc::clone(&handle.index);
                tokio::task::spawn_blocking(move || {
                    // Route on the plaintext query, score on encrypted.
                    let probe_set = ivf::probe_route(&index, &plaintext_query_f32, nprobe);
                    let sizes: Vec<usize> = probe_set
                        .iter()
                        .map(|&ci| index.clusters[ci].len())
                        .collect();
                    let query_ct = encrypt_with_nonce(key_s, &prf_key, &query_f64, beta, nonce);
                    let query_f32: Vec<f32> = query_ct.c.iter().map(|&x| x as f32).collect();
                    let dists = ivf::probe_distance(&index, &query_f32, &probe_set);
                    let hits = ivf::probe_merge(dists, k);
                    (hits, sizes)
                })
                .await
                .map_err(|_| SapIvfError::TaskPanic)?
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                // Route on host twice — once here for accounting, once
                // inside `state.score`. This method is off the
                // throughput hot path so the double-route is
                // acceptable; `probe_route` is O(n_centroids · dim) and
                // dominated by the cuVS launches.
                let probe_set = ivf::probe_route(&handle.index, &plaintext_query_f32, nprobe);
                let sizes: Vec<usize> = probe_set
                    .iter()
                    .map(|&ci| handle.index.clusters[ci].len())
                    .collect();
                let query_ct = encrypt_with_nonce(key_s, &prf_key, &query_f64, beta, nonce);
                let query_f32: Vec<f32> = query_ct.c.iter().map(|&x| x as f32).collect();
                let hits =
                    state.score(&handle.index, &plaintext_query_f32, &query_f32, nprobe, k)?;
                (hits, sizes)
            }
        };

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

    /// Batched override. Each query is encrypted (per-query nonce) and
    /// downcast to f32, then the batched f32 IVF probe is run via
    /// [`ivf::probe_batch`] — the same loop-transposed helper the
    /// plaintext IVF scorer uses. Per-query `encrypt_with_nonce`, the
    /// f64→f32 downcast, and the per-pair `l2_f32` inside `probe_batch`
    /// are unchanged, so each `out[i]` is bit-equal to
    /// `score(handle, &queries[i], k)` under the same nonce. The
    /// equivalence test's tolerance keys off the arithmetic precision
    /// (f32 post-downcast). GPU handles fall back to the inline
    /// sequential loop.
    async fn score_batch(
        &self,
        handle: &SapIvfHandle,
        queries: &[Vector],
        k: usize,
    ) -> Result<Vec<Vec<Hit>>, SapIvfError> {
        for q in queries {
            if q.0.len() != handle.dim {
                return Err(SapIvfError::DimensionMismatch {
                    query_dim: q.0.len(),
                    index_dim: handle.dim,
                });
            }
        }

        let queries_f64: Vec<Vec<f64>> = queries
            .iter()
            .map(|q| q.0.iter().map(|&x| x as f64).collect())
            .collect();

        // ThreadRng !Send; pre-generate one [u8; 16] nonce per query.
        let nonces: Vec<[u8; 16]> = {
            let mut rng = rand::rng();
            (0..queries.len())
                .map(|_| std::array::from_fn(|_| rng.random()))
                .collect()
        };

        let key_s = handle.key_s;
        let prf_key = handle.prf_key;
        let beta = handle.beta;
        let nprobe = handle.nprobe;

        let plaintext_queries_f32: Vec<Vec<f32>> = queries.iter().map(|q| q.0.clone()).collect();
        match &handle.inner {
            HandleInner::Cpu => {
                let index = Arc::clone(&handle.index);
                let hits = tokio::task::spawn_blocking(move || {
                    // Route on the plaintext queries, score with the
                    // encrypted queries. `probe_batch_split` preserves
                    // the per-cluster
                    // load-once-distance-against-all-batched-queries
                    // amortisation from `probe_batch`.
                    let encrypted_queries_f32: Vec<Vec<f32>> = queries_f64
                        .iter()
                        .zip(&nonces)
                        .map(|(qf, &n)| {
                            let ct = encrypt_with_nonce(key_s, &prf_key, qf, beta, n);
                            ct.c.iter().map(|&x| x as f32).collect()
                        })
                        .collect();
                    let routing_refs: Vec<&[f32]> =
                        plaintext_queries_f32.iter().map(|q| q.as_slice()).collect();
                    let scoring_refs: Vec<&[f32]> =
                        encrypted_queries_f32.iter().map(|q| q.as_slice()).collect();
                    ivf::probe_batch_split(&index, &routing_refs, &scoring_refs, nprobe, k)
                })
                .await
                .map_err(|_| SapIvfError::TaskPanic)?;
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

/// Per-query substep timings reported by [`SapIvfScorer::score_with_breakdown`].
/// `encode` covers query encryption + the f64→f32 downcast required by
/// the IVF probe; `route` / `distance` / `merge` come from
/// [`ivf_index::ivf::ProbeTiming`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SapIvfTiming {
    pub encode_us: u64,
    pub route_us: u64,
    pub distance_us: u64,
    pub merge_us: u64,
}

impl SapIvfScorer {
    /// Off-hot-path equivalent of [`Scorer::score`] with per-substep
    /// timings.
    pub async fn score_with_breakdown(
        &self,
        handle: &SapIvfHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, SapIvfTiming), SapIvfError> {
        if query.0.len() != handle.dim {
            return Err(SapIvfError::DimensionMismatch {
                query_dim: query.0.len(),
                index_dim: handle.dim,
            });
        }

        let query_f64: Vec<f64> = query.0.iter().map(|&x| x as f64).collect();
        let key_s = handle.key_s;
        let prf_key = handle.prf_key;
        let beta = handle.beta;
        let nprobe = handle.nprobe;

        let nonce: [u8; 16] = {
            let mut rng = rand::rng();
            std::array::from_fn(|_| rng.random())
        };

        // Dispatch on inner. CPU branch times route / distance / merge
        // manually; GPU branch breaks the timer set across the host /
        // device boundary so distance_us spans the per-cluster cuVS
        // launches and merge_us is folded into distance_us (cuVS returns
        // top-k sorted per cluster; the host merge across clusters is
        // microsecond-scale).
        let plaintext_query_f32: Vec<f32> = query.0.clone();
        let result = match &handle.inner {
            HandleInner::Cpu => {
                let index = Arc::clone(&handle.index);
                tokio::task::spawn_blocking(move || {
                    // Split route + distance manually so `route_us`
                    // reflects plaintext routing and `distance_us`
                    // reflects encrypted scoring.
                    let route_t = Instant::now();
                    let probe_set = ivf::probe_route(&index, &plaintext_query_f32, nprobe);
                    let route_us = route_t.elapsed().as_micros() as u64;

                    let encode_t = Instant::now();
                    let query_ct = encrypt_with_nonce(key_s, &prf_key, &query_f64, beta, nonce);
                    let query_f32: Vec<f32> = query_ct.c.iter().map(|&x| x as f32).collect();
                    let encode_us = encode_t.elapsed().as_micros() as u64;

                    let distance_t = Instant::now();
                    let dists = ivf::probe_distance(&index, &query_f32, &probe_set);
                    let distance_us = distance_t.elapsed().as_micros() as u64;

                    let merge_t = Instant::now();
                    let hits = ivf::probe_merge(dists, k);
                    let merge_us = merge_t.elapsed().as_micros() as u64;

                    (
                        hits,
                        SapIvfTiming {
                            encode_us,
                            route_us,
                            distance_us,
                            merge_us,
                        },
                    )
                })
                .await
                .map_err(|_| SapIvfError::TaskPanic)?
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                let encode_t = Instant::now();
                let query_ct = encrypt_with_nonce(key_s, &prf_key, &query_f64, beta, nonce);
                let query_f32: Vec<f32> = query_ct.c.iter().map(|&x| x as f32).collect();
                let encode_us = encode_t.elapsed().as_micros() as u64;

                // Time the host-side route once (the cuVS state.score
                // also routes internally; the double-route cost is
                // microsecond-scale and dwarfed by the cuVS launches).
                let route_t = Instant::now();
                let _probe_set = ivf::probe_route(&handle.index, &plaintext_query_f32, nprobe);
                let route_us = route_t.elapsed().as_micros() as u64;

                let server_t = Instant::now();
                let hits = state
                    .score(&handle.index, &plaintext_query_f32, &query_f32, nprobe, k)
                    .map_err(SapIvfError::from)?;
                let distance_us = server_t.elapsed().as_micros() as u64;

                (
                    hits,
                    SapIvfTiming {
                        encode_us,
                        route_us,
                        distance_us,
                        merge_us: 0,
                    },
                )
            }
        };

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ivf_index::distance::l2_f32;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use scorer_core::Scorer;

    use crate::crypto::keygen;

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
            .map(|(i, v)| (i as u32, l2_f32(&v.0, &query.0)))
            .collect();
        dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        dists[..k.min(dists.len())]
            .iter()
            .map(|(i, _)| *i)
            .collect()
    }

    fn cfg(beta: f64, n_centroids: usize, nprobe: usize) -> SapIvfConfig {
        SapIvfConfig {
            key: keygen(1.0, 2.0, 7),
            beta,
            upload_seed: 42,
            n_centroids,
            nprobe,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        }
    }

    #[tokio::test]
    async fn upload_and_score_roundtrip() {
        let scorer = SapIvfScorer::new();
        let vectors = random_vectors(20, 8, 1);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(0.5, 4, 2), &vectors)
            .await
            .unwrap();
        let query = random_vectors(1, 8, 99).remove(0);
        let hits = scorer.score(&handle, &query, 5).await.unwrap();
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|h| h.id < 20));
        assert!(hits.iter().all(|h| h.score.is_finite()));
    }

    #[tokio::test]
    async fn scores_are_descending() {
        let scorer = SapIvfScorer::new();
        let vectors = random_vectors(30, 8, 2);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(0.5, 5, 3), &vectors)
            .await
            .unwrap();
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

    /// At β=0 with s=1 and full probe (nprobe = n_centroids), ciphertexts equal the
    /// plaintext vectors exactly, so the result must match brute-force f32 L2 search.
    /// PlaintextScorer at full probe also matches brute-force, so this transitively
    /// verifies ID-level agreement with PlaintextScorer.
    #[tokio::test]
    async fn beta_zero_full_probe_matches_brute_force() {
        let scorer = SapIvfScorer::new();
        let vectors = random_vectors(50, 8, 3);
        let k = 5;
        let n_centroids = 8;
        // s=1.0 so ciphertexts = 1.0·m = m exactly (no scaling, no noise at β=0).
        let config = SapIvfConfig {
            key: keygen(1.0, 1.0, 7),
            beta: 0.0,
            upload_seed: 42,
            n_centroids,
            nprobe: n_centroids,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        let query = random_vectors(1, 8, 101).remove(0);
        let hits = scorer.score(&handle, &query, k).await.unwrap();
        let got: Vec<u32> = hits.iter().map(|h| h.id).collect();
        let expected = brute_force_top_k(&vectors, &query, k);
        assert_eq!(
            got, expected,
            "full probe at β=0, s=1 must match brute force"
        );
    }

    #[tokio::test]
    async fn k_capped_at_cluster_size() {
        let scorer = SapIvfScorer::new();
        let vectors = random_vectors(3, 4, 4);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(0.0, 1, 1), &vectors)
            .await
            .unwrap();
        let query = random_vectors(1, 4, 102).remove(0);
        let hits = scorer.score(&handle, &query, 100).await.unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[tokio::test]
    async fn error_on_empty_cluster() {
        let scorer = SapIvfScorer::new();
        let err = scorer
            .upload_cluster(&cfg(0.0, 1, 1), &[])
            .await
            .unwrap_err();
        assert!(matches!(err, SapIvfError::EmptyCluster));
    }

    #[tokio::test]
    async fn error_on_too_many_centroids() {
        let scorer = SapIvfScorer::new();
        let vectors = random_vectors(3, 4, 5);
        let err = scorer
            .upload_cluster(&cfg(0.0, 10, 1), &vectors)
            .await
            .unwrap_err();
        assert!(matches!(err, SapIvfError::TooManyCentroids(_)));
    }

    #[tokio::test]
    async fn error_on_invalid_beta() {
        let scorer = SapIvfScorer::new();
        let vectors = random_vectors(5, 4, 5);
        let err = scorer
            .upload_cluster(&cfg(-0.1, 2, 1), &vectors)
            .await
            .unwrap_err();
        assert!(matches!(err, SapIvfError::InvalidBeta(_)));
    }

    #[tokio::test]
    async fn error_on_dimension_mismatch() {
        let scorer = SapIvfScorer::new();
        let vectors = random_vectors(10, 4, 6);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(0.0, 2, 2), &vectors)
            .await
            .unwrap();
        let wrong = random_vectors(1, 8, 103).remove(0);
        let err = scorer.score(&handle, &wrong, 3).await.unwrap_err();
        assert!(matches!(err, SapIvfError::DimensionMismatch { .. }));
    }

    #[tokio::test]
    async fn debug_redacts_prf_key() {
        let scorer = SapIvfScorer::new();
        let vectors = random_vectors(5, 4, 7);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(0.0, 2, 2), &vectors)
            .await
            .unwrap();
        let s = format!("{:?}", handle);
        assert!(s.contains("[redacted]"));
        assert!(!s.contains("prf_key: ["));
    }

    #[tokio::test]
    async fn communication_cost_formula() {
        let scorer = SapIvfScorer::new();
        let vectors = random_vectors(10, 16, 8);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(0.0, 2, 2), &vectors)
            .await
            .unwrap();
        let cost = scorer.communication_cost(&handle, 5);
        assert_eq!(cost.query_bytes, 16 * 8);
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
    /// after k-means converges.
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
        let scorer = SapIvfScorer::new();
        let vectors = skewed_corpus_4_clusters();
        // β=0, s=1 so ciphertexts equal plaintext vectors and k-means
        // assignments mirror plaintext exactly.
        let config = SapIvfConfig {
            key: keygen(1.0, 1.0, 7),
            beta: 0.0,
            upload_seed: 42,
            n_centroids: 4,
            nprobe: 2,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let (handle, _) = scorer.upload_cluster(&config, &vectors).await.unwrap();

        let (_, cost_a) = scorer
            .score_with_realised_cost(&handle, &Vector(vec![0.0]), 10)
            .await
            .unwrap();
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
        let scorer = SapIvfScorer::new();
        let vectors = skewed_corpus_4_clusters();
        let config = SapIvfConfig {
            key: keygen(1.0, 1.0, 7),
            beta: 0.0,
            upload_seed: 42,
            n_centroids: 4,
            nprobe: 4,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let (handle, _) = scorer.upload_cluster(&config, &vectors).await.unwrap();

        let (_, cost) = scorer
            .score_with_realised_cost(&handle, &Vector(vec![15.0]), 10)
            .await
            .unwrap();
        assert_eq!(cost.response_bytes, 35 * 8);
    }

    #[tokio::test]
    async fn realised_cost_matches_communication_cost_when_uniform() {
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

        let scorer = SapIvfScorer::new();
        let config = SapIvfConfig {
            key: keygen(1.0, 1.0, 7),
            beta: 0.0,
            upload_seed: 42,
            n_centroids: 4,
            nprobe: 2,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let (handle, _) = scorer.upload_cluster(&config, &vectors).await.unwrap();

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
        let scorer = SapIvfScorer::new();
        let vectors = random_vectors(40, 8, 60);
        // β = 0, full probe, s = 1 → result is rng-independent at the
        // ranking level (matches brute force; see
        // `beta_zero_full_probe_matches_brute_force`).
        let n_centroids = 4;
        let config = SapIvfConfig {
            key: keygen(1.0, 1.0, 7),
            beta: 0.0,
            upload_seed: 42,
            n_centroids,
            nprobe: n_centroids,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        let query = random_vectors(1, 8, 61).remove(0);

        let baseline = scorer.score(&handle, &query, 5).await.unwrap();
        let (with_breakdown, timing) = scorer
            .score_with_breakdown(&handle, &query, 5)
            .await
            .unwrap();

        let baseline_ids: Vec<u32> = baseline.iter().map(|h| h.id).collect();
        let breakdown_ids: Vec<u32> = with_breakdown.iter().map(|h| h.id).collect();
        assert_eq!(baseline_ids, breakdown_ids);
        let _ = timing.encode_us + timing.route_us + timing.distance_us + timing.merge_us;
    }

    /// Per-query result from the batched override is equivalent
    /// (ranking-equal + score within 1e-6 rel tol) to a sequential
    /// `score()` call on the same input. SAP IVF downcasts the f64
    /// ciphertext to f32 before invoking [`ivf::probe_batch`], so the
    /// arithmetic precision is f32 and the 1e-6 rel-tol applies (not the
    /// 1e-12 f64 tier). β = 0 neutralises per-query nonce divergence so
    /// the trait-surface comparison is deterministic. The implementation
    /// preserves summation order, so this is effectively bit-equal
    /// today; the tolerance assertion leaves headroom for a future SIMD
    /// lift.
    #[tokio::test]
    async fn sap_ivf_score_batch_matches_score() {
        let scorer = SapIvfScorer::new();
        let vectors = random_vectors(128, 16, 27);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(0.0, 8, 3), &vectors)
            .await
            .unwrap();

        let queries = random_vectors(64, 16, 28);
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
        let scorer = SapIvfScorer::new();
        let vectors = random_vectors(40, 8, 50);
        let (_h1, b1) = scorer
            .upload_cluster(&cfg(0.0, 4, 2), &vectors)
            .await
            .unwrap();
        assert!(!b1.cache_hit, "first build must be a cold build");
        let (_h2, b2) = scorer
            .upload_cluster(&cfg(0.0, 4, 2), &vectors)
            .await
            .unwrap();
        assert!(b2.cache_hit, "second build must hit the in-memory cache");
        assert!(
            b2.build_duration < std::time::Duration::from_millis(100),
            "warm build was {:?}, expected < 100 ms",
            b2.build_duration
        );
    }

    /// Correctness gate, full-probe regime. β=0, s=1 → ciphertexts
    /// equal plaintexts; with `nprobe = n_centroids` every cluster is
    /// scored, so the result must approximately equal CPU brute force.
    /// Threshold is set-overlap ≥ 8/10 because cuVS L2Expanded admits
    /// catastrophic cancellation at the top-k boundary on identical
    /// inputs.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_full_probe_matches_brute_force() {
        let vectors = random_vectors(256, 64, 909);
        let query = random_vectors(1, 64, 910).remove(0);
        let k = 10;
        let n_centroids = 8;

        let scorer = SapIvfScorer::new();
        let config = SapIvfConfig {
            key: keygen(1.0, 1.0, 7),
            beta: 0.0,
            upload_seed: 42,
            n_centroids,
            nprobe: n_centroids,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Gpu,
            vram_budget_bytes: None,
        };
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
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

    /// Structural invariant: with `nprobe < n_centroids` every returned
    /// hit must come from one of the probed clusters. Recall is *not*
    /// asserted (it's expected < 1.0 at small nprobe).
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_partial_probe_returns_subset_of_probed_clusters() {
        let vectors = random_vectors(256, 64, 911);
        let query = random_vectors(1, 64, 912).remove(0);
        let k = 10;
        let n_centroids = 8;
        let nprobe = 2;

        let scorer = SapIvfScorer::new();
        let config = SapIvfConfig {
            key: keygen(1.0, 1.0, 7),
            beta: 0.0,
            upload_seed: 42,
            n_centroids,
            nprobe,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Gpu,
            vram_budget_bytes: None,
        };
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        let hits = scorer.score(&handle, &query, k).await.unwrap();

        // Routing runs on the plaintext query against the plaintext
        // centroids that k-means trained, so re-deriving the expected
        // probe set uses `query.0` directly (no encryption step).
        let probe_set = ivf::probe_route(&handle.index, &query.0, nprobe);
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

    /// Realised-cost gate: `effective_bytes_per_query` and
    /// `response_bytes` must reflect the *probed* cluster set, not the
    /// full corpus, when `nprobe < n_centroids`.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_realised_cost_matches_probed_clusters() {
        let vectors = random_vectors(256, 64, 913);
        let query = random_vectors(1, 64, 914).remove(0);
        let k = 10;
        let n_centroids = 8;
        let nprobe = 2;

        let scorer = SapIvfScorer::new();
        let config = SapIvfConfig {
            key: keygen(1.0, 1.0, 7),
            beta: 0.0,
            upload_seed: 42,
            n_centroids,
            nprobe,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Gpu,
            vram_budget_bytes: None,
        };
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        let (_hits, cost) = scorer
            .score_with_realised_cost(&handle, &query, k)
            .await
            .unwrap();

        let n_total: u64 = handle.index.clusters.iter().map(|c| c.len() as u64).sum();
        assert!(
            cost.effective_bytes_per_query < n_total * handle.dim as u64 * 4,
            "with nprobe < n_centroids, effective bytes must be < full-corpus bytes"
        );
        // `response_bytes` is Σ_probed min(k, m_i) × 8; with k=10 and
        // 256 vectors / 8 centroids → m_avg=32, the upper bound is
        // nprobe × k × 8 = 2 × 10 × 8 = 160. Lower bound is at least
        // one byte per probed cluster (cluster sizes are non-zero by
        // construction here).
        assert!(cost.response_bytes > 0);
        assert!(cost.response_bytes <= (nprobe as u64) * (k as u64) * 8);
    }

    /// sap-IVF GPU breakdown agrees with score(). Validates
    /// `score_with_breakdown` accepts GPU and the substep timings span
    /// the host route + cuVS launches.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_score_with_breakdown_matches_score() {
        let vectors = random_vectors(256, 64, 922);
        let query = random_vectors(1, 64, 923).remove(0);
        let k = 10;
        let n_centroids = 8;
        let nprobe = 4;

        let scorer = SapIvfScorer::new();
        let config = SapIvfConfig {
            key: keygen(1.0, 1.0, 7),
            beta: 0.0,
            upload_seed: 42,
            n_centroids,
            nprobe,
            train_seed: 42,
            max_iter: 25,
            progress: None,
            device: Device::Gpu,
            vram_budget_bytes: None,
        };
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();

        let baseline = scorer.score(&handle, &query, k).await.unwrap();
        let (with_breakdown, timing) = scorer
            .score_with_breakdown(&handle, &query, k)
            .await
            .unwrap();

        let baseline_ids: std::collections::HashSet<u32> = baseline.iter().map(|h| h.id).collect();
        let breakdown_ids: std::collections::HashSet<u32> =
            with_breakdown.iter().map(|h| h.id).collect();
        assert_eq!(baseline_ids, breakdown_ids);
        assert!(
            timing.distance_us > 0,
            "distance_us should cover cuVS launches: {timing:?}"
        );
    }
}
