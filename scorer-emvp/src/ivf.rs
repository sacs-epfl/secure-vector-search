//! EMVP + IVF scorer.
//!
//! Per-cluster EMVP encryption with shared D and per-cluster R seeds
//! (security: prevents the M̂_i − M̂_j = (M_i − M_j)·D cross-cluster
//! leakage). Plaintext-side routing on cleartext centroids — EMVP is
//! not distance-preserving, so the server cannot route. Server learns
//! the probe set; same access-pattern leakage as SAP+IVF.
//!
//! `progress.on_encrypt(i)` granularity: per-cluster (called once after
//! each cluster's M̂_i is built).

use std::collections::HashMap;
use std::fmt;
use std::io::{self, BufWriter, Seek, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use ivf_index::{
    ivf::{self, IvfIndex},
    kmeans,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use scorer_core::{
    BuildOutcome, CommunicationCost, Device, Hit, ProgressReporter, Scorer, Vector, cache,
};

use crate::EncryptedCluster;
use crate::Mhat;
use crate::crypto;
#[cfg(feature = "gpu")]
use crate::gpu;
use crate::params::{FIELD_BYTES, SEC128};

/// Configuration for `EmvpIvfScorer::upload_cluster`.
///
/// `progress.on_encrypt(i)` fires once per cluster after its M̂_i is
/// built (per-cluster granularity, not per-row). `on_init_centroid`
/// and `on_kmeans_iter` fire during k-means.
pub struct EmvpIvfConfig {
    /// Seed for the D matrix (shared across all clusters in this index).
    pub key_seed: [u8; 32],
    pub n_centroids: usize,
    pub nprobe: usize,
    pub train_seed: u64,
    pub max_iter: usize,
    /// Seed for per-cluster R seed derivation (`r_seed_i = derive(upload_seed, i)`).
    pub upload_seed: u64,
    pub progress: Option<Arc<dyn ProgressReporter>>,
    /// Compute substrate. `Cpu` runs the per-cluster
    /// `compute_products` over rayon. `Gpu` requires the `gpu` Cargo
    /// feature; without it `upload_cluster` returns
    /// [`EmvpIvfError::GpuFeatureNotEnabled`]. Encryption + k-means +
    /// decoding stay on host either way; only the per-cluster
    /// server-side block-GEMV moves to device. GPU and CPU runs share
    /// centroids and cluster assignments.
    pub device: Device,
    /// VRAM budget for the streaming cluster store on GPU runs. `None`
    /// (default) auto-detects 80 % of free VRAM via NVML at handle
    /// build. `Some(b)` pins a hard byte cap. Ignored on CPU runs.
    pub vram_budget_bytes: Option<u64>,
}

/// Opaque per-index state.
pub struct EmvpIvfHandle {
    /// Plaintext-side IVF (centroids + per-cluster (id, vector) tuples).
    /// Held by the scorer; the centroids drive client-side routing, and
    /// the per-cluster vector ids are used to map cluster-local indices
    /// back to global `VectorId`s after decryption.
    index: Arc<IvfIndex>,
    /// Per-cluster encrypted matrices (M̂_i).
    /// SCALE: ~10 GiB at 1M corpus — demand-loading is future work.
    clusters: Arc<Vec<EncryptedCluster>>,
    /// D matrix (ell0 × n), shared across clusters.
    d_mat: Arc<Vec<u64>>,
    nprobe: usize,
    dim: usize,
    inner: HandleInner,
}

pub(crate) enum HandleInner {
    Cpu,
    /// Per-cluster device-resident M̂; per-query kernel launches per
    /// probed cluster. Decoding stays on CPU. Boxed because the
    /// streaming `IvfGpuState` carries host-side per-cluster M̂
    /// payloads + LRU + sampler — clippy's `large_enum_variant` lint
    /// trips otherwise.
    #[cfg(feature = "gpu")]
    Gpu(Box<gpu::IvfGpuState>),
}

impl fmt::Debug for EmvpIvfHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device = match &self.inner {
            HandleInner::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(_) => "gpu",
        };
        f.debug_struct("EmvpIvfHandle")
            .field("device", &device)
            .field("dim", &self.dim)
            .field("n_centroids", &self.index.centroids.len())
            .field("nprobe", &self.nprobe)
            .field(
                "cluster_sizes",
                &self.clusters.iter().map(|c| c.m).collect::<Vec<_>>(),
            )
            .field("r_seeds", &"[redacted]")
            .finish()
    }
}

impl EmvpIvfHandle {
    /// NVML-sampled peak VRAM for this handle's lifetime. `None` on
    /// CPU handles or when NVML was unavailable at GPU handle build.
    pub fn peak_vram_bytes(&self) -> Option<u64> {
        match &self.inner {
            HandleInner::Cpu => None,
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => state.peak_vram_bytes(),
        }
    }

    /// Resolved VRAM budget. `None` on CPU handles.
    pub fn vram_budget_bytes(&self) -> Option<u64> {
        match &self.inner {
            HandleInner::Cpu => None,
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => Some(state.vram_budget_bytes()),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmvpIvfError {
    #[error("corpus must contain at least one vector")]
    EmptyCorpus,
    #[error("k-means produced an empty cluster (corpus too small for n_centroids?)")]
    EmptyCluster,
    #[error("n_centroids {0} exceeds corpus size")]
    TooManyCentroids(usize),
    #[error("invalid nprobe {nprobe}: must be in 1..={n_centroids}")]
    InvalidNprobe { nprobe: usize, n_centroids: usize },
    #[error("dimension too large: vectors have dim {actual}, ell0 = {ell0}")]
    DimensionTooLarge { actual: usize, ell0: usize },
    #[error("query dimension {query_dim} does not match index dimension {index_dim}")]
    DimensionMismatch { query_dim: usize, index_dim: usize },
    #[error("spawn_blocking panicked")]
    SpawnPanic,
    /// `Device::Gpu` was requested but the crate was built without the
    /// `gpu` Cargo feature.
    #[error("Device::Gpu requested but the `gpu` Cargo feature is not enabled in this build")]
    GpuFeatureNotEnabled,
    /// GPU runtime error from cudarc (kernel launch / device alloc).
    #[cfg(feature = "gpu")]
    #[error("CUDA GPU error: {0}")]
    Gpu(#[from] gpu::GpuError),
}

/// Cached state for one (n_centroids, train_seed, max_iter, N, upload_seed,
/// key_seed) combination — built once per nprobe sweep.
#[derive(Clone)]
struct CachedIndex {
    index: Arc<IvfIndex>,
    clusters: Arc<Vec<EncryptedCluster>>,
    d_mat: Arc<Vec<u64>>,
    dim: usize,
}

pub struct EmvpIvfScorer {
    cache: Mutex<HashMap<String, CachedIndex>>,
    cache_dir: Option<PathBuf>,
}

impl EmvpIvfScorer {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            cache_dir: None,
        }
    }

    pub fn with_cache_dir(dir: impl Into<PathBuf>) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            cache_dir: Some(dir.into()),
        }
    }
}

impl Default for EmvpIvfScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Derive a per-cluster R seed deterministically from `(upload_seed, cluster_idx)`.
/// ChaCha20 mixing avoids structural collisions across small offsets.
/// Byte size of the IVF index payload as written by
/// `ivf::write_index_to`, computed without serialising. The streaming
/// cache writer needs this to track file position and place the
/// 8-byte-alignment pad correctly so the subsequent per-cluster
/// `m_hat` regions land at mmap-castable offsets.
///
/// Format (mirrors ivf-index::ivf::write_index_to):
///   [u64 dim] [u64 n_centroids] [centroids: n_centroids × dim × f32]
///   [u64 n_clusters_in_index]
///     for each cluster: [u64 size] then size × ([u32 id] [dim × f32 vector])
fn ivf_payload_size(index: &IvfIndex, dim: usize) -> u64 {
    let mut size: u64 = 8 + 8 + (index.centroids.len() as u64) * (dim as u64) * 4 + 8;
    for cluster in &index.clusters {
        size += 8 + (cluster.len() as u64) * (4 + (dim as u64) * 4);
    }
    size
}

fn derive_r_seed(upload_seed: u64, cluster_idx: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&upload_seed.to_le_bytes());
    seed[8..16].copy_from_slice(&(cluster_idx as u64).to_le_bytes());
    let mut rng = ChaCha20Rng::from_seed(seed);
    let mut out = [0u8; 32];
    for b in out.iter_mut() {
        *b = rng.random();
    }
    out
}

#[async_trait]
impl Scorer for EmvpIvfScorer {
    type Config = EmvpIvfConfig;
    type ClusterHandle = EmvpIvfHandle;
    type Error = EmvpIvfError;

    async fn upload_cluster(
        &self,
        config: &EmvpIvfConfig,
        vectors: &[Vector],
    ) -> Result<(EmvpIvfHandle, BuildOutcome), EmvpIvfError> {
        let build_start = Instant::now();
        let params = &SEC128;
        let ell0 = params.ell0;

        if vectors.is_empty() {
            return Err(EmvpIvfError::EmptyCorpus);
        }
        if config.n_centroids > vectors.len() {
            return Err(EmvpIvfError::TooManyCentroids(config.n_centroids));
        }
        if config.nprobe == 0 || config.nprobe > config.n_centroids {
            return Err(EmvpIvfError::InvalidNprobe {
                nprobe: config.nprobe,
                n_centroids: config.n_centroids,
            });
        }
        let dim = vectors[0].0.len();
        if dim > ell0 {
            return Err(EmvpIvfError::DimensionTooLarge { actual: dim, ell0 });
        }

        let key_seed = config.key_seed;
        let n_centroids = config.n_centroids;
        let nprobe = config.nprobe;
        let train_seed = config.train_seed;
        let max_iter = config.max_iter;
        let upload_seed = config.upload_seed;
        let progress = config.progress.clone();

        let cache_parts: [Vec<u8>; 7] = [
            b"emvp-ivf".to_vec(),
            (n_centroids as u64).to_le_bytes().to_vec(),
            train_seed.to_le_bytes().to_vec(),
            (max_iter as u64).to_le_bytes().to_vec(),
            (vectors.len() as u64).to_le_bytes().to_vec(),
            upload_seed.to_le_bytes().to_vec(),
            key_seed.to_vec(),
        ];
        let parts_refs: Vec<&[u8]> = cache_parts.iter().map(|v| v.as_slice()).collect();
        let fp = cache::fingerprint(&parts_refs);
        let disk_path = self
            .cache_dir
            .as_deref()
            .map(|dir| dir.join(format!(".emvp-ivf-cache-{fp}.bin")));

        let cached = self.cache.lock().unwrap().get(&fp).cloned();
        let in_memory_hit = cached.is_some();
        let mut disk_hit = false;
        let cached_state = if let Some(c) = cached {
            if let Some(ref p) = progress {
                p.on_build_complete();
            }
            Some(c)
        } else {
            // Try disk cache.
            let from_disk = if let Some(ref path) = disk_path {
                if path.exists() {
                    let p = path.clone();
                    eprintln!("Loading cached EMVP+IVF index from {p:?}…");
                    let parts_owned = cache_parts.clone();
                    tokio::task::spawn_blocking(move || {
                        let refs: Vec<&[u8]> = parts_owned.iter().map(|v| v.as_slice()).collect();
                        load_emvp_ivf_cache(&p, &refs).ok().flatten()
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

            if let Some((index, clusters, d_mat)) = from_disk {
                disk_hit = true;
                let state = CachedIndex {
                    index: Arc::new(index),
                    clusters: Arc::new(clusters),
                    d_mat: Arc::new(d_mat),
                    dim,
                };
                if let Some(ref p) = progress {
                    p.on_build_complete();
                }
                self.cache.lock().unwrap().insert(fp.clone(), state.clone());
                Some(state)
            } else {
                None
            }
        };

        let cached_state = if let Some(c) = cached_state {
            c
        } else {
            // Build from scratch.
            let data: Vec<Vec<f32>> = vectors.iter().map(|v| v.0.clone()).collect();
            let progress_for_build = progress.clone();
            let save_path = disk_path.clone();
            let save_parts = cache_parts.clone();

            let (index, clusters, d_mat) = tokio::task::spawn_blocking(move || {
                let progress = progress_for_build;

                // 1. k-means + IVF over plaintext.
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
                let mut index = ivf::build_index(&data, centroids, dim);
                drop(data);

                // 2. Generate D once.
                let d_mat = crypto::generate_d(key_seed);

                // 3. Pick build mode. `save_path = Some` →
                //    stream-to-disk + mmap (when the m_hat sum exceeds
                //    physical RAM, we never accumulate Vec<u64> m_hats
                //    — each is written to file then dropped, and the
                //    post-build handle holds mmap slices into the
                //    on-disk cache instead).
                //    `save_path = None` → owned in-memory (tests,
                //    no-cache mode; corpora are tiny here).
                //
                //    Cache file layout under mmap mode (differs from
                //    the single-shot `save_emvp_ivf_cache` layout —
                //    older caches without the alignment pad won't load
                //    and will rebuild from scratch via the
                //    `.ok().flatten()` fall-through):
                //      [20  cache header]
                //      [var IVF index payload]                         (sequential read)
                //      [pad to 8B  pad bytes (0..7)]                   (alignment for mmap u64 cast)
                //      [8   u64 n_clusters]
                //      [for each cluster:
                //         [8   u64 m]
                //         [32  r_seed]
                //         [m*n*8  m_hat u64s]                          (8B-aligned, mmap-castable)
                //      ]
                //      [d_mat_len*8  d_mat u64s]
                //
                //    The per-cluster m_hat offset is recorded as we
                //    write, so reconstruction after build doesn't
                //    need to re-parse the file.
                let n_n = SEC128.n;

                let (clusters, encryption_io_err): (Vec<EncryptedCluster>, Option<io::Error>) =
                    if let Some(ref final_path) = save_path {
                        let tmp_path = final_path.with_extension("bin.tmp");
                        // Best-effort streaming. Any I/O error switches
                        // us to "no cache, accumulate owned" mode for
                        // the remainder; at multi-million-vector scale
                        // that would re-introduce the out-of-memory
                        // condition, so we surface the error and let
                        // the caller see a partial run.
                        let res: io::Result<Vec<EncryptedCluster>> = (|| {
                            let mut bytes_written: u64 = 0;
                            let f = std::fs::File::create(&tmp_path)?;
                            let mut w = BufWriter::new(f);
                            let refs: Vec<&[u8]> =
                                save_parts.iter().map(|v| v.as_slice()).collect();
                            cache::write_header(&mut w, &refs)?;
                            bytes_written += 20;
                            ivf::write_index_to(&mut w, &index)?;
                            bytes_written += ivf_payload_size(&index, dim);
                            // Pad to 8-byte alignment so subsequent
                            // m_hat regions are mmap-castable to &[u64].
                            let pad = ((8 - (bytes_written % 8)) % 8) as usize;
                            if pad > 0 {
                                w.write_all(&[0u8; 8][..pad])?;
                                bytes_written += pad as u64;
                            }
                            debug_assert_eq!(bytes_written % 8, 0);
                            w.write_all(&(index.clusters.len() as u64).to_le_bytes())?;
                            bytes_written += 8;

                            let mut metas: Vec<(usize, [u8; 32], usize, usize)> =
                                Vec::with_capacity(index.clusters.len());
                            for i in 0..index.clusters.len() {
                                let m = index.clusters[i].len();
                                let r_seed = derive_r_seed(upload_seed, i);
                                let m_hat: Vec<u64> = if m == 0 {
                                    Vec::new()
                                } else {
                                    let cluster = &index.clusters[i];
                                    let row_refs: Vec<&[f32]> =
                                        cluster.iter().map(|(_, v)| v.as_slice()).collect();
                                    crypto::encrypt_cluster(&d_mat, r_seed, &row_refs, None)
                                };
                                w.write_all(&(m as u64).to_le_bytes())?;
                                w.write_all(&r_seed)?;
                                bytes_written += 8 + 32;
                                let m_hat_byte_offset = bytes_written as usize;
                                for &v in &m_hat {
                                    w.write_all(&v.to_le_bytes())?;
                                }
                                bytes_written += (m_hat.len() as u64) * 8;
                                metas.push((m, r_seed, m_hat_byte_offset, m_hat.len()));

                                // Strip f32 vectors NOW that the cluster
                                // is on disk; query path uses only the
                                // u32 ID from each tuple.
                                for (_, v) in index.clusters[i].iter_mut() {
                                    v.clear();
                                    v.shrink_to_fit();
                                }
                                if let Some(ref p) = progress {
                                    p.on_encrypt(i);
                                }
                                // m_hat drops here — Vec<u64> returned
                                // to the allocator, NOT pushed into
                                // clusters Vec (this is what keeps peak
                                // memory bounded at large scale).
                            }

                            // d_mat at the end of the file.
                            for &v in &d_mat {
                                w.write_all(&v.to_le_bytes())?;
                            }
                            w.flush()?;
                            drop(w);
                            std::fs::rename(&tmp_path, final_path)?;

                            // Reopen read-only and mmap. Construct
                            // EncryptedCluster Vec with Mhat::Mmap
                            // entries pointing into the mmap.
                            let f_ro = std::fs::File::open(final_path)?;
                            let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f_ro)? });
                            let clusters = metas
                                .into_iter()
                                .map(|(m, r_seed, byte_offset, len_u64s)| {
                                    debug_assert_eq!(len_u64s, m * n_n);
                                    EncryptedCluster {
                                        m_hat: Mhat::Mmap {
                                            mmap: mmap.clone(),
                                            byte_offset,
                                            len_u64s,
                                        },
                                        r_seed,
                                        m,
                                    }
                                })
                                .collect();
                            Ok(clusters)
                        })();
                        match res {
                            Ok(c) => (c, None),
                            Err(e) => {
                                let _ = std::fs::remove_file(&tmp_path);
                                (Vec::new(), Some(e))
                            }
                        }
                    } else {
                        // No save_path: owned mode (tests / small
                        // corpora). In-memory m_hats, no mmap.
                        let mut clusters: Vec<EncryptedCluster> =
                            Vec::with_capacity(index.clusters.len());
                        for i in 0..index.clusters.len() {
                            let m = index.clusters[i].len();
                            let r_seed = derive_r_seed(upload_seed, i);
                            let m_hat = if m == 0 {
                                Vec::new()
                            } else {
                                let cluster = &index.clusters[i];
                                let row_refs: Vec<&[f32]> =
                                    cluster.iter().map(|(_, v)| v.as_slice()).collect();
                                crypto::encrypt_cluster(&d_mat, r_seed, &row_refs, None)
                            };
                            clusters.push(EncryptedCluster {
                                m_hat: Mhat::Owned(Arc::new(m_hat)),
                                r_seed,
                                m,
                            });
                            // Strip even in owned mode — tests pass
                            // either way and the contract is the same:
                            // query uses .0 only.
                            for (_, v) in index.clusters[i].iter_mut() {
                                v.clear();
                                v.shrink_to_fit();
                            }
                            if let Some(ref p) = progress {
                                p.on_encrypt(i);
                            }
                        }
                        (clusters, None)
                    };
                if let Some(e) = encryption_io_err {
                    eprintln!("Warning: EMVP+IVF cache write failed: {e}");
                }

                if let Some(ref p) = progress {
                    p.on_build_complete();
                }

                (index, clusters, d_mat)
            })
            .await
            .map_err(|_| EmvpIvfError::SpawnPanic)?;

            // Sanity-check: if k-means produced any empty cluster, reject the build
            // (signals corpus too small for n_centroids).
            if clusters.iter().any(|c| c.m == 0) {
                return Err(EmvpIvfError::EmptyCluster);
            }

            let state = CachedIndex {
                index: Arc::new(index),
                clusters: Arc::new(clusters),
                d_mat: Arc::new(d_mat),
                dim,
            };
            self.cache.lock().unwrap().insert(fp.clone(), state.clone());
            state
        };

        // Branch on substrate. The host index + per-cluster M̂_i are
        // built either way; GPU additionally uploads each M̂_i to its
        // own device buffer at handle construction (one-time cost
        // amortised over all queries).
        let inner = match config.device {
            Device::Cpu => HandleInner::Cpu,
            Device::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    let clusters_for_gpu = Arc::clone(&cached_state.clusters);
                    let vram_budget = config.vram_budget_bytes;
                    let state = tokio::task::spawn_blocking(move || {
                        gpu::IvfGpuState::build(clusters_for_gpu, vram_budget)
                    })
                    .await
                    .map_err(|_| EmvpIvfError::SpawnPanic)??;
                    HandleInner::Gpu(Box::new(state))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(EmvpIvfError::GpuFeatureNotEnabled);
                }
            }
        };

        let outcome = BuildOutcome {
            cache_hit: in_memory_hit || disk_hit,
            build_duration: build_start.elapsed(),
        };
        Ok((
            EmvpIvfHandle {
                index: cached_state.index,
                clusters: cached_state.clusters,
                d_mat: cached_state.d_mat,
                nprobe,
                dim: cached_state.dim,
                inner,
            },
            outcome,
        ))
    }

    async fn score(
        &self,
        handle: &EmvpIvfHandle,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<Hit>, EmvpIvfError> {
        if query.0.len() != handle.dim {
            return Err(EmvpIvfError::DimensionMismatch {
                query_dim: query.0.len(),
                index_dim: handle.dim,
            });
        }

        let params = &SEC128;
        let ell0 = params.ell0;
        let k_dim = params.k;

        // 1. Plaintext routing on cleartext centroids (client-side).
        let probe_set = ivf::probe_route(&handle.index, &query.0, handle.nprobe);

        // 2. Pad and upcast query.
        let mut q_f64: Vec<f64> = query.0.iter().map(|&x| x as f64).collect();
        q_f64.resize(ell0, 0.0);

        // 3. Generate u (k_dim field elements) on the async task.
        let u: Vec<u64> = {
            let mut rng = rand::rng();
            (0..k_dim)
                .map(|_| rng.random::<u64>() % crate::params::P)
                .collect()
        };

        let d_mat = Arc::clone(&handle.d_mat);
        let clusters = Arc::clone(&handle.clusters);
        let index = Arc::clone(&handle.index);

        let hits = match &handle.inner {
            HandleInner::Cpu => {
                tokio::task::spawn_blocking(move || {
                    // 4. Encode query once; q̂ is independent of each
                    //    cluster's M̂_i, so it is reused across probes.
                    let q_hat = crypto::encode_query(&d_mat, &q_f64, &u);

                    // 5. Compute + decode per probed cluster, in parallel.
                    let mut all_hits: Vec<(u32, f32)> = probe_set
                        .par_iter()
                        .flat_map(|&ci| {
                            let cluster = &clusters[ci];
                            let m = cluster.m;
                            if m == 0 {
                                return Vec::new();
                            }
                            let partials =
                                crypto::compute_products(cluster.m_hat.as_slice(), m, &q_hat);
                            let scores =
                                crypto::decode_scores(m, cluster.r_seed, &q_hat, &partials);
                            let membership = &index.clusters[ci];
                            scores
                                .into_iter()
                                .enumerate()
                                .map(|(local_i, score)| (membership[local_i].0, score))
                                .collect::<Vec<_>>()
                        })
                        .collect();

                    all_hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    all_hits.truncate(k);
                    all_hits
                        .into_iter()
                        .map(|(id, score)| Hit { id, score })
                        .collect::<Vec<_>>()
                })
                .await
                .map_err(|_| EmvpIvfError::SpawnPanic)?
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                // Encode q̂ on host, then dispatch one kernel launch
                // per probed cluster. The probed clusters are scored
                // sequentially through the cuda stream; cudarc's
                // `compute_products` lock-then-launch path serialises
                // them. Decoding stays in rayon-parallel CPU code.
                let q_hat =
                    tokio::task::spawn_blocking(move || crypto::encode_query(&d_mat, &q_f64, &u))
                        .await
                        .map_err(|_| EmvpIvfError::SpawnPanic)?;

                let mut per_cluster: Vec<(usize, Vec<u64>)> = Vec::with_capacity(probe_set.len());
                for &ci in &probe_set {
                    if let Some(partials) = state.compute_products(ci, &q_hat)? {
                        per_cluster.push((ci, partials));
                    }
                }

                tokio::task::spawn_blocking(move || {
                    let mut all_hits: Vec<(u32, f32)> = per_cluster
                        .into_par_iter()
                        .flat_map(|(ci, partials)| {
                            let cluster = &clusters[ci];
                            let m = cluster.m;
                            if m == 0 {
                                return Vec::new();
                            }
                            let scores =
                                crypto::decode_scores(m, cluster.r_seed, &q_hat, &partials);
                            let membership = &index.clusters[ci];
                            scores
                                .into_iter()
                                .enumerate()
                                .map(|(local_i, score)| (membership[local_i].0, score))
                                .collect::<Vec<_>>()
                        })
                        .collect();

                    all_hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    all_hits.truncate(k);
                    all_hits
                        .into_iter()
                        .map(|(id, score)| Hit { id, score })
                        .collect::<Vec<_>>()
                })
                .await
                .map_err(|_| EmvpIvfError::SpawnPanic)?
            }
        };

        Ok(hits)
    }

    fn communication_cost(&self, handle: &EmvpIvfHandle, _k: usize) -> CommunicationCost {
        let params = &SEC128;
        let query_bytes = (params.n * FIELD_BYTES) as u64;
        // Analytical estimate using the mean per-cluster response over
        // the index. The realised per-query value (which clusters were
        // actually probed, with what sizes) is captured by
        // `score_with_realised_cost`; use that path when figures need
        // honest variance accounting.
        let mean_m: f64 = if handle.clusters.is_empty() {
            0.0
        } else {
            handle.clusters.iter().map(|c| c.m as f64).sum::<f64>() / handle.clusters.len() as f64
        };
        let avg_cluster_resp = (mean_m * params.s as f64 * FIELD_BYTES as f64).round() as u64;
        let total_setup: u64 = handle
            .clusters
            .iter()
            .map(|c| (c.m * params.n * FIELD_BYTES) as u64)
            .sum();
        // GPU bandwidth proxy = bytes the matrix kernel streams per
        // query. Each probed cluster's `compute_products` reads the
        // full `m × n` encrypted matrix M_hat; analytical estimate
        // sums `m_avg × n × FIELD_BYTES` across the probe set (the
        // matrix touched), not the s × m server output shape (which
        // would under-count matrix work by a factor of n/s ≈ 17).
        let effective_bytes_per_query = (handle.nprobe as u64)
            * (mean_m.round() as u64)
            * (params.n as u64)
            * (FIELD_BYTES as u64);
        CommunicationCost {
            query_bytes,
            response_bytes: handle.nprobe as u64 * avg_cluster_resp,
            cluster_response_bytes: avg_cluster_resp,
            setup_bytes: total_setup,
            pre_query_offline_up_bytes: 0,
            pre_query_offline_down_bytes: 0,
            effective_bytes_per_query,
        }
    }

    async fn score_with_realised_cost(
        &self,
        handle: &EmvpIvfHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, CommunicationCost), EmvpIvfError> {
        // Routing happens once below (`probe_route`); the realised
        // probe set drives both the scoring path and the per-query
        // `cluster_response_bytes` accounting. We don't call
        // `Self::score` here because the routing result needs to be
        // shared between the scoring path and the cost calculation.
        if query.0.len() != handle.dim {
            return Err(EmvpIvfError::DimensionMismatch {
                query_dim: query.0.len(),
                index_dim: handle.dim,
            });
        }

        let params = &SEC128;
        let ell0 = params.ell0;
        let k_dim = params.k;

        let probe_set = ivf::probe_route(&handle.index, &query.0, handle.nprobe);
        let probed_sizes: Vec<usize> = probe_set.iter().map(|&ci| handle.clusters[ci].m).collect();

        let mut q_f64: Vec<f64> = query.0.iter().map(|&x| x as f64).collect();
        q_f64.resize(ell0, 0.0);

        let u: Vec<u64> = {
            let mut rng = rand::rng();
            (0..k_dim)
                .map(|_| rng.random::<u64>() % crate::params::P)
                .collect()
        };

        let d_mat = Arc::clone(&handle.d_mat);
        let clusters = Arc::clone(&handle.clusters);
        let index = Arc::clone(&handle.index);

        let hits = match &handle.inner {
            HandleInner::Cpu => {
                let probe_set = probe_set.clone();
                tokio::task::spawn_blocking(move || {
                    let q_hat = crypto::encode_query(&d_mat, &q_f64, &u);

                    let mut all_hits: Vec<(u32, f32)> = probe_set
                        .par_iter()
                        .flat_map(|&ci| {
                            let cluster = &clusters[ci];
                            let m = cluster.m;
                            if m == 0 {
                                return Vec::new();
                            }
                            let partials =
                                crypto::compute_products(cluster.m_hat.as_slice(), m, &q_hat);
                            let scores =
                                crypto::decode_scores(m, cluster.r_seed, &q_hat, &partials);
                            let membership = &index.clusters[ci];
                            scores
                                .into_iter()
                                .enumerate()
                                .map(|(local_i, score)| (membership[local_i].0, score))
                                .collect::<Vec<_>>()
                        })
                        .collect();

                    all_hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    all_hits.truncate(k);
                    all_hits
                        .into_iter()
                        .map(|(id, score)| Hit { id, score })
                        .collect::<Vec<_>>()
                })
                .await
                .map_err(|_| EmvpIvfError::SpawnPanic)?
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                let q_hat =
                    tokio::task::spawn_blocking(move || crypto::encode_query(&d_mat, &q_f64, &u))
                        .await
                        .map_err(|_| EmvpIvfError::SpawnPanic)?;

                let mut per_cluster: Vec<(usize, Vec<u64>)> = Vec::with_capacity(probe_set.len());
                for &ci in &probe_set {
                    if let Some(partials) = state.compute_products(ci, &q_hat)? {
                        per_cluster.push((ci, partials));
                    }
                }

                tokio::task::spawn_blocking(move || {
                    let mut all_hits: Vec<(u32, f32)> = per_cluster
                        .into_par_iter()
                        .flat_map(|(ci, partials)| {
                            let cluster = &clusters[ci];
                            let m = cluster.m;
                            if m == 0 {
                                return Vec::new();
                            }
                            let scores =
                                crypto::decode_scores(m, cluster.r_seed, &q_hat, &partials);
                            let membership = &index.clusters[ci];
                            scores
                                .into_iter()
                                .enumerate()
                                .map(|(local_i, score)| (membership[local_i].0, score))
                                .collect::<Vec<_>>()
                        })
                        .collect();

                    all_hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    all_hits.truncate(k);
                    all_hits
                        .into_iter()
                        .map(|(id, score)| Hit { id, score })
                        .collect::<Vec<_>>()
                })
                .await
                .map_err(|_| EmvpIvfError::SpawnPanic)?
            }
        };

        let mut cost = self.communication_cost(handle, k);
        let per_cluster: Vec<u64> = probed_sizes
            .iter()
            .map(|&m_i| (m_i * params.s * FIELD_BYTES) as u64)
            .collect();
        let response_bytes: u64 = per_cluster.iter().sum();
        cost.response_bytes = response_bytes;
        cost.cluster_response_bytes = if per_cluster.is_empty() {
            0
        } else {
            response_bytes / per_cluster.len() as u64
        };
        // Realised matrix-work proxy = Σ_probed m_i × n × FIELD_BYTES,
        // the bytes the EMVP matvec actually streams across the probed
        // cluster set this query (the matrix touched), not the s × m
        // result bytes (which would under-count by a factor of n/s).
        let probed_total: u64 = probed_sizes.iter().map(|&m| m as u64).sum();
        cost.effective_bytes_per_query = probed_total * (params.n as u64) * (FIELD_BYTES as u64);
        Ok((hits, cost))
    }

    /// Batched-scoring override. Cluster-major loop-transpose: for
    /// each probed cluster, [`crypto::compute_products_batch`] is
    /// invoked once with the queries that probed it — `m_hat` is read
    /// once per call across those queries — then per-query
    /// [`crypto::decode_scores`] emits (membership_id, score) pairs.
    /// Per-query pairs are bucketed by probe position so the final
    /// gather preserves the per-query path's probe-set ordering;
    /// `sort_unstable_by` then sees a bit-equal input. d_mat and
    /// `clusters` stay Arc-shared via the handle. GPU handles fall
    /// back to the inline sequential loop.
    async fn score_batch(
        &self,
        handle: &EmvpIvfHandle,
        queries: &[Vector],
        k: usize,
    ) -> Result<Vec<Vec<Hit>>, EmvpIvfError> {
        let params = &SEC128;
        let ell0 = params.ell0;
        let k_dim = params.k;

        for q in queries {
            if q.0.len() != handle.dim {
                return Err(EmvpIvfError::DimensionMismatch {
                    query_dim: q.0.len(),
                    index_dim: handle.dim,
                });
            }
        }

        // Per-query routing on cleartext centroids (client-side).
        let probe_sets: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| ivf::probe_route(&handle.index, &q.0, handle.nprobe))
            .collect();

        // Per-query: pad + upcast.
        let queries_f64: Vec<Vec<f64>> = queries
            .iter()
            .map(|q| {
                let mut v: Vec<f64> = q.0.iter().map(|&x| x as f64).collect();
                v.resize(ell0, 0.0);
                v
            })
            .collect();

        // Per-query u: ThreadRng !Send — pre-generate.
        let us: Vec<Vec<u64>> = {
            let mut rng = rand::rng();
            (0..queries.len())
                .map(|_| {
                    (0..k_dim)
                        .map(|_| rng.random::<u64>() % crate::params::P)
                        .collect()
                })
                .collect()
        };

        let d_mat = Arc::clone(&handle.d_mat);
        let clusters = Arc::clone(&handle.clusters);
        let index = Arc::clone(&handle.index);

        match &handle.inner {
            HandleInner::Cpu => {
                let hits = tokio::task::spawn_blocking(move || {
                    // 1. Per-query encode (one q_hat per query).
                    let q_hats: Vec<Vec<u64>> = queries_f64
                        .iter()
                        .zip(&us)
                        .map(|(qf, u)| crypto::encode_query(&d_mat, qf, u))
                        .collect();

                    // 2. Invert probe sets: per cluster → list of
                    //    (q_idx, position_in_probe_set) pairs.
                    let mut per_cluster_qpos: Vec<Vec<(usize, usize)>> =
                        vec![Vec::new(); clusters.len()];
                    for (q_idx, ps) in probe_sets.iter().enumerate() {
                        for (pos, &ci) in ps.iter().enumerate() {
                            per_cluster_qpos[ci].push((q_idx, pos));
                        }
                    }

                    // 3. Per cluster (parallel): one batched matvec,
                    //    then per-query decode. Output per cluster is
                    //    a list of (q_idx, pos, Vec<(membership_id, f32)>).
                    type Bin = (usize, usize, Vec<(u32, f32)>);
                    let cluster_contribs: Vec<Vec<Bin>> = per_cluster_qpos
                        .par_iter()
                        .enumerate()
                        .map(|(ci, qpos_list)| {
                            if qpos_list.is_empty() {
                                return Vec::new();
                            }
                            let cluster = &clusters[ci];
                            let m = cluster.m;
                            if m == 0 {
                                return qpos_list
                                    .iter()
                                    .map(|&(q_idx, pos)| (q_idx, pos, Vec::new()))
                                    .collect();
                            }
                            let qhat_refs: Vec<&[u64]> = qpos_list
                                .iter()
                                .map(|&(q_idx, _)| q_hats[q_idx].as_slice())
                                .collect();
                            let partials_batch = crypto::compute_products_batch(
                                cluster.m_hat.as_slice(),
                                m,
                                &qhat_refs,
                            );
                            let membership = &index.clusters[ci];
                            qpos_list
                                .iter()
                                .zip(partials_batch.iter())
                                .map(|(&(q_idx, pos), partials)| {
                                    let scores = crypto::decode_scores(
                                        m,
                                        cluster.r_seed,
                                        &q_hats[q_idx],
                                        partials,
                                    );
                                    let pairs: Vec<(u32, f32)> = scores
                                        .into_iter()
                                        .enumerate()
                                        .map(|(local_i, score)| (membership[local_i].0, score))
                                        .collect();
                                    (q_idx, pos, pairs)
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect();

                    // 4. Scatter into per-query bins indexed by probe
                    //    position so the gather preserves the per-query
                    //    path's `probe_set.par_iter().flat_map()` order.
                    let mut bins_per_query: Vec<Vec<Vec<(u32, f32)>>> = probe_sets
                        .iter()
                        .map(|ps| vec![Vec::new(); ps.len()])
                        .collect();
                    for cluster_list in cluster_contribs {
                        for (q_idx, pos, pairs) in cluster_list {
                            bins_per_query[q_idx][pos] = pairs;
                        }
                    }

                    // 5. Per query: flatten in probe-position order,
                    //    sort desc by score, truncate to k.
                    bins_per_query
                        .into_iter()
                        .map(|bins| {
                            let mut all: Vec<(u32, f32)> = bins.into_iter().flatten().collect();
                            all.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                            all.truncate(k);
                            all.into_iter()
                                .map(|(id, score)| Hit { id, score })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .await
                .map_err(|_| EmvpIvfError::SpawnPanic)?;

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

// ---------------------------------------------------------------------------
// Per-query breakdown
// ---------------------------------------------------------------------------

/// Per-query substep timings reported by [`EmvpIvfScorer::score_with_breakdown`].
/// `route` = client-side centroid scan; `encode` = D-mask + u-blind on
/// the padded f64 query; `server` = aggregate `compute_products` over
/// the probe set; `decode` = aggregate `decode_scores` over the probe
/// set; `merge` = top-k sort across all clusters.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmvpIvfTiming {
    pub route_us: u64,
    pub encode_us: u64,
    pub server_us: u64,
    pub decode_us: u64,
    pub merge_us: u64,
}

impl EmvpIvfScorer {
    /// Off-hot-path equivalent of [`Scorer::score`] with per-substep
    /// timings. The breakdown path collects all per-cluster
    /// `compute_products` results before decoding so the
    /// server and decode phases can be timed separately; this incurs a
    /// transient memory peak proportional to `nprobe × m_max × s` that
    /// `score` itself does not pay (the throughput path interleaves
    /// compute and decode inside the parallel iterator).
    pub async fn score_with_breakdown(
        &self,
        handle: &EmvpIvfHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, EmvpIvfTiming), EmvpIvfError> {
        if query.0.len() != handle.dim {
            return Err(EmvpIvfError::DimensionMismatch {
                query_dim: query.0.len(),
                index_dim: handle.dim,
            });
        }

        let params = &SEC128;
        let ell0 = params.ell0;
        let k_dim = params.k;
        let nprobe = handle.nprobe.min(handle.index.centroids.len());

        let query_full = query.0.clone();
        let d_mat = Arc::clone(&handle.d_mat);
        let clusters = Arc::clone(&handle.clusters);
        let index = Arc::clone(&handle.index);

        // u is sampled here so ThreadRng doesn't cross spawn_blocking.
        let u: Vec<u64> = {
            let mut rng = rand::rng();
            (0..k_dim)
                .map(|_| rng.random::<u64>() % crate::params::P)
                .collect()
        };

        // Dispatch on inner. CPU branch runs the five-substep timer
        // set inside one spawn_blocking (the closure can hold non-Send
        // `cluster.m_hat` Arc clones since the rayon iterator runs
        // inside the blocking task). GPU branch breaks the substeps
        // across the host / device boundary so server_us spans each
        // kernel launch's full round-trip; decode and merge stay
        // CPU-side.
        let result = match &handle.inner {
            HandleInner::Cpu => tokio::task::spawn_blocking(move || {
                let t = Instant::now();
                let mut centroid_dists: Vec<(usize, f32)> = index
                    .centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let d: f32 = c
                            .iter()
                            .zip(&query_full)
                            .map(|(a, b)| {
                                let d = *a - *b;
                                d * d
                            })
                            .sum();
                        (i, d)
                    })
                    .collect();
                centroid_dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                let probe_set: Vec<usize> = centroid_dists
                    .iter()
                    .take(nprobe)
                    .map(|(i, _)| *i)
                    .collect();
                let route_us = t.elapsed().as_micros() as u64;

                let t = Instant::now();
                let mut q_f64: Vec<f64> = query_full.iter().map(|&x| x as f64).collect();
                q_f64.resize(ell0, 0.0);
                let q_hat = crypto::encode_query(&d_mat, &q_f64, &u);
                let encode_us = t.elapsed().as_micros() as u64;

                let t = Instant::now();
                let partials_per_cluster: Vec<(usize, Vec<u64>)> = probe_set
                    .par_iter()
                    .filter_map(|&ci| {
                        let cluster = &clusters[ci];
                        if cluster.m == 0 {
                            return None;
                        }
                        let partials =
                            crypto::compute_products(cluster.m_hat.as_slice(), cluster.m, &q_hat);
                        Some((ci, partials))
                    })
                    .collect();
                let server_us = t.elapsed().as_micros() as u64;

                let t = Instant::now();
                let mut all_hits: Vec<(u32, f32)> = partials_per_cluster
                    .par_iter()
                    .flat_map(|(ci, partials)| {
                        let cluster = &clusters[*ci];
                        let scores =
                            crypto::decode_scores(cluster.m, cluster.r_seed, &q_hat, partials);
                        let membership = &index.clusters[*ci];
                        scores
                            .into_iter()
                            .enumerate()
                            .map(|(local_i, score)| (membership[local_i].0, score))
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let decode_us = t.elapsed().as_micros() as u64;

                let t = Instant::now();
                all_hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                all_hits.truncate(k);
                let hits: Vec<Hit> = all_hits
                    .into_iter()
                    .map(|(id, score)| Hit { id, score })
                    .collect();
                let merge_us = t.elapsed().as_micros() as u64;

                (
                    hits,
                    EmvpIvfTiming {
                        route_us,
                        encode_us,
                        server_us,
                        decode_us,
                        merge_us,
                    },
                )
            })
            .await
            .map_err(|_| EmvpIvfError::SpawnPanic)?,
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                let route_t = Instant::now();
                let mut centroid_dists: Vec<(usize, f32)> = index
                    .centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let d: f32 = c
                            .iter()
                            .zip(&query_full)
                            .map(|(a, b)| {
                                let d = *a - *b;
                                d * d
                            })
                            .sum();
                        (i, d)
                    })
                    .collect();
                centroid_dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                let probe_set: Vec<usize> = centroid_dists
                    .iter()
                    .take(nprobe)
                    .map(|(i, _)| *i)
                    .collect();
                let route_us = route_t.elapsed().as_micros() as u64;

                let encode_t = Instant::now();
                let mut q_f64: Vec<f64> = query_full.iter().map(|&x| x as f64).collect();
                q_f64.resize(ell0, 0.0);
                let q_hat = crypto::encode_query(&d_mat, &q_f64, &u);
                let encode_us = encode_t.elapsed().as_micros() as u64;

                // Sequential per-cluster GPU launches (the cuda
                // stream serialises them anyway). server_us sums the
                // per-launch round-trips.
                let mut server_us: u64 = 0;
                let mut partials_per_cluster: Vec<(usize, Vec<u64>)> =
                    Vec::with_capacity(probe_set.len());
                for &ci in &probe_set {
                    let t = Instant::now();
                    if let Some(partials) = state.compute_products(ci, &q_hat)? {
                        server_us = server_us.saturating_add(t.elapsed().as_micros() as u64);
                        partials_per_cluster.push((ci, partials));
                    }
                }

                tokio::task::spawn_blocking(move || -> (Vec<Hit>, u64, u64) {
                    let t = Instant::now();
                    let mut all_hits: Vec<(u32, f32)> = partials_per_cluster
                        .par_iter()
                        .flat_map(|(ci, partials)| {
                            let cluster = &clusters[*ci];
                            let scores =
                                crypto::decode_scores(cluster.m, cluster.r_seed, &q_hat, partials);
                            let membership = &index.clusters[*ci];
                            scores
                                .into_iter()
                                .enumerate()
                                .map(|(local_i, score)| (membership[local_i].0, score))
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    let decode_us = t.elapsed().as_micros() as u64;

                    let t = Instant::now();
                    all_hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    all_hits.truncate(k);
                    let hits: Vec<Hit> = all_hits
                        .into_iter()
                        .map(|(id, score)| Hit { id, score })
                        .collect();
                    let merge_us = t.elapsed().as_micros() as u64;

                    (hits, decode_us, merge_us)
                })
                .await
                .map(|(hits, decode_us, merge_us)| {
                    (
                        hits,
                        EmvpIvfTiming {
                            route_us,
                            encode_us,
                            server_us,
                            decode_us,
                            merge_us,
                        },
                    )
                })
                .map_err(|_| EmvpIvfError::SpawnPanic)?
            }
        };

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Cache I/O
// ---------------------------------------------------------------------------
//
// Layout:
//   <cache header>
//   <IVF index payload>            (ivf::write_index_to / read_index_from)
//   <u64 n_clusters>
//     for each cluster: <u64 m> <32 bytes r_seed> <m * SEC128.n u64s of m̂>
//   <SEC128.ell0 * SEC128.n u64s of D>

/// Single-shot cache writer — kept as a format reference for the
/// inline streaming write in `upload_cluster`. The production path no
/// longer calls this directly because the incremental strip in the
/// encryption loop requires interleaving the write with the build.
#[allow(dead_code)]
fn save_emvp_ivf_cache(
    path: &std::path::Path,
    index: &IvfIndex,
    clusters: &[EncryptedCluster],
    d_mat: &[u64],
    parts: &[&[u8]],
) -> io::Result<()> {
    let tmp = path.with_extension("bin.tmp");
    let mut w = BufWriter::new(std::fs::File::create(&tmp)?);
    cache::write_header(&mut w, parts)?;
    ivf::write_index_to(&mut w, index)?;
    // The loader pads to an 8-byte absolute offset before reading
    // `n_clusters` so each cluster's m_hat region lands on an 8-byte
    // boundary for `Mhat::Mmap::as_slice`. The 20-byte cache header
    // plus a variable-length IVF payload can leave the writer's
    // position at any byte alignment; emit the matching zero pad
    // here so the on-disk layout actually matches what the loader
    // expects. Without this the loader reads garbage as `n_clusters`
    // and OOM-aborts on a huge `Vec::with_capacity`.
    let pos = w.stream_position()? as usize;
    let pad = (8 - (pos % 8)) % 8;
    w.write_all(&[0u8; 8][..pad])?;
    debug_assert_eq!(
        w.stream_position()? as usize % 8,
        0,
        "`n_clusters` u64 must land at an 8-byte absolute offset",
    );
    w.write_all(&(clusters.len() as u64).to_le_bytes())?;
    for c in clusters {
        w.write_all(&(c.m as u64).to_le_bytes())?;
        w.write_all(&c.r_seed)?;
        for &v in c.m_hat.as_slice().iter() {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    for &v in d_mat {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    std::fs::rename(&tmp, path)
}

type LoadedIvfCache = (IvfIndex, Vec<EncryptedCluster>, Vec<u64>);

fn load_emvp_ivf_cache(
    path: &std::path::Path,
    parts: &[&[u8]],
) -> io::Result<Option<LoadedIvfCache>> {
    // Cache layout (mmap-backed):
    //   [20  cache header]
    //   [var IVF index payload (sequential read)]
    //   [pad 0..7 zero bytes to next 8-byte boundary]
    //   [8   u64 n_clusters]
    //   [per-cluster: 8 m + 32 r_seed + (m·n·8) m_hat]
    //   [d_mat_len·8 d_mat u64s]
    //
    // m_hat regions are 8-byte aligned by construction → mmap'd
    // and exposed as `&[u64]` via `Mhat::Mmap::as_slice()`.
    //
    // Older caches with no alignment padding accumulated m_hat in the
    // file with no alignment guarantee; loading one here typically
    // results in malformed-cluster data and the caller's
    // `.ok().flatten()` will rebuild the cache from scratch. That's
    // the intended migration path.
    let file = std::fs::File::open(path)?;
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file)? });

    // Verify header via a transient Cursor.
    {
        let mut cur = std::io::Cursor::new(&mmap[..]);
        if !cache::verify_header(&mut cur, parts)? {
            return Ok(None);
        }
    }
    let mut offset: usize = 20;

    // Sequential IVF index read.
    let index = {
        let mut cur = std::io::Cursor::new(&mmap[offset..]);
        let idx = ivf::read_index_from(&mut cur)?;
        offset += cur.position() as usize;
        idx
    };

    // Pad to 8-byte alignment.
    offset += (8 - (offset % 8)) % 8;

    if mmap.len() < offset + 8 {
        return Ok(None);
    }
    let n_clusters = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;
    // Cross-check against the IVF index we just parsed: the writer
    // emits `clusters.len() as u64` here, and that count is the
    // same as `index.clusters.len()` by construction (one
    // EncryptedCluster per IVF cluster). A mismatch means we read
    // 8 bytes from the wrong offset — typically a pre-fix cache
    // whose writer didn't emit the 8-byte alignment pad — and the
    // read value can be arbitrarily large. Return Ok(None) so the
    // caller rebuilds the cache instead of aborting on a huge
    // `Vec::with_capacity(n_clusters)` allocation.
    if n_clusters != index.clusters.len() {
        return Ok(None);
    }
    let n = SEC128.n;

    let mut clusters = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        if mmap.len() < offset + 8 + 32 {
            return Ok(None);
        }
        let m = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        let mut r_seed = [0u8; 32];
        r_seed.copy_from_slice(&mmap[offset..offset + 32]);
        offset += 32;
        let len_u64s = m * n;
        let m_hat_byte_offset = offset;
        if mmap.len() < offset + len_u64s * 8 {
            return Ok(None);
        }
        offset += len_u64s * 8;
        clusters.push(EncryptedCluster {
            m_hat: Mhat::Mmap {
                mmap: mmap.clone(),
                byte_offset: m_hat_byte_offset,
                len_u64s,
            },
            r_seed,
            m,
        });
    }

    let d_len = SEC128.ell0 * n;
    if mmap.len() < offset + d_len * 8 {
        return Ok(None);
    }
    let mut d_mat = vec![0u64; d_len];
    for (i, v) in d_mat.iter_mut().enumerate() {
        *v = u64::from_le_bytes(mmap[offset + i * 8..offset + i * 8 + 8].try_into().unwrap());
    }

    Ok(Some((index, clusters, d_mat)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn random_vectors(n: usize, d: usize, seed: u64) -> Vec<Vector> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..d)
                    .map(|_| rng.random_range(-1.0_f32..1.0_f32))
                    .collect();
                let norm = v
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(f32::EPSILON);
                Vector(v.into_iter().map(|x| x / norm).collect())
            })
            .collect()
    }

    fn cfg(n_centroids: usize, nprobe: usize) -> EmvpIvfConfig {
        EmvpIvfConfig {
            key_seed: [42u8; 32],
            n_centroids,
            nprobe,
            train_seed: 42,
            max_iter: 25,
            upload_seed: 7,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        }
    }

    #[tokio::test]
    async fn upload_and_score_roundtrip() {
        let scorer = EmvpIvfScorer::new();
        let vectors = random_vectors(20, 32, 1);
        let (handle, _build) = scorer.upload_cluster(&cfg(4, 2), &vectors).await.unwrap();
        let query = random_vectors(1, 32, 99).remove(0);
        let hits = scorer.score(&handle, &query, 5).await.unwrap();
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|h| h.id < 20));
        assert!(hits.iter().all(|h| h.score.is_finite()));
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score, "scores not descending: {hits:?}");
        }
    }

    /// At n_centroids=1 the IVF degenerates to a single cluster, so EMVP+IVF must
    /// produce bit-identical scores and IDs to flat EMVP. Catches off-by-one bugs
    /// in cluster-membership-table construction without needing the merge logic.
    #[tokio::test]
    async fn single_cluster_matches_flat() {
        use crate::{EmvpConfig, EmvpScorer};
        let vectors = random_vectors(10, 32, 9);
        let query = random_vectors(1, 32, 999).remove(0);

        let flat = EmvpScorer::new();
        let (flat_h, _build) = flat
            .upload_cluster(
                &EmvpConfig {
                    key_seed: [42u8; 32],
                    progress: None,
                    device: scorer_core::Device::Cpu,
                    vram_budget_bytes: None,
                },
                &vectors,
            )
            .await
            .unwrap();
        let flat_hits = flat.score(&flat_h, &query, 5).await.unwrap();

        let ivf = EmvpIvfScorer::new();
        let (ivf_h, _build) = ivf.upload_cluster(&cfg(1, 1), &vectors).await.unwrap();
        let ivf_hits = ivf.score(&ivf_h, &query, 5).await.unwrap();

        // Note: scores depend on the random query-encoding `u`, which is
        // generated independently per call. Compare IDs only — top-k IDs
        // must match because the underlying encrypted matrix is identical
        // (same key_seed, same r_seed: derive_r_seed(7, 0) at n_centroids=1
        // — note the flat scorer uses a different r_seed, but at full probe
        // both compute the same M̂·q̂ scoring relation so ID order matches).
        let flat_ids: Vec<u32> = flat_hits.iter().map(|h| h.id).collect();
        let ivf_ids: Vec<u32> = ivf_hits.iter().map(|h| h.id).collect();
        assert_eq!(
            flat_ids, ivf_ids,
            "single-cluster IVF must match flat: flat={flat_hits:?}, ivf={ivf_hits:?}"
        );
    }

    /// At nprobe = n_centroids, the union of probed clusters covers all corpus
    /// vectors, so the top-k must equal the brute-force nearest-neighbour result
    /// (also what flat EMVP returns).
    #[tokio::test]
    async fn full_probe_matches_brute_force() {
        let vectors = random_vectors(40, 64, 11);
        let scorer = EmvpIvfScorer::new();
        let n_centroids = 5;
        let (handle, _build) = scorer
            .upload_cluster(&cfg(n_centroids, n_centroids), &vectors)
            .await
            .unwrap();

        for qi in 0..3u64 {
            let query = random_vectors(1, 64, 1000 + qi).remove(0);

            // Brute-force ground truth by inner product (EMVP recovers IPs).
            let gt_id = vectors
                .iter()
                .enumerate()
                .map(|(id, v)| {
                    let ip: f32 = v.0.iter().zip(&query.0).map(|(a, b)| a * b).sum();
                    (id, ip)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0 as u32;

            let hits = scorer.score(&handle, &query, 1).await.unwrap();
            assert_eq!(
                hits[0].id, gt_id,
                "query {qi}: full-probe EMVP+IVF returned id={} but ground truth is id={gt_id}",
                hits[0].id
            );
        }
    }

    #[tokio::test]
    async fn k_capped_at_total_corpus_size() {
        let scorer = EmvpIvfScorer::new();
        let vectors = random_vectors(7, 16, 2);
        let (handle, _build) = scorer.upload_cluster(&cfg(2, 2), &vectors).await.unwrap();
        let query = random_vectors(1, 16, 100).remove(0);
        let hits = scorer.score(&handle, &query, 100).await.unwrap();
        assert_eq!(hits.len(), 7);
    }

    #[tokio::test]
    async fn dimension_mismatch_error() {
        let scorer = EmvpIvfScorer::new();
        let vectors = random_vectors(10, 16, 3);
        let (handle, _build) = scorer.upload_cluster(&cfg(2, 2), &vectors).await.unwrap();
        let wrong = random_vectors(1, 32, 200).remove(0);
        let err = scorer.score(&handle, &wrong, 3).await.unwrap_err();
        assert!(matches!(err, EmvpIvfError::DimensionMismatch { .. }));
    }

    #[tokio::test]
    async fn too_many_centroids_error() {
        let scorer = EmvpIvfScorer::new();
        let vectors = random_vectors(3, 16, 4);
        let err = scorer
            .upload_cluster(&cfg(10, 1), &vectors)
            .await
            .unwrap_err();
        assert!(matches!(err, EmvpIvfError::TooManyCentroids(_)));
    }

    #[tokio::test]
    async fn invalid_nprobe_error() {
        let scorer = EmvpIvfScorer::new();
        let vectors = random_vectors(20, 16, 5);
        let err = scorer
            .upload_cluster(&cfg(4, 0), &vectors)
            .await
            .unwrap_err();
        assert!(matches!(err, EmvpIvfError::InvalidNprobe { .. }));
        let err = scorer
            .upload_cluster(&cfg(4, 5), &vectors)
            .await
            .unwrap_err();
        assert!(matches!(err, EmvpIvfError::InvalidNprobe { .. }));
    }

    #[tokio::test]
    async fn empty_corpus_error() {
        let scorer = EmvpIvfScorer::new();
        let err = scorer.upload_cluster(&cfg(1, 1), &[]).await.unwrap_err();
        assert!(matches!(err, EmvpIvfError::EmptyCorpus));
    }

    #[tokio::test]
    async fn dimension_too_large_error() {
        let scorer = EmvpIvfScorer::new();
        // ell0 = 1024; corpus dim 1500 > 1024.
        let big = random_vectors(5, 1500, 6);
        let err = scorer.upload_cluster(&cfg(2, 1), &big).await.unwrap_err();
        assert!(matches!(err, EmvpIvfError::DimensionTooLarge { .. }));
    }

    /// 1D corpus designed to force uneven cluster sizes [60, 20, 15, 5]
    /// after k-means converges. Used by the realised-cost tests; same
    /// shape as scorer-plaintext / scorer-sap so the three crates'
    /// realised-cost tests follow the same template.
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
        let scorer = EmvpIvfScorer::new();
        let vectors = skewed_corpus_4_clusters();
        let (handle, _) = scorer.upload_cluster(&cfg(4, 2), &vectors).await.unwrap();

        let (_, cost_a) = scorer
            .score_with_realised_cost(&handle, &Vector(vec![0.0]), 10)
            .await
            .unwrap();
        let (_, cost_b) = scorer
            .score_with_realised_cost(&handle, &Vector(vec![30.0]), 10)
            .await
            .unwrap();

        let params = &SEC128;
        let bytes_per_m = (params.s * FIELD_BYTES) as u64;
        // Query near 0.0: probe set covers the 60- and 20-clusters
        // → realised m_total = 80, response = 80 × s × FIELD_BYTES.
        // Query near 30.0: 5- and 15-clusters → realised m_total = 20.
        assert_eq!(cost_a.response_bytes, 80 * bytes_per_m);
        assert_eq!(cost_b.response_bytes, 20 * bytes_per_m);
        assert_ne!(cost_a.response_bytes, cost_b.response_bytes);
    }

    #[tokio::test]
    async fn realised_cost_full_probe_recovers_total() {
        let scorer = EmvpIvfScorer::new();
        let vectors = skewed_corpus_4_clusters();
        let (handle, _) = scorer.upload_cluster(&cfg(4, 4), &vectors).await.unwrap();

        let (_, cost) = scorer
            .score_with_realised_cost(&handle, &Vector(vec![15.0]), 10)
            .await
            .unwrap();

        let params = &SEC128;
        let bytes_per_m = (params.s * FIELD_BYTES) as u64;
        // nprobe = 4 → every cluster probed; realised m_total = 100.
        assert_eq!(cost.response_bytes, 100 * bytes_per_m);
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

        let scorer = EmvpIvfScorer::new();
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
    async fn communication_cost_formula() {
        let scorer = EmvpIvfScorer::new();
        let vectors = random_vectors(20, 32, 7);
        let n_centroids = 4;
        let nprobe = 2;
        let (handle, _build) = scorer
            .upload_cluster(&cfg(n_centroids, nprobe), &vectors)
            .await
            .unwrap();
        let cost = scorer.communication_cost(&handle, 5);
        let params = &SEC128;
        assert_eq!(cost.query_bytes, (params.n * FIELD_BYTES) as u64);

        let total_m: usize = handle.clusters.iter().map(|c| c.m).sum();
        assert_eq!(total_m, 20, "all corpus vectors should be assigned");

        let mean_m = total_m as f64 / n_centroids as f64;
        let expected_cluster_resp = (mean_m * params.s as f64 * FIELD_BYTES as f64).round() as u64;
        assert_eq!(cost.cluster_response_bytes, expected_cluster_resp);
        assert_eq!(cost.response_bytes, nprobe as u64 * expected_cluster_resp);
        assert_eq!(cost.setup_bytes, (total_m * params.n * FIELD_BYTES) as u64);
        assert_eq!(cost.pre_query_offline_up_bytes, 0);
        assert_eq!(cost.pre_query_offline_down_bytes, 0);
        // GPU proxy: matrix work nprobe × m_avg × n × FIELD_BYTES
        // (the m × n encrypted matrix the server matvec actually reads,
        // not the s × m response).
        let expected_eff_bytes =
            (nprobe as u64) * (mean_m.round() as u64) * (params.n as u64) * (FIELD_BYTES as u64);
        assert_eq!(cost.effective_bytes_per_query, expected_eff_bytes);
    }

    #[tokio::test]
    async fn debug_redacts_secrets() {
        let scorer = EmvpIvfScorer::new();
        let vectors = random_vectors(10, 16, 8);
        let (handle, _build) = scorer.upload_cluster(&cfg(2, 2), &vectors).await.unwrap();
        let dbg = format!("{handle:?}");
        // The r_seed bytes from each cluster must not surface; check that the
        // tag we redact under appears and that no contiguous 32-byte hex/decimal
        // r_seed bleed-through exists.
        assert!(
            dbg.contains("[redacted]"),
            "Debug should mark r_seeds as redacted: {dbg}"
        );
        for cluster in handle.clusters.iter() {
            for &b in cluster.r_seed.iter() {
                let needle = format!("{b}");
                // r_seed bytes are typically 0..255; this is loose but a defence
                // against accidentally derived-Debug-leaking the array.
                let occurrences = dbg.matches(&needle).count();
                assert!(
                    occurrences < 30,
                    "Debug appears to leak r_seed bytes: {dbg}"
                );
                let _ = needle;
            }
        }
    }

    #[tokio::test]
    async fn disk_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let vectors = random_vectors(15, 16, 9);
        let query = random_vectors(1, 16, 400).remove(0);

        let scorer = EmvpIvfScorer::with_cache_dir(dir.path());
        let (h1, b1) = scorer.upload_cluster(&cfg(3, 2), &vectors).await.unwrap();
        assert!(!b1.cache_hit, "first build is cold");
        let h1_hits = scorer.score(&h1, &query, 4).await.unwrap();

        // New scorer (cleared in-memory cache) using the same cache dir → should load from disk.
        let scorer2 = EmvpIvfScorer::with_cache_dir(dir.path());
        let (h2, b2) = scorer2.upload_cluster(&cfg(3, 2), &vectors).await.unwrap();
        assert!(b2.cache_hit, "second build hits the on-disk cache");
        let h2_hits = scorer2.score(&h2, &query, 4).await.unwrap();

        let ids1: Vec<u32> = h1_hits.iter().map(|h| h.id).collect();
        let ids2: Vec<u32> = h2_hits.iter().map(|h| h.id).collect();
        assert_eq!(ids1, ids2, "disk-cache reload must produce identical top-k");
    }

    /// At full probe (nprobe = n_centroids) EMVP+IVF recovers the
    /// brute-force inner-product structurally, regardless of the
    /// per-call random `u` mask, so `score` and `score_with_breakdown`
    /// agree on the top-k IDs.
    #[tokio::test]
    async fn score_with_breakdown_matches_score() {
        let scorer = EmvpIvfScorer::new();
        let vectors = random_vectors(40, 64, 30);
        let n_centroids = 5;
        let (handle, _build) = scorer
            .upload_cluster(&cfg(n_centroids, n_centroids), &vectors)
            .await
            .unwrap();
        let query = random_vectors(1, 64, 31).remove(0);

        let baseline = scorer.score(&handle, &query, 5).await.unwrap();
        let (with_breakdown, timing) = scorer
            .score_with_breakdown(&handle, &query, 5)
            .await
            .unwrap();
        let baseline_ids: Vec<u32> = baseline.iter().map(|h| h.id).collect();
        let breakdown_ids: Vec<u32> = with_breakdown.iter().map(|h| h.id).collect();
        assert_eq!(baseline_ids, breakdown_ids);
        let _ = timing.route_us
            + timing.encode_us
            + timing.server_us
            + timing.decode_us
            + timing.merge_us;
    }

    /// Per-query result from the batched override is bit-equal to a
    /// sequential `score()` call on the same input. Same noise-cancels
    /// argument as the flat EMVP test: random `u` differs per call, but
    /// `decode_scores` recovers the same field element (plaintext
    /// distance) regardless, and `decode_field` is deterministic. The
    /// override preserves per-query bin order (probe-set position) so
    /// `sort_unstable_by` sees a bit-equal input. Integer-field tier →
    /// bit-equal assertion.
    #[tokio::test]
    async fn emvp_ivf_score_batch_matches_score() {
        let scorer = EmvpIvfScorer::new();
        let vectors = random_vectors(128, 64, 57);
        let (handle, _build) = scorer.upload_cluster(&cfg(8, 3), &vectors).await.unwrap();

        let queries = random_vectors(64, 64, 58);
        let k = 10;

        let mut from_score: Vec<Vec<Hit>> = Vec::with_capacity(queries.len());
        for q in &queries {
            from_score.push(scorer.score(&handle, q, k).await.unwrap());
        }

        for &b in &[1usize, 8, 64] {
            let chunk = &queries[..b];
            let from_batch = scorer.score_batch(&handle, chunk, k).await.unwrap();
            assert_eq!(
                from_score[..b],
                from_batch[..],
                "B={b}: batched output diverges from sequential score()"
            );
        }
    }

    #[tokio::test]
    async fn build_outcome_cache_hit_smoke() {
        let scorer = EmvpIvfScorer::new();
        let vectors = random_vectors(20, 16, 50);
        let (_h1, b1) = scorer.upload_cluster(&cfg(3, 2), &vectors).await.unwrap();
        assert!(!b1.cache_hit, "first build must be a cold build");
        let (_h2, b2) = scorer.upload_cluster(&cfg(3, 2), &vectors).await.unwrap();
        assert!(b2.cache_hit, "second build must hit the in-memory cache");
        assert!(
            b2.build_duration < std::time::Duration::from_millis(100),
            "warm build was {:?}, expected < 100 ms",
            b2.build_duration
        );
    }

    /// EMVP+IVF GPU exact-equality gate at full probe.
    ///
    /// Mersenne arithmetic is deterministic, so the GPU path must
    /// return the same top-k IDs as the CPU path on the same query at
    /// `nprobe = n_centroids`. The blinding mask `u` is freshly
    /// sampled per call but the inner-product structure is recovered
    /// exactly, so ID equality holds even though the raw partials
    /// differ between the two upload-cluster calls. Requires a working
    /// CUDA toolchain and GPU.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_full_probe_matches_cpu_exact() {
        let vectors = random_vectors(60, 32, 909);
        let query = random_vectors(1, 32, 910).remove(0);
        let k = 10;
        let n_centroids = 4;

        let cfg_cpu = EmvpIvfConfig {
            key_seed: [42u8; 32],
            n_centroids,
            nprobe: n_centroids,
            train_seed: 42,
            max_iter: 25,
            upload_seed: 7,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let cfg_gpu = EmvpIvfConfig {
            device: Device::Gpu,
            ..cfg_cpu_clone(&cfg_cpu)
        };

        let scorer_cpu = EmvpIvfScorer::new();
        let scorer_gpu = EmvpIvfScorer::new();
        let (h_cpu, _) = scorer_cpu.upload_cluster(&cfg_cpu, &vectors).await.unwrap();
        let (h_gpu, _) = scorer_gpu.upload_cluster(&cfg_gpu, &vectors).await.unwrap();

        let hits_cpu = scorer_cpu.score(&h_cpu, &query, k).await.unwrap();
        let hits_gpu = scorer_gpu.score(&h_gpu, &query, k).await.unwrap();
        assert_eq!(hits_cpu.len(), k);
        assert_eq!(hits_gpu.len(), k);
        let ids_cpu: std::collections::HashSet<u32> = hits_cpu.iter().map(|h| h.id).collect();
        let ids_gpu: std::collections::HashSet<u32> = hits_gpu.iter().map(|h| h.id).collect();
        assert_eq!(
            ids_cpu, ids_gpu,
            "GPU full-probe top-{k} ids must match CPU exactly under Mersenne arithmetic"
        );
    }

    /// Structural invariant for partial probe: every returned hit
    /// comes from one of the probed clusters' members.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_partial_probe_returns_subset_of_probed_clusters() {
        let vectors = random_vectors(60, 32, 911);
        let query = random_vectors(1, 32, 912).remove(0);
        let k = 10;
        let n_centroids = 4;
        let nprobe = 2;

        let cfg = EmvpIvfConfig {
            key_seed: [42u8; 32],
            n_centroids,
            nprobe,
            train_seed: 42,
            max_iter: 25,
            upload_seed: 7,
            progress: None,
            device: Device::Gpu,
            vram_budget_bytes: None,
        };

        let scorer = EmvpIvfScorer::new();
        let (handle, _) = scorer.upload_cluster(&cfg, &vectors).await.unwrap();
        let hits = scorer.score(&handle, &query, k).await.unwrap();

        let probe_set = ivf::probe_route(&handle.index, &query.0, nprobe);
        let allowed: std::collections::HashSet<u32> = probe_set
            .iter()
            .flat_map(|&ci| handle.index.clusters[ci].iter().map(|(id, _)| *id))
            .collect();
        for h in &hits {
            assert!(
                allowed.contains(&h.id),
                "GPU returned hit id={} not in any probed cluster (probe_set={probe_set:?})",
                h.id
            );
        }
        assert!(hits.len() <= k);
    }

    #[cfg(feature = "gpu")]
    fn cfg_cpu_clone(c: &EmvpIvfConfig) -> EmvpIvfConfig {
        EmvpIvfConfig {
            key_seed: c.key_seed,
            n_centroids: c.n_centroids,
            nprobe: c.nprobe,
            train_seed: c.train_seed,
            max_iter: c.max_iter,
            upload_seed: c.upload_seed,
            progress: None,
            device: c.device,
            vram_budget_bytes: c.vram_budget_bytes,
        }
    }

    /// EMVP+IVF GPU breakdown agrees with score(). Validates that
    /// `score_with_breakdown` accepts GPU; canonical substeps (route +
    /// encode + server + decode + merge) all land non-zero on a real
    /// GPU run with non-trivial nprobe.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_score_with_breakdown_matches_score() {
        let vectors = random_vectors(60, 32, 920);
        let query = random_vectors(1, 32, 921).remove(0);
        let k = 5;

        let mut config = cfg(4, 2);
        config.device = Device::Gpu;

        let scorer = EmvpIvfScorer::new();
        let (handle, _) = scorer.upload_cluster(&config, &vectors).await.unwrap();

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
            timing.server_us > 0,
            "server_us should cover GPU launches: {timing:?}"
        );
        assert!(
            timing.decode_us > 0,
            "decode_us should cover host decode: {timing:?}"
        );
    }
}
