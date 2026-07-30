//! Braverman–Newman + IVF scorer.
//!
//! Per-cluster BN encryption with shared `L` (the LPN subspace) and
//! per-cluster `(H, S)` seeds. Plaintext-side routing on cleartext
//! centroids — like EMVP+IVF, the BN protocol is not
//! distance-preserving, so the server cannot route. The server learns
//! the probe set; same access-pattern leakage as SAP+IVF / EMVP+IVF.
//!
//! On-disk cache stores `m_enc` and `a_l` per cluster but NOT
//! `m_plain`: the IVF index payload already carries the plaintext
//! cluster vectors, so `m_plain` is re-derived from those at load
//! time (a single quantisation pass, O(m·d) per cluster).
//!
//! `progress.on_encrypt(i)` granularity: per-cluster (called once
//! after each cluster's `M_enc + AL` is built).

use std::collections::HashMap;
use std::fmt;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use ivf_index::{
    ivf::{self, IvfIndex},
    kmeans,
};
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use scorer_core::{
    BuildOutcome, CommunicationCost, Device, Hit, ProgressReporter, Scorer, Vector, cache,
};

use crate::crypto::{
    EncryptedCluster, SparseMatrix, Storage, compute_products, compute_products_batch,
    decode_scores, domain_separate, encode_query, encrypt_cluster, fp_to_signed, generate_subspace,
    quantise_vector, regenerate_h_s, verify_response,
};
#[cfg(feature = "gpu")]
use crate::crypto::{decode_scores_with_gpu_dense, l_transpose_times_vec};

/// One probed cluster's GPU decode bundle, accumulated per query
/// before the final compose-and-rank `spawn_blocking`:
/// `(cluster_id, server_response, dense_gpu, cached_s_mat)`.
#[cfg(feature = "gpu")]
type GpuClusterDecode = (usize, Vec<u64>, Vec<u64>, Arc<SparseMatrix>);
#[cfg(feature = "gpu")]
use crate::gpu;
use crate::params::{BnTmParams, FIELD_BYTES};

/// Configuration for `BnTmIvfScorer::upload_cluster`.
pub struct BnTmIvfConfig {
    pub params: BnTmParams,
    /// Seed for the LPN subspace `L` — shared across all clusters in
    /// this index (security: clusters share the LPN subspace, not the
    /// noise).
    pub key_seed: [u8; 32],
    pub n_centroids: usize,
    pub nprobe: usize,
    pub train_seed: u64,
    pub max_iter: usize,
    /// Seed for per-cluster `h_seed` derivation. Same security pattern
    /// as EMVP+IVF's R-seed: shared `L`, per-cluster `(H, S)`. Prevents
    /// `M_enc_i − M_enc_j = M_i − M_j + (small)` cross-cluster leakage.
    pub upload_seed: u64,
    /// Whether per-query Protocol 2 (Freivalds) verification runs.
    pub verification_enabled: bool,
    /// f32 → F_p quantisation scale (mirrors `BnTmConfig::quantisation_q`).
    /// Default `params::Q` (= 2^20); a sweep knob. Change ⇒ disk cache
    /// invalidates (Q is in the fingerprint).
    pub quantisation_q: u64,
    pub progress: Option<Arc<dyn ProgressReporter>>,
    /// Compute substrate. `Cpu` runs the CPU per-cluster matvec under
    /// rayon. `Gpu` requires the `gpu` Cargo feature; without it
    /// `upload_cluster` returns [`BnTmIvfError::GpuFeatureNotEnabled`].
    /// Encryption + k-means + verification + decoding stay on host
    /// either way; only the per-cluster `compute_products` matvec moves
    /// to device. Same `[ivf]` parity invariant: GPU and CPU runs share
    /// centroids and cluster assignments.
    pub device: Device,
    /// VRAM budget for the streaming cluster store on GPU runs. `None`
    /// (default) auto-detects 80 % of free VRAM via NVML at handle
    /// build. `Some(b)` pins a hard byte cap. Ignored on CPU runs.
    pub vram_budget_bytes: Option<u64>,
}

/// Opaque per-index state.
pub struct BnTmIvfHandle {
    /// Plaintext-side IVF — drives client-side routing and cluster-id
    /// → global VectorId remapping after decryption.
    index: Arc<IvfIndex>,
    /// Per-cluster encrypted state.
    clusters: Arc<Vec<EncryptedCluster>>,
    /// `L` (n × n_1), shared across clusters, derived from `key_seed`.
    l_subspace: Arc<Vec<u64>>,
    nprobe: usize,
    dim: usize,
    params: BnTmParams,
    verification_enabled: bool,
    /// Quantisation scale baked into every cluster's `m_enc`; queries
    /// must quantise at the same Q.
    quantisation_q: u64,
    inner: HandleInner,
}

pub(crate) enum HandleInner {
    Cpu,
    /// Per-cluster device-resident M_enc; per-query kernel launches
    /// per probed cluster. Verification + decoding stay on CPU. Boxed
    /// because the streaming `IvfGpuState` carries
    /// `Arc<Vec<EncryptedCluster>>` + LRU + sampler + s_mats —
    /// clippy's `large_enum_variant` lint trips otherwise.
    #[cfg(feature = "gpu")]
    Gpu(Box<gpu::IvfGpuState>),
}

impl BnTmIvfHandle {
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

impl fmt::Debug for BnTmIvfHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device = match &self.inner {
            HandleInner::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(_) => "gpu",
        };
        f.debug_struct("BnTmIvfHandle")
            .field("device", &device)
            .field("dim", &self.dim)
            .field("n_centroids", &self.index.centroids.len())
            .field("nprobe", &self.nprobe)
            .field("n", &self.params.n())
            .field("n_1", &self.params.n1())
            .field("verification_enabled", &self.verification_enabled)
            .field(
                "cluster_sizes",
                &self.clusters.iter().map(|c| c.m).collect::<Vec<_>>(),
            )
            .field("h_seeds", &"[redacted]")
            .field("l_subspace", &"[redacted]")
            .field("a_l", &"[redacted]")
            .field("m_plain", &"[redacted]")
            .field("h_mat", &"[redacted]")
            .field("s_mat", &"[redacted]")
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BnTmIvfError {
    #[error("corpus must contain at least one vector")]
    EmptyCorpus,
    #[error("k-means produced an empty cluster (corpus too small for n_centroids?)")]
    EmptyCluster,
    #[error("n_centroids {0} exceeds corpus size")]
    TooManyCentroids(usize),
    #[error("invalid nprobe {nprobe}: must be in 1..={n_centroids}")]
    InvalidNprobe { nprobe: usize, n_centroids: usize },
    #[error("dimension too large: vectors have dim {actual}, n = {n}")]
    DimensionTooLarge { actual: usize, n: usize },
    #[error("query dimension {query_dim} does not match index dimension {index_dim}")]
    DimensionMismatch { query_dim: usize, index_dim: usize },
    #[error("Protocol 2 verification failed at cluster {cluster}, trial {trial}")]
    VerificationFailed { cluster: usize, trial: usize },
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

/// Cached state per (n_centroids, train_seed, max_iter, N, upload_seed,
/// key_seed, params, quantisation_q) tuple — built once per nprobe
/// sweep at a fixed Q.
#[derive(Clone)]
struct CachedIndex {
    index: Arc<IvfIndex>,
    clusters: Arc<Vec<EncryptedCluster>>,
    l_subspace: Arc<Vec<u64>>,
    dim: usize,
    quantisation_q: u64,
}

pub struct BnTmIvfScorer {
    cache: Mutex<HashMap<String, CachedIndex>>,
    cache_dir: Option<PathBuf>,
}

impl BnTmIvfScorer {
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

impl Default for BnTmIvfScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Byte size of the IVF index payload as written by
/// `ivf::write_index_to`, computed without serialising. The streaming
/// cache writer needs this to track file position and place the
/// 8-byte-alignment pad correctly so the subsequent per-cluster
/// `m_enc` / `a_l` / `m_plain` regions land at mmap-castable offsets.
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

/// Derive a per-cluster `h_seed` deterministically from
/// `(upload_seed, cluster_idx)`. ChaCha20 mixing avoids structural
/// collisions across small offsets — same pattern as EMVP+IVF's
/// R-seed derivation.
fn derive_h_seed(upload_seed: u64, cluster_idx: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&upload_seed.to_le_bytes());
    seed[8..16].copy_from_slice(&(cluster_idx as u64).to_le_bytes());
    let mut rng = ChaCha20Rng::from_seed(seed);
    rng.random()
}

#[async_trait]
impl Scorer for BnTmIvfScorer {
    type Config = BnTmIvfConfig;
    type ClusterHandle = BnTmIvfHandle;
    type Error = BnTmIvfError;

    async fn upload_cluster(
        &self,
        config: &BnTmIvfConfig,
        vectors: &[Vector],
    ) -> Result<(BnTmIvfHandle, BuildOutcome), BnTmIvfError> {
        let build_start = Instant::now();
        let params = config.params;
        let n = params.n();

        if vectors.is_empty() {
            return Err(BnTmIvfError::EmptyCorpus);
        }
        if config.n_centroids > vectors.len() {
            return Err(BnTmIvfError::TooManyCentroids(config.n_centroids));
        }
        if config.nprobe == 0 || config.nprobe > config.n_centroids {
            return Err(BnTmIvfError::InvalidNprobe {
                nprobe: config.nprobe,
                n_centroids: config.n_centroids,
            });
        }
        let dim = vectors[0].0.len();
        if dim > n {
            return Err(BnTmIvfError::DimensionTooLarge { actual: dim, n });
        }

        let key_seed = config.key_seed;
        let n_centroids = config.n_centroids;
        let nprobe = config.nprobe;
        let train_seed = config.train_seed;
        let max_iter = config.max_iter;
        let upload_seed = config.upload_seed;
        let verification_enabled = config.verification_enabled;
        let quantisation_q = config.quantisation_q;
        let progress = config.progress.clone();

        // Cache fingerprint covers everything that affects the encrypted
        // index. `n1` stands in for the parameter-set discriminant: it
        // determines the a_l / h_mat region sizes, so two different δ
        // configs (or any future parameter set with a different n_1)
        // must never share a cache file. quantisation_q is in the
        // fingerprint because the encrypted matrix bakes Q in (a
        // quantisation sweep would otherwise reuse a stale cache).
        //
        // Tag bumped to "bntm-ivf2" (Plan 28): the on-disk layout gained
        // a new per-cluster `h_mat` region, so old "bntm-ivf"-tagged
        // cache files must not be mistaken for the new layout — the tag
        // change forces a different fingerprint/filename, so old caches
        // are silently skipped and rebuilt rather than misread.
        let cache_parts: [Vec<u8>; 9] = [
            b"bntm-ivf2".to_vec(),
            (n_centroids as u64).to_le_bytes().to_vec(),
            train_seed.to_le_bytes().to_vec(),
            (max_iter as u64).to_le_bytes().to_vec(),
            (vectors.len() as u64).to_le_bytes().to_vec(),
            upload_seed.to_le_bytes().to_vec(),
            key_seed.to_vec(),
            (params.n1() as u64).to_le_bytes().to_vec(),
            quantisation_q.to_le_bytes().to_vec(),
        ];
        let parts_refs: Vec<&[u8]> = cache_parts.iter().map(|v| v.as_slice()).collect();
        let fp = cache::fingerprint(&parts_refs);
        let disk_path = self
            .cache_dir
            .as_deref()
            .map(|dir| dir.join(format!(".bntm-ivf-cache-{fp}.bin")));

        // In-memory cache hit?
        let cached = self.cache.lock().unwrap().get(&fp).cloned();
        let in_memory_hit = cached.is_some();
        let mut disk_hit = false;
        let cached_state = if let Some(c) = cached {
            if let Some(ref p) = progress {
                p.on_build_complete();
            }
            Some(c)
        } else {
            // Disk cache hit?
            let from_disk = if let Some(ref path) = disk_path {
                if path.exists() {
                    let p = path.clone();
                    eprintln!("Loading cached BN+IVF index from {p:?}…");
                    let parts_owned = cache_parts.clone();
                    tokio::task::spawn_blocking(move || {
                        let refs: Vec<&[u8]> = parts_owned.iter().map(|v| v.as_slice()).collect();
                        load_bntm_ivf_cache(&p, params, &refs).ok().flatten()
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
            if let Some((index, clusters, l_subspace)) = from_disk {
                disk_hit = true;
                let state = CachedIndex {
                    index: Arc::new(index),
                    clusters: Arc::new(clusters),
                    l_subspace: Arc::new(l_subspace),
                    dim,
                    quantisation_q,
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

            let (index, clusters, l_subspace) = tokio::task::spawn_blocking(move || {
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

                // 2. Generate L once (shared across clusters).
                let l_subspace = generate_subspace(key_seed, n, params.n1());
                let n1 = params.n1();

                // 3. Pick build mode. `save_path = Some` →
                //    stream-to-disk + mmap (m_enc + a_l + m_plain sum
                //    exceeds physical RAM at large scale, so we never
                //    accumulate `Vec<u64>`s — each cluster's three
                //    matrices are written to file then dropped, and
                //    the post-build handle holds mmap slices into the
                //    on-disk cache instead).
                //    `save_path = None` → owned in-memory (tests,
                //    no-cache mode; corpora are tiny here).
                //
                //    Cache file layout under mmap mode (different
                //    from the earlier single-shot
                //    `save_bntm_ivf_cache` layout — old caches won't
                //    load and will rebuild from scratch via the
                //    `.ok().flatten()` fall-through). `m_plain` is
                //    now persisted to disk so the load path can
                //    mmap it instead of re-quantising the IVF
                //    payload's f32 vectors into a fresh 70 GB
                //    `Vec<u64>` at scale:
                //      [20  cache header]
                //      [var IVF index payload]                         (sequential read)
                //      [pad to 8B  pad bytes (0..7)]                   (alignment for mmap u64 cast)
                //      [8   u64 n_clusters]
                //      [for each cluster:
                //         [8   u64 m]
                //         [32  h_seed]
                //         [m*n*8     m_enc   u64s]                     (8B-aligned, mmap-castable)
                //         [m*n_1*8   a_l     u64s]
                //         [m*n*8     m_plain u64s]
                //      ]
                //      [n*n_1*8  l_subspace u64s]
                //
                //    Per-field byte offsets are recorded as we write,
                //    so reconstruction after build doesn't re-parse
                //    the file.
                let (clusters, encryption_io_err): (Vec<EncryptedCluster>, Option<io::Error>) =
                    if let Some(ref final_path) = save_path {
                        let tmp_path = final_path.with_extension("bin.tmp");
                        // Best-effort streaming. Any I/O error mid-stream
                        // surfaces as a warning; the caller sees a
                        // partial run (at scale, falling back to owned
                        // accumulation would re-introduce the OOM).
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
                            // u64 regions are mmap-castable to &[u64].
                            let pad = ((8 - (bytes_written % 8)) % 8) as usize;
                            if pad > 0 {
                                w.write_all(&[0u8; 8][..pad])?;
                                bytes_written += pad as u64;
                            }
                            debug_assert_eq!(bytes_written % 8, 0);
                            w.write_all(&(index.clusters.len() as u64).to_le_bytes())?;
                            bytes_written += 8;

                            // Per-cluster metadata for mmap
                            // reconstruction: (m, h_seed, m_enc_off,
                            // a_l_off, m_plain_off, h_mat_off).
                            let mut metas: Vec<(usize, [u8; 32], usize, usize, usize, usize)> =
                                Vec::with_capacity(index.clusters.len());
                            for i in 0..index.clusters.len() {
                                let m = index.clusters[i].len();
                                let h_seed = derive_h_seed(upload_seed, i);
                                let enc = if m == 0 {
                                    EncryptedCluster {
                                        m_enc: Storage::Owned(Arc::new(Vec::new())),
                                        h_seed,
                                        m: 0,
                                        a_l: Storage::Owned(Arc::new(Vec::new())),
                                        m_plain: Storage::Owned(Arc::new(Vec::new())),
                                        h_mat: Storage::Owned(Arc::new(Vec::new())),
                                        s_mat: Arc::new(SparseMatrix::empty(0, n)),
                                    }
                                } else {
                                    let m_q: Vec<u64> = index.clusters[i]
                                        .iter()
                                        .flat_map(|(_, v)| quantise_vector(v, n, quantisation_q))
                                        .collect();
                                    encrypt_cluster(&m_q, m, &l_subspace, h_seed, params)
                                };

                                w.write_all(&(m as u64).to_le_bytes())?;
                                w.write_all(&h_seed)?;
                                bytes_written += 8 + 32;

                                let m_enc_off = bytes_written as usize;
                                for &v in enc.m_enc.as_slice() {
                                    w.write_all(&v.to_le_bytes())?;
                                }
                                bytes_written += (enc.m_enc.len() as u64) * 8;

                                let a_l_off = bytes_written as usize;
                                for &v in enc.a_l.as_slice() {
                                    w.write_all(&v.to_le_bytes())?;
                                }
                                bytes_written += (enc.a_l.len() as u64) * 8;

                                let m_plain_off = bytes_written as usize;
                                for &v in enc.m_plain.as_slice() {
                                    w.write_all(&v.to_le_bytes())?;
                                }
                                bytes_written += (enc.m_plain.len() as u64) * 8;

                                // h_mat (dense, m × n_1): new Plan-28
                                // region so the load path can mmap it
                                // instead of regenerating H from
                                // h_seed on every query. S is not
                                // persisted — it's cheap enough to
                                // regenerate once at load time (see
                                // `load_bntm_ivf_cache`).
                                let h_mat_off = bytes_written as usize;
                                for &v in enc.h_mat.as_slice() {
                                    w.write_all(&v.to_le_bytes())?;
                                }
                                bytes_written += (enc.h_mat.len() as u64) * 8;

                                metas.push((m, h_seed, m_enc_off, a_l_off, m_plain_off, h_mat_off));

                                // Strip f32 vectors NOW that the
                                // cluster is on disk; query path uses
                                // only the u32 ID from each tuple.
                                for (_, v) in index.clusters[i].iter_mut() {
                                    v.clear();
                                    v.shrink_to_fit();
                                }
                                if let Some(ref p) = progress {
                                    p.on_encrypt(i);
                                }
                                // enc drops here — three `Vec<u64>`s
                                // returned to the allocator, NOT
                                // pushed into clusters Vec (this is
                                // the fix for the OOM).
                            }

                            // l_subspace at the end of the file.
                            for &v in &l_subspace {
                                w.write_all(&v.to_le_bytes())?;
                            }
                            w.flush()?;
                            drop(w);
                            std::fs::rename(&tmp_path, final_path)?;

                            // Reopen read-only and mmap. Construct
                            // EncryptedCluster Vec with Storage::Mmap
                            // entries pointing into the mmap.
                            let f_ro = std::fs::File::open(final_path)?;
                            let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f_ro)? });
                            let clusters = metas
                                .into_iter()
                                .map(|(m, h_seed, m_enc_off, a_l_off, m_plain_off, h_mat_off)| {
                                    // S is cheap (sparse, μ-rate); regenerate
                                    // once here (per cluster, at handle-build
                                    // time) rather than persisting it to
                                    // disk. H comes straight from the mmap
                                    // region written above — no regen.
                                    let s_mat = if m == 0 {
                                        SparseMatrix::empty(0, n)
                                    } else {
                                        regenerate_h_s(h_seed, m, params).1
                                    };
                                    EncryptedCluster {
                                        m_enc: Storage::Mmap {
                                            mmap: mmap.clone(),
                                            byte_offset: m_enc_off,
                                            len_u64s: m * n,
                                        },
                                        h_seed,
                                        m,
                                        a_l: Storage::Mmap {
                                            mmap: mmap.clone(),
                                            byte_offset: a_l_off,
                                            len_u64s: m * n1,
                                        },
                                        m_plain: Storage::Mmap {
                                            mmap: mmap.clone(),
                                            byte_offset: m_plain_off,
                                            len_u64s: m * n,
                                        },
                                        h_mat: Storage::Mmap {
                                            mmap: mmap.clone(),
                                            byte_offset: h_mat_off,
                                            len_u64s: m * n1,
                                        },
                                        s_mat: Arc::new(s_mat),
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
                        // corpora). Same shape as the earlier owned
                        // path, no mmap.
                        let mut clusters: Vec<EncryptedCluster> =
                            Vec::with_capacity(index.clusters.len());
                        for i in 0..index.clusters.len() {
                            let m = index.clusters[i].len();
                            let h_seed = derive_h_seed(upload_seed, i);
                            let enc = if m == 0 {
                                EncryptedCluster {
                                    m_enc: Storage::Owned(Arc::new(Vec::new())),
                                    h_seed,
                                    m: 0,
                                    a_l: Storage::Owned(Arc::new(Vec::new())),
                                    m_plain: Storage::Owned(Arc::new(Vec::new())),
                                    h_mat: Storage::Owned(Arc::new(Vec::new())),
                                    s_mat: Arc::new(SparseMatrix::empty(0, n)),
                                }
                            } else {
                                let m_q: Vec<u64> = index.clusters[i]
                                    .iter()
                                    .flat_map(|(_, v)| quantise_vector(v, n, quantisation_q))
                                    .collect();
                                encrypt_cluster(&m_q, m, &l_subspace, h_seed, params)
                            };
                            clusters.push(enc);
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
                    eprintln!("Warning: BN+IVF cache write failed: {e}");
                }

                if let Some(ref p) = progress {
                    p.on_build_complete();
                }

                (index, clusters, l_subspace)
            })
            .await
            .map_err(|_| BnTmIvfError::SpawnPanic)?;

            if clusters.iter().any(|c| c.m == 0) {
                return Err(BnTmIvfError::EmptyCluster);
            }

            let state = CachedIndex {
                index: Arc::new(index),
                clusters: Arc::new(clusters),
                l_subspace: Arc::new(l_subspace),
                dim,
                quantisation_q,
            };
            self.cache.lock().unwrap().insert(fp.clone(), state.clone());
            state
        };

        // Branch on substrate. The host-side per-cluster M_enc + L are
        // built either way; GPU additionally uploads each M_enc to its
        // own device buffer at handle construction time (one-time per
        // probe sweep).
        let inner = match config.device {
            Device::Cpu => HandleInner::Cpu,
            Device::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    let clusters_for_gpu = Arc::clone(&cached_state.clusters);
                    let vram_budget = config.vram_budget_bytes;
                    let state = tokio::task::spawn_blocking(move || {
                        // Per-cluster `h_mat` / `s_mat` are already
                        // materialised on each `EncryptedCluster` (Plan
                        // 28 — at build time or, on a disk-cache hit,
                        // once at load time), so `IvfGpuState::build`
                        // reads them directly: no `regenerate_h_s` call
                        // anywhere on this path any more. `S` stays
                        // host-side (Arc-shared, never device-resident);
                        // `H` is uploaded lazily per cluster on the
                        // streaming store's upload-miss / LRU-evict path.
                        gpu::IvfGpuState::build(Arc::clone(&clusters_for_gpu), params, vram_budget)
                    })
                    .await
                    .map_err(|_| BnTmIvfError::SpawnPanic)??;
                    HandleInner::Gpu(Box::new(state))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(BnTmIvfError::GpuFeatureNotEnabled);
                }
            }
        };

        let outcome = BuildOutcome {
            cache_hit: in_memory_hit || disk_hit,
            build_duration: build_start.elapsed(),
        };
        Ok((
            BnTmIvfHandle {
                index: cached_state.index,
                clusters: cached_state.clusters,
                l_subspace: cached_state.l_subspace,
                nprobe,
                dim: cached_state.dim,
                params,
                verification_enabled,
                quantisation_q: cached_state.quantisation_q,
                inner,
            },
            outcome,
        ))
    }

    async fn score(
        &self,
        handle: &BnTmIvfHandle,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<Hit>, BnTmIvfError> {
        if query.0.len() != handle.dim {
            return Err(BnTmIvfError::DimensionMismatch {
                query_dim: query.0.len(),
                index_dim: handle.dim,
            });
        }

        let params = handle.params;
        let n = params.n();

        // 1. Plaintext routing on cleartext centroids (client-side).
        let probe_set = ivf_index::ivf::probe_route(&handle.index, &query.0, handle.nprobe);

        // 2. Quantise query.
        let v_q = quantise_vector(&query.0, n, handle.quantisation_q);

        // 3. Per-query mask seed sampled on the async task (ThreadRng
        //    is !Send; only Copy bytes cross spawn_blocking).
        let q_seed: [u8; 32] = rand::rng().random();

        let clusters = Arc::clone(&handle.clusters);
        let index = Arc::clone(&handle.index);
        let l_subspace = Arc::clone(&handle.l_subspace);
        let verification_enabled = handle.verification_enabled;

        let result: Result<Vec<Hit>, BnTmIvfError> = match &handle.inner {
            HandleInner::Cpu => tokio::task::spawn_blocking(move || {
                let mut q_rng = ChaCha20Rng::from_seed(q_seed);
                let encoded = encode_query(&v_q, &l_subspace, &mut q_rng, params);

                let mut all_hits: Vec<(u32, i64)> = Vec::new();
                for &ci in &probe_set {
                    let cluster = &clusters[ci];
                    let m = cluster.m;
                    if m == 0 {
                        continue;
                    }
                    let r = compute_products(cluster.m_enc.as_slice(), m, n, &encoded.v_enc);
                    if verification_enabled {
                        let mut v_rng =
                            ChaCha20Rng::from_seed(domain_separate(q_seed, b"protocol2"));
                        if let Err(trial) = verify_response(
                            &r,
                            cluster.m_enc.as_slice(),
                            m,
                            n,
                            &encoded.v_enc,
                            params.verification_trials(),
                            &mut v_rng,
                        ) {
                            return Err(BnTmIvfError::VerificationFailed { cluster: ci, trial });
                        }
                    }
                    let mv = decode_scores(&r, cluster, &l_subspace, &encoded, params);
                    let membership = &index.clusters[ci];
                    for (local_i, score) in mv.iter().enumerate() {
                        all_hits.push((membership[local_i].0, fp_to_signed(*score)));
                    }
                }

                all_hits.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(*score));
                all_hits.truncate(k);
                Ok(all_hits
                    .into_iter()
                    .map(|(id, score)| Hit {
                        id,
                        score: score as f32,
                    })
                    .collect::<Vec<Hit>>())
            })
            .await
            .map_err(|_| BnTmIvfError::SpawnPanic)?,
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                // Encode q on host, then dispatch one kernel launch per
                // probed cluster. Verification + decode stay on CPU
                // and run sequentially after each launch returns —
                // mirrors the emvp-IVF GPU pattern. Sequential is
                // intentional: a verify failure surfaces with the
                // exact cluster id, matching the CPU error variant.
                let l_for_encode = Arc::clone(&l_subspace);
                let v_q_for_encode = v_q.clone();
                let encoded = tokio::task::spawn_blocking(move || {
                    let mut q_rng = ChaCha20Rng::from_seed(q_seed);
                    encode_query(&v_q_for_encode, &l_for_encode, &mut q_rng, params)
                })
                .await
                .map_err(|_| BnTmIvfError::SpawnPanic)?;

                let mut per_cluster_r: Vec<(usize, Vec<u64>)> = Vec::with_capacity(probe_set.len());
                for &ci in &probe_set {
                    if let Some(r) = state.compute_products(ci, &encoded.v_enc)? {
                        per_cluster_r.push((ci, r));
                    }
                }

                // Per-query `L⊤ · v_enc` (shared across clusters),
                // then per-probed-cluster dense GPU decode + S
                // accessor. Both must happen here so owned data can
                // cross into `spawn_blocking` for the verify + compose
                // + rank step.
                let lt_venc = {
                    let l = Arc::clone(&l_subspace);
                    let v = encoded.v_enc.clone();
                    let n_local = n;
                    let n1_local = params.n1();
                    tokio::task::spawn_blocking(move || {
                        l_transpose_times_vec(&l, n_local, n1_local, &v)
                    })
                    .await
                    .map_err(|_| BnTmIvfError::SpawnPanic)?
                };
                let mut per_cluster_decode: Vec<GpuClusterDecode> =
                    Vec::with_capacity(per_cluster_r.len());
                for (ci, r) in per_cluster_r {
                    let dense = state
                        .decode_dense_terms(ci, &encoded.g, &lt_venc)?
                        .expect("non-empty cluster from compute_products has dense_terms");
                    let s_mat = state
                        .s_mat_arc(ci)
                        .expect("non-empty cluster from compute_products has cached S");
                    per_cluster_decode.push((ci, r, dense, s_mat));
                }

                tokio::task::spawn_blocking(move || {
                    let mut all_hits: Vec<(u32, i64)> = Vec::new();
                    for (ci, r, dense_gpu, s_mat) in per_cluster_decode {
                        let cluster = &clusters[ci];
                        let m = cluster.m;
                        if m == 0 {
                            continue;
                        }
                        if verification_enabled {
                            let mut v_rng =
                                ChaCha20Rng::from_seed(domain_separate(q_seed, b"protocol2"));
                            if let Err(trial) = verify_response(
                                &r,
                                cluster.m_enc.as_slice(),
                                m,
                                n,
                                &encoded.v_enc,
                                params.verification_trials(),
                                &mut v_rng,
                            ) {
                                return Err(BnTmIvfError::VerificationFailed {
                                    cluster: ci,
                                    trial,
                                });
                            }
                        }
                        let mv = decode_scores_with_gpu_dense(
                            &r, cluster, &s_mat, &encoded, &dense_gpu, params,
                        );
                        let membership = &index.clusters[ci];
                        for (local_i, score) in mv.iter().enumerate() {
                            all_hits.push((membership[local_i].0, fp_to_signed(*score)));
                        }
                    }

                    all_hits.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(*score));
                    all_hits.truncate(k);
                    Ok(all_hits
                        .into_iter()
                        .map(|(id, score)| Hit {
                            id,
                            score: score as f32,
                        })
                        .collect::<Vec<Hit>>())
                })
                .await
                .map_err(|_| BnTmIvfError::SpawnPanic)?
            }
        };

        result
    }

    fn communication_cost(&self, handle: &BnTmIvfHandle, _k: usize) -> CommunicationCost {
        let params = handle.params;
        let n = params.n();
        // v_enc has length n.
        let query_bytes = (n * FIELD_BYTES) as u64;
        // Per-cluster response: r ∈ F_p^m (direct delivery — the
        // structural BN win over EMVP's m × s coded scalars).
        // Analytical estimate uses the per-handle mean; per-query
        // realised bytes (which clusters were actually probed, with
        // what sizes) come from `score_with_realised_cost`.
        let mean_m: f64 = if handle.clusters.is_empty() {
            0.0
        } else {
            handle.clusters.iter().map(|c| c.m as f64).sum::<f64>() / handle.clusters.len() as f64
        };
        let avg_cluster_resp = (mean_m * FIELD_BYTES as f64).round() as u64;
        let total_setup: u64 = handle
            .clusters
            .iter()
            .map(|c| (c.m * n * FIELD_BYTES) as u64)
            .sum();
        CommunicationCost {
            query_bytes,
            response_bytes: handle.nprobe as u64 * avg_cluster_resp,
            cluster_response_bytes: avg_cluster_resp,
            setup_bytes: total_setup,
            pre_query_offline_up_bytes: 0,
            pre_query_offline_down_bytes: 0,
            // GPU bandwidth proxy = bytes the matrix kernel streams
            // per query. Each probed cluster's
            // `compute_products(m_enc, m, n, v_enc)` reads the full
            // `m × n` encrypted matrix; the analytical estimate sums
            // `m_avg × n × FIELD_BYTES` over the probe set. An earlier
            // version mirrored `response_bytes` (the m-scalar result
            // returned to the client), which understated BN by a factor
            // of n=1024.
            effective_bytes_per_query: (handle.nprobe as u64)
                * (mean_m as u64)
                * (n as u64)
                * (FIELD_BYTES as u64),
        }
    }

    async fn score_with_realised_cost(
        &self,
        handle: &BnTmIvfHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, CommunicationCost), BnTmIvfError> {
        // Routing happens once below; the realised probe set drives
        // both the encrypted compute and the per-query
        // `cluster_response_bytes` accounting. Inlining the scoring
        // path (vs. calling Self::score and re-routing) avoids a
        // second centroid scan per query.
        if query.0.len() != handle.dim {
            return Err(BnTmIvfError::DimensionMismatch {
                query_dim: query.0.len(),
                index_dim: handle.dim,
            });
        }

        let params = handle.params;
        let n = params.n();

        let probe_set = ivf_index::ivf::probe_route(&handle.index, &query.0, handle.nprobe);
        let probed_sizes: Vec<usize> = probe_set.iter().map(|&ci| handle.clusters[ci].m).collect();

        let v_q = quantise_vector(&query.0, n, handle.quantisation_q);
        let q_seed: [u8; 32] = rand::rng().random();

        let clusters = Arc::clone(&handle.clusters);
        let index = Arc::clone(&handle.index);
        let l_subspace = Arc::clone(&handle.l_subspace);
        let verification_enabled = handle.verification_enabled;

        let hits: Vec<Hit> = match &handle.inner {
            HandleInner::Cpu => {
                let probe_set_for_compute = probe_set.clone();
                tokio::task::spawn_blocking(move || -> Result<Vec<Hit>, BnTmIvfError> {
                    let mut q_rng = ChaCha20Rng::from_seed(q_seed);
                    let encoded = encode_query(&v_q, &l_subspace, &mut q_rng, params);

                    let mut all_hits: Vec<(u32, i64)> = Vec::new();
                    for &ci in &probe_set_for_compute {
                        let cluster = &clusters[ci];
                        let m = cluster.m;
                        if m == 0 {
                            continue;
                        }
                        let r = compute_products(cluster.m_enc.as_slice(), m, n, &encoded.v_enc);
                        if verification_enabled {
                            let mut v_rng =
                                ChaCha20Rng::from_seed(domain_separate(q_seed, b"protocol2"));
                            if let Err(trial) = verify_response(
                                &r,
                                cluster.m_enc.as_slice(),
                                m,
                                n,
                                &encoded.v_enc,
                                params.verification_trials(),
                                &mut v_rng,
                            ) {
                                return Err(BnTmIvfError::VerificationFailed {
                                    cluster: ci,
                                    trial,
                                });
                            }
                        }
                        let mv = decode_scores(&r, cluster, &l_subspace, &encoded, params);
                        let membership = &index.clusters[ci];
                        for (local_i, score) in mv.iter().enumerate() {
                            all_hits.push((membership[local_i].0, fp_to_signed(*score)));
                        }
                    }

                    all_hits.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(*score));
                    all_hits.truncate(k);
                    Ok(all_hits
                        .into_iter()
                        .map(|(id, score)| Hit {
                            id,
                            score: score as f32,
                        })
                        .collect::<Vec<Hit>>())
                })
                .await
                .map_err(|_| BnTmIvfError::SpawnPanic)??
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                let l_for_encode = Arc::clone(&l_subspace);
                let v_q_for_encode = v_q.clone();
                let encoded = tokio::task::spawn_blocking(move || {
                    let mut q_rng = ChaCha20Rng::from_seed(q_seed);
                    encode_query(&v_q_for_encode, &l_for_encode, &mut q_rng, params)
                })
                .await
                .map_err(|_| BnTmIvfError::SpawnPanic)?;

                let mut per_cluster_r: Vec<(usize, Vec<u64>)> = Vec::with_capacity(probe_set.len());
                for &ci in &probe_set {
                    if let Some(r) = state.compute_products(ci, &encoded.v_enc)? {
                        per_cluster_r.push((ci, r));
                    }
                }

                // Same GPU decode dispatch as `score()` above —
                // `L⊤·v_enc` once, dense terms + cached `S` per
                // cluster, then verify + compose + rank inside
                // `spawn_blocking`.
                let lt_venc = {
                    let l = Arc::clone(&l_subspace);
                    let v = encoded.v_enc.clone();
                    let n_local = n;
                    let n1_local = params.n1();
                    tokio::task::spawn_blocking(move || {
                        l_transpose_times_vec(&l, n_local, n1_local, &v)
                    })
                    .await
                    .map_err(|_| BnTmIvfError::SpawnPanic)?
                };
                let mut per_cluster_decode: Vec<GpuClusterDecode> =
                    Vec::with_capacity(per_cluster_r.len());
                for (ci, r) in per_cluster_r {
                    let dense = state
                        .decode_dense_terms(ci, &encoded.g, &lt_venc)?
                        .expect("non-empty cluster from compute_products has dense_terms");
                    let s_mat = state
                        .s_mat_arc(ci)
                        .expect("non-empty cluster from compute_products has cached S");
                    per_cluster_decode.push((ci, r, dense, s_mat));
                }

                tokio::task::spawn_blocking(move || -> Result<Vec<Hit>, BnTmIvfError> {
                    let mut all_hits: Vec<(u32, i64)> = Vec::new();
                    for (ci, r, dense_gpu, s_mat) in per_cluster_decode {
                        let cluster = &clusters[ci];
                        let m = cluster.m;
                        if m == 0 {
                            continue;
                        }
                        if verification_enabled {
                            let mut v_rng =
                                ChaCha20Rng::from_seed(domain_separate(q_seed, b"protocol2"));
                            if let Err(trial) = verify_response(
                                &r,
                                cluster.m_enc.as_slice(),
                                m,
                                n,
                                &encoded.v_enc,
                                params.verification_trials(),
                                &mut v_rng,
                            ) {
                                return Err(BnTmIvfError::VerificationFailed {
                                    cluster: ci,
                                    trial,
                                });
                            }
                        }
                        let mv = decode_scores_with_gpu_dense(
                            &r, cluster, &s_mat, &encoded, &dense_gpu, params,
                        );
                        let membership = &index.clusters[ci];
                        for (local_i, score) in mv.iter().enumerate() {
                            all_hits.push((membership[local_i].0, fp_to_signed(*score)));
                        }
                    }

                    all_hits.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(*score));
                    all_hits.truncate(k);
                    Ok(all_hits
                        .into_iter()
                        .map(|(id, score)| Hit {
                            id,
                            score: score as f32,
                        })
                        .collect::<Vec<Hit>>())
                })
                .await
                .map_err(|_| BnTmIvfError::SpawnPanic)??
            }
        };

        let mut cost = self.communication_cost(handle, k);
        let n = handle.params.n();
        let per_cluster: Vec<u64> = probed_sizes
            .iter()
            .map(|&m_i| (m_i * FIELD_BYTES) as u64)
            .collect();
        let response_bytes: u64 = per_cluster.iter().sum();
        cost.response_bytes = response_bytes;
        cost.cluster_response_bytes = if per_cluster.is_empty() {
            0
        } else {
            response_bytes / per_cluster.len() as u64
        };
        // Realised matrix-work proxy = Σ_probed m_i × n ×
        // FIELD_BYTES, the bytes the BN matvec actually streams
        // across the probed cluster set this query. An earlier version
        // mirrored realised `response_bytes` (m-scalar results), which
        // under-counted matrix work by a factor of n.
        let probed_total: u64 = probed_sizes.iter().map(|&m| m as u64).sum();
        cost.effective_bytes_per_query = probed_total * (n as u64) * (FIELD_BYTES as u64);
        Ok((hits, cost))
    }

    /// Batched `score` override. Cluster-major loop-transpose: per-query
    /// probe_route on the async task, then invert to per-cluster
    /// `(q_idx, probe_pos)` lists. Each non-empty cluster runs one
    /// [`crypto::compute_products_batch`] over its probing queries
    /// (`m_enc` read once across that subset), per-query verify (when
    /// `verification_enabled`) + [`crypto::decode_scores`] produce one
    /// `(membership_id, score)` bin per probing query, scattered into
    /// `query_results[q_idx][probe_pos]`. Final per-query gather walks
    /// `probe_set` in order so the concatenated `(id, signed_score)`
    /// stream is byte-identical to the per-query path's pre-sort input;
    /// `sort_unstable_by_key` then sees a bit-equal slice and returns a
    /// bit-equal Vec<Hit>. `clusters`, `index`, and `l_subspace` stay
    /// Arc-shared via the handle. GPU handles fall back to the inline
    /// sequential loop.
    ///
    /// Cluster iteration order in the outer loop is ascending cluster
    /// id — different from the per-query probe-set order, but the
    /// per-(q, ci) bins are scattered by probe_pos before the per-query
    /// gather, so the final pre-sort sequence per query still matches.
    /// Under verification-on with a malicious server, the first
    /// surfaced failure is the lowest-id cluster across the batch
    /// rather than the per-query "first probe-position with a flip" —
    /// both are correct `BnTmIvfError::VerificationFailed { cluster,
    /// trial }` reports, just different first-failure conventions.
    async fn score_batch(
        &self,
        handle: &BnTmIvfHandle,
        queries: &[Vector],
        k: usize,
    ) -> Result<Vec<Vec<Hit>>, BnTmIvfError> {
        for q in queries {
            if q.0.len() != handle.dim {
                return Err(BnTmIvfError::DimensionMismatch {
                    query_dim: q.0.len(),
                    index_dim: handle.dim,
                });
            }
        }

        let params = handle.params;
        let n = params.n();

        // Per-query routing on cleartext centroids (client-side). Done
        // on the async task — cheap relative to the cryptographic
        // matvec, but matches the per-query path's pre-spawn route.
        let probe_sets: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| ivf::probe_route(&handle.index, &q.0, handle.nprobe))
            .collect();

        // Per-query quantise.
        let v_qs: Vec<Vec<u64>> = queries
            .iter()
            .map(|q| quantise_vector(&q.0, n, handle.quantisation_q))
            .collect();

        // Per-query q_seed: ThreadRng is !Send — sample before
        // spawn_blocking, only `Copy` bytes cross.
        let q_seeds: Vec<[u8; 32]> = (0..queries.len()).map(|_| rand::rng().random()).collect();

        let clusters = Arc::clone(&handle.clusters);
        let index = Arc::clone(&handle.index);
        let l_subspace = Arc::clone(&handle.l_subspace);
        let verification_enabled = handle.verification_enabled;
        let big_b = queries.len();

        match &handle.inner {
            HandleInner::Cpu => {
                tokio::task::spawn_blocking(move || -> Result<Vec<Vec<Hit>>, BnTmIvfError> {
                    // Per-query encode.
                    let encodeds: Vec<crate::crypto::QueryEncoded> = v_qs
                        .iter()
                        .zip(&q_seeds)
                        .map(|(v_q, q_seed)| {
                            let mut rng = ChaCha20Rng::from_seed(*q_seed);
                            encode_query(v_q, &l_subspace, &mut rng, params)
                        })
                        .collect();

                    // Invert probe_sets → per-cluster (q_idx, probe_pos) lists.
                    let mut cluster_probers: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
                    for (q_idx, probe_set) in probe_sets.iter().enumerate() {
                        for (probe_pos, &ci) in probe_set.iter().enumerate() {
                            cluster_probers
                                .entry(ci)
                                .or_default()
                                .push((q_idx, probe_pos));
                        }
                    }

                    // Per-query bins indexed by probe_pos. Empty bin = "this
                    // probe slot is empty / cluster skipped" (matches the
                    // per-query path's `continue` on `m == 0`).
                    let mut query_results: Vec<Vec<Vec<(u32, i64)>>> = probe_sets
                        .iter()
                        .map(|ps| vec![Vec::new(); ps.len()])
                        .collect();

                    // Iterate clusters in ascending id order. Verification
                    // surfacing is deterministic; bit-equality of the
                    // final output is independent of cluster order because
                    // each (q, ci) bin is scattered to its own probe_pos
                    // slot before the per-query gather.
                    let mut cluster_ids: Vec<usize> = cluster_probers.keys().copied().collect();
                    cluster_ids.sort_unstable();

                    for ci in cluster_ids {
                        let probers = &cluster_probers[&ci];
                        let cluster = &clusters[ci];
                        let m = cluster.m;
                        if m == 0 {
                            continue;
                        }
                        // Gather v_encs for probing queries in slot order.
                        let v_encs: Vec<&[u64]> = probers
                            .iter()
                            .map(|&(q_idx, _)| encodeds[q_idx].v_enc.as_slice())
                            .collect();

                        // Batched matvec: m_enc read once across the
                        // probing queries.
                        let rs = compute_products_batch(cluster.m_enc.as_slice(), m, n, &v_encs);

                        let membership = &index.clusters[ci];
                        for (slot_idx, &(q_idx, probe_pos)) in probers.iter().enumerate() {
                            let r = &rs[slot_idx];
                            let encoded = &encodeds[q_idx];
                            let q_seed = &q_seeds[q_idx];
                            if verification_enabled {
                                let mut v_rng =
                                    ChaCha20Rng::from_seed(domain_separate(*q_seed, b"protocol2"));
                                if let Err(trial) = verify_response(
                                    r,
                                    cluster.m_enc.as_slice(),
                                    m,
                                    n,
                                    &encoded.v_enc,
                                    params.verification_trials(),
                                    &mut v_rng,
                                ) {
                                    return Err(BnTmIvfError::VerificationFailed {
                                        cluster: ci,
                                        trial,
                                    });
                                }
                            }
                            let mv = decode_scores(r, cluster, &l_subspace, encoded, params);
                            let bin: Vec<(u32, i64)> = mv
                                .iter()
                                .enumerate()
                                .map(|(local_i, score)| {
                                    (membership[local_i].0, fp_to_signed(*score))
                                })
                                .collect();
                            query_results[q_idx][probe_pos] = bin;
                        }
                    }

                    // Per-query gather in probe-set order → sort → truncate.
                    let mut out: Vec<Vec<Hit>> = Vec::with_capacity(big_b);
                    for bins in query_results {
                        let mut all_hits: Vec<(u32, i64)> = Vec::new();
                        for bin in bins {
                            all_hits.extend(bin);
                        }
                        all_hits.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(*score));
                        all_hits.truncate(k);
                        out.push(
                            all_hits
                                .into_iter()
                                .map(|(id, score)| Hit {
                                    id,
                                    score: score as f32,
                                })
                                .collect(),
                        );
                    }
                    Ok(out)
                })
                .await
                .map_err(|_| BnTmIvfError::SpawnPanic)?
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(_) => {
                let mut out = Vec::with_capacity(big_b);
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

/// Per-query substep timings reported by [`BnTmIvfScorer::score_with_breakdown`].
/// `route` = client-side centroid scan; `encode` = quantisation + LG/T
/// mask construction; `server` = aggregate `compute_products` over the
/// probe set; `verify` = aggregate Protocol 2 trials when
/// `verification_enabled`, otherwise 0; `decode` = aggregate
/// `decode_scores`; `merge` = top-k sort across clusters.
#[derive(Debug, Clone, Copy, Default)]
pub struct BnTmIvfTiming {
    pub route_us: u64,
    pub encode_us: u64,
    pub server_us: u64,
    pub verify_us: u64,
    pub decode_us: u64,
    pub merge_us: u64,
}

impl BnTmIvfScorer {
    /// Off-hot-path equivalent of [`Scorer::score`] with per-substep
    /// timings. Substeps are accumulated across the probe set; the
    /// breakdown path matches `score`'s sequential per-cluster
    /// ordering so verification failures surface deterministically.
    pub async fn score_with_breakdown(
        &self,
        handle: &BnTmIvfHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, BnTmIvfTiming), BnTmIvfError> {
        if query.0.len() != handle.dim {
            return Err(BnTmIvfError::DimensionMismatch {
                query_dim: query.0.len(),
                index_dim: handle.dim,
            });
        }

        let params = handle.params;
        let n = params.n();
        let nprobe = handle.nprobe.min(handle.index.centroids.len());

        let q_seed: [u8; 32] = rand::rng().random();
        let clusters = Arc::clone(&handle.clusters);
        let index = Arc::clone(&handle.index);
        let l_subspace = Arc::clone(&handle.l_subspace);
        let verification_enabled = handle.verification_enabled;
        let quantisation_q = handle.quantisation_q;
        let query_vec = query.0.clone();

        // Dispatch on inner. The CPU branch keeps
        // route → encode → per-cluster (server, verify, decode) → merge
        // inside a single spawn_blocking. The GPU branch breaks the
        // loop across the host/device boundary so the per-cluster
        // server_us measurement spans each kernel launch's full
        // round-trip — the same shape `run_eval_with_verify_us`
        // measures, so verification-overhead-us stays meaningful on
        // GPU runs.
        let result: Result<(Vec<Hit>, BnTmIvfTiming), BnTmIvfError> = match &handle.inner {
            HandleInner::Cpu => tokio::task::spawn_blocking(move || {
                let t = Instant::now();
                let mut centroid_dists: Vec<(usize, f32)> = index
                    .centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let d: f32 = c
                            .iter()
                            .zip(&query_vec)
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
                let v_q = quantise_vector(&query_vec, n, quantisation_q);
                let mut q_rng = ChaCha20Rng::from_seed(q_seed);
                let encoded = encode_query(&v_q, &l_subspace, &mut q_rng, params);
                let encode_us = t.elapsed().as_micros() as u64;

                let mut server_us: u64 = 0;
                let mut verify_us: u64 = 0;
                let mut decode_us: u64 = 0;
                let mut all_hits: Vec<(u32, i64)> = Vec::new();
                for &ci in &probe_set {
                    let cluster = &clusters[ci];
                    let m = cluster.m;
                    if m == 0 {
                        continue;
                    }
                    let t = Instant::now();
                    let r = compute_products(cluster.m_enc.as_slice(), m, n, &encoded.v_enc);
                    server_us = server_us.saturating_add(t.elapsed().as_micros() as u64);

                    if verification_enabled {
                        let t = Instant::now();
                        let mut v_rng =
                            ChaCha20Rng::from_seed(domain_separate(q_seed, b"protocol2"));
                        if let Err(trial) = verify_response(
                            &r,
                            cluster.m_enc.as_slice(),
                            m,
                            n,
                            &encoded.v_enc,
                            params.verification_trials(),
                            &mut v_rng,
                        ) {
                            return Err(BnTmIvfError::VerificationFailed { cluster: ci, trial });
                        }
                        verify_us = verify_us.saturating_add(t.elapsed().as_micros() as u64);
                    }

                    let t = Instant::now();
                    let mv = decode_scores(&r, cluster, &l_subspace, &encoded, params);
                    let membership = &index.clusters[ci];
                    for (local_i, score) in mv.iter().enumerate() {
                        all_hits.push((membership[local_i].0, fp_to_signed(*score)));
                    }
                    decode_us = decode_us.saturating_add(t.elapsed().as_micros() as u64);
                }

                let t = Instant::now();
                all_hits.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(*score));
                all_hits.truncate(k);
                let hits: Vec<Hit> = all_hits
                    .into_iter()
                    .map(|(id, score)| Hit {
                        id,
                        score: score as f32,
                    })
                    .collect();
                let merge_us = t.elapsed().as_micros() as u64;

                Ok((
                    hits,
                    BnTmIvfTiming {
                        route_us,
                        encode_us,
                        server_us,
                        verify_us,
                        decode_us,
                        merge_us,
                    },
                ))
            })
            .await
            .map_err(|_| BnTmIvfError::SpawnPanic)?,
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                let l_for_route = Arc::clone(&l_subspace);
                let index_for_route = Arc::clone(&index);
                let query_for_route = query_vec.clone();
                let (probe_set, encoded, route_us, encode_us) =
                    tokio::task::spawn_blocking(move || {
                        let t = Instant::now();
                        let mut centroid_dists: Vec<(usize, f32)> = index_for_route
                            .centroids
                            .iter()
                            .enumerate()
                            .map(|(i, c)| {
                                let d: f32 = c
                                    .iter()
                                    .zip(&query_for_route)
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
                        let v_q = quantise_vector(&query_for_route, n, quantisation_q);
                        let mut q_rng = ChaCha20Rng::from_seed(q_seed);
                        let encoded = encode_query(&v_q, &l_for_route, &mut q_rng, params);
                        let encode_us = t.elapsed().as_micros() as u64;
                        (probe_set, encoded, route_us, encode_us)
                    })
                    .await
                    .map_err(|_| BnTmIvfError::SpawnPanic)?;

                // Per-cluster GPU dispatch + collect r vectors with
                // per-launch wall-clock timing. We hold the timing in a
                // u64 accumulator (saturating add) so a measurement
                // glitch can't panic mid-sweep.
                let mut server_us: u64 = 0;
                let mut per_cluster_r: Vec<(usize, Vec<u64>)> = Vec::with_capacity(probe_set.len());
                for &ci in &probe_set {
                    let t = Instant::now();
                    if let Some(r) = state.compute_products(ci, &encoded.v_enc)? {
                        server_us = server_us.saturating_add(t.elapsed().as_micros() as u64);
                        per_cluster_r.push((ci, r));
                    } else {
                        server_us = server_us.saturating_add(t.elapsed().as_micros() as u64);
                    }
                }

                // Pre-spawn portion of `decode_us` — shared `L⊤·v_enc`
                // for the query plus per-cluster GPU dense terms + S
                // accessor. The post-spawn portion (sparse + compose +
                // rank) is timed inside the closure and summed.
                let t_decode_pre = Instant::now();
                let lt_venc = l_transpose_times_vec(&l_subspace, n, params.n1(), &encoded.v_enc);
                let mut per_cluster_decode: Vec<GpuClusterDecode> =
                    Vec::with_capacity(per_cluster_r.len());
                for (ci, r) in per_cluster_r {
                    let dense = state
                        .decode_dense_terms(ci, &encoded.g, &lt_venc)?
                        .expect("non-empty cluster from compute_products has dense_terms");
                    let s_mat = state
                        .s_mat_arc(ci)
                        .expect("non-empty cluster from compute_products has cached S");
                    per_cluster_decode.push((ci, r, dense, s_mat));
                }
                let decode_pre_us = t_decode_pre.elapsed().as_micros() as u64;

                tokio::task::spawn_blocking(move || -> Result<_, BnTmIvfError> {
                    let mut verify_us: u64 = 0;
                    let mut decode_us: u64 = decode_pre_us;
                    let mut all_hits: Vec<(u32, i64)> = Vec::new();
                    for (ci, r, dense_gpu, s_mat) in per_cluster_decode {
                        let cluster = &clusters[ci];
                        let m = cluster.m;
                        if m == 0 {
                            continue;
                        }
                        if verification_enabled {
                            let t = Instant::now();
                            let mut v_rng =
                                ChaCha20Rng::from_seed(domain_separate(q_seed, b"protocol2"));
                            if let Err(trial) = verify_response(
                                &r,
                                cluster.m_enc.as_slice(),
                                m,
                                n,
                                &encoded.v_enc,
                                params.verification_trials(),
                                &mut v_rng,
                            ) {
                                return Err(BnTmIvfError::VerificationFailed {
                                    cluster: ci,
                                    trial,
                                });
                            }
                            verify_us = verify_us.saturating_add(t.elapsed().as_micros() as u64);
                        }

                        let t = Instant::now();
                        let mv = decode_scores_with_gpu_dense(
                            &r, cluster, &s_mat, &encoded, &dense_gpu, params,
                        );
                        let membership = &index.clusters[ci];
                        for (local_i, score) in mv.iter().enumerate() {
                            all_hits.push((membership[local_i].0, fp_to_signed(*score)));
                        }
                        decode_us = decode_us.saturating_add(t.elapsed().as_micros() as u64);
                    }

                    let t = Instant::now();
                    all_hits.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(*score));
                    all_hits.truncate(k);
                    let hits: Vec<Hit> = all_hits
                        .into_iter()
                        .map(|(id, score)| Hit {
                            id,
                            score: score as f32,
                        })
                        .collect();
                    let merge_us = t.elapsed().as_micros() as u64;

                    Ok((
                        hits,
                        BnTmIvfTiming {
                            route_us,
                            encode_us,
                            server_us,
                            verify_us,
                            decode_us,
                            merge_us,
                        },
                    ))
                })
                .await
                .map_err(|_| BnTmIvfError::SpawnPanic)?
            }
        };

        result
    }
}

// ---------------------------------------------------------------------------
// Cache I/O
// ---------------------------------------------------------------------------
//
// mmap cache layout:
//   <20  cache header>
//   <var IVF index payload>                       (sequential read)
//   <pad 0..7 zero bytes to next 8-byte boundary> (alignment for mmap u64 cast)
//   <8   u64 n_clusters>
//   <per-cluster:
//      <8   u64 m>
//      <32  h_seed>
//      <m*n*8     m_enc   u64s>                   (8B-aligned, mmap-castable)
//      <m*n_1*8   a_l     u64s>
//      <m*n*8     m_plain u64s>
//   >
//   <n*n_1*8 l_subspace u64s>
//
// `m_plain` is persisted (an earlier layout re-derived it from the
// IVF payload's f32 vectors at load time). The load path mmaps the
// file, so none of `m_enc` / `a_l` / `m_plain` are copied into heap
// `Vec`s — the in-memory handle holds offsets only. Older caches
// have no pad and a different per-cluster layout, so they fail to
// load gracefully via the caller's `.ok().flatten()` and rebuild from
// scratch.

/// Single-shot cache writer — kept as a format reference for the
/// inline streaming write in `upload_cluster`. The production path no
/// longer calls this directly because the incremental f32-vector
/// strip in the encryption loop requires interleaving the write with
/// the build, and the post-build handle now holds `Storage::Mmap`-backed
/// clusters built from `metas` rather than the in-memory
/// `clusters: &[EncryptedCluster]` shape this fn assumes.
#[allow(dead_code)]
fn save_bntm_ivf_cache(
    path: &std::path::Path,
    index: &IvfIndex,
    clusters: &[EncryptedCluster],
    l_subspace: &[u64],
    parts: &[&[u8]],
) -> io::Result<()> {
    let tmp = path.with_extension("bin.tmp");
    let mut w = BufWriter::new(std::fs::File::create(&tmp)?);
    let mut bytes_written: u64 = 0;
    cache::write_header(&mut w, parts)?;
    bytes_written += 20;
    let dim = index.centroids.first().map_or(0, |c| c.len());
    ivf::write_index_to(&mut w, index)?;
    bytes_written += ivf_payload_size(index, dim);
    let pad = ((8 - (bytes_written % 8)) % 8) as usize;
    if pad > 0 {
        w.write_all(&[0u8; 8][..pad])?;
    }
    w.write_all(&(clusters.len() as u64).to_le_bytes())?;
    for c in clusters {
        w.write_all(&(c.m as u64).to_le_bytes())?;
        w.write_all(&c.h_seed)?;
        for &v in c.m_enc.as_slice() {
            w.write_all(&v.to_le_bytes())?;
        }
        for &v in c.a_l.as_slice() {
            w.write_all(&v.to_le_bytes())?;
        }
        for &v in c.m_plain.as_slice() {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    for &v in l_subspace {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    std::fs::rename(&tmp, path)
}

type LoadedIvfCache = (IvfIndex, Vec<EncryptedCluster>, Vec<u64>);

fn load_bntm_ivf_cache(
    path: &std::path::Path,
    params: BnTmParams,
    parts: &[&[u8]],
) -> io::Result<Option<LoadedIvfCache>> {
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
    let n = params.n();
    let n1 = params.n1();

    if n_clusters != index.clusters.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "cache cluster count {} ≠ IVF index cluster count {}",
                n_clusters,
                index.clusters.len()
            ),
        ));
    }

    let mut clusters = Vec::with_capacity(n_clusters);
    for ci in 0..n_clusters {
        if mmap.len() < offset + 8 + 32 {
            return Ok(None);
        }
        let m = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        let mut h_seed = [0u8; 32];
        h_seed.copy_from_slice(&mmap[offset..offset + 32]);
        offset += 32;

        let cluster_vecs = &index.clusters[ci];
        if cluster_vecs.len() != m {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "cache m={m} ≠ IVF cluster {ci} membership size {}",
                    cluster_vecs.len()
                ),
            ));
        }

        let m_enc_len = m * n;
        let m_enc_off = offset;
        if mmap.len() < offset + m_enc_len * 8 {
            return Ok(None);
        }
        offset += m_enc_len * 8;

        let a_l_len = m * n1;
        let a_l_off = offset;
        if mmap.len() < offset + a_l_len * 8 {
            return Ok(None);
        }
        offset += a_l_len * 8;

        let m_plain_len = m * n;
        let m_plain_off = offset;
        if mmap.len() < offset + m_plain_len * 8 {
            return Ok(None);
        }
        offset += m_plain_len * 8;

        // h_mat (Plan 28): dense m × n_1, same shape/region pattern as
        // a_l. Persisted to disk (unlike S) so the load path never
        // needs to regenerate H.
        let h_mat_len = m * n1;
        let h_mat_off = offset;
        if mmap.len() < offset + h_mat_len * 8 {
            return Ok(None);
        }
        offset += h_mat_len * 8;

        // S is not persisted — cheap enough (sparse, μ-rate) to
        // regenerate once here, at handle-build time, rather than
        // once per query (the bug this plan fixes) or bloating the
        // cache file. Empty clusters get an empty S with no RNG cost.
        let s_mat = if m == 0 {
            SparseMatrix::empty(0, n)
        } else {
            regenerate_h_s(h_seed, m, params).1
        };

        clusters.push(EncryptedCluster {
            m_enc: Storage::Mmap {
                mmap: mmap.clone(),
                byte_offset: m_enc_off,
                len_u64s: m_enc_len,
            },
            h_seed,
            m,
            a_l: Storage::Mmap {
                mmap: mmap.clone(),
                byte_offset: a_l_off,
                len_u64s: a_l_len,
            },
            m_plain: Storage::Mmap {
                mmap: mmap.clone(),
                byte_offset: m_plain_off,
                len_u64s: m_plain_len,
            },
            h_mat: Storage::Mmap {
                mmap: mmap.clone(),
                byte_offset: h_mat_off,
                len_u64s: h_mat_len,
            },
            s_mat: Arc::new(s_mat),
        });
    }

    let l_len = n * n1;
    if mmap.len() < offset + l_len * 8 {
        return Ok(None);
    }
    let mut l_subspace = vec![0u64; l_len];
    for (i, v) in l_subspace.iter_mut().enumerate() {
        *v = u64::from_le_bytes(mmap[offset + i * 8..offset + i * 8 + 8].try_into().unwrap());
    }

    Ok(Some((index, clusters, l_subspace)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Q;

    fn random_unit_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vector> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..count)
            .map(|_| {
                let v: Vec<f32> = (0..dim)
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

    fn cfg(n_centroids: usize, nprobe: usize, verification_enabled: bool) -> BnTmIvfConfig {
        BnTmIvfConfig {
            params: BnTmParams::Sec128,
            key_seed: [42u8; 32],
            n_centroids,
            nprobe,
            train_seed: 42,
            max_iter: 25,
            upload_seed: 7,
            verification_enabled,
            quantisation_q: Q,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        }
    }

    #[tokio::test]
    async fn upload_and_score_smoke() {
        let scorer = BnTmIvfScorer::new();
        let vectors = random_unit_vectors(40, 32, 1);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(4, 2, false), &vectors)
            .await
            .unwrap();
        let q = random_unit_vectors(1, 32, 2).pop().unwrap();
        let hits = scorer.score(&handle, &q, 5).await.unwrap();
        assert_eq!(hits.len(), 5);
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    /// Honest server: verification on, every query passes across
    /// nprobe clusters.
    #[tokio::test]
    async fn ivf_score_with_verification_passes_on_honest_server() {
        let scorer = BnTmIvfScorer::new();
        let vectors = random_unit_vectors(40, 32, 3);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(4, 2, true), &vectors)
            .await
            .unwrap();
        for s in 4..8u64 {
            let q = random_unit_vectors(1, 32, s).pop().unwrap();
            let hits = scorer.score(&handle, &q, 5).await.unwrap();
            assert_eq!(hits.len(), 5);
        }
    }

    /// Per-query result from the batched IVF override is bit-equal to a
    /// sequential `score()` call on the same input. Same correctness
    /// argument as the flat test: per-(q, ci) the
    /// batched matvec is byte-identical to per-query
    /// `compute_products`, `decode_scores` is deterministic on
    /// `(r, cluster, encoded, params)`, and the per-query gather walks
    /// the bins in probe-position order so the pre-sort sequence
    /// matches the per-query path's `all_hits` exactly — therefore
    /// `sort_unstable_by_key` sees a bit-equal input and produces a
    /// bit-equal Vec<Hit>. `verification_enabled = false` so the test
    /// exercises bit-equality of the matvec + decode + rank stack
    /// (verification is covered by the IVF honest-server test above).
    #[tokio::test]
    async fn bntm_ivf_score_batch_matches_score() {
        let scorer = BnTmIvfScorer::new();
        let vectors = random_unit_vectors(64, 32, 77);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(8, 3, false), &vectors)
            .await
            .unwrap();

        let queries = random_unit_vectors(64, 32, 78);
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
                "B={b}: batched IVF output diverges from sequential score()"
            );
        }
    }

    /// Verification-on path through the IVF batched override.
    /// Independent of the equivalence test: `verification_enabled = true`
    /// exercises the per-(cluster, query) Freivalds trial in the
    /// closure on every probed cluster. A divergence between the
    /// batched `r` and what the verifier recomputes would fail this on
    /// an honest server.
    #[tokio::test]
    async fn bntm_ivf_score_batch_verification_passes_on_honest_server() {
        let scorer = BnTmIvfScorer::new();
        let vectors = random_unit_vectors(40, 32, 79);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(4, 2, true), &vectors)
            .await
            .unwrap();
        let queries = random_unit_vectors(8, 32, 80);
        let k = 5;
        let hits = scorer.score_batch(&handle, &queries, k).await.unwrap();
        assert_eq!(hits.len(), queries.len());
        for h in &hits {
            assert_eq!(h.len(), k);
        }
    }

    /// `nprobe = n_centroids` (full probe) with a single cluster
    /// matches the flat scorer's top-k IDs. The construction is
    /// zero-noise so this is exact equality; any drift here is a port
    /// bug.
    #[tokio::test]
    async fn single_cluster_full_probe_matches_flat_scorer() {
        use crate::{BnTmConfig, BnTmScorer};
        let vectors = random_unit_vectors(8, 32, 11);
        let q = random_unit_vectors(1, 32, 12).pop().unwrap();

        // IVF with n_centroids=1 (single cluster), nprobe=1.
        let ivf = BnTmIvfScorer::new();
        let (ivf_handle, _build) = ivf
            .upload_cluster(&cfg(1, 1, false), &vectors)
            .await
            .unwrap();
        let ivf_hits = ivf.score(&ivf_handle, &q, 5).await.unwrap();

        // Flat scorer over the same corpus, same params, same key_seed.
        let flat = BnTmScorer::new();
        let flat_cfg = BnTmConfig {
            params: BnTmParams::Sec128,
            key_seed: [42u8; 32],
            verification_enabled: false,
            quantisation_q: Q,
            progress: None,
            device: scorer_core::Device::Cpu,
            vram_budget_bytes: None,
        };
        let (flat_handle, _build) = flat.upload_cluster(&flat_cfg, &vectors).await.unwrap();
        let flat_hits = flat.score(&flat_handle, &q, 5).await.unwrap();

        let ivf_ids: Vec<u32> = ivf_hits.iter().map(|h| h.id).collect();
        let flat_ids: Vec<u32> = flat_hits.iter().map(|h| h.id).collect();
        assert_eq!(
            ivf_ids, flat_ids,
            "single-cluster IVF must match flat top-k IDs (§7.2 zero-noise)"
        );
    }

    #[tokio::test]
    async fn k_capped_at_total_cluster_membership() {
        let scorer = BnTmIvfScorer::new();
        let vectors = random_unit_vectors(20, 16, 13);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(4, 4, false), &vectors)
            .await
            .unwrap();
        let q = random_unit_vectors(1, 16, 14).pop().unwrap();
        let hits = scorer.score(&handle, &q, 10_000).await.unwrap();
        // Full probe sees all 20 vectors.
        assert_eq!(hits.len(), 20);
    }

    #[tokio::test]
    async fn empty_corpus_surfaces_as_error() {
        let scorer = BnTmIvfScorer::new();
        let vectors: Vec<Vector> = Vec::new();
        let result = scorer.upload_cluster(&cfg(4, 2, false), &vectors).await;
        assert!(matches!(result, Err(BnTmIvfError::EmptyCorpus)));
    }

    #[tokio::test]
    async fn too_many_centroids_surfaces_as_error() {
        let scorer = BnTmIvfScorer::new();
        let vectors = random_unit_vectors(5, 16, 17);
        let result = scorer.upload_cluster(&cfg(10, 2, false), &vectors).await;
        assert!(matches!(result, Err(BnTmIvfError::TooManyCentroids(10))));
    }

    #[tokio::test]
    async fn invalid_nprobe_surfaces_as_error() {
        let scorer = BnTmIvfScorer::new();
        let vectors = random_unit_vectors(20, 16, 19);
        let result = scorer.upload_cluster(&cfg(4, 5, false), &vectors).await;
        assert!(matches!(
            result,
            Err(BnTmIvfError::InvalidNprobe {
                nprobe: 5,
                n_centroids: 4
            })
        ));
    }

    #[tokio::test]
    async fn dimension_mismatch_surfaces_as_error() {
        let scorer = BnTmIvfScorer::new();
        let vectors = random_unit_vectors(20, 16, 21);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(4, 2, false), &vectors)
            .await
            .unwrap();
        let q = Vector(vec![0.5_f32; 8]); // wrong dim
        let result = scorer.score(&handle, &q, 5).await;
        assert!(matches!(
            result,
            Err(BnTmIvfError::DimensionMismatch {
                query_dim: 8,
                index_dim: 16,
            })
        ));
    }

    #[tokio::test]
    async fn debug_redacts_secrets() {
        let scorer = BnTmIvfScorer::new();
        let vectors = random_unit_vectors(8, 8, 23);
        let (handle, _build) = scorer
            .upload_cluster(&cfg(2, 1, false), &vectors)
            .await
            .unwrap();
        let dbg = format!("{handle:?}");
        assert!(dbg.contains("[redacted]"));
    }

    /// Disk cache roundtrip: build, save to tempdir, reload, score
    /// must match.
    #[tokio::test]
    async fn disk_cache_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let scorer1 = BnTmIvfScorer::with_cache_dir(tmp.path().to_path_buf());
        let vectors = random_unit_vectors(40, 32, 31);
        let q = random_unit_vectors(1, 32, 32).pop().unwrap();
        let (handle1, b1) = scorer1
            .upload_cluster(&cfg(4, 2, false), &vectors)
            .await
            .unwrap();
        assert!(!b1.cache_hit, "first build is cold");
        let hits1 = scorer1.score(&handle1, &q, 5).await.unwrap();

        // Fresh scorer, same cache dir → should hit the on-disk cache.
        let scorer2 = BnTmIvfScorer::with_cache_dir(tmp.path().to_path_buf());
        let (handle2, b2) = scorer2
            .upload_cluster(&cfg(4, 2, false), &vectors)
            .await
            .unwrap();
        assert!(b2.cache_hit, "second build hits the on-disk cache");
        let hits2 = scorer2.score(&handle2, &q, 5).await.unwrap();

        let ids1: Vec<u32> = hits1.iter().map(|h| h.id).collect();
        let ids2: Vec<u32> = hits2.iter().map(|h| h.id).collect();
        assert_eq!(ids1, ids2, "cache reload must reproduce top-k IDs");
    }

    /// `score_with_breakdown` agrees with `score` on top-k IDs at full
    /// probe (recovery is structural; the per-call mask cancels). Run
    /// without verification so the test stays fast and the timing
    /// surface is unambiguous.
    #[tokio::test]
    async fn score_with_breakdown_matches_score_no_verify() {
        let scorer = BnTmIvfScorer::new();
        let vectors = random_unit_vectors(40, 32, 60);
        let n_centroids = 4;
        let (handle, _build) = scorer
            .upload_cluster(&cfg(n_centroids, n_centroids, false), &vectors)
            .await
            .unwrap();
        let q = random_unit_vectors(1, 32, 61).pop().unwrap();
        let baseline = scorer.score(&handle, &q, 5).await.unwrap();
        let (with_breakdown, timing) = scorer.score_with_breakdown(&handle, &q, 5).await.unwrap();

        let baseline_ids: Vec<u32> = baseline.iter().map(|h| h.id).collect();
        let breakdown_ids: Vec<u32> = with_breakdown.iter().map(|h| h.id).collect();
        assert_eq!(baseline_ids, breakdown_ids);
        assert_eq!(timing.verify_us, 0, "no verify when disabled");
        let _ = timing.route_us
            + timing.encode_us
            + timing.server_us
            + timing.decode_us
            + timing.merge_us;
    }

    #[tokio::test]
    async fn build_outcome_cache_hit_smoke() {
        let scorer = BnTmIvfScorer::new();
        let vectors = random_unit_vectors(40, 32, 41);
        let (_h1, b1) = scorer
            .upload_cluster(&cfg(4, 2, false), &vectors)
            .await
            .unwrap();
        assert!(!b1.cache_hit, "first build must be a cold build");
        let (_h2, b2) = scorer
            .upload_cluster(&cfg(4, 2, false), &vectors)
            .await
            .unwrap();
        assert!(b2.cache_hit, "second build must hit the in-memory cache");
        assert!(
            b2.build_duration < std::time::Duration::from_millis(100),
            "warm build was {:?}, expected < 100 ms",
            b2.build_duration
        );
    }

    /// 1D corpus designed to force uneven cluster sizes [60, 20, 15, 5]
    /// after k-means converges. Mirrors the realised-cost templates in
    /// scorer-plaintext / scorer-sap / scorer-emvp.
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
        let scorer = BnTmIvfScorer::new();
        let vectors = skewed_corpus_4_clusters();
        // verification=false to skip Protocol 2 — the cost-accounting
        // path is identical either way; faster test.
        let (handle, _) = scorer
            .upload_cluster(&cfg(4, 2, false), &vectors)
            .await
            .unwrap();

        let (_, cost_a) = scorer
            .score_with_realised_cost(&handle, &Vector(vec![0.0]), 10)
            .await
            .unwrap();
        let (_, cost_b) = scorer
            .score_with_realised_cost(&handle, &Vector(vec![30.0]), 10)
            .await
            .unwrap();

        // Per-cluster bytes = m_i × FIELD_BYTES (no s factor — BN's
        // structural advantage over EMVP).
        // Query A: probe set [60, 20] → 80 × FIELD_BYTES.
        // Query B: probe set [5, 15]  → 20 × FIELD_BYTES.
        assert_eq!(cost_a.response_bytes, 80 * FIELD_BYTES as u64);
        assert_eq!(cost_b.response_bytes, 20 * FIELD_BYTES as u64);
        assert_ne!(cost_a.response_bytes, cost_b.response_bytes);
    }

    #[tokio::test]
    async fn realised_cost_full_probe_recovers_total() {
        let scorer = BnTmIvfScorer::new();
        let vectors = skewed_corpus_4_clusters();
        let (handle, _) = scorer
            .upload_cluster(&cfg(4, 4, false), &vectors)
            .await
            .unwrap();

        let (_, cost) = scorer
            .score_with_realised_cost(&handle, &Vector(vec![15.0]), 10)
            .await
            .unwrap();
        // Σ m_i = 100; per-cluster = m_i × FIELD_BYTES.
        assert_eq!(cost.response_bytes, 100 * FIELD_BYTES as u64);
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

        let scorer = BnTmIvfScorer::new();
        let (handle, _) = scorer
            .upload_cluster(&cfg(4, 2, false), &vectors)
            .await
            .unwrap();

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

    /// bntm-IVF GPU exact-equality at full probe.
    ///
    /// Mersenne arithmetic is deterministic, so the GPU path must
    /// return the same top-k IDs as the CPU path on the same query at
    /// `nprobe = n_centroids`. Verification stays on (default) so a
    /// kernel-arithmetic divergence would surface as
    /// `BnTmIvfError::VerificationFailed` rather than a silent
    /// ranking mismatch — same defence-in-depth as the flat-BN GPU
    /// gate. Requires the rapids conda env (`docs/envs/README.md`).
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_full_probe_matches_cpu_exact() {
        let vectors = random_unit_vectors(60, 32, 909);
        let q = random_unit_vectors(1, 32, 910).pop().unwrap();
        let k = 10;
        let n_centroids = 4;

        let mut cfg_cpu = cfg(n_centroids, n_centroids, false);
        cfg_cpu.device = Device::Cpu;
        let mut cfg_gpu = cfg(n_centroids, n_centroids, false);
        cfg_gpu.device = Device::Gpu;

        let scorer_cpu = BnTmIvfScorer::new();
        let scorer_gpu = BnTmIvfScorer::new();
        let (h_cpu, _) = scorer_cpu.upload_cluster(&cfg_cpu, &vectors).await.unwrap();
        let (h_gpu, _) = scorer_gpu.upload_cluster(&cfg_gpu, &vectors).await.unwrap();

        let hits_cpu = scorer_cpu.score(&h_cpu, &q, k).await.unwrap();
        let hits_gpu = scorer_gpu.score(&h_gpu, &q, k).await.unwrap();
        assert_eq!(hits_cpu.len(), k);
        assert_eq!(hits_gpu.len(), k);
        let ids_cpu: std::collections::HashSet<u32> = hits_cpu.iter().map(|h| h.id).collect();
        let ids_gpu: std::collections::HashSet<u32> = hits_gpu.iter().map(|h| h.id).collect();
        assert_eq!(
            ids_cpu, ids_gpu,
            "GPU full-probe top-{k} ids must match CPU exactly under Mersenne arithmetic"
        );
    }

    /// Structural invariant: with `nprobe < n_centroids` every
    /// returned hit comes from one of the probed clusters' members.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_partial_probe_returns_subset_of_probed_clusters() {
        let vectors = random_unit_vectors(60, 32, 911);
        let q = random_unit_vectors(1, 32, 912).pop().unwrap();
        let k = 10;
        let n_centroids = 4;
        let nprobe = 2;

        let mut config = cfg(n_centroids, nprobe, false);
        config.device = Device::Gpu;

        let scorer = BnTmIvfScorer::new();
        let (handle, _) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        let hits = scorer.score(&handle, &q, k).await.unwrap();

        let probe_set = ivf_index::ivf::probe_route(&handle.index, &q.0, nprobe);
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

    /// Verification still passes on the GPU path with `verification_enabled = true`.
    /// Catches kernel divergence as a `VerificationFailed` rather than
    /// a silent ranking mismatch at sweep time.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_score_with_verification_passes_on_honest_server() {
        let vectors = random_unit_vectors(20, 32, 913);
        let q = random_unit_vectors(1, 32, 914).pop().unwrap();

        let mut config = cfg(4, 4, true);
        config.device = Device::Gpu;

        let scorer = BnTmIvfScorer::new();
        let (handle, _) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        let hits = scorer.score(&handle, &q, 5).await.unwrap();
        assert_eq!(hits.len(), 5);
    }
}
