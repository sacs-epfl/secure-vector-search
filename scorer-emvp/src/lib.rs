use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use rand::RngExt;
use scorer_core::{
    BuildOutcome, CommunicationCost, Device, Hit, ProgressReporter, Scorer, Vector, cache,
};

// Public so `examples/emvp_compute_products_simd_bench.rs` can reach
// `compute_products_avx512`. The scalar `compute_products` was already
// `pub fn`; the SIMD sibling lives at the same level. No external API
// on scorer-emvp depends on the privacy of this module.
pub mod crypto;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
pub mod ivf;
pub mod params;

pub use ivf::{EmvpIvfConfig, EmvpIvfError, EmvpIvfHandle, EmvpIvfScorer};
pub use params::{EmvpParams, Q, SEC128};

/// Encrypted single-cluster state shared between the flat scorer
/// (single cluster) and the IVF scorer (`Vec<EncryptedCluster>`).
/// On-disk byte layout matches the original flat-EMVP cache so the
/// flat-cache path is a straight refactor with no format change.
#[derive(Clone)]
pub(crate) struct EncryptedCluster {
    /// M̂ = M_0 · D + R, row-major m × n field elements. Storage is
    /// either an owned `Vec<u64>` (small corpora / tests / flat path)
    /// or a mmap-backed slice into the on-disk cluster cache (large-
    /// scale IVF builds where the per-cluster m_hat sum exceeds
    /// physical RAM).
    pub m_hat: Mhat,
    /// Seed for the blinding matrix R (per-cluster).
    pub r_seed: [u8; 32],
    /// Number of vectors in this cluster (row count of M).
    pub m: usize,
}

/// Storage backend for an encrypted cluster's `m_hat` matrix.
///
/// `Owned` keeps the `m × n` `u64`s in a heap-allocated `Vec`. Used
/// for the flat scorer and for the IVF path when no cache directory is
/// configured.
///
/// `Mmap` stores a byte-offset into a memory-mapped on-disk cluster
/// cache file; `as_slice()` returns a zero-copy `&[u64]` backed by the
/// page cache. Used for the IVF path on cache-write or cache-load so
/// the in-memory handle never holds the corpus-sized encrypted matrix
/// (tens of GB at multi-million-vector scale, exceeding physical RAM).
/// Per-cluster m_hat regions are 8-byte aligned by the cache file
/// layout (header + 4-byte pad + per-cluster header is 40 bytes,
/// m_hat is `m × n × 8 B`); the unsafe cast in `as_slice()` is sound
/// under that invariant.
#[derive(Clone)]
pub(crate) enum Mhat {
    Owned(Arc<Vec<u64>>),
    Mmap {
        mmap: Arc<memmap2::Mmap>,
        byte_offset: usize,
        len_u64s: usize,
    },
}

impl Mhat {
    pub fn as_slice(&self) -> &[u64] {
        match self {
            Mhat::Owned(v) => v.as_slice(),
            Mhat::Mmap {
                mmap,
                byte_offset,
                len_u64s,
            } => {
                let bytes = &mmap[*byte_offset..*byte_offset + *len_u64s * 8];
                debug_assert_eq!(
                    bytes.as_ptr() as usize % 8,
                    0,
                    "m_hat mmap region must be 8-byte aligned (file format invariant)"
                );
                // SAFETY: u64 has no invalid bit patterns; mmap regions
                // are 8-byte aligned by cache file layout construction.
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, *len_u64s) }
            }
        }
    }
}

impl std::fmt::Debug for Mhat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mhat::Owned(v) => write!(f, "Mhat::Owned(len={})", v.len()),
            Mhat::Mmap { len_u64s, .. } => write!(f, "Mhat::Mmap(len={len_u64s})"),
        }
    }
}

/// Per-cluster configuration for EmvpScorer.
pub struct EmvpConfig {
    /// Seed for the D matrix (same seed → same D across clusters).
    pub key_seed: [u8; 32],
    /// Optional progress reporter. `on_encrypt(i)` fires per encrypted row
    /// (rayon worker threads); `on_build_complete` fires once when M̂ is
    /// ready (whether built or loaded from cache).
    pub progress: Option<Arc<dyn ProgressReporter>>,
    /// Compute substrate. `Cpu` runs the rayon-parallel
    /// `compute_products`. `Gpu` requires the `gpu` Cargo feature;
    /// without it `upload_cluster` returns
    /// [`EmvpError::GpuFeatureNotEnabled`]. EMVP encryption + decoding
    /// stay on host either way; only the server-side block-GEMV
    /// (`compute_products`) moves to device.
    pub device: Device,
    /// VRAM budget for the streaming cluster store on GPU runs. `None`
    /// (default) auto-detects 80 % of free VRAM via NVML at handle
    /// build. `Some(b)` pins a hard byte cap. Ignored on CPU runs.
    pub vram_budget_bytes: Option<u64>,
}

/// Opaque per-cluster state.
pub struct EmvpHandle {
    cluster: EncryptedCluster,
    /// D matrix (ell0 × n), derived from key_seed. Shared across clusters
    /// in the IVF case; held here as a single Arc for the flat case so the
    /// score path doesn't have to know the difference.
    d_mat: Arc<Vec<u64>>,
    /// Original embedding dimension (before padding to ell0).
    dim: usize,
    inner: HandleInner,
}

pub(crate) enum HandleInner {
    Cpu,
    /// Device-resident M̂; per-query kernel launch for
    /// `compute_products`. Decoding stays on CPU. Boxed because the
    /// streaming `FlatGpuState` carries the host-side M̂ + LRU +
    /// sampler that push the variant size well above `Cpu`'s
    /// unit-variant — clippy's `large_enum_variant` lint trips
    /// otherwise.
    #[cfg(feature = "gpu")]
    Gpu(Box<gpu::FlatGpuState>),
}

impl fmt::Debug for EmvpHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device = match &self.inner {
            HandleInner::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(_) => "gpu",
        };
        f.debug_struct("EmvpHandle")
            .field("device", &device)
            .field("m", &self.cluster.m)
            .field("dim", &self.dim)
            .field("r_seed", &"[redacted]")
            .finish()
    }
}

impl EmvpHandle {
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
pub enum EmvpError {
    #[error("dimension too large: vectors have dim {actual}, ell0 = {ell0}")]
    DimensionTooLarge { actual: usize, ell0: usize },
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

// (corpus_len, dim, fingerprint)
type CacheKey = (usize, usize, String);
// (m_hat, r_seed, d_mat). D matrix is cached too: it depends only on key_seed
// (already part of the fingerprint), and regenerating it from PRG is O(ell0·n)
// — ~200 ms at SEC128 (1024 × 1292) every call without caching.
type CacheEntry = (Arc<Vec<u64>>, [u8; 32], Arc<Vec<u64>>);

/// EMVP scorer with optional disk cache for M̂.
///
/// The encrypted cluster (M̂) is expensive to compute (O(m·ell0·n) field muls).
/// `with_cache_dir` persists M̂ + r_seed to disk so subsequent runs skip
/// encryption. Cache filename: `.emvp-cache-<16hex>.bin`, where the hex
/// is a fingerprint over the parameter tuple.
pub struct EmvpScorer {
    cache: Mutex<HashMap<CacheKey, CacheEntry>>,
    cache_dir: Option<PathBuf>,
}

impl EmvpScorer {
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

impl Default for EmvpScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Cache I/O
// ---------------------------------------------------------------------------
//
// File layout: 20-byte cache header (see scorer_core::cache) followed by
// the existing payload `[m: u64][n: u64][r_seed: 32 bytes][m*n u64s]`.

pub(crate) fn save_emvp_cache(
    path: &std::path::Path,
    m_hat: &[u64],
    r_seed: &[u8; 32],
    n: usize,
    parts: &[&[u8]],
) -> io::Result<()> {
    let m = m_hat.len() / n;
    let tmp = path.with_extension("bin.tmp");
    let file = std::fs::File::create(&tmp)?;
    let mut w = io::BufWriter::new(file);
    cache::write_header(&mut w, parts)?;
    w.write_all(&(m as u64).to_le_bytes())?;
    w.write_all(&(n as u64).to_le_bytes())?;
    w.write_all(r_seed)?;
    for &v in m_hat {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    std::fs::rename(&tmp, path)
}

pub(crate) fn load_emvp_cache(
    path: &std::path::Path,
    n: usize,
    parts: &[&[u8]],
) -> Option<(Vec<u64>, [u8; 32])> {
    let mut file = std::fs::File::open(path).ok()?;
    if !cache::verify_header(&mut file, parts).ok()? {
        return None;
    }
    let mut buf8 = [0u8; 8];

    file.read_exact(&mut buf8).ok()?;
    let m = u64::from_le_bytes(buf8) as usize;

    file.read_exact(&mut buf8).ok()?;
    let stored_n = u64::from_le_bytes(buf8) as usize;
    if stored_n != n {
        return None;
    }

    let mut r_seed = [0u8; 32];
    file.read_exact(&mut r_seed).ok()?;

    let mut m_hat = vec![0u64; m * n];
    for v in m_hat.iter_mut() {
        file.read_exact(&mut buf8).ok()?;
        *v = u64::from_le_bytes(buf8);
    }
    Some((m_hat, r_seed))
}

// ---------------------------------------------------------------------------
// Score → Hit conversion (shared by CPU and GPU paths, flat and IVF)
// ---------------------------------------------------------------------------

pub(crate) fn top_k_from_scores(scores: Vec<f32>, k_capped: usize) -> Vec<Hit> {
    let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    indexed.truncate(k_capped);
    indexed
        .into_iter()
        .map(|(id, score)| Hit {
            id: id as u32,
            score,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Scorer impl
// ---------------------------------------------------------------------------

#[async_trait]
impl Scorer for EmvpScorer {
    type Config = EmvpConfig;
    type ClusterHandle = EmvpHandle;
    type Error = EmvpError;

    async fn upload_cluster(
        &self,
        config: &EmvpConfig,
        vectors: &[Vector],
    ) -> Result<(EmvpHandle, BuildOutcome), EmvpError> {
        let build_start = Instant::now();
        let params = &SEC128;
        let ell0 = params.ell0;
        let n = params.n;
        let dim = vectors.first().map_or(0, |v| v.0.len());

        if dim > ell0 {
            return Err(EmvpError::DimensionTooLarge { actual: dim, ell0 });
        }

        let key_seed = config.key_seed;
        let m = vectors.len();
        let cache_parts: [Vec<u8>; 4] = [
            b"emvp-flat".to_vec(),
            (m as u64).to_le_bytes().to_vec(),
            (dim as u64).to_le_bytes().to_vec(),
            key_seed.to_vec(),
        ];
        let parts_refs: Vec<&[u8]> = cache_parts.iter().map(|v| v.as_slice()).collect();
        let fp = cache::fingerprint(&parts_refs);
        let cache_key: CacheKey = (m, dim, fp.clone());
        let progress = config.progress.clone();

        let disk_path = self
            .cache_dir
            .as_deref()
            .map(|dir| dir.join(format!(".emvp-cache-{fp}.bin")));

        let cached = self.cache.lock().unwrap().get(&cache_key).cloned();
        let in_memory_hit = cached.is_some();
        let mut disk_hit = false;

        let (m_hat_arc, r_seed, d_mat_arc) = if let Some(entry) = cached {
            if let Some(ref p) = progress {
                p.on_build_complete();
            }
            entry
        } else {
            let from_disk = if let Some(ref path) = disk_path {
                if path.exists() {
                    let p = path.clone();
                    eprintln!("Loading cached EMVP index from {p:?}…");
                    let parts_owned = cache_parts.clone();
                    tokio::task::spawn_blocking(move || {
                        let refs: Vec<&[u8]> = parts_owned.iter().map(|v| v.as_slice()).collect();
                        load_emvp_cache(&p, SEC128.n, &refs)
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

            let (m_hat_vec, r_seed) = if let Some((mh, rs)) = from_disk {
                disk_hit = true;
                if let Some(ref p) = progress {
                    p.on_build_complete();
                }
                (mh, rs)
            } else {
                let r_seed: [u8; 32] = rand::rng().random();

                // Clone the corpus once into an owned Vec<Vector> for
                // the spawn_blocking move (unavoidable given
                // Scorer::upload_cluster takes &[Vector] and
                // spawn_blocking needs 'static). encrypt_cluster
                // quantises and zero-pads to ell0 per-row internally,
                // so no full upcast or quantised matrix is materialised
                // here. Zero-padding to ell0 is handled per-row inside
                // encrypt_cluster.
                let data: Vec<Vector> = vectors.to_vec();

                let save_path = disk_path.clone();
                let save_parts = cache_parts.clone();
                let progress_for_blocking = progress.clone();
                let (m_hat_vec, _) = tokio::task::spawn_blocking(move || {
                    let d_mat = crypto::generate_d(key_seed);
                    let row_refs: Vec<&[f32]> = data.iter().map(|v| v.0.as_slice()).collect();
                    let on_row: Option<Arc<dyn Fn(usize) + Send + Sync>> =
                        progress_for_blocking.as_ref().map(|p| {
                            let p = Arc::clone(p);
                            Arc::new(move |i: usize| p.on_encrypt(i))
                                as Arc<dyn Fn(usize) + Send + Sync>
                        });
                    let m_hat = crypto::encrypt_cluster(
                        &d_mat,
                        r_seed,
                        &row_refs,
                        on_row
                            .as_deref()
                            .map(|cb| cb as &(dyn Fn(usize) + Send + Sync)),
                    );
                    drop(row_refs);
                    drop(data);
                    if let Some(ref p) = progress_for_blocking {
                        p.on_build_complete();
                    }
                    if let Some(ref path) = save_path {
                        let refs: Vec<&[u8]> = save_parts.iter().map(|v| v.as_slice()).collect();
                        if let Err(e) = save_emvp_cache(path, &m_hat, &r_seed, n, &refs) {
                            eprintln!("Warning: could not save EMVP cache to {path:?}: {e}");
                        }
                    }
                    (m_hat, d_mat)
                })
                .await
                .map_err(|_| EmvpError::SpawnPanic)?;

                (m_hat_vec, r_seed)
            };

            let arc = Arc::new(m_hat_vec);
            let d_mat_vec = tokio::task::spawn_blocking(move || crypto::generate_d(key_seed))
                .await
                .map_err(|_| EmvpError::SpawnPanic)?;
            let d_arc = Arc::new(d_mat_vec);
            self.cache
                .lock()
                .unwrap()
                .insert(cache_key, (Arc::clone(&arc), r_seed, Arc::clone(&d_arc)));
            (arc, r_seed, d_arc)
        };

        // Branch on substrate. Encryption + D generation above are
        // device-agnostic; only the per-query `compute_products`
        // server-side matvec moves to GPU. We upload M̂ to the device
        // once at handle construction so the hot-path query latency is
        // bounded by H2D q̂ + kernel + D2H partials.
        let inner = match config.device {
            Device::Cpu => HandleInner::Cpu,
            Device::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    let m_hat_for_gpu = Arc::clone(&m_hat_arc);
                    let vram_budget = config.vram_budget_bytes;
                    let state = tokio::task::spawn_blocking(move || {
                        gpu::FlatGpuState::build(&m_hat_for_gpu, m, vram_budget)
                    })
                    .await
                    .map_err(|_| EmvpError::SpawnPanic)??;
                    HandleInner::Gpu(Box::new(state))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(EmvpError::GpuFeatureNotEnabled);
                }
            }
        };

        let outcome = BuildOutcome {
            cache_hit: in_memory_hit || disk_hit,
            build_duration: build_start.elapsed(),
        };
        Ok((
            EmvpHandle {
                cluster: EncryptedCluster {
                    m_hat: Mhat::Owned(m_hat_arc),
                    r_seed,
                    m,
                },
                d_mat: d_mat_arc,
                dim,
                inner,
            },
            outcome,
        ))
    }

    async fn score(
        &self,
        handle: &EmvpHandle,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<Hit>, EmvpError> {
        let params = &SEC128;
        let ell0 = params.ell0;
        let k_dim = params.k;

        if query.0.len() > ell0 {
            return Err(EmvpError::DimensionTooLarge {
                actual: query.0.len(),
                ell0,
            });
        }

        // Upcast and pad query to ell0 on the async task.
        let mut q_f64: Vec<f64> = query.0.iter().map(|&x| x as f64).collect();
        q_f64.resize(ell0, 0.0);

        // Generate random u (k_dim field elements) on the async task.
        // Scoped block: ThreadRng is !Send; must not be held across the spawn_blocking await.
        let u: Vec<u64> = {
            let mut rng = rand::rng();
            (0..k_dim)
                .map(|_| rng.random::<u64>() % crate::params::P)
                .collect()
        };

        let m_hat = handle.cluster.m_hat.clone();
        let d_mat = Arc::clone(&handle.d_mat);
        let r_seed = handle.cluster.r_seed;
        let m = handle.cluster.m;
        let k_capped = k.min(m);

        // Per-substrate dispatch. The CPU branch keeps the original
        // self-contained spawn_blocking (encode → compute → decode →
        // sort). The GPU branch encodes on host, runs `compute_products`
        // on device, then decodes on host — same end-to-end shape, just
        // with the heavy block-GEMV offloaded.
        let hits = match &handle.inner {
            HandleInner::Cpu => tokio::task::spawn_blocking(move || {
                let q_hat = crypto::encode_query(&d_mat, &q_f64, &u);
                let partials = crypto::compute_products(m_hat.as_slice(), m, &q_hat);
                let scores = crypto::decode_scores(m, r_seed, &q_hat, &partials);

                top_k_from_scores(scores, k_capped)
            })
            .await
            .map_err(|_| EmvpError::SpawnPanic)?,
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                // Decode stays on CPU; only the server-side matvec is
                // offloaded. `state.compute_products` does the H2D q̂
                // copy + kernel launch + D2H partials copy under a
                // per-handle Mutex.
                let q_hat =
                    tokio::task::spawn_blocking(move || crypto::encode_query(&d_mat, &q_f64, &u))
                        .await
                        .map_err(|_| EmvpError::SpawnPanic)?;
                let partials = state.compute_products(&q_hat)?;
                tokio::task::spawn_blocking(move || {
                    let scores = crypto::decode_scores(m, r_seed, &q_hat, &partials);
                    top_k_from_scores(scores, k_capped)
                })
                .await
                .map_err(|_| EmvpError::SpawnPanic)?
            }
        };

        Ok(hits)
    }

    fn communication_cost(&self, handle: &EmvpHandle, _k: usize) -> CommunicationCost {
        let params = &SEC128;
        let query_bytes = (params.n * crate::params::FIELD_BYTES) as u64;
        let cluster_response_bytes = crypto::cluster_response_bytes(handle.cluster.m);
        let setup_bytes = crypto::setup_bytes(handle.cluster.m);
        // GPU bandwidth proxy = bytes the matrix kernel streams across
        // per query. The `compute_products` matvec reads the full
        // `m × n` encrypted matrix M_hat each scoring call, so the
        // proxy is `m × n × FIELD_BYTES` (the matrix touched), not the
        // `m × s × FIELD_BYTES` server output shape (which would
        // under-count matrix work by a factor of n/s ≈ 17×).
        let effective_bytes_per_query =
            (handle.cluster.m * params.n * crate::params::FIELD_BYTES) as u64;
        CommunicationCost {
            query_bytes,
            response_bytes: cluster_response_bytes,
            cluster_response_bytes,
            setup_bytes,
            pre_query_offline_up_bytes: 0,
            pre_query_offline_down_bytes: 0,
            effective_bytes_per_query,
        }
    }

    /// Batched-scoring override. Per query: pad+upcast, generate `u`
    /// (per-call !Send), encode → `q_hat`. Across the batch: one
    /// [`crypto::compute_products_batch`] call reads `m_hat` once and
    /// produces per-query partials in the same layout `compute_products`
    /// would return, then per-query [`crypto::decode_scores`] +
    /// `top_k_from_scores`. Per-(si, i) inner fold is byte-identical to
    /// the per-query path, and `decode_scores` is a deterministic
    /// function of `(m, r_seed, q_hat, partials)`, so each `out[i]` is
    /// bit-equal to `score(handle, &queries[i], k)` despite per-query
    /// random `u` (the noise cancels in decode per EMVP correctness).
    /// d_mat stays Arc-shared via the handle — no double-cache.
    /// GPU handles fall back to the inline sequential loop.
    async fn score_batch(
        &self,
        handle: &EmvpHandle,
        queries: &[Vector],
        k: usize,
    ) -> Result<Vec<Vec<Hit>>, EmvpError> {
        let params = &SEC128;
        let ell0 = params.ell0;
        let k_dim = params.k;

        for q in queries {
            if q.0.len() > ell0 {
                return Err(EmvpError::DimensionTooLarge {
                    actual: q.0.len(),
                    ell0,
                });
            }
        }

        // Per-query: upcast + pad to ell0.
        let queries_f64: Vec<Vec<f64>> = queries
            .iter()
            .map(|q| {
                let mut v: Vec<f64> = q.0.iter().map(|&x| x as f64).collect();
                v.resize(ell0, 0.0);
                v
            })
            .collect();

        // Per-query u: ThreadRng !Send — pre-generate before spawn_blocking.
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

        let m_hat = handle.cluster.m_hat.clone();
        let d_mat = Arc::clone(&handle.d_mat);
        let r_seed = handle.cluster.r_seed;
        let m = handle.cluster.m;
        let k_capped = k.min(m);

        match &handle.inner {
            HandleInner::Cpu => {
                let hits = tokio::task::spawn_blocking(move || {
                    // Per-query encode.
                    let q_hats: Vec<Vec<u64>> = queries_f64
                        .iter()
                        .zip(&us)
                        .map(|(qf, u)| crypto::encode_query(&d_mat, qf, u))
                        .collect();
                    let qhat_refs: Vec<&[u64]> = q_hats.iter().map(|q| q.as_slice()).collect();

                    // Single batched matvec: m_hat read once across queries.
                    let partials_batch =
                        crypto::compute_products_batch(m_hat.as_slice(), m, &qhat_refs);

                    // Per-query decode + top-k.
                    q_hats
                        .iter()
                        .zip(partials_batch.iter())
                        .map(|(q_hat, partials)| {
                            let scores = crypto::decode_scores(m, r_seed, q_hat, partials);
                            top_k_from_scores(scores, k_capped)
                        })
                        .collect::<Vec<_>>()
                })
                .await
                .map_err(|_| EmvpError::SpawnPanic)?;
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

/// Per-query substep timings reported by [`EmvpScorer::score_with_breakdown`].
/// Canonical taxonomy fields: `encode` (D-mask + u-blind), `server`
/// (matvec M̂·q̂), `decode` (decode coded scalars + top-k). EMVP flat
/// has no `route` or `merge` phase.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmvpTiming {
    pub encode_us: u64,
    pub server_us: u64,
    pub decode_us: u64,
}

impl EmvpScorer {
    /// Off-hot-path equivalent of [`Scorer::score`] with per-substep
    /// timings. Per-call `Instant::now()` pairs add nanosecond-scale
    /// overhead per substep; do not use for throughput benchmarks.
    pub async fn score_with_breakdown(
        &self,
        handle: &EmvpHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, EmvpTiming), EmvpError> {
        let params = &SEC128;
        let ell0 = params.ell0;
        let k_dim = params.k;

        if query.0.len() > ell0 {
            return Err(EmvpError::DimensionTooLarge {
                actual: query.0.len(),
                ell0,
            });
        }

        let mut q_f64: Vec<f64> = query.0.iter().map(|&x| x as f64).collect();
        q_f64.resize(ell0, 0.0);

        let u: Vec<u64> = {
            let mut rng = rand::rng();
            (0..k_dim)
                .map(|_| rng.random::<u64>() % crate::params::P)
                .collect()
        };

        let d_mat = Arc::clone(&handle.d_mat);
        let r_seed = handle.cluster.r_seed;
        let m = handle.cluster.m;
        let k_capped = k.min(m);

        // Dispatch on inner. Both branches measure encode (q̂
        // construction) + server (matvec) + decode (subtract R, lift,
        // top-k). On GPU, server_us spans the H2D q̂ + kernel + D2H
        // partials round-trip; decode stays on CPU.
        let result = match &handle.inner {
            HandleInner::Cpu => {
                let m_hat = handle.cluster.m_hat.clone();
                tokio::task::spawn_blocking(move || {
                    let t = Instant::now();
                    let q_hat = crypto::encode_query(&d_mat, &q_f64, &u);
                    let encode_us = t.elapsed().as_micros() as u64;

                    let t = Instant::now();
                    let partials = crypto::compute_products(m_hat.as_slice(), m, &q_hat);
                    let server_us = t.elapsed().as_micros() as u64;

                    let t = Instant::now();
                    let scores = crypto::decode_scores(m, r_seed, &q_hat, &partials);
                    let hits = top_k_from_scores(scores, k_capped);
                    let decode_us = t.elapsed().as_micros() as u64;

                    (
                        hits,
                        EmvpTiming {
                            encode_us,
                            server_us,
                            decode_us,
                        },
                    )
                })
                .await
                .map_err(|_| EmvpError::SpawnPanic)?
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                // Encoding is a few microseconds — small enough to
                // run inline (not spawn_blocking) on the async task.
                // The host-side timer pair under encode_us captures
                // it cleanly.
                let encode_t = Instant::now();
                let q_hat = crypto::encode_query(&d_mat, &q_f64, &u);
                let encode_us = encode_t.elapsed().as_micros() as u64;

                let server_t = Instant::now();
                let partials = state.compute_products(&q_hat)?;
                let server_us = server_t.elapsed().as_micros() as u64;

                let decode_t = Instant::now();
                let scores = crypto::decode_scores(m, r_seed, &q_hat, &partials);
                let hits = top_k_from_scores(scores, k_capped);
                let decode_us = decode_t.elapsed().as_micros() as u64;

                (
                    hits,
                    EmvpTiming {
                        encode_us,
                        server_us,
                        decode_us,
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
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn random_vectors(n: usize, d: usize, seed: u64) -> Vec<Vector> {
        use rand::RngExt;
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

    fn cfg() -> EmvpConfig {
        EmvpConfig {
            key_seed: [42u8; 32],
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        }
    }

    #[tokio::test]
    async fn upload_and_score_roundtrip() {
        let scorer = EmvpScorer::new();
        let vectors = random_vectors(10, 32, 1);
        let (handle, _build) = scorer.upload_cluster(&cfg(), &vectors).await.unwrap();
        assert_eq!(handle.cluster.m, 10);

        let query = random_vectors(1, 32, 99).remove(0);
        let hits = scorer.score(&handle, &query, 3).await.unwrap();
        assert_eq!(hits.len(), 3);
        assert!(hits[0].score >= hits[1].score);
        assert!(hits[1].score >= hits[2].score);
        assert!(hits.iter().all(|h| h.id < 10));
    }

    #[tokio::test]
    async fn score_caps_k_at_cluster_size() {
        let scorer = EmvpScorer::new();
        let vectors = random_vectors(3, 16, 2);
        let (handle, _build) = scorer.upload_cluster(&cfg(), &vectors).await.unwrap();
        let query = random_vectors(1, 16, 100).remove(0);
        let hits = scorer.score(&handle, &query, 10).await.unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[tokio::test]
    async fn communication_cost_formula() {
        let scorer = EmvpScorer::new();
        let m = 5;
        let vectors = random_vectors(m, 8, 3);
        let (handle, _build) = scorer.upload_cluster(&cfg(), &vectors).await.unwrap();
        let cost = scorer.communication_cost(&handle, 10);
        let params = &SEC128;
        assert_eq!(
            cost.query_bytes,
            (params.n * crate::params::FIELD_BYTES) as u64
        );
        assert_eq!(
            cost.cluster_response_bytes,
            (m * params.s * crate::params::FIELD_BYTES) as u64
        );
        assert_eq!(
            cost.setup_bytes,
            (m * params.n * crate::params::FIELD_BYTES) as u64
        );
        assert_eq!(cost.response_bytes, cost.cluster_response_bytes);
        assert_eq!(cost.pre_query_offline_up_bytes, 0);
        assert_eq!(cost.pre_query_offline_down_bytes, 0);
        // GPU proxy: matrix work the server kernel streams per query —
        // full `m × n` encrypted matrix M_hat.
        assert_eq!(
            cost.effective_bytes_per_query,
            (m * params.n * crate::params::FIELD_BYTES) as u64
        );
    }

    #[tokio::test]
    async fn exact_recall_nearest_neighbour() {
        let scorer = EmvpScorer::new();
        let vectors = random_vectors(20, 64, 7);
        let (handle, _build) = scorer.upload_cluster(&cfg(), &vectors).await.unwrap();

        for qi in 0..5u64 {
            let query = random_vectors(1, 64, 200 + qi).remove(0);

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
                "query {qi}: EMVP returned id={} but ground truth is id={gt_id}",
                hits[0].id
            );
        }
    }

    #[tokio::test]
    async fn disk_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let vectors = random_vectors(8, 16, 5);
        let query = random_vectors(1, 16, 300).remove(0);

        // First call: compute and save.
        let (hits_first, b1) = {
            let scorer = EmvpScorer::with_cache_dir(dir.path());
            let (handle, build) = scorer.upload_cluster(&cfg(), &vectors).await.unwrap();
            (scorer.score(&handle, &query, 3).await.unwrap(), build)
        };
        assert!(!b1.cache_hit, "first build is cold");

        // Second call: load from disk.
        let (hits_second, b2) = {
            let scorer = EmvpScorer::with_cache_dir(dir.path());
            let (handle, build) = scorer.upload_cluster(&cfg(), &vectors).await.unwrap();
            (scorer.score(&handle, &query, 3).await.unwrap(), build)
        };
        assert!(b2.cache_hit, "second build hits the on-disk cache");

        // Results must be identical.
        assert_eq!(hits_first.len(), hits_second.len());
        for (a, b) in hits_first.iter().zip(&hits_second) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.score, b.score);
        }
    }

    /// EMVP recovers the inner-product structurally regardless of `u`,
    /// so `score` and `score_with_breakdown` produce the same top-k
    /// IDs even though each call samples a fresh `u`.
    #[tokio::test]
    async fn score_with_breakdown_matches_score() {
        let scorer = EmvpScorer::new();
        let vectors = random_vectors(20, 64, 30);
        let (handle, _build) = scorer.upload_cluster(&cfg(), &vectors).await.unwrap();
        let query = random_vectors(1, 64, 31).remove(0);

        let baseline = scorer.score(&handle, &query, 5).await.unwrap();
        let (with_breakdown, timing) = scorer
            .score_with_breakdown(&handle, &query, 5)
            .await
            .unwrap();

        let baseline_ids: Vec<u32> = baseline.iter().map(|h| h.id).collect();
        let breakdown_ids: Vec<u32> = with_breakdown.iter().map(|h| h.id).collect();
        assert_eq!(baseline_ids, breakdown_ids);
        let _ = timing.encode_us + timing.server_us + timing.decode_us;
    }

    /// Per-query result from the batched override is bit-equal to a
    /// sequential `score()` call on the same input. EMVP's noise terms
    /// cancel exactly in `decode_scores` (the `block_sum - r_dot` field
    /// element is the encoded plaintext distance regardless of the
    /// per-call random `u`), so end-to-end scores are bit-equal even
    /// though `score` and `score_batch` each sample fresh `u`s. Uses
    /// integer-field tolerance (bit-equal, no rel-tol).
    #[tokio::test]
    async fn emvp_score_batch_matches_score() {
        let scorer = EmvpScorer::new();
        let vectors = random_vectors(32, 64, 47);
        let (handle, _build) = scorer.upload_cluster(&cfg(), &vectors).await.unwrap();

        let queries = random_vectors(64, 64, 48);
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
        let scorer = EmvpScorer::new();
        let vectors = random_vectors(6, 16, 11);
        let (_h1, b1) = scorer.upload_cluster(&cfg(), &vectors).await.unwrap();
        assert!(!b1.cache_hit, "first build must be a cold build");
        let (_h2, b2) = scorer.upload_cluster(&cfg(), &vectors).await.unwrap();
        assert!(b2.cache_hit, "second build must hit the in-memory cache");
        assert!(
            b2.build_duration < std::time::Duration::from_millis(100),
            "warm build was {:?}, expected < 100 ms",
            b2.build_duration
        );
    }

    /// Flat-EMVP GPU exact-equality gate.
    ///
    /// Mersenne field arithmetic is deterministic (no fp rounding,
    /// no catastrophic cancellation), so the GPU kernel must produce
    /// *exactly* the same top-k as the CPU path on the same query.
    /// Verifies: (a) the Mersenne reduction matches
    /// `crypto::compute_products` to the bit, (b) the H2D / D2H
    /// transfers preserve u64 ordering, (c) the (i, si) → output
    /// linearisation matches the CPU `s × m` block-major layout.
    /// Requires a working CUDA toolchain and GPU.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_matches_cpu_exact() {
        let vectors = random_vectors(64, 64, 909);
        let query = random_vectors(1, 64, 910).remove(0);
        let k = 10;

        let scorer_cpu = EmvpScorer::new();
        let scorer_gpu = EmvpScorer::new();

        let cfg_cpu = EmvpConfig {
            key_seed: [42u8; 32],
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let cfg_gpu = EmvpConfig {
            key_seed: [42u8; 32],
            progress: None,
            device: Device::Gpu,
            vram_budget_bytes: None,
        };

        let (h_cpu, _) = scorer_cpu.upload_cluster(&cfg_cpu, &vectors).await.unwrap();
        let (h_gpu, _) = scorer_gpu.upload_cluster(&cfg_gpu, &vectors).await.unwrap();

        // Each `score` call samples its own random `u` (the blinding
        // mask), but the underlying inner products are recovered
        // structurally — both paths return the same ranking up to
        // ties. Compare the set of returned IDs at full k to give
        // ties room; for non-tied corpora the ordering is also
        // exact, but we only assert set equality here.
        let hits_cpu = scorer_cpu.score(&h_cpu, &query, k).await.unwrap();
        let hits_gpu = scorer_gpu.score(&h_gpu, &query, k).await.unwrap();
        assert_eq!(hits_cpu.len(), k);
        assert_eq!(hits_gpu.len(), k);
        let ids_cpu: std::collections::HashSet<u32> = hits_cpu.iter().map(|h| h.id).collect();
        let ids_gpu: std::collections::HashSet<u32> = hits_gpu.iter().map(|h| h.id).collect();
        assert_eq!(
            ids_cpu, ids_gpu,
            "GPU top-{k} ids must match CPU exactly under Mersenne arithmetic"
        );
    }

    /// Kernel-correctness gate (lower-level).
    ///
    /// Direct comparison of the GPU `compute_products` partials
    /// vector against the CPU `crypto::compute_products` output for
    /// a synthetic (m_hat, q_hat) pair. Skips the full encryption /
    /// decoding pipeline so a kernel-arithmetic divergence shows up
    /// as a u64 mismatch rather than as a sorted-top-k tiebreak
    /// surprise.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_kernel_partials_match_cpu_exact() {
        use crate::params::P;
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let m: usize = 32;
        let n = SEC128.n;

        // Synthetic field elements that exercise the full F_p range
        // (not just small magnitudes), drawn from a deterministic RNG
        // for reproducibility.
        let mut rng = ChaCha20Rng::seed_from_u64(1234);
        let mut m_hat = vec![0u64; m * n];
        for v in m_hat.iter_mut() {
            *v = rng.random::<u64>() % P;
        }
        let mut q_hat = vec![0u64; n];
        for v in q_hat.iter_mut() {
            *v = rng.random::<u64>() % P;
        }

        let cpu = crypto::compute_products(&m_hat, m, &q_hat);

        // GPU path: build the FlatGpuState directly, bypassing
        // EmvpScorer. The state's `compute_products` matches the
        // CPU function's signature shape modulo the partial-output
        // ordering (s × m block-major in both).
        let gpu_state =
            tokio::task::spawn_blocking(move || gpu::FlatGpuState::build(&m_hat, m, None).unwrap())
                .await
                .unwrap();
        let q_hat_for_gpu = q_hat.clone();
        let gpu = tokio::task::spawn_blocking(move || {
            gpu_state.compute_products(&q_hat_for_gpu).unwrap()
        })
        .await
        .unwrap();

        assert_eq!(cpu.len(), gpu.len(), "partials length mismatch");
        // Find the first divergence (if any) for a more useful error
        // than `assert_eq!(cpu, gpu)` which would dump 76·32 u64s.
        for (idx, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
            assert_eq!(
                a,
                b,
                "partials[{idx}] mismatch: cpu={a}, gpu={b} (block si={}, row i={})",
                idx / m,
                idx % m
            );
        }
    }

    /// Flat-EMVP GPU breakdown agrees with score(). Validates
    /// `score_with_breakdown` accepts GPU and reports the canonical
    /// encode / server / decode substeps with non-zero timings.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_score_with_breakdown_matches_score() {
        let vectors = random_vectors(64, 64, 920);
        let query = random_vectors(1, 64, 921).remove(0);
        let k = 5;

        let scorer = EmvpScorer::new();
        let cfg_gpu = EmvpConfig {
            key_seed: [42u8; 32],
            progress: None,
            device: Device::Gpu,
            vram_budget_bytes: None,
        };
        let (handle, _build) = scorer.upload_cluster(&cfg_gpu, &vectors).await.unwrap();

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
            "server_us should cover kernel launch: {timing:?}"
        );
        assert!(
            timing.decode_us > 0,
            "decode_us should cover host decode: {timing:?}"
        );
    }
}
