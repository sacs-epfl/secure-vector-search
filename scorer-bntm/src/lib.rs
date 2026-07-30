//! Braverman–Newman trapdoored-matrix scorer.
//!
//! Implements the trapdoored-matrix scheme of Braverman & Newman,
//! *Practical Secure Delegated Linear Algebra with Trapdoored
//! Matrices* (arXiv:2502.13060v3) — specifically Protocol 1 (§6), the
//! simple (non-recursive) client protocol, faithfully: `(L, H, S)` are
//! cached at cluster initialisation and reused across queries, and δ
//! is retuned within the paper's own §7 security floor rather than
//! left at an arbitrary constant (Plan 28). No public reference
//! implementation exists to validate the concrete parameter tuple
//! against (Phase 0 search closed negative), so `(n, n_1, μ, δ, ε)`
//! are chosen conservatively rather than paper-validated — see
//! `params.rs`. Protocol 3 (§7's recursive, further-sublinear
//! construction) is not implemented; it's a materially bigger lift,
//! tracked as a follow-up plan.

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use scorer_core::{BuildOutcome, CommunicationCost, Device, Hit, ProgressReporter, Scorer, Vector};

mod crypto;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
pub mod ivf;
pub mod params;

pub use ivf::{BnTmIvfConfig, BnTmIvfError, BnTmIvfHandle, BnTmIvfScorer};

pub use params::{BnTmParams, FIELD_BYTES, P, Q};

use crypto::{
    EncryptedCluster, compute_products, compute_products_batch, decode_scores, domain_separate,
    encode_query, encrypt_cluster, fp_to_signed, generate_subspace, quantise_vector,
    verify_response,
};
#[cfg(feature = "gpu")]
use crypto::{SparseMatrix, decode_scores_with_gpu_dense, l_transpose_times_vec};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Per-cluster configuration for `BnTmScorer` (flat).
pub struct BnTmConfig {
    pub params: BnTmParams,
    /// Seed for the LPN subspace `L`. One key per scorer instance,
    /// reusable across all clusters in the IVF variant (same rationale
    /// as EMVP).
    pub key_seed: [u8; 32],
    /// Whether per-query Protocol 2 (Freivalds) verification runs.
    /// Default `false` to align all schemes under an honest-but-curious
    /// threat model; set `true` to measure the optional malicious-server
    /// detection cost. The verification-overhead measurement that
    /// originally motivated this column fired at 30 % of `latency-us`;
    /// the infrastructure remains available even though the headline
    /// figures no longer carry it.
    pub verification_enabled: bool,
    /// f32 → F_p quantisation scale: `v_q[i] = round(v[i] · Q) mod P`.
    /// Default `params::Q` (= 2^20), inherited from EMVP and confirmed
    /// by sweep — vary at run time to exercise the
    /// recall-vs-quantisation-error tradeoff. Change ⇒ disk cache
    /// invalidates (the encrypted matrix bakes Q in).
    pub quantisation_q: u64,
    pub progress: Option<Arc<dyn ProgressReporter>>,
    /// Compute substrate. `Cpu` runs the rayon-parallel
    /// `compute_products`. `Gpu` requires the `gpu` Cargo feature;
    /// without it `upload_cluster` returns
    /// [`BnTmError::GpuFeatureNotEnabled`]. Encryption, query encoding,
    /// Protocol 2 verification, and decoding stay on host either way;
    /// only the server-side `M_enc · v_enc` matvec moves to device.
    pub device: Device,
    /// VRAM budget for the streaming cluster store on GPU runs. `None`
    /// (default) auto-detects 80 % of free VRAM via NVML at handle
    /// build. `Some(b)` pins a hard byte cap. Ignored on CPU runs.
    pub vram_budget_bytes: Option<u64>,
}

impl Default for BnTmConfig {
    fn default() -> Self {
        Self {
            params: BnTmParams::Sec128,
            key_seed: [0u8; 32],
            verification_enabled: false,
            quantisation_q: Q,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// Opaque per-cluster state.
pub struct BnTmHandle {
    cluster: EncryptedCluster,
    /// `L` (n × n_1), derived from `key_seed`. Shared across clusters
    /// in the IVF case; held here as a single `Arc` for the flat case
    /// so the score path is uniform.
    l_subspace: Arc<Vec<u64>>,
    params: BnTmParams,
    /// Original embedding dimension before zero-pad to `n`.
    dim: usize,
    verification_enabled: bool,
    /// Quantisation scale baked into `cluster.m_enc` at upload time;
    /// queries must quantise at the same Q to recover M·v.
    quantisation_q: u64,
    inner: HandleInner,
}

pub(crate) enum HandleInner {
    Cpu,
    /// Device-resident `M_enc`; per-query kernel launch for
    /// `compute_products`. Verification + decoding stay on CPU. Boxed
    /// because the streaming `FlatGpuState` carries host-side
    /// M_enc/AL/H + LRU + sampler — clippy's `large_enum_variant`
    /// lint trips otherwise.
    #[cfg(feature = "gpu")]
    Gpu(Box<gpu::FlatGpuState>),
}

impl BnTmHandle {
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

impl fmt::Debug for BnTmHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device = match &self.inner {
            HandleInner::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(_) => "gpu",
        };
        f.debug_struct("BnTmHandle")
            .field("device", &device)
            .field("m", &self.cluster.m)
            .field("dim", &self.dim)
            .field("n", &self.params.n())
            .field("n_1", &self.params.n1())
            .field("verification_enabled", &self.verification_enabled)
            .field("h_seed", &"[redacted]")
            .field("l_subspace", &"[redacted]")
            .field("a_l", &"[redacted]")
            .field("m_plain", &"[redacted]")
            .field("h_mat", &"[redacted]")
            .field("s_mat", &"[redacted]")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum BnTmError {
    #[error("dimension too large: vectors have dim {actual}, n = {n}")]
    DimensionTooLarge { actual: usize, n: usize },
    #[error("f32 → F_p quantisation overflow at scale Q = {q}: |v_i| · Q would exceed P/2")]
    QuantisationOverflow { q: u64 },
    #[error("Protocol 2 verification failed at trial {trial}")]
    VerificationFailed { trial: usize },
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

// ---------------------------------------------------------------------------
// Scorer
// ---------------------------------------------------------------------------

/// Flat (single-cluster) BN scorer. The IVF wrapper lives in `ivf.rs`.
pub struct BnTmScorer;

impl BnTmScorer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BnTmScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Scorer for BnTmScorer {
    type Config = BnTmConfig;
    type ClusterHandle = BnTmHandle;
    type Error = BnTmError;

    async fn upload_cluster(
        &self,
        config: &BnTmConfig,
        vectors: &[Vector],
    ) -> Result<(BnTmHandle, BuildOutcome), BnTmError> {
        let build_start = Instant::now();
        let params = config.params;
        let n = params.n();
        let dim = vectors.first().map_or(0, |v| v.0.len());

        if dim > n {
            return Err(BnTmError::DimensionTooLarge { actual: dim, n });
        }

        // Quantise vectors on the async task before spawn_blocking; the
        // upcast is O(m·d) and shouldn't be hidden inside the blocking
        // closure (CLAUDE.md scorer checklist).
        let m = vectors.len();
        let q = config.quantisation_q;
        let m_q: Vec<u64> = vectors
            .iter()
            .flat_map(|v| quantise_vector(&v.0, n, q))
            .collect();
        debug_assert_eq!(m_q.len(), m * n);

        // Sample h_seed on the async task: ThreadRng is !Send and
        // cannot cross spawn_blocking.
        let h_seed: [u8; 32] = sample_h_seed();
        let key_seed = config.key_seed;
        let progress = config.progress.clone();
        let verification_enabled = config.verification_enabled;
        let device = config.device;

        let mut handle = tokio::task::spawn_blocking(move || {
            let l_subspace = generate_subspace(key_seed, n, params.n1());
            let cluster = encrypt_cluster(&m_q, m, &l_subspace, h_seed, params);
            if let Some(p) = progress {
                p.on_build_complete();
            }
            BnTmHandle {
                cluster,
                l_subspace: Arc::new(l_subspace),
                params,
                dim,
                verification_enabled,
                quantisation_q: q,
                inner: HandleInner::Cpu,
            }
        })
        .await
        .map_err(|_| BnTmError::SpawnPanic)?;

        // Branch on substrate. The CPU encryption above is
        // device-agnostic; only the per-query `compute_products`
        // matvec moves to GPU. Upload `M_enc` to device once at handle
        // build so the hot-path query latency is bounded by H2D
        // `v_enc` + kernel + D2H `r`.
        handle.inner = match device {
            Device::Cpu => HandleInner::Cpu,
            Device::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    let m_enc_clone = handle.cluster.m_enc.clone();
                    let a_l_clone = handle.cluster.a_l.clone();
                    let h_mat_clone = handle.cluster.h_mat.clone();
                    let s_mat_clone = Arc::clone(&handle.cluster.s_mat);
                    let cluster_m = handle.cluster.m;
                    let vram_budget = config.vram_budget_bytes;
                    let state = tokio::task::spawn_blocking(move || {
                        // (H, S) are already materialised on
                        // `handle.cluster` (Plan 28) — no need to
                        // regenerate here. H is uploaded to device; S
                        // is retained on the GpuState so the GPU decode
                        // dispatch can compute `S · v_enc` per query.
                        gpu::FlatGpuState::build(
                            m_enc_clone.as_slice(),
                            a_l_clone.as_slice(),
                            h_mat_clone.as_slice(),
                            s_mat_clone,
                            cluster_m,
                            params,
                            vram_budget,
                        )
                    })
                    .await
                    .map_err(|_| BnTmError::SpawnPanic)??;
                    HandleInner::Gpu(Box::new(state))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(BnTmError::GpuFeatureNotEnabled);
                }
            }
        };

        let outcome = BuildOutcome {
            // Flat BN has no cache; every upload is a cold build.
            cache_hit: false,
            build_duration: build_start.elapsed(),
        };
        Ok((handle, outcome))
    }

    async fn score(
        &self,
        handle: &BnTmHandle,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<Hit>, BnTmError> {
        let params = handle.params;
        let n = params.n();
        if query.0.len() > n {
            return Err(BnTmError::DimensionTooLarge {
                actual: query.0.len(),
                n,
            });
        }

        let v_q = quantise_vector(&query.0, n, handle.quantisation_q);

        // Per-query RNG seed sampled on the async task (ThreadRng is
        // !Send). The same seed is used for both encode_query and the
        // verification check, derived from independent strands so they
        // don't share entropy in a way that would weaken either.
        let q_seed: [u8; 32] = sample_h_seed();

        let cluster = handle.cluster.clone();
        let l_subspace = Arc::clone(&handle.l_subspace);
        let m = cluster.m;
        let verification_enabled = handle.verification_enabled;
        let k_capped = k.min(m);

        // GPU runs `compute_products` on device, then hands `r` back
        // for host-side verify + decode. The CPU branch keeps the
        // self-contained spawn_blocking — encoding is cheap, but
        // bundling it with the matvec keeps the timing boundary clean
        // for substep accounting.
        let r_and_encoded: Result<(Vec<u64>, crate::crypto::QueryEncoded), BnTmError> =
            match &handle.inner {
                HandleInner::Cpu => {
                    let cluster_for_encode = cluster.clone();
                    let l_for_encode = Arc::clone(&l_subspace);
                    let v_q_for_encode = v_q.clone();
                    tokio::task::spawn_blocking(move || {
                        let mut rng = ChaCha20Rng::from_seed(q_seed);
                        let encoded =
                            encode_query(&v_q_for_encode, &l_for_encode, &mut rng, params);
                        let r = compute_products(
                            cluster_for_encode.m_enc.as_slice(),
                            m,
                            n,
                            &encoded.v_enc,
                        );
                        Ok((r, encoded))
                    })
                    .await
                    .map_err(|_| BnTmError::SpawnPanic)?
                }
                #[cfg(feature = "gpu")]
                HandleInner::Gpu(state) => {
                    let l_for_encode = Arc::clone(&l_subspace);
                    let v_q_for_encode = v_q.clone();
                    let encoded = tokio::task::spawn_blocking(move || {
                        let mut rng = ChaCha20Rng::from_seed(q_seed);
                        encode_query(&v_q_for_encode, &l_for_encode, &mut rng, params)
                    })
                    .await
                    .map_err(|_| BnTmError::SpawnPanic)?;
                    let r = state.compute_products(&encoded.v_enc)?;
                    Ok((r, encoded))
                }
            };
        let (r, encoded) = r_and_encoded?;

        // For the GPU path, pre-compute the dense decode terms here
        // because `state` is borrowed from `&handle.inner` and can't
        // cross a `spawn_blocking` boundary. CPU-only builds skip the
        // entire branch.
        #[cfg(feature = "gpu")]
        let gpu_decode: Option<(Vec<u64>, Arc<SparseMatrix>)> = match &handle.inner {
            HandleInner::Cpu => None,
            HandleInner::Gpu(state) => {
                let l_for_lt = Arc::clone(&l_subspace);
                let v_for_lt = encoded.v_enc.clone();
                let n_local = n;
                let n1_local = params.n1();
                let lt_venc = tokio::task::spawn_blocking(move || {
                    l_transpose_times_vec(&l_for_lt, n_local, n1_local, &v_for_lt)
                })
                .await
                .map_err(|_| BnTmError::SpawnPanic)?;
                let dense_gpu = state.decode_dense_terms(&encoded.g, &lt_venc)?;
                Some((dense_gpu, state.s_mat_arc()))
            }
        };

        let result: Result<Vec<Hit>, BnTmError> = tokio::task::spawn_blocking(move || {
            // Optional Protocol 2 verification before decoding — the
            // server claims r, the client verifies, and only then
            // decodes. Verification stays on host even on GPU runs
            // (moving it to device is future polish).
            if verification_enabled {
                let mut v_rng = ChaCha20Rng::from_seed(domain_separate(q_seed, b"protocol2"));
                if let Err(trial) = verify_response(
                    &r,
                    cluster.m_enc.as_slice(),
                    m,
                    n,
                    &encoded.v_enc,
                    params.verification_trials(),
                    &mut v_rng,
                ) {
                    return Err(BnTmError::VerificationFailed { trial });
                }
            }

            // Decode to F_p, lift to signed, rank descending. The GPU
            // dispatch reuses the pre-computed `dense_gpu` + cached
            // `S`; CPU dispatch keeps the original self-contained
            // path with its per-query `(H, S)` regen inside
            // `decode_scores`.
            #[cfg(feature = "gpu")]
            let mv = if let Some((dense_gpu, s_mat)) = gpu_decode {
                decode_scores_with_gpu_dense(&r, &cluster, &s_mat, &encoded, &dense_gpu, params)
            } else {
                decode_scores(&r, &cluster, &l_subspace, &encoded, params)
            };
            #[cfg(not(feature = "gpu"))]
            let mv = decode_scores(&r, &cluster, &l_subspace, &encoded, params);
            let scored: Vec<(usize, i64)> = mv
                .iter()
                .enumerate()
                .map(|(id, s)| (id, fp_to_signed(*s)))
                .collect();
            let mut indexed = scored;
            indexed.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(*score));
            indexed.truncate(k_capped);
            Ok(indexed
                .into_iter()
                .map(|(id, score)| Hit {
                    id: id as u32,
                    // f32 cast for the trait surface; sufficient
                    // precision for ranking, magnitude not used
                    // cross-scheme.
                    score: score as f32,
                })
                .collect::<Vec<Hit>>())
        })
        .await
        .map_err(|_| BnTmError::SpawnPanic)?;

        result
    }

    fn communication_cost(&self, handle: &BnTmHandle, _k: usize) -> CommunicationCost {
        let params = handle.params;
        let m = handle.cluster.m;
        let n = params.n();

        // v_enc has length n; query bytes = n × FIELD_BYTES.
        let query_bytes = (n * FIELD_BYTES) as u64;
        // r has length m; per-cluster response bytes = m × FIELD_BYTES.
        let cluster_response_bytes = (m * FIELD_BYTES) as u64;
        // Setup: M_enc upload, m × n field elements per cluster.
        let setup_bytes = (m * n * FIELD_BYTES) as u64;

        CommunicationCost {
            query_bytes,
            // Flat scorer: nprobe = 1, response is one cluster's worth.
            response_bytes: cluster_response_bytes,
            cluster_response_bytes,
            setup_bytes,
            pre_query_offline_up_bytes: 0,
            pre_query_offline_down_bytes: 0,
            // GPU bandwidth proxy = bytes the matrix kernel streams
            // across per query. For BN flat the server reads the full
            // `m × n` encrypted matrix M_enc per scoring call, so the
            // proxy is `m × n × FIELD_BYTES`. An earlier version used
            // `m × FIELD_BYTES` (the response shape), which
            // mis-categorised BN as bandwidth-efficient in cross-scheme
            // comparisons — see the bntm-ivf entry in
            // `effective_bytes_per_query` for the matching matrix-work
            // definition.
            effective_bytes_per_query: (m * n * FIELD_BYTES) as u64,
        }
    }

    /// Batched `score` override. Per query: quantise, sample q_seed
    /// (per-call !Send), encode → `v_enc`. Across the batch: one
    /// [`crypto::compute_products_batch`] call reads `M_enc` once and
    /// produces per-query `r` vectors in the same layout
    /// `compute_products` would, then per-query verify (when
    /// `verification_enabled`) + [`crypto::decode_scores`] +
    /// rank-by-magnitude. Per-(i, q_idx) inner fold is byte-identical
    /// to the per-query path; verification and decode are deterministic
    /// functions of `(r, cluster, v_enc, params)` plus the per-query
    /// `q_seed`, so each `out[i]` is bit-equal to
    /// `score(handle, &queries[i], k)`. GPU handles fall back to the
    /// inline sequential loop.
    async fn score_batch(
        &self,
        handle: &BnTmHandle,
        queries: &[Vector],
        k: usize,
    ) -> Result<Vec<Vec<Hit>>, BnTmError> {
        let params = handle.params;
        let n = params.n();

        for q in queries {
            if q.0.len() > n {
                return Err(BnTmError::DimensionTooLarge {
                    actual: q.0.len(),
                    n,
                });
            }
        }

        // Per-query quantise on the async task (O(B·n) — cheap, but
        // mirrors the per-query path's pre-spawn quantise so the timing
        // boundary stays clean).
        let v_qs: Vec<Vec<u64>> = queries
            .iter()
            .map(|q| quantise_vector(&q.0, n, handle.quantisation_q))
            .collect();

        // Per-query q_seed: ThreadRng is !Send — sample before
        // spawn_blocking, move only `Copy` bytes into the closure.
        let q_seeds: Vec<[u8; 32]> = (0..queries.len()).map(|_| sample_h_seed()).collect();

        let cluster = handle.cluster.clone();
        let l_subspace = Arc::clone(&handle.l_subspace);
        let m = cluster.m;
        let verification_enabled = handle.verification_enabled;
        let k_capped = k.min(m);
        let big_b = queries.len();

        match &handle.inner {
            HandleInner::Cpu => tokio::task::spawn_blocking(move || {
                // Per-query encode.
                let encodeds: Vec<crate::crypto::QueryEncoded> = v_qs
                    .iter()
                    .zip(&q_seeds)
                    .map(|(v_q, q_seed)| {
                        let mut rng = ChaCha20Rng::from_seed(*q_seed);
                        encode_query(v_q, &l_subspace, &mut rng, params)
                    })
                    .collect();

                // Single batched matvec: m_enc read once across queries.
                let v_enc_refs: Vec<&[u64]> = encodeds.iter().map(|e| e.v_enc.as_slice()).collect();
                let rs = compute_products_batch(cluster.m_enc.as_slice(), m, n, &v_enc_refs);

                // Per-query verify + decode + top-k.
                let mut out: Vec<Vec<Hit>> = Vec::with_capacity(big_b);
                for ((r, encoded), q_seed) in rs.iter().zip(&encodeds).zip(&q_seeds) {
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
                            return Err(BnTmError::VerificationFailed { trial });
                        }
                    }
                    let mv = decode_scores(r, &cluster, &l_subspace, encoded, params);
                    let mut scored: Vec<(usize, i64)> = mv
                        .iter()
                        .enumerate()
                        .map(|(id, s)| (id, fp_to_signed(*s)))
                        .collect();
                    scored.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(*score));
                    scored.truncate(k_capped);
                    out.push(
                        scored
                            .into_iter()
                            .map(|(id, score)| Hit {
                                id: id as u32,
                                score: score as f32,
                            })
                            .collect(),
                    );
                }
                Ok(out)
            })
            .await
            .map_err(|_| BnTmError::SpawnPanic)?,
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

/// Per-query substep timings reported by [`BnTmScorer::score_with_breakdown`].
/// `encode` covers quantisation + LG-and-T mask construction; `server` =
/// matvec `M_enc · v_enc`; `verify` = Protocol 2 (Freivalds) when
/// `verification_enabled = true`, otherwise 0; `decode` = subtract
/// corrections + lift to signed + top-k.
#[derive(Debug, Clone, Copy, Default)]
pub struct BnTmTiming {
    pub encode_us: u64,
    pub server_us: u64,
    pub verify_us: u64,
    pub decode_us: u64,
}

impl BnTmScorer {
    /// Off-hot-path equivalent of [`Scorer::score`] with per-substep
    /// timings. The breakdown path follows the same sequencing as
    /// `score` (server → verify → decode); the only deviation is
    /// quantisation moves inside `spawn_blocking` so that the
    /// `encode_us` measurement is contiguous.
    pub async fn score_with_breakdown(
        &self,
        handle: &BnTmHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, BnTmTiming), BnTmError> {
        let params = handle.params;
        let n = params.n();
        if query.0.len() > n {
            return Err(BnTmError::DimensionTooLarge {
                actual: query.0.len(),
                n,
            });
        }

        let q_seed: [u8; 32] = sample_h_seed();
        let cluster = handle.cluster.clone();
        let l_subspace = Arc::clone(&handle.l_subspace);
        let verification_enabled = handle.verification_enabled;
        let quantisation_q = handle.quantisation_q;
        let query_vec = query.0.clone();
        let m = cluster.m;
        let k_capped = k.min(m);

        // Dispatch on inner. The CPU branch keeps the self-contained
        // spawn_blocking. The GPU branch breaks the breakdown across
        // the host/device boundary so the server_us measurement spans
        // the full round-trip (H2D v_enc → kernel → D2H r) — the same
        // thing run_eval_with_verify_us measures via score_canonical,
        // so the verification-overhead-us column stays meaningful when
        // BN runs on GPU.
        let result: Result<(Vec<Hit>, BnTmTiming), BnTmError> = match &handle.inner {
            HandleInner::Cpu => tokio::task::spawn_blocking(move || {
                let t = Instant::now();
                let v_q = quantise_vector(&query_vec, n, quantisation_q);
                let mut rng = ChaCha20Rng::from_seed(q_seed);
                let encoded = encode_query(&v_q, &l_subspace, &mut rng, params);
                let encode_us = t.elapsed().as_micros() as u64;

                let t = Instant::now();
                let r = compute_products(cluster.m_enc.as_slice(), m, n, &encoded.v_enc);
                let server_us = t.elapsed().as_micros() as u64;

                let verify_us = if verification_enabled {
                    let t = Instant::now();
                    let mut v_rng = ChaCha20Rng::from_seed(domain_separate(q_seed, b"protocol2"));
                    if let Err(trial) = verify_response(
                        &r,
                        cluster.m_enc.as_slice(),
                        m,
                        n,
                        &encoded.v_enc,
                        params.verification_trials(),
                        &mut v_rng,
                    ) {
                        return Err(BnTmError::VerificationFailed { trial });
                    }
                    t.elapsed().as_micros() as u64
                } else {
                    0
                };

                let (hits, decode_us) =
                    decode_and_topk(&r, &cluster, &l_subspace, &encoded, params, k_capped);

                Ok((
                    hits,
                    BnTmTiming {
                        encode_us,
                        server_us,
                        verify_us,
                        decode_us,
                    },
                ))
            })
            .await
            .map_err(|_| BnTmError::SpawnPanic)?,
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                let l_for_encode = Arc::clone(&l_subspace);
                let query_for_encode = query_vec.clone();
                let (encoded, encode_us) = tokio::task::spawn_blocking(move || {
                    let t = Instant::now();
                    let v_q = quantise_vector(&query_for_encode, n, quantisation_q);
                    let mut rng = ChaCha20Rng::from_seed(q_seed);
                    let encoded = encode_query(&v_q, &l_for_encode, &mut rng, params);
                    (encoded, t.elapsed().as_micros() as u64)
                })
                .await
                .map_err(|_| BnTmError::SpawnPanic)?;

                let t_server = Instant::now();
                let r = state.compute_products(&encoded.v_enc)?;
                let server_us = t_server.elapsed().as_micros() as u64;

                // `decode_us` spans the full GPU decode dispatch
                // (`L⊤·v_enc` + H2D + 2 dense kernels + add_mod + D2H +
                // sparse on host + compose + rank). Pre-spawn portion
                // happens here (needs `state`); post-spawn portion is
                // timed inside the closure.
                let t_decode_pre = Instant::now();
                let lt_venc = l_transpose_times_vec(&l_subspace, n, params.n1(), &encoded.v_enc);
                let dense_gpu = state.decode_dense_terms(&encoded.g, &lt_venc)?;
                let s_mat = state.s_mat_arc();
                let decode_pre_us = t_decode_pre.elapsed().as_micros() as u64;

                let cluster_for_verify = cluster.clone();
                tokio::task::spawn_blocking(move || {
                    let verify_us = if verification_enabled {
                        let t = Instant::now();
                        let mut v_rng =
                            ChaCha20Rng::from_seed(domain_separate(q_seed, b"protocol2"));
                        if let Err(trial) = verify_response(
                            &r,
                            cluster_for_verify.m_enc.as_slice(),
                            m,
                            n,
                            &encoded.v_enc,
                            params.verification_trials(),
                            &mut v_rng,
                        ) {
                            return Err(BnTmError::VerificationFailed { trial });
                        }
                        t.elapsed().as_micros() as u64
                    } else {
                        0
                    };

                    let (hits, decode_post_us) = decode_and_topk_gpu(
                        &r,
                        &cluster_for_verify,
                        &s_mat,
                        &encoded,
                        &dense_gpu,
                        params,
                        k_capped,
                    );

                    Ok((
                        hits,
                        BnTmTiming {
                            encode_us,
                            server_us,
                            verify_us,
                            decode_us: decode_pre_us + decode_post_us,
                        },
                    ))
                })
                .await
                .map_err(|_| BnTmError::SpawnPanic)?
            }
        };

        result
    }
}

/// Shared decode + top-k path for `score_with_breakdown` (CPU
/// branch). Returns the hits and the wall-clock `decode_us` covering
/// decode_scores + sort + truncate.
fn decode_and_topk(
    r: &[u64],
    cluster: &EncryptedCluster,
    l_subspace: &[u64],
    encoded: &crate::crypto::QueryEncoded,
    params: BnTmParams,
    k_capped: usize,
) -> (Vec<Hit>, u64) {
    let t = Instant::now();
    let mv = decode_scores(r, cluster, l_subspace, encoded, params);
    let scored: Vec<(usize, i64)> = mv
        .iter()
        .enumerate()
        .map(|(id, s)| (id, fp_to_signed(*s)))
        .collect();
    let mut indexed = scored;
    indexed.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(*score));
    indexed.truncate(k_capped);
    let hits: Vec<Hit> = indexed
        .into_iter()
        .map(|(id, score)| Hit {
            id: id as u32,
            score: score as f32,
        })
        .collect();
    (hits, t.elapsed().as_micros() as u64)
}

/// GPU sibling of `decode_and_topk`. Takes the pre-computed
/// `dense_gpu` (the `AL · G + H · (L⊤ · v_enc)` partial from
/// `FlatGpuState::decode_dense_terms`) and the cached `s_mat`, runs
/// the sparse + compose pieces on host, and ranks. The returned
/// `decode_us` is just the in-closure portion; the caller adds its
/// own pre-spawn timing for the L⊤·v_enc + GPU dispatch.
#[cfg(feature = "gpu")]
fn decode_and_topk_gpu(
    r: &[u64],
    cluster: &EncryptedCluster,
    s_mat: &SparseMatrix,
    encoded: &crate::crypto::QueryEncoded,
    dense_gpu: &[u64],
    params: BnTmParams,
    k_capped: usize,
) -> (Vec<Hit>, u64) {
    let t = Instant::now();
    let mv = decode_scores_with_gpu_dense(r, cluster, s_mat, encoded, dense_gpu, params);
    let scored: Vec<(usize, i64)> = mv
        .iter()
        .enumerate()
        .map(|(id, s)| (id, fp_to_signed(*s)))
        .collect();
    let mut indexed = scored;
    indexed.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(*score));
    indexed.truncate(k_capped);
    let hits: Vec<Hit> = indexed
        .into_iter()
        .map(|(id, score)| Hit {
            id: id as u32,
            score: score as f32,
        })
        .collect();
    (hits, t.elapsed().as_micros() as u64)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sample a 32-byte seed from a thread-local RNG. Used for `h_seed`
/// (cluster init) and per-query encode/verify masks. Kept on the
/// async task — `ThreadRng` is `!Send` and must not cross
/// `spawn_blocking`.
fn sample_h_seed() -> [u8; 32] {
    use rand::RngExt;
    rand::rng().random()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use scorer_core::Vector;

    fn random_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vector> {
        use rand::RngExt;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..count)
            .map(|_| {
                let v: Vec<f32> = (0..dim)
                    .map(|_| rng.random_range(-1.0_f32..1.0_f32))
                    .collect();
                Vector(v)
            })
            .collect()
    }

    #[tokio::test]
    async fn upload_and_score_smoke() {
        let scorer = BnTmScorer::new();
        let config = BnTmConfig {
            params: BnTmParams::Sec128,
            key_seed: [1u8; 32],
            verification_enabled: false,
            quantisation_q: Q,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let vectors = random_vectors(8, 32, 1);
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        assert_eq!(handle.cluster.m, 8);
        let q = random_vectors(1, 32, 2).pop().unwrap();
        let hits = scorer.score(&handle, &q, 3).await.unwrap();
        assert_eq!(hits.len(), 3);
        // Hits in descending score order.
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    /// Honest server: verification on, every query passes.
    #[tokio::test]
    async fn score_with_verification_passes_on_honest_server() {
        let scorer = BnTmScorer::new();
        let config = BnTmConfig {
            params: BnTmParams::Sec128,
            key_seed: [2u8; 32],
            verification_enabled: true,
            quantisation_q: Q,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let vectors = random_vectors(16, 64, 3);
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        for s in 4..12 {
            let q = random_vectors(1, 64, s).pop().unwrap();
            let hits = scorer.score(&handle, &q, 5).await.unwrap();
            assert_eq!(hits.len(), 5);
        }
    }

    /// `k > cluster_size` is silently capped.
    #[tokio::test]
    async fn k_capped_at_cluster_size() {
        let scorer = BnTmScorer::new();
        let config = BnTmConfig {
            key_seed: [5u8; 32],
            verification_enabled: false,
            ..Default::default()
        };
        let vectors = random_vectors(3, 16, 7);
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        let q = random_vectors(1, 16, 8).pop().unwrap();
        let hits = scorer.score(&handle, &q, 100).await.unwrap();
        assert_eq!(hits.len(), 3);
    }

    /// Dimension > n surfaces as `BnTmError::DimensionTooLarge`, not
    /// a panic inside `spawn_blocking`.
    #[tokio::test]
    async fn dim_too_large_surfaces_as_error() {
        let scorer = BnTmScorer::new();
        let config = BnTmConfig {
            key_seed: [9u8; 32],
            ..Default::default()
        };
        // Dim slightly above n=1024.
        let vectors = vec![Vector(vec![0.5_f32; 1025])];
        let result = scorer.upload_cluster(&config, &vectors).await;
        assert!(matches!(
            result,
            Err(BnTmError::DimensionTooLarge {
                actual: 1025,
                n: 1024
            })
        ));
    }

    #[tokio::test]
    async fn debug_redacts_secrets() {
        let scorer = BnTmScorer::new();
        let config = BnTmConfig {
            key_seed: [0xDE; 32],
            verification_enabled: false,
            ..Default::default()
        };
        let vectors = random_vectors(4, 8, 11);
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        let dbg = format!("{handle:?}");
        // No 32-byte hex blob from key_seed/h_seed, no raw vectors.
        assert!(dbg.contains("[redacted]"));
        assert!(!dbg.contains("DE")); // crude but catches the obvious leak
    }

    /// `communication_cost` matches the analytical formula at known
    /// parameters.
    #[tokio::test]
    async fn communication_cost_formula() {
        let scorer = BnTmScorer::new();
        let config = BnTmConfig {
            key_seed: [13u8; 32],
            verification_enabled: false,
            ..Default::default()
        };
        let vectors = random_vectors(10, 32, 13);
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        let cost = scorer.communication_cost(&handle, 5);
        let n = handle.params.n();
        let m = handle.cluster.m;
        assert_eq!(cost.query_bytes, (n * FIELD_BYTES) as u64);
        assert_eq!(cost.cluster_response_bytes, (m * FIELD_BYTES) as u64);
        assert_eq!(cost.response_bytes, cost.cluster_response_bytes);
        assert_eq!(cost.setup_bytes, (m * n * FIELD_BYTES) as u64);
        assert_eq!(cost.pre_query_offline_up_bytes, 0);
        assert_eq!(cost.pre_query_offline_down_bytes, 0);
        // GPU proxy: matrix work the server kernel streams per query
        // — full `m × n` encrypted matrix M_enc.
        assert_eq!(cost.effective_bytes_per_query, (m * n * FIELD_BYTES) as u64);
    }

    /// `score_with_breakdown` agrees with `score` on top-k IDs; the BN
    /// recovery is structural (mask cancels) so different per-call
    /// `q_seed`s do not change the ranking.
    #[tokio::test]
    async fn score_with_breakdown_matches_score_no_verify() {
        let scorer = BnTmScorer::new();
        let config = BnTmConfig {
            params: BnTmParams::Sec128,
            key_seed: [33u8; 32],
            verification_enabled: false,
            quantisation_q: Q,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let vectors = random_vectors(8, 32, 33);
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        let q = random_vectors(1, 32, 34).pop().unwrap();

        let baseline = scorer.score(&handle, &q, 5).await.unwrap();
        let (with_breakdown, timing) = scorer.score_with_breakdown(&handle, &q, 5).await.unwrap();
        let baseline_ids: Vec<u32> = baseline.iter().map(|h| h.id).collect();
        let breakdown_ids: Vec<u32> = with_breakdown.iter().map(|h| h.id).collect();
        assert_eq!(baseline_ids, breakdown_ids);
        assert_eq!(timing.verify_us, 0, "no verify when disabled");
        let _ = timing.encode_us + timing.server_us + timing.decode_us;
    }

    /// With verification on, the verify substep must report a positive
    /// time (Protocol 2 runs `verification_trials` rounds).
    #[tokio::test]
    async fn score_with_breakdown_runs_verify_when_enabled() {
        let scorer = BnTmScorer::new();
        let config = BnTmConfig {
            params: BnTmParams::Sec128,
            key_seed: [44u8; 32],
            verification_enabled: true,
            quantisation_q: Q,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let vectors = random_vectors(8, 32, 44);
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        let q = random_vectors(1, 32, 45).pop().unwrap();
        let (_hits, timing) = scorer.score_with_breakdown(&handle, &q, 3).await.unwrap();
        // verify_us is wall-clock; on a fast machine it could
        // round to 0 microseconds, but the trial count is positive
        // so this assertion is loose by design.
        let _ = timing.verify_us;
    }

    /// Per-query result from the batched override is bit-equal to a
    /// sequential `score()` call on the same input. BN's per-query
    /// noise terms (`L·G + T`) and (`H·L⊤ + S`) cancel structurally in
    /// `decode_scores` (the recovered `Mv` is a deterministic function
    /// of the cluster matrix and the plaintext query, independent of
    /// the per-query random `q_seed`), and `fp_to_signed` + the
    /// rank-by-magnitude sort are deterministic, so end-to-end scores
    /// are bit-equal even though `score` and `score_batch` each sample
    /// fresh `q_seed`s. Run with `verification_enabled = false` so the
    /// test is portable across the verify-on/off sweeps; bit-equality
    /// of the matvec is what the override has to preserve.
    #[tokio::test]
    async fn bntm_score_batch_matches_score() {
        let scorer = BnTmScorer::new();
        let config = BnTmConfig {
            params: BnTmParams::Sec128,
            key_seed: [55u8; 32],
            verification_enabled: false,
            quantisation_q: Q,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let vectors = random_vectors(32, 64, 55);
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();

        let queries = random_vectors(64, 64, 56);
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

    /// Verification-on path: the batched override must still pass
    /// Protocol 2 on every per-query response. Independent test from
    /// the equivalence one because `verification_enabled = true`
    /// introduces an early-error code path in the closure; if either
    /// the per-query verify trial seed (derived via `domain_separate`
    /// from `q_seed`) or the batched `r` vector diverges from the
    /// per-query path, the verify would fail on an honest server here.
    #[tokio::test]
    async fn bntm_score_batch_verification_passes_on_honest_server() {
        let scorer = BnTmScorer::new();
        let config = BnTmConfig {
            params: BnTmParams::Sec128,
            key_seed: [66u8; 32],
            verification_enabled: true,
            quantisation_q: Q,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };
        let vectors = random_vectors(16, 64, 66);
        let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        let queries = random_vectors(8, 64, 67);
        let k = 5;
        let hits = scorer.score_batch(&handle, &queries, k).await.unwrap();
        assert_eq!(hits.len(), queries.len());
        for h in &hits {
            assert_eq!(h.len(), k);
        }
    }

    /// Flat BN has no cache; every upload is a cold build, so both calls
    /// report `cache_hit = false`.
    #[tokio::test]
    async fn build_outcome_cache_hit_smoke() {
        // Flat variant has no cache layer by design — only build_duration
        // is asserted; cache_hit is structurally always false here.
        let scorer = BnTmScorer::new();
        let config = BnTmConfig {
            key_seed: [21u8; 32],
            verification_enabled: false,
            ..Default::default()
        };
        let vectors = random_vectors(8, 32, 21);
        let (_h1, b1) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        assert!(!b1.cache_hit);
        let (_h2, b2) = scorer.upload_cluster(&config, &vectors).await.unwrap();
        assert!(!b2.cache_hit, "flat BN has no cache; both builds are cold");
    }

    /// Flat-BN-GPU exact-equality gate.
    ///
    /// Mersenne field arithmetic is deterministic, so the GPU
    /// `compute_products` kernel must produce *exactly* the same r
    /// vector as the CPU path on identical (m_enc, v_enc) inputs.
    /// Stronger threshold than the cuVS-based scorers (8/10). Verifies
    /// (a) the Mersenne reduction matches the CPU `mersenne_reduce`
    /// cadence, (b) the periodic in-flight reduction at j&31==31
    /// matches CPU, (c) H2D / D2H preserve u64 ordering, (d) the
    /// (i, j) → output mapping matches the CPU row-major layout.
    /// Requires the rapids conda env (`docs/envs/README.md`).
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_kernel_partials_match_cpu_exact() {
        use crate::params::P;
        use rand::RngExt;
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let params = BnTmParams::Sec128;
        let n = params.n();
        let m: usize = 32;

        // Synthetic field elements that exercise the full F_p range.
        let mut rng = ChaCha20Rng::seed_from_u64(1234);
        let mut m_enc = vec![0u64; m * n];
        for v in m_enc.iter_mut() {
            *v = rng.random::<u64>() % P;
        }
        let mut v_enc = vec![0u64; n];
        for v in v_enc.iter_mut() {
            *v = rng.random::<u64>() % P;
        }

        let cpu = compute_products(&m_enc, m, n, &v_enc);

        // `FlatGpuState::build` also takes AL, H, and a cached sparse
        // `S`. This test only exercises the server-side matvec kernel,
        // which reads `m_enc_dev` only — zero-filled / empty
        // placeholders satisfy the upload.
        let n1 = params.n1();
        let a_l_placeholder = vec![0u64; m * n1];
        let h_placeholder = vec![0u64; m * n1];
        let s_placeholder = std::sync::Arc::new(SparseMatrix::empty(m, n));

        let m_enc_for_gpu = m_enc.clone();
        let gpu_state = tokio::task::spawn_blocking(move || {
            gpu::FlatGpuState::build(
                &m_enc_for_gpu,
                &a_l_placeholder,
                &h_placeholder,
                s_placeholder,
                m,
                params,
            )
            .unwrap()
        })
        .await
        .unwrap();
        let v_enc_for_gpu = v_enc.clone();
        let gpu = tokio::task::spawn_blocking(move || {
            gpu_state.compute_products(&v_enc_for_gpu).unwrap()
        })
        .await
        .unwrap();

        assert_eq!(cpu.len(), gpu.len(), "r length mismatch");
        for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
            assert_eq!(a, b, "r[{i}] mismatch: cpu={a}, gpu={b}");
        }
    }

    /// Exercise m values that aren't a multiple of the thread-block
    /// size (256), the cross-warp reduction step (m large enough to
    /// span multiple warps), and very small m (one block, partial
    /// warp). The warp-per-row layout means per-block work is fixed
    /// at N=1024; this test varies m only. Bit-equality with the CPU
    /// reference is the gate.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_kernel_partials_match_cpu_exact_varied_m() {
        use crate::params::P;
        use rand::RngExt;
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let params = BnTmParams::Sec128;
        let n = params.n();

        for &m in &[1usize, 7, 256, 257, 300, 511, 1024, 1500] {
            let mut rng = ChaCha20Rng::seed_from_u64(2400 + m as u64);
            let mut m_enc = vec![0u64; m * n];
            for v in m_enc.iter_mut() {
                *v = rng.random::<u64>() % P;
            }
            let mut v_enc = vec![0u64; n];
            for v in v_enc.iter_mut() {
                *v = rng.random::<u64>() % P;
            }

            let cpu = compute_products(&m_enc, m, n, &v_enc);

            // AL / H / S placeholders (see sibling test).
            let n1 = params.n1();
            let a_l_placeholder = vec![0u64; m * n1];
            let h_placeholder = vec![0u64; m * n1];
            let s_placeholder = std::sync::Arc::new(SparseMatrix::empty(m, n));

            let m_enc_for_gpu = m_enc.clone();
            let gpu_state = tokio::task::spawn_blocking(move || {
                gpu::FlatGpuState::build(
                    &m_enc_for_gpu,
                    &a_l_placeholder,
                    &h_placeholder,
                    s_placeholder,
                    m,
                    params,
                )
                .unwrap()
            })
            .await
            .unwrap();
            let v_enc_for_gpu = v_enc.clone();
            let gpu = tokio::task::spawn_blocking(move || {
                gpu_state.compute_products(&v_enc_for_gpu).unwrap()
            })
            .await
            .unwrap();

            assert_eq!(cpu.len(), gpu.len(), "m={m}: r length mismatch");
            for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
                assert_eq!(a, b, "m={m} r[{i}] mismatch: cpu={a}, gpu={b}");
            }
        }
    }

    /// Focused unit test for the `decode_dense_terms` method. Gates
    /// kernel correctness of `bntm_decode_dense` + `bntm_add_mod_vec`
    /// ahead of the end-to-end `gpu_decode_matches_cpu_exact`.
    ///
    /// Synthetic AL and H matrices in F_p, two synthetic length-n_1
    /// query vectors; the GPU result must be bit-identical to a
    /// straight CPU reference computed with `mersenne_reduce` and
    /// `add_mod`. Varying `m` exercises the kernel's full-block
    /// (m ≥ 256), partial-block (m = 1, 7, 257), and cross-warp
    /// (m = 300, 1024, 1500) regimes.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_dense_terms_matches_cpu_exact() {
        use crate::crypto::{add_mod, mersenne_reduce};
        use crate::params::P;
        use rand::RngExt;
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let params = BnTmParams::Sec128;
        let n1 = params.n1();

        fn cpu_dense_reference(mat: &[u64], m: usize, n1: usize, v: &[u64]) -> Vec<u64> {
            (0..m)
                .map(|i| {
                    let mut acc: u128 = 0;
                    for j in 0..n1 {
                        acc += (mat[i * n1 + j] as u128) * (v[j] as u128);
                        if j & 31 == 31 {
                            acc = mersenne_reduce(acc) as u128;
                        }
                    }
                    mersenne_reduce(acc)
                })
                .collect()
        }

        for &m in &[1usize, 7, 256, 257, 300, 1024, 1500] {
            let mut rng = ChaCha20Rng::seed_from_u64(3500 + m as u64);

            let mut al = vec![0u64; m * n1];
            for v in al.iter_mut() {
                *v = rng.random::<u64>() % P;
            }
            let mut h = vec![0u64; m * n1];
            for v in h.iter_mut() {
                *v = rng.random::<u64>() % P;
            }
            let mut g = vec![0u64; n1];
            for v in g.iter_mut() {
                *v = rng.random::<u64>() % P;
            }
            let mut lt_venc = vec![0u64; n1];
            for v in lt_venc.iter_mut() {
                *v = rng.random::<u64>() % P;
            }

            // CPU reference: (AL · g) + (H · lt_venc) mod P.
            let cpu_al_g = cpu_dense_reference(&al, m, n1, &g);
            let cpu_h_v = cpu_dense_reference(&h, m, n1, &lt_venc);
            let cpu: Vec<u64> = cpu_al_g
                .iter()
                .zip(&cpu_h_v)
                .map(|(a, b)| add_mod(*a, *b))
                .collect();

            // GPU: synthesise a FlatGpuState with arbitrary M_enc +
            // empty S (decode_dense_terms reads neither).
            let n = params.n();
            let m_enc_placeholder = vec![0u64; m * n];
            let al_for_gpu = al.clone();
            let h_for_gpu = h.clone();
            let s_placeholder = std::sync::Arc::new(SparseMatrix::empty(m, n));
            let gpu_state = tokio::task::spawn_blocking(move || {
                gpu::FlatGpuState::build(
                    &m_enc_placeholder,
                    &al_for_gpu,
                    &h_for_gpu,
                    s_placeholder,
                    m,
                    params,
                )
                .unwrap()
            })
            .await
            .unwrap();
            let g_for_gpu = g.clone();
            let lt_for_gpu = lt_venc.clone();
            let gpu = tokio::task::spawn_blocking(move || {
                gpu_state
                    .decode_dense_terms(&g_for_gpu, &lt_for_gpu)
                    .unwrap()
            })
            .await
            .unwrap();

            assert_eq!(cpu.len(), gpu.len(), "m={m} length mismatch");
            for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
                assert_eq!(a, b, "m={m} dense[{i}] mismatch: cpu={a}, gpu={b}");
            }
        }
    }

    /// End-to-end decode: GPU dispatch (`decode_dense_terms` +
    /// `decode_scores_with_gpu_dense`) must produce a bit-identical
    /// `Mv` to CPU `decode_scores` over the same
    /// `(r, cluster, encoded, params)`.
    ///
    /// Builds a real `EncryptedCluster` via `encrypt_cluster`, encodes
    /// a query, runs the server matvec on host, and dispatches the
    /// decode through both code paths. Varies `m ∈ {1, 7, 256, 257,
    /// 300, 1024, 1500}` so the kernel exercises full-block,
    /// partial-block, and cross-warp regimes.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_decode_matches_cpu_exact() {
        use crate::crypto::{
            decode_scores as crypto_decode_scores,
            decode_scores_with_gpu_dense as crypto_decode_with_gpu_dense, encode_query,
            encrypt_cluster, generate_subspace, l_transpose_times_vec,
        };
        use crate::params::P;
        use rand::RngExt;
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let params = BnTmParams::Sec128;
        let n = params.n();
        let n1 = params.n1();

        for &m in &[1usize, 7, 256, 257, 300, 1024, 1500] {
            let mut rng = ChaCha20Rng::seed_from_u64(7700 + m as u64);

            // Quantised corpus: m × n field elements in [0, P).
            let mut m_matrix = vec![0u64; m * n];
            for v in m_matrix.iter_mut() {
                *v = rng.random::<u64>() % P;
            }
            let key_seed: [u8; 32] = [0x11; 32];
            let h_seed: [u8; 32] = {
                let mut s = [0u8; 32];
                for b in s.iter_mut() {
                    *b = rng.random::<u8>();
                }
                s
            };
            let l_subspace = generate_subspace(key_seed, n, n1);
            let cluster = encrypt_cluster(&m_matrix, m, &l_subspace, h_seed, params);

            // Quantised query vector + per-query mask state.
            let mut v_q = vec![0u64; n];
            for v in v_q.iter_mut() {
                *v = rng.random::<u64>() % P;
            }
            let mut q_rng =
                ChaCha20Rng::from_seed([0xa5u8; 32].map(|b| b ^ (m as u8).wrapping_mul(31)));
            let encoded = encode_query(&v_q, &l_subspace, &mut q_rng, params);

            // Server response on host.
            let r = compute_products(cluster.m_enc.as_slice(), m, n, &encoded.v_enc);

            // CPU reference.
            let cpu_mv = crypto_decode_scores(&r, &cluster, &l_subspace, &encoded, params);

            // GPU dispatch. (H, S) come straight from the cluster's own
            // cached trapdoor material (Plan 28) rather than a fresh
            // regeneration.
            let m_enc_for_gpu = cluster.m_enc.as_slice().to_vec();
            let a_l_for_gpu = cluster.a_l.as_slice().to_vec();
            let h_for_gpu = cluster.h_mat.as_slice().to_vec();
            let s_for_gpu = std::sync::Arc::clone(&cluster.s_mat);
            let gpu_state = tokio::task::spawn_blocking(move || {
                gpu::FlatGpuState::build(
                    &m_enc_for_gpu,
                    &a_l_for_gpu,
                    &h_for_gpu,
                    s_for_gpu,
                    m,
                    params,
                )
                .unwrap()
            })
            .await
            .unwrap();

            let lt_venc = l_transpose_times_vec(&l_subspace, n, n1, &encoded.v_enc);
            let g_for_gpu = encoded.g.clone();
            let lt_for_gpu = lt_venc.clone();
            let dense_gpu = tokio::task::spawn_blocking(move || {
                gpu_state
                    .decode_dense_terms(&g_for_gpu, &lt_for_gpu)
                    .unwrap()
            })
            .await
            .unwrap();
            let gpu_mv = crypto_decode_with_gpu_dense(
                &r,
                &cluster,
                &cluster.s_mat,
                &encoded,
                &dense_gpu,
                params,
            );

            assert_eq!(cpu_mv.len(), gpu_mv.len(), "m={m}: Mv length mismatch");
            for (i, (a, b)) in cpu_mv.iter().zip(&gpu_mv).enumerate() {
                assert_eq!(a, b, "m={m} Mv[{i}] mismatch: cpu={a}, gpu={b}");
            }
        }
    }

    /// Flat-BN end-to-end GPU vs CPU.
    ///
    /// At small corpus + fixed key_seed / verification_enabled = false,
    /// GPU `score` must return the same top-k IDs as CPU. Each call
    /// samples its own per-query seed (via `sample_h_seed`), but BN
    /// recovers M·v structurally regardless of the trapdoor mask, so
    /// IDs match.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_score_matches_cpu() {
        let vectors = random_vectors(20, 64, 909);
        let q = random_vectors(1, 64, 910).pop().unwrap();
        let k = 5;

        let scorer_cpu = BnTmScorer::new();
        let scorer_gpu = BnTmScorer::new();

        let cfg_cpu = BnTmConfig {
            key_seed: [42u8; 32],
            verification_enabled: false,
            ..Default::default()
        };
        let cfg_gpu = BnTmConfig {
            device: Device::Gpu,
            ..BnTmConfig {
                key_seed: [42u8; 32],
                verification_enabled: false,
                ..Default::default()
            }
        };

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
            "GPU top-{k} ids must match CPU exactly under Mersenne arithmetic"
        );
    }

    /// Verification still passes on the honest server when GPU
    /// computes the response. Catches kernel-arithmetic divergence
    /// that would otherwise surface as a confusing
    /// `BnTmError::VerificationFailed` at sweep time.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_score_with_verification_passes_on_honest_server() {
        let vectors = random_vectors(8, 32, 911);
        let q = random_vectors(1, 32, 912).pop().unwrap();

        let scorer = BnTmScorer::new();
        let cfg = BnTmConfig {
            key_seed: [99u8; 32],
            verification_enabled: true,
            device: Device::Gpu,
            ..Default::default()
        };
        let (handle, _) = scorer.upload_cluster(&cfg, &vectors).await.unwrap();

        // Should not return BnTmError::VerificationFailed — Freivalds
        // checks the GPU-produced r against the same M_enc / v_enc
        // and matches when the kernel is bit-identical to CPU.
        let hits = scorer.score(&handle, &q, 3).await.unwrap();
        assert_eq!(hits.len(), 3);
    }

    /// Flat-BN GPU breakdown agrees with score(). The BN measurement
    /// path (`run_eval_with_verify_us` in eval-harness) routes through
    /// `score_with_breakdown` for the per-query
    /// verification-overhead-us column, so this contract is
    /// load-bearing on GPU. A regression that rejected GPU here would
    /// fail the harness immediately.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_score_with_breakdown_matches_score() {
        let vectors = random_vectors(20, 64, 920);
        let q = random_vectors(1, 64, 921).pop().unwrap();
        let k = 5;

        let scorer = BnTmScorer::new();
        let cfg = BnTmConfig {
            key_seed: [42u8; 32],
            verification_enabled: true,
            device: Device::Gpu,
            ..Default::default()
        };
        let (handle, _) = scorer.upload_cluster(&cfg, &vectors).await.unwrap();

        let baseline = scorer.score(&handle, &q, k).await.unwrap();
        let (with_breakdown, timing) = scorer.score_with_breakdown(&handle, &q, k).await.unwrap();

        let baseline_ids: std::collections::HashSet<u32> = baseline.iter().map(|h| h.id).collect();
        let breakdown_ids: std::collections::HashSet<u32> =
            with_breakdown.iter().map(|h| h.id).collect();
        assert_eq!(baseline_ids, breakdown_ids);
        assert!(
            timing.server_us > 0,
            "server_us should cover kernel launch: {timing:?}"
        );
        assert!(
            timing.verify_us > 0,
            "verify_us should cover Protocol 2 trials: {timing:?}"
        );
        assert!(
            timing.decode_us > 0,
            "decode_us should cover host decode: {timing:?}"
        );
    }
}
