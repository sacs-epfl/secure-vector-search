use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use rand::RngExt;
use scorer_core::{BuildOutcome, CommunicationCost, Device, Hit, ProgressReporter, Scorer, Vector};

pub(crate) mod cluster;
mod crypto;
// Public only so `examples/l2_simd_bench.rs` can reach `l2_simd`. No
// external API on scorer-sap depends on these functions, but `pub mod`
// is required for examples/ to compile against them.
pub mod distance;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
pub mod ivf;

pub use crypto::{SapKey, keygen};
pub use ivf::{SapIvfConfig, SapIvfError, SapIvfHandle, SapIvfScorer};

/// Per-cluster configuration passed to `upload_cluster`.
pub struct SapConfig {
    /// Pre-generated key. Use [`keygen`] to produce one.
    pub key: SapKey,
    /// Perturbation parameter β ≥ 0. β = 0 gives exact (no perturbation).
    pub beta: f64,
    /// Seed for the parallel encryption RNG. Each vector `i` is encrypted with
    /// `ChaCha20Rng::seed_from_u64(upload_seed ^ i)`, matching the prototype's
    /// `build` for reproducibility.
    pub upload_seed: u64,
    /// Optional progress reporter. `on_encrypt(i)` fires once per encrypted
    /// vector (per-row granularity); other milestones are unused.
    pub progress: Option<Arc<dyn ProgressReporter>>,
    /// Compute substrate. `Cpu` runs the flat scan on host. `Gpu`
    /// requires the `gpu` Cargo feature; without it `upload_cluster`
    /// returns [`SapError::GpuFeatureNotEnabled`]. SAP encryption itself
    /// stays on CPU either way; only the L2 scan over ciphertexts moves
    /// to device. There is no `Default` impl for `SapConfig` (the `key`
    /// field has no sane default — a zero-filled `prf_key` would be an
    /// unsafe footgun), so callers must set this field explicitly.
    pub device: Device,
    /// VRAM budget for the streaming cluster store on GPU runs. `None`
    /// (default) auto-detects 80 % of free VRAM via NVML at handle
    /// build. `Some(b)` pins a hard byte cap. Ignored on CPU runs.
    pub vram_budget_bytes: Option<u64>,
}

/// Opaque per-cluster state. Holds the encrypted entries and the key material
/// needed to encrypt queries against this cluster.
pub struct SapClusterHandle {
    entries: Arc<Vec<crypto::Ciphertext>>,
    key_s: f64,
    prf_key: [u8; 32], // redacted in Debug — see impl below
    beta: f64,
    dim: usize,
    inner: HandleInner,
}

enum HandleInner {
    Cpu,
    /// Flat-SAP single cuVS BfIndex over the entire encrypted corpus.
    /// Encryption stays on host; only the per-query L2 scan over
    /// ciphertexts moves to device. Per-query host-to-device and
    /// device-to-host transfer dominates at small corpora.
    /// Boxed because the streaming `FlatGpuState` carries host-side
    /// vectors + the LRU + NVML sampler; without Box this pushes the
    /// `HandleInner` size well above the `Cpu` unit variant and trips
    /// clippy's `large_enum_variant` gate.
    #[cfg(feature = "gpu")]
    Gpu(Box<gpu::FlatGpuState>),
}

impl fmt::Debug for SapClusterHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device = match &self.inner {
            HandleInner::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(_) => "gpu",
        };
        f.debug_struct("SapClusterHandle")
            .field("device", &device)
            .field("dim", &self.dim)
            .field("cluster_size", &self.entries.len())
            .field("beta", &self.beta)
            .field("key_s", &self.key_s)
            .field("prf_key", &"[redacted]")
            .finish()
    }
}

impl SapClusterHandle {
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
pub enum SapError {
    #[error("cluster must contain at least one vector")]
    EmptyCluster,
    #[error("invalid beta {0}: must be >= 0.0")]
    InvalidBeta(f64),
    #[error("query dimension {query_dim} does not match cluster dimension {cluster_dim}")]
    DimensionMismatch {
        query_dim: usize,
        cluster_dim: usize,
    },
    #[error("spawn_blocking task panicked")]
    TaskPanic,
    /// `Device::Gpu` was requested but the crate was built without the
    /// `gpu` Cargo feature. Surfaced as a clean error rather than
    /// silently falling back to CPU — silent fallback would mislabel
    /// rows as GPU-measured when they ran on CPU.
    #[error("Device::Gpu requested but the `gpu` Cargo feature is not enabled in this build")]
    GpuFeatureNotEnabled,
    /// GPU runtime error from cuVS (build / search / device alloc).
    #[cfg(feature = "gpu")]
    #[error("cuVS GPU error: {0}")]
    Gpu(#[from] gpu::GpuError),
}

pub struct SapScorer;

#[async_trait]
impl Scorer for SapScorer {
    type Config = SapConfig;
    type ClusterHandle = SapClusterHandle;
    type Error = SapError;

    async fn upload_cluster(
        &self,
        config: &SapConfig,
        vectors: &[Vector],
    ) -> Result<(SapClusterHandle, BuildOutcome), SapError> {
        let build_start = Instant::now();
        if vectors.is_empty() {
            return Err(SapError::EmptyCluster);
        }
        if config.beta < 0.0 {
            return Err(SapError::InvalidBeta(config.beta));
        }

        let dim = vectors[0].0.len();
        let key_s = config.key.s;
        let prf_key = config.key.prf_key;
        let beta = config.beta;
        let upload_seed = config.upload_seed;
        let progress = config.progress.clone();

        // The f32 → f64 upcast is streamed per-row inside
        // `build_cluster` rather than materialised up front: at an 8.8M
        // corpus a full `Vec<Vec<f64>>` pre-allocates ~54 GB and can
        // OOM the machine. Here we only move an owned `Vec<Vector>` into
        // the blocking closure. The clone is the same shape as the
        // caller's corpus — unavoidable given `upload_cluster` takes
        // `&[Vector]` and spawn_blocking needs `'static` data.
        let data: Vec<Vector> = vectors.to_vec();
        let entries = tokio::task::spawn_blocking(move || {
            let p_for_rows = progress.clone();
            let entries = cluster::build_cluster(key_s, prf_key, &data, beta, upload_seed, |i| {
                if let Some(ref p) = p_for_rows {
                    p.on_encrypt(i);
                }
            });
            drop(data);
            if let Some(ref p) = progress {
                p.on_build_complete();
            }
            entries
        })
        .await
        .map_err(|_| SapError::TaskPanic)?;
        let entries = Arc::new(entries);

        // Branch on substrate. Encryption above is identical for CPU
        // and GPU; only the per-query L2 scan differs. The host-side
        // `entries` Arc is kept either way (used by `score` on CPU;
        // useful for per-query accounting on GPU and to keep `Debug`
        // self-consistent).
        let inner = match config.device {
            Device::Cpu => HandleInner::Cpu,
            Device::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    let entries_for_gpu = Arc::clone(&entries);
                    let vram_budget = config.vram_budget_bytes;
                    let state = tokio::task::spawn_blocking(move || {
                        gpu::FlatGpuState::build(&entries_for_gpu, vram_budget)
                    })
                    .await
                    .map_err(|_| SapError::TaskPanic)??;
                    HandleInner::Gpu(Box::new(state))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(SapError::GpuFeatureNotEnabled);
                }
            }
        };

        let outcome = BuildOutcome {
            // Flat SAP has no cache; every upload is a cold build.
            cache_hit: false,
            build_duration: build_start.elapsed(),
        };
        Ok((
            SapClusterHandle {
                entries,
                key_s,
                prf_key,
                beta,
                dim,
                inner,
            },
            outcome,
        ))
    }

    async fn score(
        &self,
        handle: &SapClusterHandle,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<Hit>, SapError> {
        if query.0.len() != handle.dim {
            return Err(SapError::DimensionMismatch {
                query_dim: query.0.len(),
                cluster_dim: handle.dim,
            });
        }

        let query_f64: Vec<f64> = query.0.iter().map(|&x| x as f64).collect();
        let key_s = handle.key_s;
        let prf_key = handle.prf_key;
        let beta = handle.beta;

        // ThreadRng is !Send and cannot be captured in the spawn_blocking closure.
        // Generate the nonce here, then move the [u8; 16] (Copy + Send) in.
        let nonce: [u8; 16] = {
            let mut rng = rand::rng();
            std::array::from_fn(|_| rng.random())
        };

        match &handle.inner {
            HandleInner::Cpu => {
                let entries = Arc::clone(&handle.entries);
                let hits = tokio::task::spawn_blocking(move || {
                    let query_ct =
                        crypto::encrypt_with_nonce(key_s, &prf_key, &query_f64, beta, nonce);
                    cluster::knn_with_scores(&entries, &query_ct, k)
                })
                .await
                .map_err(|_| SapError::TaskPanic)?;
                Ok(hits)
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                // Encryption stays on CPU (the SAP semantics — `score`
                // sees the encrypted query). The cuVS launch is the
                // device-side work. Mutex-guarded `state.score` is
                // !Send-friendly because we don't hold the lock across
                // an await.
                let query_ct = crypto::encrypt_with_nonce(key_s, &prf_key, &query_f64, beta, nonce);
                state.score(&query_ct, k).map_err(Into::into)
            }
        }
    }

    fn communication_cost(&self, handle: &SapClusterHandle, k: usize) -> CommunicationCost {
        let response_bytes = k as u64 * 8;
        // GPU bandwidth proxy. SAP flat scans the entire cluster of
        // perturbed-fp32 vectors; the matrix the kernel streams is
        // m × dim × 4 bytes (fp32 — what a GPU SAP path would consume).
        let effective_bytes_per_query = handle.entries.len() as u64 * handle.dim as u64 * 4;
        CommunicationCost {
            query_bytes: handle.dim as u64 * 8,
            response_bytes,
            cluster_response_bytes: response_bytes,
            setup_bytes: 0,
            pre_query_offline_up_bytes: 0,
            pre_query_offline_down_bytes: 0,
            effective_bytes_per_query,
        }
    }

    /// Batched override. Loop-transposes the per-query
    /// `knn_with_scores` scan via [`cluster::knn_with_scores_batch`]: each
    /// entry ciphertext is read once per call and contributes f64
    /// distances to every batched query, instead of once per query.
    /// Encryption stays per-query (per-call nonces matter for the SAP
    /// scheme); the only fused work is the distance scan and per-query
    /// top-k. Per-pair `l2` is bit-equal to the per-query path, so each
    /// `out[i]` is bit-equal to `score(handle, &queries[i], k)`. GPU
    /// handles fall back to the inline sequential loop.
    async fn score_batch(
        &self,
        handle: &SapClusterHandle,
        queries: &[Vector],
        k: usize,
    ) -> Result<Vec<Vec<Hit>>, SapError> {
        for q in queries {
            if q.0.len() != handle.dim {
                return Err(SapError::DimensionMismatch {
                    query_dim: q.0.len(),
                    cluster_dim: handle.dim,
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

        match &handle.inner {
            HandleInner::Cpu => {
                let entries = Arc::clone(&handle.entries);
                let hits = tokio::task::spawn_blocking(move || {
                    let query_cts: Vec<crypto::Ciphertext> = queries_f64
                        .iter()
                        .zip(&nonces)
                        .map(|(qf, &n)| crypto::encrypt_with_nonce(key_s, &prf_key, qf, beta, n))
                        .collect();
                    let qct_refs: Vec<&crypto::Ciphertext> = query_cts.iter().collect();
                    cluster::knn_with_scores_batch(&entries, &qct_refs, k)
                })
                .await
                .map_err(|_| SapError::TaskPanic)?;
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

/// Per-query substep timings reported by [`SapScorer::score_with_breakdown`].
/// `encode` = `encrypt_with_nonce` on the query; `server` = the
/// distance-on-ciphertexts scan (`l2` over `entries`); `merge` = top-k.
#[derive(Debug, Clone, Copy, Default)]
pub struct SapTiming {
    pub encode_us: u64,
    pub server_us: u64,
    pub merge_us: u64,
}

impl SapScorer {
    /// Off-hot-path equivalent of [`Scorer::score`] with per-substep
    /// timings. Per-call `Instant::now()` pairs add nanosecond-scale
    /// overhead per substep; do not use for throughput benchmarks.
    pub async fn score_with_breakdown(
        &self,
        handle: &SapClusterHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, SapTiming), SapError> {
        if query.0.len() != handle.dim {
            return Err(SapError::DimensionMismatch {
                query_dim: query.0.len(),
                cluster_dim: handle.dim,
            });
        }

        let query_f64: Vec<f64> = query.0.iter().map(|&x| x as f64).collect();
        let key_s = handle.key_s;
        let prf_key = handle.prf_key;
        let beta = handle.beta;

        let nonce: [u8; 16] = {
            let mut rng = rand::rng();
            std::array::from_fn(|_| rng.random())
        };

        // Dispatch on inner. Both branches measure encode (host-side
        // ciphertext construction) + server (the distance scan) + merge
        // (top-k). On GPU, the cuVS launch round-trip is the server
        // measurement; merge is folded into server because `state.score`
        // already returns top-k hits sorted (cuVS does the per-launch
        // top-k internally).
        let result = match &handle.inner {
            HandleInner::Cpu => {
                let entries = Arc::clone(&handle.entries);
                tokio::task::spawn_blocking(move || {
                    let t = Instant::now();
                    let query_ct =
                        crypto::encrypt_with_nonce(key_s, &prf_key, &query_f64, beta, nonce);
                    let encode_us = t.elapsed().as_micros() as u64;

                    let t = Instant::now();
                    let mut dists: Vec<(u32, f64)> = entries
                        .iter()
                        .enumerate()
                        .map(|(i, ct)| (i as u32, distance::l2(&query_ct.c, &ct.c)))
                        .collect();
                    let server_us = t.elapsed().as_micros() as u64;

                    let t = Instant::now();
                    let k_capped = k.min(entries.len());
                    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                    let hits: Vec<Hit> = dists[..k_capped]
                        .iter()
                        .map(|(id, dist)| Hit {
                            id: *id,
                            score: -(*dist as f32),
                        })
                        .collect();
                    let merge_us = t.elapsed().as_micros() as u64;

                    (
                        hits,
                        SapTiming {
                            encode_us,
                            server_us,
                            merge_us,
                        },
                    )
                })
                .await
                .map_err(|_| SapError::TaskPanic)?
            }
            #[cfg(feature = "gpu")]
            HandleInner::Gpu(state) => {
                let encode_t = Instant::now();
                let query_ct = crypto::encrypt_with_nonce(key_s, &prf_key, &query_f64, beta, nonce);
                let encode_us = encode_t.elapsed().as_micros() as u64;

                let server_t = Instant::now();
                let hits = state.score(&query_ct, k).map_err(SapError::from)?;
                let server_us = server_t.elapsed().as_micros() as u64;

                // cuVS returns top-k already sorted; merge cost is
                // folded into server_us. Reporting merge_us = 0
                // honestly reflects the available substep boundary.
                (
                    hits,
                    SapTiming {
                        encode_us,
                        server_us,
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

    fn brute_force_ranking(vectors: &[Vector], query: &Vector) -> Vec<u32> {
        let mut dists: Vec<(u32, f64)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f64 =
                    v.0.iter()
                        .zip(&query.0)
                        .map(|(a, b)| {
                            let d = *a as f64 - *b as f64;
                            d * d
                        })
                        .sum::<f64>()
                        .sqrt();
                (i as u32, d)
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        dists.iter().map(|(i, _)| *i).collect()
    }

    fn default_config(beta: f64) -> SapConfig {
        SapConfig {
            key: keygen(1.0, 2.0, 7),
            beta,
            upload_seed: 42,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        }
    }

    #[tokio::test]
    async fn upload_and_score_roundtrip() {
        let scorer = SapScorer;
        let vectors = random_vectors(20, 8, 1);
        let (handle, _build) = scorer
            .upload_cluster(&default_config(0.5), &vectors)
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
        let scorer = SapScorer;
        let vectors = random_vectors(30, 8, 2);
        let (handle, _build) = scorer
            .upload_cluster(&default_config(0.5), &vectors)
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

    #[tokio::test]
    async fn beta_zero_exact_ordering() {
        // At β=0, perturbation radius = key.s * 0 / 4 = 0 → ciphertext = s*m exactly.
        // L2(s*q, s*m_i) = s * L2(q, m_i), so ranking equals plaintext f64 ranking.
        let scorer = SapScorer;
        let vectors = random_vectors(15, 6, 3);
        let (handle, _build) = scorer
            .upload_cluster(&default_config(0.0), &vectors)
            .await
            .unwrap();
        let query = random_vectors(1, 6, 101).remove(0);
        let k = 5;
        let hits = scorer.score(&handle, &query, k).await.unwrap();
        let got: Vec<u32> = hits.iter().map(|h| h.id).collect();
        let expected = brute_force_ranking(&vectors, &query);
        assert_eq!(got, expected[..k]);
    }

    #[tokio::test]
    async fn k_capped_at_cluster_size() {
        let scorer = SapScorer;
        let vectors = random_vectors(3, 4, 4);
        let (handle, _build) = scorer
            .upload_cluster(&default_config(0.0), &vectors)
            .await
            .unwrap();
        let query = random_vectors(1, 4, 102).remove(0);
        let hits = scorer.score(&handle, &query, 100).await.unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[tokio::test]
    async fn error_on_empty_cluster() {
        let scorer = SapScorer;
        let err = scorer
            .upload_cluster(&default_config(0.5), &[])
            .await
            .unwrap_err();
        assert!(matches!(err, SapError::EmptyCluster));
    }

    #[tokio::test]
    async fn error_on_invalid_beta() {
        let scorer = SapScorer;
        let err = scorer
            .upload_cluster(&default_config(-0.1), &random_vectors(5, 4, 5))
            .await
            .unwrap_err();
        assert!(matches!(err, SapError::InvalidBeta(_)));
    }

    #[tokio::test]
    async fn error_on_dimension_mismatch() {
        let scorer = SapScorer;
        let vectors = random_vectors(5, 4, 6);
        let (handle, _build) = scorer
            .upload_cluster(&default_config(0.0), &vectors)
            .await
            .unwrap();
        let wrong_query = random_vectors(1, 8, 103).remove(0);
        let err = scorer.score(&handle, &wrong_query, 3).await.unwrap_err();
        assert!(matches!(err, SapError::DimensionMismatch { .. }));
    }

    #[tokio::test]
    async fn debug_redacts_prf_key() {
        let scorer = SapScorer;
        let vectors = random_vectors(5, 4, 7);
        let (handle, _build) = scorer
            .upload_cluster(&default_config(0.0), &vectors)
            .await
            .unwrap();
        let s = format!("{:?}", handle);
        assert!(s.contains("[redacted]"));
        assert!(!s.contains("prf_key: ["));
    }

    #[tokio::test]
    async fn communication_cost_formula() {
        let scorer = SapScorer;
        let vectors = random_vectors(10, 16, 8);
        let (handle, _build) = scorer
            .upload_cluster(&default_config(0.0), &vectors)
            .await
            .unwrap();
        let cost = scorer.communication_cost(&handle, 5);
        assert_eq!(cost.query_bytes, 16 * 8);
        assert_eq!(cost.response_bytes, 5 * 8);
        assert_eq!(cost.cluster_response_bytes, 5 * 8);
        assert_eq!(cost.setup_bytes, 0);
        assert_eq!(cost.pre_query_offline_up_bytes, 0);
        assert_eq!(cost.pre_query_offline_down_bytes, 0);
        // GPU bandwidth proxy: m × dim × 4 = 10 × 16 × 4 = 640.
        assert_eq!(cost.effective_bytes_per_query, 10 * 16 * 4);
    }

    /// At β = 0 the score is rank-deterministic regardless of the
    /// per-query nonce; both `score` and `score_with_breakdown` must
    /// return the same hit IDs in the same order.
    #[tokio::test]
    async fn score_with_breakdown_matches_score() {
        let scorer = SapScorer;
        let vectors = random_vectors(20, 8, 30);
        let (handle, _build) = scorer
            .upload_cluster(&default_config(0.0), &vectors)
            .await
            .unwrap();
        let query = random_vectors(1, 8, 31).remove(0);

        let baseline = scorer.score(&handle, &query, 5).await.unwrap();
        let (with_breakdown, timing) = scorer
            .score_with_breakdown(&handle, &query, 5)
            .await
            .unwrap();

        let baseline_ids: Vec<u32> = baseline.iter().map(|h| h.id).collect();
        let breakdown_ids: Vec<u32> = with_breakdown.iter().map(|h| h.id).collect();
        assert_eq!(baseline_ids, breakdown_ids);
        let _ = timing.encode_us + timing.server_us + timing.merge_us;
    }

    /// Per-query result from the batched override is equivalent
    /// (ranking-equal + score within 1e-12 rel tol for f64) to a
    /// sequential `score()` call on the same input. β = 0 neutralises
    /// per-query nonce divergence (perturbation radius = key.s × 0 / 4
    /// = 0 → ciphertext = key.s · vector regardless of nonce), so the
    /// trait-surface comparison is deterministic. The implementation
    /// preserves summation order, so this is effectively bit-equal
    /// today; the tolerance assertion leaves headroom for a future SIMD
    /// lift.
    #[tokio::test]
    async fn sap_score_batch_matches_score() {
        let scorer = SapScorer;
        let vectors = random_vectors(64, 16, 17);
        let (handle, _build) = scorer
            .upload_cluster(&default_config(0.0), &vectors)
            .await
            .unwrap();

        let queries = random_vectors(64, 16, 18);
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
                        diff <= 1e-12_f32 * denom,
                        "B={b} q={q_idx} rank={rank}: score diff {diff} > 1e-12 * {denom}"
                    );
                }
            }
        }
    }

    /// Flat SAP has no cache; every upload is a cold build, so both calls
    /// report `cache_hit = false`.
    #[tokio::test]
    async fn build_outcome_cache_hit_smoke() {
        // Flat variant has no cache layer by design — only build_duration
        // is asserted; cache_hit is structurally always false here.
        let scorer = SapScorer;
        let vectors = random_vectors(20, 8, 9);
        let (_h1, b1) = scorer
            .upload_cluster(&default_config(0.0), &vectors)
            .await
            .unwrap();
        assert!(!b1.cache_hit);
        let (_h2, b2) = scorer
            .upload_cluster(&default_config(0.0), &vectors)
            .await
            .unwrap();
        assert!(!b2.cache_hit, "flat SAP has no cache; both builds are cold");
    }

    /// Correctness gate for flat-SAP-GPU. At β=0 with s=1 the ciphertext
    /// equals the plaintext, so the result must approximately equal CPU
    /// brute-force L2. Threshold is 8/10 set overlap rather than exact
    /// equality — cuVS computes L2 via L2Expanded which admits
    /// catastrophic cancellation at the top-k boundary. Requires a CUDA
    /// build environment with cuVS available.
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_flat_matches_brute_force() {
        let vectors = random_vectors(256, 64, 909);
        let query = random_vectors(1, 64, 910).remove(0);
        let k = 10;

        let scorer = SapScorer;
        // s=1, β=0 → ciphertext = plaintext exactly, no perturbation.
        let cfg = SapConfig {
            key: keygen(1.0, 1.0, 7),
            beta: 0.0,
            upload_seed: 42,
            progress: None,
            device: Device::Gpu,
            vram_budget_bytes: None,
        };
        let (handle, _build) = scorer.upload_cluster(&cfg, &vectors).await.unwrap();
        let hits = scorer.score(&handle, &query, k).await.unwrap();
        assert_eq!(hits.len(), k);

        let got: std::collections::HashSet<u32> = hits.iter().map(|h| h.id).collect();
        // CPU reference: brute-force L2 on the plaintext (β=0, s=1
        // means ciphertexts are identical to plaintexts up to fp64↔fp32
        // round-trip).
        let mut dists: Vec<(u32, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, ivf_index::distance::l2_f32(&v.0, &query.0)))
            .collect();
        dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        let expected: std::collections::HashSet<u32> = dists[..k].iter().map(|(i, _)| *i).collect();
        let overlap = got.intersection(&expected).count();
        assert!(
            overlap >= 8,
            "GPU flat overlap with CPU brute force was {}/10; expected ≥ 8/10. \
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

    /// GPU breakdown agrees with GPU score on hits. Validates
    /// `score_with_breakdown` accepts GPU and produces the same top-k
    /// IDs as `score()`. server_us should be > 0 (cuVS launch
    /// round-trip).
    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn gpu_score_with_breakdown_matches_score() {
        let vectors = random_vectors(256, 64, 920);
        let query = random_vectors(1, 64, 921).remove(0);
        let k = 5;

        let scorer = SapScorer;
        let cfg = SapConfig {
            key: keygen(1.0, 1.0, 7),
            beta: 0.0,
            upload_seed: 42,
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
        assert_eq!(baseline_ids, breakdown_ids);
        assert!(
            timing.server_us > 0,
            "server_us should cover cuVS launch: {timing:?}"
        );
    }
}
