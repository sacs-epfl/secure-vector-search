use std::time::Duration;

use async_trait::async_trait;

pub mod cache;
pub mod progress;

pub use progress::ProgressReporter;

/// A dense f32 embedding vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector(pub Vec<f32>);

/// Opaque index of a vector within its cluster.
pub type VectorId = u32;

/// A nearest-neighbour hit: vector index plus similarity score.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: VectorId,
    pub score: f32,
}

/// Estimated wire bytes for a single query against one cluster.
#[derive(Debug, Clone, Copy)]
pub struct CommunicationCost {
    /// Query ciphertext sent to the server per probe.
    pub query_bytes: u64,
    /// Total per-query server response (may span nprobe clusters).
    pub response_bytes: u64,
    /// Server response for one cluster (response_bytes / nprobe for IVF scorers).
    pub cluster_response_bytes: u64,
    /// One-time per-client setup traffic (e.g. encrypted database upload).
    /// Paid once at corpus-fixation, reused across every query from that
    /// client. 0 for schemes with no setup phase.
    pub setup_bytes: u64,
    /// Per-query offline upload — work the client sends to the server
    /// *before* the online round-trip but consumed by exactly one query.
    /// Tiptoe's BFV-encrypted query token is the canonical example;
    /// 0 for every other scheme today.
    pub pre_query_offline_up_bytes: u64,
    /// Per-query offline download — server response in the offline phase
    /// (e.g. Tiptoe's apply-hint result, BFV-encrypted h_s). 0 for
    /// every scheme except Tiptoe.
    pub pre_query_offline_down_bytes: u64,
    /// Analytical proxy for GPU throughput under bandwidth-bound
    /// scaling: total bytes the matrix kernel streams across per query
    /// (not wire bytes). Captures Tiptoe's ~4× bandwidth penalty
    /// cleanly even when Tiptoe itself only has analytical-proxy GPU
    /// coverage.
    pub effective_bytes_per_query: u64,
}

/// Device axis for evaluation. Re-exported from `scorer-core` so every
/// candidate scorer crate names the value the same way and CSV /
/// metadata strings don't drift.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    #[default]
    Cpu,
    Gpu,
}

impl Device {
    pub fn as_str(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
        }
    }
}

/// Outcome of a single `Scorer::upload_cluster` call. Distinguishes cache hits
/// (warm reload, microseconds) from cold builds (full encryption pipeline).
///
/// `cache_hit = true` means the scorer returned without rebuilding — either an
/// in-memory cache hit on the same scorer instance or a disk-cache reload. For
/// IVF scorers covering multiple clusters, `true` iff every cluster's state
/// was loaded from cache.
///
/// `build_duration` is wall-clock time inside `upload_cluster`, measured by
/// the scorer with a single `Instant::now()` pair so cache hits don't include
/// task-spawn noise.
#[derive(Debug, Clone, Copy)]
pub struct BuildOutcome {
    pub cache_hit: bool,
    pub build_duration: Duration,
}

/// Shared error type for scorer-core. Each scorer crate defines its own.
#[derive(Debug, thiserror::Error)]
pub enum ScorerError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// The backend abstraction. Each scheme (plaintext, SAP, EMVP, Tiptoe)
/// implements this trait.
///
/// Not object-safe — associated types prevent `dyn Scorer`. Call sites use
/// `fn bench<S: Scorer>(…)` generics throughout.
#[async_trait]
pub trait Scorer: Send + Sync {
    type Config: Send + Sync;
    /// Per-cluster state produced by `upload_cluster` and consumed by `score`.
    type ClusterHandle: Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Preprocess or encrypt `vectors` under `config` and return an opaque
    /// handle representing that cluster's backend state, plus a [`BuildOutcome`]
    /// reporting cache-hit status and build wall-clock time.
    async fn upload_cluster(
        &self,
        config: &Self::Config,
        vectors: &[Vector],
    ) -> Result<(Self::ClusterHandle, BuildOutcome), Self::Error>;

    /// Return the top-`k` hits for `query` against `handle`, in descending
    /// score order.
    async fn score(
        &self,
        handle: &Self::ClusterHandle,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<Hit>, Self::Error>;

    /// Score all `queries` against `handle`. Defaults to sequential calls to
    /// `score`; schemes with native batch protocols (e.g. EMVP) should override
    /// to fuse work across queries.
    ///
    /// **Override contract**:
    ///
    /// - **Ordering**: `out[i]` corresponds to `queries[i]`.
    /// - **Per-query semantics**: each `Vec<Hit>` is ranking-equal to a
    ///   sequential `score(handle, &queries[i], k)` call, with per-result
    ///   score within 1e-6 relative tolerance for f32 schemes (plaintext) and
    ///   1e-12 relative tolerance for f64 schemes (SAP); bit-equal for
    ///   integer-field schemes (EMVP, BN). Batched GEMM may reassociate
    ///   summation.
    /// - **Cost accounting**: no `score_batch_with_realised_cost` analogue.
    ///   Batched per-row cost is reported by the eval-harness via the
    ///   `wallclock-us` and `amortised-latency-us` raw.csv columns;
    ///   `latency-us` stays populated only at `batch-size = 1` for backwards
    ///   compatibility.
    /// - **Concurrency**: implementations may internally use rayon, CUDA
    ///   streams, etc. The call returns once all `B` results are ready;
    ///   streaming output is not part of the contract.
    async fn score_batch(
        &self,
        handle: &Self::ClusterHandle,
        queries: &[Vector],
        k: usize,
    ) -> Result<Vec<Vec<Hit>>, Self::Error> {
        let mut out = Vec::with_capacity(queries.len());
        for q in queries {
            out.push(self.score(handle, q, k).await?);
        }
        Ok(out)
    }

    /// Analytical estimate of wire bytes for one `score` call returning `k`
    /// results. Sync because it is derived from handle parameters, not I/O.
    fn communication_cost(&self, handle: &Self::ClusterHandle, k: usize) -> CommunicationCost;

    /// Per-query realised online cost, alongside the hits. Default impl pairs
    /// `score()` with the analytical `communication_cost()` — correct for any
    /// scheme whose per-query response shape is fixed by handle parameters
    /// alone (flat scorers; Tiptoe at a fixed quantisation; SAP/SAP+IVF where
    /// the response is k×8 bytes regardless of which clusters were probed
    /// because the merged top-k is the only thing returned).
    ///
    /// Override when realised cost varies per query — the canonical case being
    /// IVF schemes where probed cluster sizes are uneven (cv = 34.7 % at our
    /// 100k MS MARCO defaults).
    async fn score_with_realised_cost(
        &self,
        handle: &Self::ClusterHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, CommunicationCost), Self::Error> {
        let hits = self.score(handle, query, k).await?;
        let cost = self.communication_cost(handle, k);
        Ok((hits, cost))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockHandle {
        cluster_size: usize,
    }

    struct MockScorer;

    #[async_trait]
    impl Scorer for MockScorer {
        type Config = ();
        type ClusterHandle = MockHandle;
        type Error = ScorerError;

        async fn upload_cluster(
            &self,
            _config: &(),
            vectors: &[Vector],
        ) -> Result<(MockHandle, BuildOutcome), ScorerError> {
            let start = std::time::Instant::now();
            let handle = MockHandle {
                cluster_size: vectors.len(),
            };
            Ok((
                handle,
                BuildOutcome {
                    cache_hit: false,
                    build_duration: start.elapsed(),
                },
            ))
        }

        async fn score(
            &self,
            handle: &MockHandle,
            _query: &Vector,
            k: usize,
        ) -> Result<Vec<Hit>, ScorerError> {
            let k = k.min(handle.cluster_size);
            Ok((0..k as u32).map(|id| Hit { id, score: 1.0 }).collect())
        }

        fn communication_cost(&self, _handle: &MockHandle, _k: usize) -> CommunicationCost {
            CommunicationCost {
                query_bytes: 0,
                response_bytes: 0,
                cluster_response_bytes: 0,
                setup_bytes: 0,
                pre_query_offline_up_bytes: 0,
                pre_query_offline_down_bytes: 0,
                effective_bytes_per_query: 0,
            }
        }
    }

    #[tokio::test]
    async fn upload_and_score_roundtrip() {
        let scorer = MockScorer;
        let vectors = vec![
            Vector(vec![1.0, 0.0]),
            Vector(vec![0.0, 1.0]),
            Vector(vec![1.0, 1.0]),
        ];
        let (handle, build) = scorer.upload_cluster(&(), &vectors).await.unwrap();
        assert_eq!(handle.cluster_size, 3);
        assert!(!build.cache_hit);

        let hits = scorer
            .score(&handle, &Vector(vec![1.0, 0.0]), 2)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 0);
        assert_eq!(hits[1].id, 1);
        assert!(hits.iter().all(|h| h.score == 1.0));
    }

    struct OverridingScorer;

    #[async_trait]
    impl Scorer for OverridingScorer {
        type Config = ();
        type ClusterHandle = MockHandle;
        type Error = ScorerError;

        async fn upload_cluster(
            &self,
            _config: &(),
            vectors: &[Vector],
        ) -> Result<(MockHandle, BuildOutcome), ScorerError> {
            let start = std::time::Instant::now();
            Ok((
                MockHandle {
                    cluster_size: vectors.len(),
                },
                BuildOutcome {
                    cache_hit: false,
                    build_duration: start.elapsed(),
                },
            ))
        }

        async fn score(
            &self,
            handle: &MockHandle,
            _query: &Vector,
            k: usize,
        ) -> Result<Vec<Hit>, ScorerError> {
            let k = k.min(handle.cluster_size);
            Ok((0..k as u32).map(|id| Hit { id, score: 1.0 }).collect())
        }

        // Structurally different from the default impl (reverse the input,
        // then reverse the output) so the test exercises the override path
        // rather than the inherited default. Observable behavior must match.
        async fn score_batch(
            &self,
            handle: &MockHandle,
            queries: &[Vector],
            k: usize,
        ) -> Result<Vec<Vec<Hit>>, ScorerError> {
            let mut out = Vec::with_capacity(queries.len());
            for q in queries.iter().rev() {
                out.push(self.score(handle, q, k).await?);
            }
            out.reverse();
            Ok(out)
        }

        fn communication_cost(&self, _handle: &MockHandle, _k: usize) -> CommunicationCost {
            CommunicationCost {
                query_bytes: 0,
                response_bytes: 0,
                cluster_response_bytes: 0,
                setup_bytes: 0,
                pre_query_offline_up_bytes: 0,
                pre_query_offline_down_bytes: 0,
                effective_bytes_per_query: 0,
            }
        }
    }

    #[tokio::test]
    async fn score_batch_override_matches_default() {
        let default_scorer = MockScorer;
        let overriding = OverridingScorer;
        let vectors = vec![
            Vector(vec![0.0, 0.0]),
            Vector(vec![1.0, 0.0]),
            Vector(vec![0.0, 1.0]),
        ];
        let (h_default, _) = default_scorer.upload_cluster(&(), &vectors).await.unwrap();
        let (h_override, _) = overriding.upload_cluster(&(), &vectors).await.unwrap();

        let queries: Vec<Vector> = (0..5)
            .map(|i| Vector(vec![i as f32, (i + 1) as f32]))
            .collect();

        let from_default = default_scorer
            .score_batch(&h_default, &queries, 2)
            .await
            .unwrap();
        let from_override = overriding
            .score_batch(&h_override, &queries, 2)
            .await
            .unwrap();

        assert_eq!(from_default, from_override);
    }

    #[tokio::test]
    async fn score_batch_delegates_to_score() {
        let scorer = MockScorer;
        let vectors = vec![Vector(vec![0.0]), Vector(vec![1.0]), Vector(vec![2.0])];
        let (handle, _build) = scorer.upload_cluster(&(), &vectors).await.unwrap();

        let queries = vec![Vector(vec![0.0]), Vector(vec![1.0])];
        let results = scorer.score_batch(&handle, &queries, 2).await.unwrap();

        assert_eq!(results.len(), 2);
        for hits in &results {
            assert_eq!(hits.len(), 2);
            assert_eq!(hits[0].id, 0);
            assert_eq!(hits[1].id, 1);
        }
    }

    #[tokio::test]
    async fn default_score_with_realised_cost_matches_score_plus_cost() {
        let scorer = MockScorer;
        let vectors = vec![
            Vector(vec![1.0, 0.0]),
            Vector(vec![0.0, 1.0]),
            Vector(vec![1.0, 1.0]),
        ];
        let (handle, _build) = scorer.upload_cluster(&(), &vectors).await.unwrap();
        let query = Vector(vec![1.0, 0.0]);

        let hits_only = scorer.score(&handle, &query, 2).await.unwrap();
        let cost_only = scorer.communication_cost(&handle, 2);
        let (hits_pair, cost_pair) = scorer
            .score_with_realised_cost(&handle, &query, 2)
            .await
            .unwrap();

        assert_eq!(hits_only, hits_pair);
        assert_eq!(cost_only.query_bytes, cost_pair.query_bytes);
        assert_eq!(cost_only.response_bytes, cost_pair.response_bytes);
        assert_eq!(
            cost_only.cluster_response_bytes,
            cost_pair.cluster_response_bytes
        );
        assert_eq!(cost_only.setup_bytes, cost_pair.setup_bytes);
        assert_eq!(
            cost_only.pre_query_offline_up_bytes,
            cost_pair.pre_query_offline_up_bytes
        );
        assert_eq!(
            cost_only.pre_query_offline_down_bytes,
            cost_pair.pre_query_offline_down_bytes
        );
        assert_eq!(
            cost_only.effective_bytes_per_query,
            cost_pair.effective_bytes_per_query
        );
    }

    #[test]
    fn device_as_str_roundtrip() {
        assert_eq!(Device::Cpu.as_str(), "cpu");
        assert_eq!(Device::Gpu.as_str(), "gpu");
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[tokio::test]
    async fn score_caps_k_at_cluster_size() {
        let scorer = MockScorer;
        let vectors = vec![Vector(vec![1.0])];
        let (handle, _build) = scorer.upload_cluster(&(), &vectors).await.unwrap();
        // k larger than the cluster; must not panic or pad.
        let hits = scorer.score(&handle, &Vector(vec![1.0]), 10).await.unwrap();
        assert_eq!(hits.len(), 1);
    }
}
