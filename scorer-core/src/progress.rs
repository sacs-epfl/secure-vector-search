//! Unified progress-reporting interface for scorers.
//!
//! Each scorer's `Config` carries `progress: Option<Arc<dyn ProgressReporter>>`.
//! The eval-harness binary supplies an `IndicatifProgress` impl; tests
//! pass `None`. Default-impl methods let consumers override only what
//! they care about.

/// Progress milestones a scorer emits while building a cluster/index.
pub trait ProgressReporter: Send + Sync {
    /// Called once per centroid during k-means++ initialisation (1-based).
    fn on_init_centroid(&self, _idx: usize) {}

    /// Called after each Lloyd's iteration (1-based).
    fn on_kmeans_iter(&self, _iter: usize) {}

    /// Called per encryption work unit. Granularity is scorer-specific:
    /// per-row for flat scorers (EmvpScorer, SapScorer); per-cluster
    /// for IVF scorers (EmvpIvfScorer); per SimplePIR hint row for
    /// TiptoeScorer. Each scorer documents its convention in the
    /// doc-comment above its Config.
    fn on_encrypt(&self, _idx: usize) {}

    /// Tell the reporter that a subsequent phase will emit `additional`
    /// extra ticks on top of the originally-budgeted total. Used when
    /// the size of a heavy phase isn't known up-front (e.g. TiptoeScorer
    /// only learns `m_max` after k-means + IVF finish). Default no-op.
    fn on_phase_extend(&self, _additional: usize) {}

    /// Called once when the cluster/index is fully built (whether built
    /// from scratch or loaded from cache).
    fn on_build_complete(&self) {}
}
