//! Full-corpus C1 recall test.
//!
//! With `n_centroids = 1` the routing layer is a no-op (every vector
//! lives in the only cluster) and the entire `Scorer::score` pipeline
//! reduces to SimplePIR-LHE + BFV apply-hint correctness on the
//! production crypto path. We assert that Tiptoe's top-`k` IDs match a
//! plaintext brute-force ranking with `recall@10 ≥ 0.99` on a synthetic
//! but full-`d = 768` corpus.
//!
//! ### Why 0.99 and not 1.00
//!
//! At the Tiptoe-text parameters (`n = 2048`, `log2_q = 64`,
//! `σ = 81920`, `P = 2¹⁷`) the SimplePIR-LHE decryption-failure
//! probability per coordinate is cryptographically negligible. The 0.99
//! threshold is a margin for that residual failure probability, not for
//! quantisation rounding — the synthetic corpus is constructed to
//! quantise losslessly (see `random_grid_vectors`). Larger-scale recall
//! drift that this test would miss is caught by the paired Rust/Go
//! validation gate (`analysis/tiptoe_diff.py`).
//!
//! ### Why `quantisation_bits = 3`
//!
//! At `d = 768` the Go reference's defensive precondition
//! `2·4^q·d < P` (= 2¹⁷ = 131072 for text-search) forces `q ≤ 3`:
//! - `q = 3`: `2·4³·768 = 98 304 < 131 072` ✓
//! - `q = 4`: `2·4⁴·768 = 393 216 > 131 072` ✗
//!
//! Picking a larger `q` here would surface as
//! `EncodingError::QuantisationOverflow`, not a recall miss, so the
//! threshold is well-defined at `q = 3`.
//!
//! ### Sizing for CI
//!
//! Production LWE+BFV parameters mean each query runs the full crypto
//! pipeline (LHE encrypt of an `n × dC`-byte matrix, BFV apply-hint
//! over `m_max × n_lwe / poly_degree` ciphertexts). We pick `N = 64`
//! and 10 queries — wall-clock ≈ 10s release / ≈ 100s debug — small
//! enough for CI without dropping the per-query LWE noise check
//! below the 1-miss-out-of-100 granularity that the 0.99 threshold
//! requires.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use scorer_core::{Scorer, Vector};
use scorer_tiptoe::{TiptoeConfig, TiptoeScorer};
use std::collections::HashSet;

/// Random vectors whose `f32` components land **exactly** on the
/// `q = 3` quantisation grid (`{-1, -6/7, …, 6/7, 1}`). This makes
/// `quantise_component` a faithful identity (no rounding error), so
/// the only difference between brute-force and Tiptoe rankings can
/// come from LHE+BFV noise — which is exactly what the 0.99 recall
/// threshold absorbs. Random unit-norm or uniform-`[-1, 1]` data at
/// `d = 768`, `q = 3` carries enough quantisation rounding noise
/// (~7% recall hit empirically) to swamp the much smaller LWE
/// decryption-failure rate, defeating the test's purpose of isolating
/// LHE correctness from data-quantisation effects.
///
/// Worst-case inner product is `d · mag² = 768 · 49 = 37 632`, well
/// under `P/2 = 65 536`, so no `mod P` wrap can flip a sign.
fn random_grid_vectors(n: usize, d: usize, seed: u64) -> Vec<Vector> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mag = 7_i32; // = 2^3 − 1
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..d)
                .map(|_| {
                    let q: i32 = rng.random_range(-mag..=mag);
                    q as f32 / mag as f32
                })
                .collect();
            Vector(v)
        })
        .collect()
}

fn brute_force_top_k(vectors: &[Vector], query: &Vector, k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = v.0.iter().zip(&query.0).map(|(a, b)| a * b).sum();
            (i as u32, dot)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

#[tokio::test]
async fn tiptoe_recall_full_corpus_c1() {
    const N: usize = 64;
    const DIM: usize = 768;
    const N_QUERIES: usize = 10;
    const K: usize = 10;
    const Q_BITS: u8 = 3;

    let vectors = random_grid_vectors(N, DIM, 0xC1_C0_C0);
    let queries = random_grid_vectors(N_QUERIES, DIM, 0xC1_BA_BE);

    let scorer = TiptoeScorer::new();
    let cfg = TiptoeConfig {
        n_centroids: 1,
        quantisation_bits: Q_BITS,
        ..Default::default()
    };
    let (handle, _build) = scorer
        .upload_cluster(&cfg, &vectors)
        .await
        .expect("upload_cluster");

    let mut total_recall = 0.0_f64;
    for (qi, q) in queries.iter().enumerate() {
        let plaintext: HashSet<u32> = brute_force_top_k(&vectors, q, K).into_iter().collect();
        let hits = scorer.score(&handle, q, K).await.expect("score");
        let tiptoe: HashSet<u32> = hits.iter().map(|h| h.id).collect();
        assert_eq!(tiptoe.len(), K, "q{qi}: expected {K} unique hits");
        let overlap = plaintext.intersection(&tiptoe).count();
        total_recall += overlap as f64 / K as f64;
    }
    let mean_recall = total_recall / N_QUERIES as f64;
    assert!(
        mean_recall >= 0.99,
        "mean recall@{K} = {mean_recall:.4}, expected >= 0.99 \
         at q={Q_BITS}, n_centroids=1, N={N}, dim={DIM}"
    );
}
