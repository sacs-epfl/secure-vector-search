//! `CommunicationCost` formula test for Tiptoe.
//!
//! Tiptoe is the only scheme where `pre_query_offline_*_bytes ≠ 0`;
//! every other scorer's `communication_cost_formula` test asserts
//! `== 0` inline in its own `mod tests`. This file lives separately
//! because Tiptoe has no in-crate `mod tests` block (its tests are
//! all integration-level under `tests/`).
//!
//! The per-query offline cost is split into directional `_up` and
//! `_down` halves. The BFV-encrypted query token is upload
//! (`pre_query_offline_up_bytes`); the BFV apply-hint result
//! (`chunks · NUM_LIMBS_64 · bfv_ct_bytes`) is download
//! (`pre_query_offline_down_bytes`), not part of `response_bytes`.
//! The actual online response is the LWE answer `m_max × 8` bytes.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use scorer_core::{Scorer, Vector};
use scorer_tiptoe::pir::lwe::Params as LweParams;
use scorer_tiptoe::{TiptoeConfig, TiptoeScorer};

const NUM_LIMBS_64: u64 = 8;

fn random_vectors(n: usize, d: usize, seed: u64) -> Vec<Vector> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..d)
                .map(|_| rng.random_range(-1.0_f32..1.0_f32))
                .collect();
            Vector(v)
        })
        .collect()
}

#[tokio::test]
async fn communication_cost_splits_token_out_of_setup() {
    let scorer = TiptoeScorer::new();
    let vectors = random_vectors(20, 8, 17);
    let cfg = TiptoeConfig {
        n_centroids: 2,
        ..Default::default()
    };
    let (handle, _build) = scorer.upload_cluster(&cfg, &vectors).await.unwrap();
    let cost = scorer.communication_cost(&handle, 10);

    let lwe = LweParams::tiptoe_text();
    let bfv = scorer_tiptoe::bfv::tiptoe_text_params();
    let degree = bfv.degree();
    let coeff_bytes = bfv.moduli_sizes().iter().sum::<usize>().div_ceil(8);
    let bfv_ct_bytes = (2 * degree * coeff_bytes) as u64;

    let m_max = handle.encoded().m_max() as u64;
    let expected_hint = m_max * lwe.n as u64 * 8;
    let expected_token = lwe.n as u64 * bfv_ct_bytes;
    let chunks = m_max.div_ceil(degree as u64);
    let expected_apply_hint = chunks * NUM_LIMBS_64 * bfv_ct_bytes;
    let expected_lwe_answer = m_max * 8;

    assert_eq!(cost.setup_bytes, expected_hint, "setup must be hint only");
    assert_eq!(
        cost.pre_query_offline_up_bytes, expected_token,
        "pre_query_offline_up must be the §6.3 BFV token"
    );
    assert_eq!(
        cost.pre_query_offline_down_bytes, expected_apply_hint,
        "pre_query_offline_down must be the §6.2 apply-hint result"
    );
    assert_eq!(
        cost.response_bytes, expected_lwe_answer,
        "response_bytes must be the online LWE answer m_max × 8"
    );
    assert_ne!(
        cost.pre_query_offline_up_bytes, 0,
        "Tiptoe has a non-zero §6.3 token"
    );
    assert_ne!(
        cost.pre_query_offline_down_bytes, 0,
        "Tiptoe has a non-zero §6.2 apply-hint result"
    );
}
