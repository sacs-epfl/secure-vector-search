//! Verification-toggle timing contract for the BN scorers.
//!
//! The eval harness derives the per-query `verification-overhead-us` CSV
//! column from `BnTmTiming::verify_us` (flat) / `BnTmIvfTiming::verify_us`
//! (IVF), populated via `score_with_breakdown`. This test pins that timing
//! source directly:
//!
//!   - `verification_enabled = true`  ⇒ verify_us > 0
//!   - `verification_enabled = false` ⇒ verify_us == 0
//!
//! A refactor that silently zeroed verify_us (or wrote a non-zero value
//! when verification is off) would corrupt the reported verification
//! overhead without any other test catching it.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use scorer_bntm::{BnTmConfig, BnTmIvfConfig, BnTmIvfScorer, BnTmParams, BnTmScorer, Q};
use scorer_core::{Device, Scorer, Vector};

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

fn flat_cfg(verification_enabled: bool) -> BnTmConfig {
    BnTmConfig {
        params: BnTmParams::Sec128,
        key_seed: [1u8; 32],
        verification_enabled,
        quantisation_q: Q,
        progress: None,
        device: Device::Cpu,
        vram_budget_bytes: None,
    }
}

fn ivf_cfg(verification_enabled: bool) -> BnTmIvfConfig {
    BnTmIvfConfig {
        params: BnTmParams::Sec128,
        key_seed: [2u8; 32],
        n_centroids: 4,
        nprobe: 2,
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
async fn bntm_flat_verify_us_tracks_verification_enabled() {
    let scorer = BnTmScorer::new();
    let vectors = random_unit_vectors(16, 32, 1);
    let query = random_unit_vectors(1, 32, 2).into_iter().next().unwrap();

    let (handle_on, _) = scorer
        .upload_cluster(&flat_cfg(true), &vectors)
        .await
        .unwrap();
    let (_, timing_on) = scorer
        .score_with_breakdown(&handle_on, &query, 5)
        .await
        .unwrap();
    assert!(
        timing_on.verify_us > 0,
        "verification_enabled=true must populate verify_us > 0, got {}",
        timing_on.verify_us
    );

    let (handle_off, _) = scorer
        .upload_cluster(&flat_cfg(false), &vectors)
        .await
        .unwrap();
    let (_, timing_off) = scorer
        .score_with_breakdown(&handle_off, &query, 5)
        .await
        .unwrap();
    assert_eq!(
        timing_off.verify_us, 0,
        "verification_enabled=false must zero verify_us, got {}",
        timing_off.verify_us
    );
}

#[tokio::test]
async fn bntm_ivf_verify_us_tracks_verification_enabled() {
    let scorer = BnTmIvfScorer::new();
    let vectors = random_unit_vectors(40, 32, 3);
    let query = random_unit_vectors(1, 32, 4).into_iter().next().unwrap();

    let (handle_on, _) = scorer
        .upload_cluster(&ivf_cfg(true), &vectors)
        .await
        .unwrap();
    let (_, timing_on) = scorer
        .score_with_breakdown(&handle_on, &query, 5)
        .await
        .unwrap();
    assert!(
        timing_on.verify_us > 0,
        "verification_enabled=true must populate verify_us > 0, got {}",
        timing_on.verify_us
    );

    let (handle_off, _) = scorer
        .upload_cluster(&ivf_cfg(false), &vectors)
        .await
        .unwrap();
    let (_, timing_off) = scorer
        .score_with_breakdown(&handle_off, &query, 5)
        .await
        .unwrap();
    assert_eq!(
        timing_off.verify_us, 0,
        "verification_enabled=false must zero verify_us, got {}",
        timing_off.verify_us
    );
}
