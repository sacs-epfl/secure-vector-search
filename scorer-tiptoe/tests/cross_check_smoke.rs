//! Structural smoke test for the in-house Tiptoe port.
//!
//! This is the lowest-fidelity port-correctness check — it asserts that
//! a tiny synthetic Tiptoe scorer produces well-formed output (correct
//! shape, finite scores, monotonically non-increasing, no panics)
//! without comparing to hand-coded expected values. The full validation
//! gate (paired Rust/Go runs + `analysis/tiptoe_diff.py`) lives
//! elsewhere.
//!
//! Corpus size and query count are kept small because every query
//! triggers `n_lwe = 2048` BFV encryptions for the query token; that
//! dominates the runtime even at low `d`.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use scorer_core::{Scorer, Vector};
use scorer_tiptoe::{TiptoeConfig, TiptoeScorer};

/// Generate `n` unit-normalised f32 vectors of dimension `d`. Avoids
/// pure-zero vectors that can collapse k-means.
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

#[tokio::test]
async fn upload_and_score_produces_well_formed_hits() {
    // Small corpus, small d → quantised inner products stay well below
    // P/2 even at quantisation_bits=4, so dequantisation is faithful.
    let n = 20;
    let d = 8;
    let n_centroids = 4;
    let k = 5;

    let vectors = random_vectors(n, d, 1);
    let scorer = TiptoeScorer::new();
    let config = TiptoeConfig {
        n_centroids,
        ..Default::default()
    };
    let (handle, _build) = scorer
        .upload_cluster(&config, &vectors)
        .await
        .expect("upload");

    let queries = random_vectors(3, d, 99);
    for (qi, q) in queries.iter().enumerate() {
        let hits = scorer.score(&handle, q, k).await.expect("score");

        // 1. Shape: ≤ k hits, all ids are valid corpus indices.
        assert!(hits.len() <= k, "q{qi}: returned {} > k={k}", hits.len());
        for h in &hits {
            assert!(h.id < n as u32, "q{qi}: id {} out of range", h.id);
            assert!(h.score.is_finite(), "q{qi}: non-finite score {}", h.score);
        }

        // 2. Monotonically non-increasing scores (top-k contract).
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score, "q{qi}: not descending: {hits:?}");
        }
    }
}

#[tokio::test]
async fn k_capped_at_routed_cluster_size() {
    // The routed cluster has at most `cluster_size` vectors; if the
    // caller asks for more, we return `cluster_size` per the
    // scorer invariant.
    let vectors = random_vectors(20, 8, 2);
    let scorer = TiptoeScorer::new();
    let (handle, _build) = scorer
        .upload_cluster(
            &TiptoeConfig {
                n_centroids: 4,
                ..Default::default()
            },
            &vectors,
        )
        .await
        .unwrap();

    let q = random_vectors(1, 8, 200).remove(0);
    let hits = scorer.score(&handle, &q, 1000).await.unwrap();
    // n_centroids=4 over 20 vectors → cluster_size ≤ 20 / 4 (with
    // reasonable balance), so the cap is well below 1000 either way.
    assert!(hits.len() <= 20);
    assert!(!hits.is_empty(), "should not be empty");
}

#[tokio::test]
async fn dimension_mismatch_surfaces_as_error() {
    let vectors = random_vectors(20, 8, 3);
    let scorer = TiptoeScorer::new();
    let (handle, _build) = scorer
        .upload_cluster(
            &TiptoeConfig {
                n_centroids: 2,
                ..Default::default()
            },
            &vectors,
        )
        .await
        .unwrap();

    let wrong = random_vectors(1, 16, 300).remove(0);
    let err = scorer
        .score(&handle, &wrong, 3)
        .await
        .expect_err("should fail");
    assert!(
        format!("{err}").contains("dimension"),
        "expected dimension-mismatch error, got: {err}"
    );
}

#[tokio::test]
async fn too_many_centroids_surfaces_as_error() {
    let vectors = random_vectors(3, 4, 4);
    let scorer = TiptoeScorer::new();
    let err = scorer
        .upload_cluster(
            &TiptoeConfig {
                n_centroids: 100,
                ..Default::default()
            },
            &vectors,
        )
        .await
        .expect_err("should fail");
    assert!(
        format!("{err}").contains("exceeds corpus size"),
        "expected too-many-centroids error, got: {err}"
    );
}

#[tokio::test]
async fn empty_corpus_surfaces_as_error() {
    let scorer = TiptoeScorer::new();
    let err = scorer
        .upload_cluster(&TiptoeConfig::default(), &[])
        .await
        .expect_err("should fail");
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("empty") || msg.contains("at least one"),
        "expected empty-corpus error, got: {err}"
    );
}

#[tokio::test]
async fn cache_round_trip_skips_rebuild_on_disk_hit() {
    // Two `with_cache_dir` scorers against the same tmp dir + params:
    // the second's `upload_cluster` must hit the disk cache, skipping
    // the SimplePIR preprocess. We can't time it cheaply in a unit
    // test, so we assert the cache file appears after the first build
    // and that scoring against the second handle still produces
    // well-formed hits identical to the first.
    let vectors = random_vectors(20, 8, 7);
    let tmp = tempfile::tempdir().expect("tmp");
    let cfg = TiptoeConfig {
        n_centroids: 4,
        ..Default::default()
    };

    let scorer1 = TiptoeScorer::with_cache_dir(tmp.path());
    let (handle1, b1) = scorer1.upload_cluster(&cfg, &vectors).await.unwrap();
    assert!(!b1.cache_hit, "first build is cold");

    // Cache file lands.
    let cache_files: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(".tiptoe-cache-")
        })
        .collect();
    assert_eq!(
        cache_files.len(),
        1,
        "expected exactly one .tiptoe-cache-*.bin in {:?}, got {:?}",
        tmp.path(),
        cache_files
    );

    // Fresh scorer instance, same dir → disk-cache hit.
    let scorer2 = TiptoeScorer::with_cache_dir(tmp.path());
    let (handle2, b2) = scorer2.upload_cluster(&cfg, &vectors).await.unwrap();
    assert!(b2.cache_hit, "second build hits the on-disk cache");

    let q = random_vectors(1, 8, 700).remove(0);
    let hits1 = scorer1.score(&handle1, &q, 5).await.unwrap();
    let hits2 = scorer2.score(&handle2, &q, 5).await.unwrap();
    let ids1: Vec<u32> = hits1.iter().map(|h| h.id).collect();
    let ids2: Vec<u32> = hits2.iter().map(|h| h.id).collect();
    assert_eq!(
        ids1, ids2,
        "fresh build and disk-cache load must produce the same top-k IDs"
    );
}

#[tokio::test]
async fn debug_redacts_secrets_or_omits_them() {
    // The handle holds no key material, so Debug is uneventful; verify
    // it doesn't leak any secret intermediates.
    let vectors = random_vectors(20, 8, 5);
    let scorer = TiptoeScorer::new();
    let (handle, _build) = scorer
        .upload_cluster(
            &TiptoeConfig {
                n_centroids: 2,
                ..Default::default()
            },
            &vectors,
        )
        .await
        .unwrap();
    let dbg = format!("{handle:?}");
    // No "secret" / "key" references should slip through into debug.
    let lower = dbg.to_lowercase();
    assert!(
        !lower.contains("secret") && !lower.contains("key"),
        "Debug must not mention any secret state: {dbg}"
    );
    // ChaCha20Rng helper would otherwise warn-as-unused; suppress.
    let _ = ChaCha20Rng::seed_from_u64(0);
}

#[tokio::test]
async fn score_with_breakdown_matches_score() {
    let n = 20;
    let d = 8;
    let n_centroids = 4;
    let k = 5;

    let vectors = random_vectors(n, d, 70);
    let scorer = TiptoeScorer::new();
    let config = TiptoeConfig {
        n_centroids,
        ..Default::default()
    };
    let (handle, _build) = scorer.upload_cluster(&config, &vectors).await.unwrap();
    let q = random_vectors(1, d, 71).remove(0);

    let baseline = scorer.score(&handle, &q, k).await.unwrap();
    let (with_breakdown, timing) = scorer.score_with_breakdown(&handle, &q, k).await.unwrap();

    // Tiptoe samples fresh LWE/BFV secrets per call, so scores are not
    // byte-identical, but the structural recovery (round to nearest Δ)
    // produces the same Z_p values, so top-k IDs match.
    let baseline_ids: Vec<u32> = baseline.iter().map(|h| h.id).collect();
    let breakdown_ids: Vec<u32> = with_breakdown.iter().map(|h| h.id).collect();
    assert_eq!(baseline_ids, breakdown_ids);
    let _ = timing.route_us
        + timing.lwe_encrypt_us
        + timing.server_us
        + timing.bfv_decompress_us
        + timing.decode_us;
}

#[tokio::test]
async fn build_outcome_cache_hit_smoke() {
    let vectors = random_vectors(20, 8, 50);
    let scorer = TiptoeScorer::new();
    let cfg = TiptoeConfig {
        n_centroids: 2,
        ..Default::default()
    };
    let (_h1, b1) = scorer.upload_cluster(&cfg, &vectors).await.unwrap();
    assert!(!b1.cache_hit, "first build must be a cold build");
    let (_h2, b2) = scorer.upload_cluster(&cfg, &vectors).await.unwrap();
    assert!(b2.cache_hit, "second build must hit the in-memory cache");
    assert!(
        b2.build_duration < std::time::Duration::from_millis(100),
        "warm build was {:?}, expected < 100 ms",
        b2.build_duration
    );
}
