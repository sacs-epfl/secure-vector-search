use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use scorer_core::{Device, Scorer, Vector};
use scorer_plaintext::{PlaintextConfig, PlaintextScorer};
use scorer_sap::{SapConfig, SapScorer, keygen};

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

/// At β=0, SAP ciphertext is exactly s·m, preserving L2 ordering. With nprobe =
/// n_centroids (full probe), plaintext IVF gives exact results. Both scorers must
/// therefore return identical top-10 rankings → recall@10 = 1.0.
#[tokio::test]
async fn sap_beta_zero_matches_plaintext_recall_at_10() {
    let n = 200;
    let dim = 32;
    let k = 10;
    let n_centroids = 16;

    let corpus = random_vectors(n, dim, 1);
    let queries = random_vectors(10, dim, 2);

    // Plaintext with full probe: exact IVF = brute force.
    let pt = PlaintextScorer::new();
    let pt_config = PlaintextConfig {
        n_centroids,
        nprobe: n_centroids,
        train_seed: 42,
        max_iter: 25,
        ..Default::default()
    };
    let (pt_handle, _) = pt.upload_cluster(&pt_config, &corpus).await.unwrap();

    // SAP at β=0.
    let sap = SapScorer;
    let sap_config = SapConfig {
        key: keygen(1.0, 2.0, 7),
        beta: 0.0,
        upload_seed: 42,
        progress: None,
        device: Device::Cpu,
        vram_budget_bytes: None,
    };
    let (sap_handle, _) = sap.upload_cluster(&sap_config, &corpus).await.unwrap();

    for (qi, query) in queries.iter().enumerate() {
        let pt_hits = pt.score(&pt_handle, query, k).await.unwrap();
        let sap_hits = sap.score(&sap_handle, query, k).await.unwrap();

        let pt_ids: Vec<u32> = pt_hits.iter().map(|h| h.id).collect();
        let sap_ids: Vec<u32> = sap_hits.iter().map(|h| h.id).collect();

        let recall = eval_harness::recall_at_k(&pt_ids, &sap_ids, k);
        assert_eq!(
            recall, 1.0,
            "query {qi}: SAP β=0 recall@{k} = {recall:.3} (expected 1.0)\n\
             plaintext top-{k}: {pt_ids:?}\n\
             sap top-{k}:       {sap_ids:?}",
        );
    }
}
