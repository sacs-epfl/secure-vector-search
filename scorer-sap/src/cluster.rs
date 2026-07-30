use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use scorer_core::{Hit, Vector};

use crate::crypto::{Ciphertext, encrypt_with_nonce};
use crate::distance::l2;

/// Encrypt all vectors in `data` in parallel. Each vector gets its own seeded RNG
/// (`base_seed ^ index`) so the build is deterministic and fully parallelisable.
/// `on_row` is called once per encrypted vector (rayon worker thread; index is
/// 0-based) and is intended for indicatif progress bars.
///
/// Takes `&[Vector]` (f32) and streams the `f32 → f64` upcast inside the
/// per-row map. Materialising the full `Vec<Vec<f64>>` up front would
/// pre-allocate ~54 GB at an 8.8M × 768 corpus and can OOM the build, so
/// the upcast lives inside the parallel encryption loop as per-row
/// scratch that's freed as soon as `encrypt_with_nonce` returns.
pub fn build_cluster<F: Fn(usize) + Sync>(
    key_s: f64,
    prf_key: [u8; 32],
    data: &[Vector],
    beta: f64,
    base_seed: u64,
    on_row: F,
) -> Vec<Ciphertext> {
    data.par_iter()
        .enumerate()
        .map(|(i, v)| {
            let mut rng = ChaCha20Rng::seed_from_u64(base_seed ^ i as u64);
            let nonce: [u8; 16] = std::array::from_fn(|_| rng.random());
            let m_f64: Vec<f64> = v.0.iter().map(|&x| x as f64).collect();
            let ct = encrypt_with_nonce(key_s, &prf_key, &m_f64, beta, nonce);
            on_row(i);
            ct
        })
        .collect()
}

/// Return the `k` nearest ciphertexts to `query_ct` by L2 distance, as `Hit`s.
/// `k` is capped at `entries.len()`. Scores are negated distances (higher = better).
pub fn knn_with_scores(entries: &[Ciphertext], query_ct: &Ciphertext, k: usize) -> Vec<Hit> {
    let k = k.min(entries.len());
    let mut dists: Vec<(u32, f64)> = entries
        .iter()
        .enumerate()
        .map(|(i, ct)| (i as u32, l2(&query_ct.c, &ct.c)))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    dists[..k]
        .iter()
        .map(|(id, dist)| Hit {
            id: *id,
            score: -(*dist as f32),
        })
        .collect()
}

/// Batched [`knn_with_scores`] over multiple query ciphertexts. Per-query
/// result is bit-equal to calling `knn_with_scores` in a loop: same `l2`
/// per (query, entry) pair, same per-query sort and top-k. The
/// amortisation is memory-bandwidth: each entry's `Vec<f64>` is iterated
/// once per call and contributes distances to every batched query in
/// entries-iteration order, instead of once per query.
pub fn knn_with_scores_batch(
    entries: &[Ciphertext],
    query_cts: &[&Ciphertext],
    k: usize,
) -> Vec<Vec<Hit>> {
    let b = query_cts.len();
    if b == 0 {
        return Vec::new();
    }
    let k_capped = k.min(entries.len());

    // Per-query dist buffer, populated entries-major so each entry's
    // ciphertext is loaded once and contributes to every batched query.
    // Buffer order matches knn_with_scores's `iter().enumerate().map().collect()`
    // (entries-iteration order), so per-query sort sees a bit-equal input.
    let mut dists: Vec<Vec<(u32, f64)>> =
        (0..b).map(|_| Vec::with_capacity(entries.len())).collect();
    for (i, entry) in entries.iter().enumerate() {
        let id = i as u32;
        for (q_idx, qct) in query_cts.iter().enumerate() {
            dists[q_idx].push((id, l2(&qct.c, &entry.c)));
        }
    }

    dists
        .into_iter()
        .map(|mut q_dists| {
            q_dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            q_dists[..k_capped]
                .iter()
                .map(|(id, dist)| Hit {
                    id: *id,
                    score: -(*dist as f32),
                })
                .collect()
        })
        .collect()
}
