//! Per-query scoring orchestration: route → encrypt → server LHE +
//! BFV compression → BFV decrypt → top-k.
//!
//! Adapted to our single-bin matrix layout (`cluster.rs`) and the
//! in-house LWE / SimplePIR / BFV stack (`pir/`, `bfv/`). The function
//! [`score_query`] is sync and runs entirely on a blocking task; the
//! async [`crate::TiptoeScorer::score`] wrapper drives RNG seed
//! generation and `spawn_blocking`.
//!
//! Every call generates fresh LWE and BFV secrets (single-use token
//! semantics): no per-query state survives between calls, and the
//! handle holds **no** key material.

use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_chacha_fhe::ChaCha20Rng as ChaCha20RngFhe;
// rand_chacha_fhe is the rand-0.9-flavoured `rand_chacha` crate; its
// `seed_from_u64` lives on the rand-0.9 `SeedableRng` trait, which we
// pull in under an alias to avoid clashing with the rand-0.10 one.
use rand_fhe::SeedableRng as SeedableRngFhe;
use scorer_core::Hit;

use crate::TiptoeError;
use crate::TiptoeHandle;
use crate::bfv::SecretKey as BfvSecretKey;
use crate::bfv::compress::{apply_hint, encrypt_secret_under_bfv, recover_h_s};
use crate::encoding::max_magnitude;
use crate::pir::lwe::gen_secret as lwe_gen_secret;
use crate::pir::simplepir::{encrypt_lhe, server_answer};
use crate::routing::{build_augmented_query, route};

/// Sortable `f32` score recovered from a single recovered `M·q̃`
/// component. The raw value lives in `Z_p`; we map back to a signed
/// integer (negative-mod-P convention) and divide by `mag²` so the
/// magnitude is comparable across `quantisation_bits` choices.
#[inline]
fn score_to_f32(z_p: u64, p: u64, quantisation_bits: u8) -> f32 {
    let signed = if z_p >= p / 2 {
        z_p as i64 - p as i64
    } else {
        z_p as i64
    };
    let mag = max_magnitude(quantisation_bits) as f32;
    signed as f32 / (mag * mag)
}

/// Run one Tiptoe query end-to-end.
///
/// The caller supplies independent seeds for the LWE-side and BFV-side
/// RNGs; the async wrapper draws both from the OS RNG once per call.
/// Splitting the seeds keeps test runs deterministic and avoids any
/// accidental correlation between the rand-0.10 and rand-0.9 RNG
/// streams (the two `fhe` deps live in different version namespaces;
/// see `Cargo.toml`).
pub fn score_query(
    handle: &TiptoeHandle,
    query: &[f32],
    k: usize,
    lwe_seed: u64,
    bfv_seed: u64,
) -> Result<Vec<Hit>, TiptoeError> {
    if query.len() != handle.dim {
        return Err(TiptoeError::DimensionMismatch {
            query_dim: query.len(),
            index_dim: handle.dim,
        });
    }

    let mut lwe_rng = ChaCha20Rng::seed_from_u64(lwe_seed);
    let mut bfv_rng = <ChaCha20RngFhe as SeedableRngFhe>::seed_from_u64(bfv_seed);

    // 1. Plaintext routing: pick the nearest centroid.
    let cluster_idx = route(query, &handle.index.centroids);
    let cluster_size = handle.encoded.cluster_membership(cluster_idx).len();

    // 2. Build the augmented query q̃ ∈ Z_p^{n_centroids·d}.
    let p = handle.lwe_params.p();
    let q_tilde = build_augmented_query(
        query,
        cluster_idx,
        &handle.encoded,
        handle.quantisation_bits,
        p,
    )
    .map_err(TiptoeError::QuantisationOverflow)?;

    // 3. Generate fresh per-query keys (single-use semantics).
    let lwe_secret = lwe_gen_secret(&handle.lwe_params, &mut lwe_rng);
    let bfv_sk = BfvSecretKey::random(&handle.bfv_params, &mut bfv_rng);

    // 4. LWE-encrypt the augmented query under the SimplePIR LHE flavour.
    let ct = encrypt_lhe(
        &handle.pir_setup,
        &lwe_secret,
        &q_tilde,
        &handle.lwe_params,
        &mut lwe_rng,
    );

    // 5. Server: M · ct ∈ Z_q^{m_max} (the SimplePIR answer for the LHE
    //    flavour; carries Δ·(M·q̃) + H·s + small noise).
    let m_max = handle.encoded.m_max();
    let width = handle.encoded.width();
    let ans = server_answer(handle.encoded.matrix(), m_max, width, &ct);

    // 6. BFV compression server hot path: encrypt the LWE secret
    //    under BFV (the query token), then apply the precomputed hint
    //    to obtain Enc_BFV(H·s) limb-decomposed.
    let enc_secret =
        encrypt_secret_under_bfv(&lwe_secret, &bfv_sk, &handle.bfv_params, &mut bfv_rng)?;
    let bfv_answers = apply_hint(&handle.compressed_hint, &enc_secret)?;

    // 7. Client: BFV-decrypt and recombine limbs to recover H·s mod 2^64.
    let h_s = recover_h_s(&bfv_answers, &bfv_sk, m_max, &handle.bfv_params)?;

    // 8. SimplePIR decode: (ans − H·s) rounded to the nearest multiple
    //    of Δ, mod P → M·q̃ ∈ Z_p^{m_max}.
    let m_q_tilde: Vec<u64> = ans
        .iter()
        .zip(h_s.iter())
        .map(|(&a, &h)| handle.lwe_params.round(a.wrapping_sub(h)))
        .collect();

    // 9. The first `cluster_size` rows of M·q̃ are the inner-product
    //    scores for the routed cluster; the remaining `m_max −
    //    cluster_size` rows are zero-padding from the matrix layout.
    let membership = handle.encoded.cluster_membership(cluster_idx);
    let mut scored: Vec<(u32, f32)> = m_q_tilde[..cluster_size]
        .iter()
        .zip(membership.iter())
        .map(|(&z, &id)| (id, score_to_f32(z, p, handle.quantisation_bits)))
        .collect();

    // 10. Top-k by score descending. Cap k at cluster size per the
    //     CLAUDE.md scorer invariant ("k > cluster_size is capped, not
    //     rejected").
    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);

    Ok(scored
        .into_iter()
        .map(|(id, score)| Hit { id, score })
        .collect())
}

/// Per-query substep timings reported by [`score_query_with_breakdown`].
/// `route` (centroid scan), `lwe_encrypt` (build augmented query + LWE
/// secret + LWE-encrypt), `server` (server matvec), `bfv_decompress`
/// (BFV compression layer: BFV secret + encrypt LWE secret under BFV +
/// apply hint + recover H·s), `decode` (SimplePIR decode + score
/// recovery + top-k).
#[derive(Debug, Clone, Copy, Default)]
pub struct TiptoeTiming {
    pub route_us: u64,
    pub lwe_encrypt_us: u64,
    pub server_us: u64,
    pub bfv_decompress_us: u64,
    pub decode_us: u64,
}

/// Off-hot-path equivalent of [`score_query`] with per-substep
/// timings. Mirrors the same protocol — fresh LWE and BFV per-query
/// secrets, single-cluster routing, BFV limb-decomposition recovery —
/// with `Instant::now()` pairs around each substep boundary. Do not
/// use from the throughput path.
pub fn score_query_with_breakdown(
    handle: &TiptoeHandle,
    query: &[f32],
    k: usize,
    lwe_seed: u64,
    bfv_seed: u64,
) -> Result<(Vec<Hit>, TiptoeTiming), TiptoeError> {
    if query.len() != handle.dim {
        return Err(TiptoeError::DimensionMismatch {
            query_dim: query.len(),
            index_dim: handle.dim,
        });
    }

    let mut lwe_rng = ChaCha20Rng::seed_from_u64(lwe_seed);
    let mut bfv_rng = <ChaCha20RngFhe as SeedableRngFhe>::seed_from_u64(bfv_seed);

    // 1. route
    let t = Instant::now();
    let cluster_idx = route(query, &handle.index.centroids);
    let cluster_size = handle.encoded.cluster_membership(cluster_idx).len();
    let route_us = t.elapsed().as_micros() as u64;

    // 2. lwe_encrypt: build q̃, draw LWE secret, LWE-encrypt.
    let t = Instant::now();
    let p = handle.lwe_params.p();
    let q_tilde = build_augmented_query(
        query,
        cluster_idx,
        &handle.encoded,
        handle.quantisation_bits,
        p,
    )
    .map_err(TiptoeError::QuantisationOverflow)?;
    let lwe_secret = lwe_gen_secret(&handle.lwe_params, &mut lwe_rng);
    let ct = encrypt_lhe(
        &handle.pir_setup,
        &lwe_secret,
        &q_tilde,
        &handle.lwe_params,
        &mut lwe_rng,
    );
    let lwe_encrypt_us = t.elapsed().as_micros() as u64;

    // 3. server: M · ct.
    let t = Instant::now();
    let m_max = handle.encoded.m_max();
    let width = handle.encoded.width();
    let ans = server_answer(handle.encoded.matrix(), m_max, width, &ct);
    let server_us = t.elapsed().as_micros() as u64;

    // 4. bfv_decompress: BFV secret + encrypt LWE secret + apply
    //    hint + recover H·s. The BFV compression layer.
    let t = Instant::now();
    let bfv_sk = BfvSecretKey::random(&handle.bfv_params, &mut bfv_rng);
    let enc_secret =
        encrypt_secret_under_bfv(&lwe_secret, &bfv_sk, &handle.bfv_params, &mut bfv_rng)?;
    let bfv_answers = apply_hint(&handle.compressed_hint, &enc_secret)?;
    let h_s = recover_h_s(&bfv_answers, &bfv_sk, m_max, &handle.bfv_params)?;
    let bfv_decompress_us = t.elapsed().as_micros() as u64;

    // 5. decode: SimplePIR round + score-to-f32 + top-k.
    let t = Instant::now();
    let m_q_tilde: Vec<u64> = ans
        .iter()
        .zip(h_s.iter())
        .map(|(&a, &h)| handle.lwe_params.round(a.wrapping_sub(h)))
        .collect();
    let membership = handle.encoded.cluster_membership(cluster_idx);
    let mut scored: Vec<(u32, f32)> = m_q_tilde[..cluster_size]
        .iter()
        .zip(membership.iter())
        .map(|(&z, &id)| (id, score_to_f32(z, p, handle.quantisation_bits)))
        .collect();
    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    let hits: Vec<Hit> = scored
        .into_iter()
        .map(|(id, score)| Hit { id, score })
        .collect();
    let decode_us = t.elapsed().as_micros() as u64;

    Ok((
        hits,
        TiptoeTiming {
            route_us,
            lwe_encrypt_us,
            server_us,
            bfv_decompress_us,
            decode_us,
        },
    ))
}
