//! BFV second-layer compression — Tiptoe-specific glue.
//!
//! Computes `Enc_BFV(H · s)` from a SimplePIR hint matrix
//! `H ∈ Z_q^{r × n_lwe}` and an LWE secret `s ∈ {0,1,2}^{n_lwe}`,
//! shrinking the per-query online download from ~4096× plaintext to
//! ~1× BFV-ciphertext per limb per chunk.
//!
//! ### Decomposition
//!
//! Each entry of `H` is split into `MAX_LIMBS_64 = 16` 4-bit limbs. The
//! server only computes the **top 8** limbs (`NUM_LIMBS_64`); the
//! budget calculation
//! `secret_dim · max_secret · max_limb = 2048 · 2 · 15 = 61440 < t = 65537`
//! keeps every limb product within the plaintext modulus.
//!
//! Rows of `H` are partitioned into chunks of `N` rows each (where
//! `N = BfvParameters::degree() = 2048`). For each `(limb, chunk, col)`,
//! a BFV plaintext is built whose `i`-th polynomial coefficient holds
//! the `(chunk·N + i, col)` entry of `H`'s limb. The server's encrypted
//! secret is one BFV ciphertext per LWE secret coordinate, with
//! coefficient 0 = `s[j]` and zeros elsewhere.
//!
//! ### Server hot path
//!
//! For each `(limb, chunk)`, the server computes the inner product
//! `sum_j Enc_BFV(s[j]) · pt[limb][chunk · n_lwe + j]`. Because each
//! `Enc_BFV(s[j])` only has coefficient 0 nonzero in plaintext space,
//! polynomial multiplication mod (X^N + 1) yields a result whose
//! coefficient `i` equals `sum_j s[j] · limb_b(H[chunk·N + i, j])` —
//! exactly the limb at row `chunk·N + i` of `H · s`.
//!
//! ### Client recovery
//!
//! The client BFV-decrypts each `(limb, chunk)` ciphertext to get the
//! limb's polynomial coefficients in `Z_t`. Each coefficient is mapped
//! back to a signed integer via `from_modulo_p`, scaled by
//! `1 << (4 · limb_idx)`, and summed into the recovered `H · s` vector
//! (in `Z_q = Z_{2^64}`, wrapping).

use std::sync::Arc;

use fhe::bfv::{BfvParameters, Ciphertext, Encoding, Plaintext, SecretKey, dot_product_scalar};
use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter};
use rand_fhe::{CryptoRng, RngCore};

use crate::pir::lwe::SecretKey as LweSecretKey;

/// 4 bits per limb.
pub const BITS_PER_LIMB: usize = 4;

/// Total number of limbs covering a 64-bit hint entry.
pub const MAX_LIMBS_64: usize = 64 / BITS_PER_LIMB;

/// Number of high-order limbs the server actually computes over for
/// 64-bit ciphertext-modulus values.
pub const NUM_LIMBS_64: usize = 8;

/// BFV plaintext encoding of one limb of the hint matrix `H`, per
/// `(chunk, column)` pair. The outer index walks limbs (top-first); the
/// inner Vec is `chunks * n_lwe` plaintexts in row-major
/// `(chunk, col)` order.
pub struct CompressedHint {
    pts: Vec<Vec<Plaintext>>,
    r: usize,
    n_lwe: usize,
    chunks: usize,
}

impl std::fmt::Debug for CompressedHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompressedHint")
            .field("r", &self.r)
            .field("n_lwe", &self.n_lwe)
            .field("chunks", &self.chunks)
            .field("limbs", &self.pts.len())
            .finish()
    }
}

impl CompressedHint {
    pub fn r(&self) -> usize {
        self.r
    }
    pub fn n_lwe(&self) -> usize {
        self.n_lwe
    }
    pub fn chunks(&self) -> usize {
        self.chunks
    }
    pub fn num_limbs(&self) -> usize {
        self.pts.len()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CompressError {
    #[error("hint length {actual} doesn't match r * n_lwe = {expected} (r={r}, n_lwe={n_lwe})")]
    HintLengthMismatch {
        actual: usize,
        expected: usize,
        r: usize,
        n_lwe: usize,
    },

    #[error("encrypted-secret length {actual} doesn't match expected n_lwe = {expected}")]
    EncSecretLengthMismatch { actual: usize, expected: usize },

    #[error(
        "BFV answers shape mismatch: got {limbs} limbs × varying chunks, expected {expected_limbs} × {expected_chunks}"
    )]
    AnswersShapeMismatch {
        limbs: usize,
        expected_limbs: usize,
        expected_chunks: usize,
    },

    #[error("LWE secret value {value} at index {index} is out of {{0,1,2}} range")]
    LweSecretOutOfRange { index: usize, value: u8 },

    #[error("BFV operation failed: {0}")]
    Fhe(#[from] fhe::Error),
}

/// Extract the `chunk_idx`-th 4-bit limb from a 64-bit value.
#[inline]
fn get_chunk(v: u64, chunk_idx: usize) -> u64 {
    let mask = (1u64 << BITS_PER_LIMB) - 1;
    (v >> (chunk_idx * BITS_PER_LIMB)) & mask
}

/// Build the per-limb plaintext decomposition of a SimplePIR hint
/// matrix `H ∈ Z_q^{r × n_lwe}` (row-major, `q = 2^64`).
///
/// Only the top [`NUM_LIMBS_64`] limbs are computed; the outer index
/// walks them top-first (`limb_idx = MAX_LIMBS_64 - b - 1`) so that the
/// client's limb-scaling formula in [`recover_h_s`] lines up.
pub fn decompose_hint(
    hint: &[u64],
    r: usize,
    n_lwe: usize,
    params: &Arc<BfvParameters>,
) -> Result<CompressedHint, CompressError> {
    if hint.len() != r * n_lwe {
        return Err(CompressError::HintLengthMismatch {
            actual: hint.len(),
            expected: r * n_lwe,
            r,
            n_lwe,
        });
    }

    let n = params.degree();
    let chunks = r.div_ceil(n);
    let mut pts: Vec<Vec<Plaintext>> = Vec::with_capacity(NUM_LIMBS_64);

    let mut vals = vec![0u64; n];
    for b in 0..NUM_LIMBS_64 {
        let limb_idx = MAX_LIMBS_64 - b - 1;
        let mut limb_pts = Vec::with_capacity(chunks * n_lwe);

        for chunk in 0..chunks {
            for col in 0..n_lwe {
                vals.iter_mut().for_each(|v| *v = 0);
                let row_start = chunk * n;
                let row_end = (row_start + n).min(r);
                for row in row_start..row_end {
                    let i = row - row_start;
                    vals[i] = get_chunk(hint[row * n_lwe + col], limb_idx);
                }
                let pt = Plaintext::try_encode(&vals, Encoding::poly(), params)?;
                limb_pts.push(pt);
            }
        }

        pts.push(limb_pts);
    }

    Ok(CompressedHint {
        pts,
        r,
        n_lwe,
        chunks,
    })
}

/// Encrypt each LWE secret coordinate as its own BFV ciphertext, with
/// coefficient 0 of the polynomial holding `s[j]` and the remaining
/// `N - 1` coefficients zero.
///
/// The output is the "query token": `n_lwe` BFV ciphertexts under
/// `bfv_sk`, single-use per query. Tiptoe's privacy model treats each
/// token as fresh per query, so callers must generate `lwe_secret` and
/// `bfv_sk` fresh per call.
pub fn encrypt_secret_under_bfv<R: RngCore + CryptoRng>(
    lwe_secret: &LweSecretKey,
    bfv_sk: &SecretKey,
    params: &Arc<BfvParameters>,
    rng: &mut R,
) -> Result<Vec<Ciphertext>, CompressError> {
    let n = params.degree();
    let s = lwe_secret.as_slice();
    let mut out = Vec::with_capacity(s.len());
    let mut vals = vec![0u64; n];

    for (j, &s_j) in s.iter().enumerate() {
        if s_j > 2 {
            return Err(CompressError::LweSecretOutOfRange {
                index: j,
                value: s_j,
            });
        }
        vals[0] = s_j as u64;
        let pt = Plaintext::try_encode(&vals, Encoding::poly(), params)?;
        let ct: Ciphertext = bfv_sk.try_encrypt(&pt, rng)?;
        out.push(ct);
        // Reset only the slot we touched (the rest stay zero).
        vals[0] = 0;
    }

    Ok(out)
}

/// Compute `Enc_BFV(limb_b(H · s))` for every `(limb, chunk)` pair.
///
/// For each limb `b` and chunk `chunk_i`, the result ciphertext is the
/// homomorphic inner product
/// `Σⱼ enc_secret[j] · pts[b][chunk_i · n_lwe + j]`, computed via
/// [`fhe::bfv::dot_product_scalar`] (the optimised plain×cipher
/// inner-product primitive).
///
/// Output layout: `out[b][chunk_i]` is the BFV ciphertext encoding
/// limb `b` of `H · s` for chunk `chunk_i`. `out.len() == NUM_LIMBS_64`,
/// `out[b].len() == compressed.chunks()`.
pub fn apply_hint(
    compressed: &CompressedHint,
    enc_secret: &[Ciphertext],
) -> Result<Vec<Vec<Ciphertext>>, CompressError> {
    if enc_secret.len() != compressed.n_lwe {
        return Err(CompressError::EncSecretLengthMismatch {
            actual: enc_secret.len(),
            expected: compressed.n_lwe,
        });
    }

    let n_lwe = compressed.n_lwe;
    let chunks = compressed.chunks;
    let mut out = Vec::with_capacity(compressed.pts.len());

    for limb_pts in compressed.pts.iter() {
        let mut chunk_out = Vec::with_capacity(chunks);
        for chunk_i in 0..chunks {
            let pt_slice = &limb_pts[chunk_i * n_lwe..(chunk_i + 1) * n_lwe];
            let ct = dot_product_scalar(enc_secret.iter(), pt_slice.iter())?;
            chunk_out.push(ct);
        }
        out.push(chunk_out);
    }

    Ok(out)
}

/// Recover `H · s ∈ Z_q^r` (with `q = 2^64`) by decrypting and summing
/// the per-limb BFV answers.
///
/// For each limb the decrypted polynomial coefficients are mapped back
/// to signed integers via "if `v > t/2`, treat as `v - t`", scaled by
/// `1 << (BITS_PER_LIMB · limb_idx)`, and summed into the output. The
/// final `Z_q` representation comes from reinterpreting the wrapping-
/// `i64` accumulator as `u64`.
pub fn recover_h_s(
    bfv_answers: &[Vec<Ciphertext>],
    bfv_sk: &SecretKey,
    r: usize,
    params: &Arc<BfvParameters>,
) -> Result<Vec<u64>, CompressError> {
    let n = params.degree();
    let chunks = r.div_ceil(n);

    if bfv_answers.len() != NUM_LIMBS_64 || bfv_answers.iter().any(|v| v.len() != chunks) {
        return Err(CompressError::AnswersShapeMismatch {
            limbs: bfv_answers.len(),
            expected_limbs: NUM_LIMBS_64,
            expected_chunks: chunks,
        });
    }

    let t = params.plaintext();
    let mut acc = vec![0i64; r];

    for (b, limb_cts) in bfv_answers.iter().enumerate() {
        let limb_idx = MAX_LIMBS_64 - b - 1;
        // Limb scale fits in u64 because limb_idx <= 15 → shift <= 60.
        let scale = 1i64 << (BITS_PER_LIMB * limb_idx);

        for (chunk_i, ct) in limb_cts.iter().enumerate() {
            let pt = bfv_sk.try_decrypt(ct)?;
            let coeffs = Vec::<u64>::try_decode(&pt, Encoding::poly())?;

            let row_start = chunk_i * n;
            let row_end = (row_start + n).min(r);
            for (i, acc_row) in acc[row_start..row_end].iter_mut().enumerate() {
                let signed = from_modulo_p(t, coeffs[i]);
                *acc_row = acc_row.wrapping_add(signed.wrapping_mul(scale));
            }
        }
    }

    Ok(acc.into_iter().map(|v| v as u64).collect())
}

/// Map a `Z_t` value back to a signed integer using the "if `v > t/2`,
/// treat as `v - t`" convention. Panics if `v >= t` (an internal
/// invariant that would indicate a malformed BFV decode).
#[inline]
fn from_modulo_p(t: u64, v: u64) -> i64 {
    debug_assert!(v < t, "BFV-decoded coefficient {v} >= t = {t}");
    if v > t / 2 {
        v as i64 - t as i64
    } else {
        v as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bfv::tiptoe_text_params;
    use crate::pir::lwe::{Params as LweParams, gen_secret as lwe_gen_secret};
    use rand_chacha_fhe::ChaCha20Rng;
    use rand_fhe::SeedableRng;

    fn fhe_rng() -> ChaCha20Rng {
        ChaCha20Rng::seed_from_u64(42)
    }

    /// rand 0.10-flavoured RNG for the LWE-side helpers (which take the
    /// workspace `rand` traits, not the renamed `rand_fhe` traits).
    fn lwe_rng() -> rand_chacha::ChaCha20Rng {
        use rand::SeedableRng;
        rand_chacha::ChaCha20Rng::seed_from_u64(42)
    }

    #[test]
    fn get_chunk_extracts_4_bit_chunks_low_to_high() {
        // 0x_F_E_D_C_B_A_9_8_7_6_5_4_3_2_1_0 — limb i is hex digit i.
        let v: u64 = 0xFEDC_BA98_7654_3210;
        for i in 0..MAX_LIMBS_64 {
            assert_eq!(get_chunk(v, i), i as u64, "limb {i}");
        }
    }

    #[test]
    fn get_chunk_zero_and_max() {
        assert_eq!(get_chunk(0, 0), 0);
        assert_eq!(get_chunk(0, 15), 0);
        assert_eq!(get_chunk(u64::MAX, 0), 0xF);
        assert_eq!(get_chunk(u64::MAX, 15), 0xF);
    }

    #[test]
    fn from_modulo_p_signed_mapping() {
        let t: u64 = 65537;
        assert_eq!(from_modulo_p(t, 0), 0);
        assert_eq!(from_modulo_p(t, 1), 1);
        assert_eq!(from_modulo_p(t, t / 2), (t / 2) as i64);
        // Values > t/2 are negative.
        assert_eq!(from_modulo_p(t, t / 2 + 1), -((t / 2) as i64));
        assert_eq!(from_modulo_p(t, t - 1), -1);
    }

    #[test]
    fn decompose_hint_top_8_limbs_recombine_to_top_32_bits() {
        // Plaintext-only sanity: the 8 limbs we compute correspond to the
        // top 32 bits of each entry. Reassembling them must agree with
        // `entry & 0xFFFF_FFFF_0000_0000`.
        let params = tiptoe_text_params();
        let r = 3;
        let n_lwe = 4;
        // Hand-picked entries that exercise both halves of the value.
        let hint: Vec<u64> = vec![
            0xFEDC_BA98_7654_3210,
            0x0123_4567_89AB_CDEF,
            0x0000_0000_FFFF_FFFF,
            0xFFFF_FFFF_0000_0000,
            0xDEAD_BEEF_CAFE_BABE,
            0x1111_2222_3333_4444,
            0xAAAA_5555_AAAA_5555,
            0,
            u64::MAX,
            0x1, // small
            0x8000_0000_0000_0000,
            0x0FFF_FFFF_FFFF_FFFF,
        ];
        assert_eq!(hint.len(), r * n_lwe);

        let compressed = decompose_hint(&hint, r, n_lwe, &params).expect("decompose");
        assert_eq!(compressed.num_limbs(), NUM_LIMBS_64);
        assert_eq!(compressed.r(), r);
        assert_eq!(compressed.n_lwe(), n_lwe);
        // r=3 < N=2048 → exactly one chunk.
        assert_eq!(compressed.chunks(), 1);

        // Walk the encoded plaintexts back to limb values via decode and
        // recombine. Tests the encoding direction of the trick without
        // touching encryption.
        for (b, limb_pts) in compressed.pts.iter().enumerate() {
            let limb_idx = MAX_LIMBS_64 - b - 1;
            for (col, pt) in limb_pts.iter().enumerate() {
                // chunk=0 (single chunk for r=3 < N=2048).
                let coeffs = Vec::<u64>::try_decode(pt, Encoding::poly()).expect("decode");
                for (row, &expected_entry) in
                    hint.iter().skip(col).step_by(n_lwe).take(r).enumerate()
                {
                    let expected_limb = get_chunk(expected_entry, limb_idx);
                    assert_eq!(coeffs[row], expected_limb, "b={b} col={col} row={row}");
                }
            }
        }

        // Reassemble: only the top NUM_LIMBS_64 limbs are recoverable;
        // the bottom NUM_LIMBS_64 bits are dropped by design.
        for col in 0..n_lwe {
            for row in 0..r {
                let mut reassembled: u64 = 0;
                for (b, limb_pts) in compressed.pts.iter().enumerate() {
                    let limb_idx = MAX_LIMBS_64 - b - 1;
                    let coeffs =
                        Vec::<u64>::try_decode(&limb_pts[col], Encoding::poly()).expect("decode");
                    reassembled |= coeffs[row] << (BITS_PER_LIMB * limb_idx);
                }
                let mask: u64 = !((1u64 << (BITS_PER_LIMB * (MAX_LIMBS_64 - NUM_LIMBS_64))) - 1);
                let expected = hint[row * n_lwe + col] & mask;
                assert_eq!(reassembled, expected, "row={row} col={col}");
            }
        }
    }

    #[test]
    fn decompose_hint_rejects_size_mismatch() {
        let params = tiptoe_text_params();
        let hint = vec![0u64; 5]; // claim 2×3=6
        let err = decompose_hint(&hint, 2, 3, &params).expect_err("size mismatch");
        match err {
            CompressError::HintLengthMismatch {
                actual, expected, ..
            } => {
                assert_eq!(actual, 5);
                assert_eq!(expected, 6);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    /// Compute `H · s mod 2^64` directly (oracle for the compression).
    fn h_times_s_oracle(hint: &[u64], r: usize, n_lwe: usize, s: &[u8]) -> Vec<u64> {
        assert_eq!(hint.len(), r * n_lwe);
        assert_eq!(s.len(), n_lwe);
        let mut out = vec![0u64; r];
        for i in 0..r {
            let mut acc: u64 = 0;
            for j in 0..n_lwe {
                acc = acc.wrapping_add(hint[i * n_lwe + j].wrapping_mul(s[j] as u64));
            }
            out[i] = acc;
        }
        out
    }

    /// Mask off the bits the limb decomposition discards. With the top
    /// `NUM_LIMBS_64` of `MAX_LIMBS_64` limbs computed, anything below
    /// bit `BITS_PER_LIMB * (MAX_LIMBS_64 - NUM_LIMBS_64) = 32` is
    /// dropped.
    const RECOVERED_MASK: u64 = !((1u64 << (BITS_PER_LIMB * (MAX_LIMBS_64 - NUM_LIMBS_64))) - 1);

    #[test]
    fn apply_hint_recovers_top_32_bits_of_h_times_s_small() {
        // End-to-end limb-decomposition composition on a tiny synthetic
        // H. We pick H entries whose bottom 32 bits are zero so the
        // top-8-limbs recovery is exact; this isolates BFV/limb
        // correctness from the deliberate truncation.
        let params = tiptoe_text_params();
        let lwe_params = LweParams::tiptoe_text();
        let n_lwe = lwe_params.n; // 2048

        let r = 4;
        // Build H with entries shifted into the top 32 bits so the
        // bottom-half discard is harmless. Values are small enough not
        // to overflow the limb-product budget once multiplied by the
        // ternary secret and summed.
        let hint: Vec<u64> = (0..r * n_lwe)
            .map(|k| ((k as u64 * 17 + 3) & 0xFFFF) << 32)
            .collect();

        let mut lwe_rng = lwe_rng();
        let mut bfv_rng = fhe_rng();

        let lwe_secret = lwe_gen_secret(&lwe_params, &mut lwe_rng);
        let bfv_sk = SecretKey::random(&params, &mut bfv_rng);

        let compressed = decompose_hint(&hint, r, n_lwe, &params).expect("decompose");
        let enc_secret = encrypt_secret_under_bfv(&lwe_secret, &bfv_sk, &params, &mut bfv_rng)
            .expect("encrypt secret");
        let answers = apply_hint(&compressed, &enc_secret).expect("apply hint");
        let recovered = recover_h_s(&answers, &bfv_sk, r, &params).expect("recover");

        let oracle = h_times_s_oracle(&hint, r, n_lwe, lwe_secret.as_slice());
        for i in 0..r {
            assert_eq!(
                recovered[i] & RECOVERED_MASK,
                oracle[i] & RECOVERED_MASK,
                "row {i}: recovered={:016x} oracle={:016x}",
                recovered[i],
                oracle[i]
            );
        }
    }

    #[test]
    fn apply_hint_compose_with_simplepir_decrypts_to_m_db_times_q() {
        // Compose with SimplePIR: build a small SimplePIR setup, run an
        // LHE query end-to-end using compress.rs to recover H · s, and
        // assert the SimplePIR decode lands on `M_db · q mod P`.
        //
        // The limb decomposition recovers `H · s` only up to the LWE
        // round budget — the bottom 32 bits of each H entry are dropped
        // in exchange for ~4096× compression. The missing bits'
        // contribution to `H · s` is bounded by
        // `n_lwe · 2 · 2^32 ≈ 2^45 < Δ/2 = 2^46`, so SimplePIR's
        // round-to-nearest-Δ absorbs the discrepancy. This test is the
        // empirical check that the budget works at our parameters —
        // any narrower assertion (e.g., bitmask equality on the top 32
        // bits) would over-constrain the recovery.
        use crate::pir::simplepir::{encrypt_lhe, preprocess, server_answer};

        let bfv_params = tiptoe_text_params();
        let lwe_params = LweParams::tiptoe_text();
        let n_lwe = lwe_params.n;

        let r = 2;
        let c = 3;
        let m_db: Vec<u64> = (0..r * c)
            .map(|i| (i as u64 * 23 + 7) % lwe_params.p())
            .collect();
        let seed = [9u8; 32];
        let setup = preprocess(&m_db, r, c, seed, &lwe_params);
        assert_eq!(setup.hint.len(), r * n_lwe);

        let mut lwe_rng = lwe_rng();
        let mut bfv_rng = fhe_rng();

        let lwe_secret = lwe_gen_secret(&lwe_params, &mut lwe_rng);
        let bfv_sk = SecretKey::random(&bfv_params, &mut bfv_rng);

        // Query token: BFV-encrypt the LWE secret coordinates.
        let enc_secret = encrypt_secret_under_bfv(&lwe_secret, &bfv_sk, &bfv_params, &mut bfv_rng)
            .expect("encrypt secret");

        // Server recovers H · s under BFV (the compression hot path).
        let compressed = decompose_hint(&setup.hint, r, n_lwe, &bfv_params).expect("decompose");
        let answers = apply_hint(&compressed, &enc_secret).expect("apply hint");
        let recovered_h_s = recover_h_s(&answers, &bfv_sk, r, &bfv_params).expect("recover");

        // LHE flow: client encrypts q, server returns M_db · ct, client
        // subtracts the recovered H · s and rounds.
        let query: Vec<u64> = vec![1, 2, 3];
        let ct = encrypt_lhe(&setup, &lwe_secret, &query, &lwe_params, &mut lwe_rng);
        let ans = server_answer(&m_db, r, c, &ct);

        let recovered_plaintext: Vec<u64> = ans
            .iter()
            .zip(recovered_h_s.iter())
            .map(|(&a, &h)| lwe_params.round(a.wrapping_sub(h)))
            .collect();

        let p = lwe_params.p();
        let expected: Vec<u64> = (0..r)
            .map(|i| {
                let mut acc: u64 = 0;
                for (j, &q_j) in query.iter().enumerate() {
                    acc = acc.wrapping_add(m_db[i * c + j].wrapping_mul(q_j));
                }
                acc % p
            })
            .collect();
        assert_eq!(recovered_plaintext, expected);
    }

    #[test]
    fn apply_hint_rejects_wrong_secret_length() {
        let params = tiptoe_text_params();
        let lwe_params = LweParams::tiptoe_text();
        let r = 1;
        let n_lwe = lwe_params.n;
        let hint = vec![0u64; r * n_lwe];

        let compressed = decompose_hint(&hint, r, n_lwe, &params).expect("decompose");

        let mut bfv_rng = fhe_rng();
        let bfv_sk = SecretKey::random(&params, &mut bfv_rng);
        let pt = Plaintext::try_encode(&[0u64], Encoding::poly(), &params).unwrap();
        let bogus_ct: Ciphertext = bfv_sk.try_encrypt(&pt, &mut bfv_rng).unwrap();

        let bad = vec![bogus_ct; n_lwe - 1];
        let err = apply_hint(&compressed, &bad).expect_err("len mismatch");
        match err {
            CompressError::EncSecretLengthMismatch { actual, expected } => {
                assert_eq!(actual, n_lwe - 1);
                assert_eq!(expected, n_lwe);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
}
