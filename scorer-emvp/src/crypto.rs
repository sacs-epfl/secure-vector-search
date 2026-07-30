// EMVP (encrypted matrix-vector product) primitives. D and R are
// pseudorandom matrices derived from the key seed. Client decode is
// O(m·n): the client replays R and subtracts R·q̂ to recover each
// score, rather than using a near-linear trapdoor. Server-side compute
// and communication costs are unaffected, and decoding correctness is
// maintained.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::params::{FIELD_BYTES, P, Q, SEC128};

// ---------------------------------------------------------------------------
// Mersenne prime arithmetic  (p = 2^61 - 1)
// ---------------------------------------------------------------------------

#[inline]
fn fp_reduce(x: u128) -> u64 {
    let lo = (x & P as u128) as u64;
    let hi = (x >> 61) as u64;
    let v = lo + hi;
    if v >= P { v - P } else { v }
}

#[inline]
pub(crate) fn fp_add(a: u64, b: u64) -> u64 {
    let v = a + b;
    if v >= P { v - P } else { v }
}

#[inline]
fn fp_mul(a: u64, b: u64) -> u64 {
    fp_reduce(a as u128 * b as u128)
}

#[inline]
fn fp_neg(a: u64) -> u64 {
    if a == 0 { 0 } else { P - a }
}

// ---------------------------------------------------------------------------
// Quantisation
// ---------------------------------------------------------------------------

/// f32 → F_p: round(v * Q) mod p.
pub(crate) fn quantize(v: f32) -> u64 {
    let vi = (v as f64 * Q as f64).round() as i64;
    let p = P as i64;
    ((vi % p + p) % p) as u64
}

/// Field element → f32 score (signed interpret, divide by Q²).
fn decode_field(v: u64) -> f32 {
    let signed: i64 = if v > P / 2 {
        v as i64 - P as i64
    } else {
        v as i64
    };
    (signed as f64 / (Q as f64 * Q as f64)) as f32
}

// ---------------------------------------------------------------------------
// D matrix  (ell0 × n, row-major, pseudorandom from key seed)
// ---------------------------------------------------------------------------

/// Build the D matrix from `key_seed`.
///
/// D = [I_{ell0} | D'] in row-major order (ell0 × n).
/// The identity block ensures D · q̂ = q when q̂ = (q - D'·u | u).
/// D' (the right ell0 × k_dim block) is pseudorandom from key_seed.
pub(crate) fn generate_d(key_seed: [u8; 32]) -> Vec<u64> {
    let params = &SEC128;
    let ell0 = params.ell0;
    let n = params.n;
    let k_dim = params.k; // n - ell0

    let mut d = vec![0u64; ell0 * n];

    // Identity block: D[i][i] = 1 for i < ell0.
    for i in 0..ell0 {
        d[i * n + i] = 1;
    }

    // D' block: pseudorandom ell0 × k_dim elements.
    let mut rng = ChaCha20Rng::from_seed(key_seed);
    for i in 0..ell0 {
        for j in 0..k_dim {
            d[i * n + ell0 + j] = rng.random::<u64>() % P;
        }
    }

    d
}

// ---------------------------------------------------------------------------
// Per-row R RNG  (independent stream per row via seed derivation)
// ---------------------------------------------------------------------------

fn row_rng(r_seed: [u8; 32], row: usize) -> ChaCha20Rng {
    let mut seed = r_seed;
    let row_bytes = (row as u64).to_le_bytes();
    for i in 0..8 {
        seed[i] ^= row_bytes[i];
    }
    ChaCha20Rng::from_seed(seed)
}

// ---------------------------------------------------------------------------
// Cluster encryption
// ---------------------------------------------------------------------------

/// Encrypt `vectors` (each padded to ell0 and cast to f64) under `d_mat` and `r_seed`.
/// Returns M̂ = M_0 · D + R as a flat m × n row-major array of field elements.
///
/// `on_row`: optional callback invoked once per completed row (from rayon worker threads;
/// must be `Send + Sync`).
/// Takes `&[&[f32]]` (borrowed-row slice) and streams both the
/// `f32 → F_p` quantisation and zero-pad-to-ell0 inside the per-row
/// parallel map. No full upcast or quantised matrix is materialised:
/// quantised scratch is a per-row `Vec<u64>` of length ell0 (~8 KB),
/// freed at the end of each parallel iteration. This keeps peak memory
/// bounded at large corpus scale.
pub fn encrypt_cluster(
    d_mat: &[u64],
    r_seed: [u8; 32],
    vectors: &[&[f32]],
    on_row: Option<&(dyn Fn(usize) + Send + Sync)>,
) -> Vec<u64> {
    let params = &SEC128;
    let m = vectors.len();
    let n = params.n;
    let ell0 = params.ell0;

    // Compute M̂ row-by-row in parallel.
    // Row i: M̂[i][j] = (sum_l M0[i][l] * D[l*n+j]) + R[i][j]
    // where M0[i][l] = quantize(vectors[i][l]) for l < vectors[i].len(),
    // 0 otherwise (zero-pad to ell0).
    let mut m_hat = vec![0u64; m * n];
    m_hat.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let v = vectors[i];
        let take = v.len().min(ell0);
        let mut m0_row = vec![0u64; ell0];
        for (l, &x) in v.iter().take(take).enumerate() {
            m0_row[l] = quantize(x);
        }

        let mut r_rng = row_rng(r_seed, i);
        for j in 0..n {
            let dot = (0..ell0).fold(0u64, |acc, l| {
                fp_add(acc, fp_mul(m0_row[l], d_mat[l * n + j]))
            });
            row[j] = fp_add(dot, r_rng.random::<u64>() % P);
        }
        if let Some(cb) = on_row {
            cb(i);
        }
    });

    m_hat
}

// ---------------------------------------------------------------------------
// Query encoding
// ---------------------------------------------------------------------------

/// Encode `query` (ell0 f64 values) as q̂ = (q - D'·u | u), an n-element vector.
///
/// D = [I_{ell0} | D'] where D'[i][j] = d_mat[i*n + ell0 + j] (ell0 × k_dim).
/// `u`: k_dim random field elements generated on the calling async task.
pub fn encode_query(d_mat: &[u64], query: &[f64], u: &[u64]) -> Vec<u64> {
    let params = &SEC128;
    let n = params.n;
    let ell0 = params.ell0;
    let k_dim = params.k;

    let mut q_hat = vec![0u64; n];

    // First ell0 components: q[i] - D'[i] · u
    for i in 0..ell0 {
        let dp_u = (0..k_dim).fold(0u64, |acc, j| {
            fp_add(acc, fp_mul(d_mat[i * n + ell0 + j], u[j]))
        });
        q_hat[i] = fp_add(quantize(query[i] as f32), fp_neg(dp_u));
    }

    // Last k_dim components: u
    q_hat[ell0..].copy_from_slice(u);
    q_hat
}

// ---------------------------------------------------------------------------
// Server computation
// ---------------------------------------------------------------------------

/// Compute s independent block dot products: result[si * m + i] = M̂_block_si[i] · q̂_si.
///
/// `m_hat`: flat m × n row-major encrypted matrix.
/// Returns a flat s × m array (block-major).
pub fn compute_products(m_hat: &[u64], m: usize, q_hat: &[u64]) -> Vec<u64> {
    let params = &SEC128;
    let s = params.s;
    let b = params.b;
    let n = params.n;

    let mut results = vec![0u64; s * m];
    // Parallelise over blocks; each block's m dot products run sequentially.
    results
        .par_chunks_mut(m)
        .enumerate()
        .for_each(|(si, chunk)| {
            let qblock = &q_hat[si * b..(si + 1) * b];
            chunk.iter_mut().enumerate().for_each(|(i, out)| {
                let row_base = i * n + si * b;
                *out = m_hat[row_base..row_base + b]
                    .iter()
                    .zip(qblock)
                    .fold(0u64, |acc, (&a, &bv)| fp_add(acc, fp_mul(a, bv)));
            });
        });
    results
}

// ---------------------------------------------------------------------------
// AVX-512 SIMD for EMVP `compute_products`.
//
// EMVP's inner extent is b=17, which doesn't fit AVX-512's natural
// u64-lane widths (8 or 16). The kernel here SIMD-izes the 17-element
// dot product as "two full 8-lane chunks + one scalar tail." Per
// (si, i) the SIMD work is fixed (2 lane-decomposed MAC iterations + a
// tail mul); horizontal reduce + final fold are paid once per output
// cell, so per-row overhead is high relative to the 17 muls.
//
// Whether this kernel beats scalar depends on per-cell wallclock at
// EMVP's m × s × ell0 shape — the lane-decomp's amortisation case
// (long inner extent) is not satisfied here. The micro-bench in
// `scorer-emvp/examples/emvp_compute_products_simd_bench.rs` decides
// whether to wire this into `compute_products`. Until then the SIMD
// path lives next to the scalar code; the production `compute_products`
// keeps the scalar fold.
//
// Bit-equality with scalar is pinned by
// `compute_products_avx512_matches_scalar` (cfg-gated to AVX-512
// builds). The helpers are self-contained here because EMVP has its
// own field-arithmetic module and there is no shared Mersenne SIMD
// kernel to reuse.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn, dead_code)]
unsafe fn avx512_fold(
    v: std::arch::x86_64::__m512i,
    mask: std::arch::x86_64::__m512i,
) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    let lo = _mm512_and_si512(v, mask);
    let hi = _mm512_srli_epi64::<61>(v);
    _mm512_add_epi64(lo, hi)
}

/// AVX-512 inner dot product for a 17-element EMVP block. Two
/// `u64x8` SIMD MAC iterations cover the leading 16 elements; element
/// 16 is folded in via a scalar tail mul. Output is `dot(m_chunk,
/// q_chunk) mod P`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn, dead_code)]
unsafe fn avx512_dot_b17(m_chunk: &[u64], q_chunk: &[u64]) -> u64 {
    use std::arch::x86_64::*;
    debug_assert_eq!(m_chunk.len(), 17);
    debug_assert_eq!(q_chunk.len(), 17);

    let mask = _mm512_set1_epi64(P as i64);

    let mut acc_ll_v = _mm512_setzero_si512();
    let mut acc_cross_v = _mm512_setzero_si512();
    let mut acc_hh_v = _mm512_setzero_si512();

    for k_start in [0usize, 8] {
        let a_v = _mm512_loadu_si512(m_chunk.as_ptr().add(k_start) as *const _);
        let b_v = _mm512_loadu_si512(q_chunk.as_ptr().add(k_start) as *const _);

        let a_hi = _mm512_srli_epi64::<32>(a_v);
        let b_hi = _mm512_srli_epi64::<32>(b_v);

        let prod_ll = _mm512_mul_epu32(a_v, b_v);
        let prod_lh = _mm512_mul_epu32(a_v, b_hi);
        let prod_hl = _mm512_mul_epu32(a_hi, b_v);
        let prod_hh = _mm512_mul_epu32(a_hi, b_hi);
        let prod_cross = _mm512_add_epi64(prod_lh, prod_hl);

        let prod_ll_f = avx512_fold(prod_ll, mask);
        let prod_cross_f = avx512_fold(prod_cross, mask);

        acc_ll_v = _mm512_add_epi64(acc_ll_v, prod_ll_f);
        acc_cross_v = _mm512_add_epi64(acc_cross_v, prod_cross_f);
        acc_hh_v = _mm512_add_epi64(acc_hh_v, prod_hh);
    }

    let mut buf_ll = [0u64; 8];
    let mut buf_cross = [0u64; 8];
    let mut buf_hh = [0u64; 8];
    _mm512_storeu_si512(buf_ll.as_mut_ptr() as *mut _, acc_ll_v);
    _mm512_storeu_si512(buf_cross.as_mut_ptr() as *mut _, acc_cross_v);
    _mm512_storeu_si512(buf_hh.as_mut_ptr() as *mut _, acc_hh_v);

    let mut ll_sum: u128 = 0;
    let mut cross_sum: u128 = 0;
    let mut hh_sum: u128 = 0;
    for k in 0..8 {
        ll_sum += buf_ll[k] as u128;
        cross_sum += buf_cross[k] as u128;
        hh_sum += buf_hh[k] as u128;
    }

    let r_ll = fp_reduce(ll_sum) as u128;
    let r_cross = fp_reduce(cross_sum) as u128;
    let r_hh = fp_reduce(hh_sum) as u128;

    // Tail (k=16): scalar mul + reduce.
    let r_tail = fp_reduce((m_chunk[16] as u128) * (q_chunk[16] as u128)) as u128;

    // Recombine. Under P = 2^61 − 1: 2^61 ≡ 1, so 2^64 ≡ 8 (mod P).
    let total = r_ll + (r_cross << 32) + r_hh * 8 + r_tail;
    fp_reduce(total)
}

/// AVX-512-accelerated `compute_products`. Same external contract as
/// [`compute_products`]; uses [`avx512_dot_b17`] for each (si, i)
/// cell. **Not wired into the production dispatch** — the production
/// `compute_products` keeps the scalar fold until a bench confirms
/// this kernel is meaningfully faster at the EMVP shape.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[allow(dead_code)]
pub fn compute_products_avx512(m_hat: &[u64], m: usize, q_hat: &[u64]) -> Vec<u64> {
    let params = &SEC128;
    let s = params.s;
    let b = params.b;
    let n = params.n;
    debug_assert_eq!(b, 17, "compute_products_avx512 is hard-coded for b=17");

    let mut results = vec![0u64; s * m];
    results
        .par_chunks_mut(m)
        .enumerate()
        .for_each(|(si, chunk)| {
            let qblock = &q_hat[si * b..(si + 1) * b];
            chunk.iter_mut().enumerate().for_each(|(i, out)| {
                let row_base = i * n + si * b;
                let m_chunk = &m_hat[row_base..row_base + b];
                // SAFETY: cfg(target_feature = "avx512f") guarantees
                // AVX-512 on the build target.
                *out = unsafe { avx512_dot_b17(m_chunk, qblock) };
            });
        });
    results
}

/// Batched [`compute_products`] over B queries against the same `m_hat`.
/// Returns one `Vec<u64>` of length `s × m` per query (matching the
/// shape `compute_products` would have produced individually). Per-query
/// partials are bit-equal to a sequential per-query call: same
/// per-(si, i) fold of the same `m_row[..b]` against the same
/// `qhat[si*b..(si+1)*b]`, in the same iteration order. The amortisation
/// is memory-bandwidth on `m_hat`: each row block is read once per call
/// and contributes to every batched query's partials.
pub fn compute_products_batch(m_hat: &[u64], m: usize, q_hats: &[&[u64]]) -> Vec<Vec<u64>> {
    let big_b = q_hats.len();
    if big_b == 0 {
        return Vec::new();
    }

    let params = &SEC128;
    let s = params.s;
    let b = params.b;
    let n = params.n;

    // Parallelise over s-blocks (matches `compute_products`). For each
    // block, iterate `m_hat` rows once and fold against every query's
    // qblock. Per-block output is laid out m-major then query-minor so
    // the scatter step below is a tight (i, q_idx) loop.
    let per_block: Vec<Vec<u64>> = (0..s)
        .into_par_iter()
        .map(|si| {
            let mut block_out = vec![0u64; m * big_b];
            for i in 0..m {
                let row_base = i * n + si * b;
                let m_row = &m_hat[row_base..row_base + b];
                for (q_idx, qhat) in q_hats.iter().enumerate() {
                    let qblock = &qhat[si * b..(si + 1) * b];
                    block_out[i * big_b + q_idx] = m_row
                        .iter()
                        .zip(qblock)
                        .fold(0u64, |acc, (&a, &bv)| fp_add(acc, fp_mul(a, bv)));
                }
            }
            block_out
        })
        .collect();

    // Scatter into per-query `s × m` block-major outputs, matching
    // `compute_products`'s layout exactly.
    let mut results: Vec<Vec<u64>> = (0..big_b).map(|_| vec![0u64; s * m]).collect();
    for (si, block_out) in per_block.into_iter().enumerate() {
        let base = si * m;
        for i in 0..m {
            for q_idx in 0..big_b {
                results[q_idx][base + i] = block_out[i * big_b + q_idx];
            }
        }
    }
    results
}

/// Communication cost: s × m × FIELD_BYTES bytes per cluster probe.
pub fn cluster_response_bytes(m: usize) -> u64 {
    (SEC128.s * m * FIELD_BYTES) as u64
}

/// One-time setup cost: m × n × FIELD_BYTES bytes.
pub fn setup_bytes(m: usize) -> u64 {
    (m * SEC128.n * FIELD_BYTES) as u64
}

// ---------------------------------------------------------------------------
// Client decoding
// ---------------------------------------------------------------------------

/// Decode scores from s × m partial results (block-major).
///
/// Replays row_rng(r_seed, i) for each row to recompute R[i]·q̂ (O(m·n) total).
/// score[i] = decode_field(sum_blocks(partials[si*m+i]) - R[i]·q̂).
pub fn decode_scores(m: usize, r_seed: [u8; 32], q_hat: &[u64], partials: &[u64]) -> Vec<f32> {
    let params = &SEC128;
    let s = params.s;
    let n = params.n;

    (0..m)
        .into_par_iter()
        .map(|i| {
            let block_sum = (0..s).fold(0u64, |acc, si| fp_add(acc, partials[si * m + i]));

            let mut r_rng = row_rng(r_seed, i);
            let r_dot = (0..n).fold(0u64, |acc, j| {
                fp_add(acc, fp_mul(r_rng.random::<u64>() % P, q_hat[j]))
            });

            decode_field(fp_add(block_sum, fp_neg(r_dot)))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests for field arithmetic and quantisation
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp_add_wraps() {
        assert_eq!(fp_add(P - 1, 1), 0);
        assert_eq!(fp_add(P - 1, P - 1), P - 2);
    }

    /// Bit-equality gate for [`compute_products_avx512`]. Pinned to
    /// AVX-512 builds; runs the SIMD path against the production scalar
    /// [`compute_products`] on a fixed-seed `m=64`, SEC128 fixture
    /// (s=76, b=17, n=1292) and asserts byte-equal outputs across
    /// every `(si, i)` cell.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    #[test]
    fn compute_products_avx512_matches_scalar() {
        let params = &SEC128;
        let n = params.n;
        let m = 64usize;

        let mut rng = ChaCha20Rng::seed_from_u64(0xC0DE_BEEF ^ 0xE);
        let m_hat: Vec<u64> = (0..m * n).map(|_| rng.random_range(0..P)).collect();
        let q_hat: Vec<u64> = (0..n).map(|_| rng.random_range(0..P)).collect();

        let scalar = compute_products(&m_hat, m, &q_hat);
        let simd = compute_products_avx512(&m_hat, m, &q_hat);

        assert_eq!(
            simd, scalar,
            "compute_products_avx512 vs scalar drift on EMVP SEC128 m={m}"
        );
    }

    #[test]
    fn fp_mul_identity() {
        assert_eq!(fp_mul(1, 42), 42);
        assert_eq!(fp_mul(P - 1, P - 1), 1); // (-1)^2 = 1 mod p
    }

    #[test]
    fn quantize_roundtrip() {
        // For small exact integers, quantize / Q should recover the value.
        let v = 0.5f32;
        let q = quantize(v);
        // q = round(v * Q) = round(0.5 * 2^20) = 524288
        assert_eq!(q, (v * Q as f32).round() as u64);
    }

    #[test]
    fn encode_decode_single_vector() {
        // Smoke test: encrypt a single-vector cluster, encode a query, and verify
        // that the decoded score matches the f32 inner product (up to quantisation).
        let params = &SEC128;
        let dim = params.ell0;

        // Constant vectors for determinism.
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32) * 0.01).collect();
        let q: Vec<f64> = (0..dim)
            .map(|i| ((dim - i) as f64 / dim as f64) * 0.01)
            .collect();

        let key_seed = [0u8; 32];
        let r_seed = [1u8; 32];
        let d_mat = generate_d(key_seed);
        let row_refs: [&[f32]; 1] = [v.as_slice()];
        let m_hat = encrypt_cluster(&d_mat, r_seed, &row_refs, None);

        let u = vec![0u64; params.k]; // zero u → no masking, simplifies verification
        let q_hat = encode_query(&d_mat, &q, &u);
        let partials = compute_products(&m_hat, 1, &q_hat);
        let scores = decode_scores(1, r_seed, &q_hat, &partials);

        let ip_f32: f32 = v.iter().zip(&q).map(|(&a, &b)| a * b as f32).sum();
        let decoded = scores[0];

        // Allow up to 1e-3 relative error (quantisation at Q=2^20).
        let rel_err = ((decoded - ip_f32) / ip_f32.abs().max(1e-9)).abs();
        assert!(
            rel_err < 1e-3,
            "decoded {decoded} vs ip_f32 {ip_f32}, rel_err {rel_err}"
        );
    }
}
