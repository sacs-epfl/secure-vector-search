//! Mersenne-field matvec SIMD micro-bench (NEON on aarch64, AVX-512 on
//! x86_64). Validates the projected 2–3× helper-level speedup against
//! scalar `dense_matvec` / `compute_products` before any wholesale
//! refactor lands.
//!
//! Two production shapes:
//!   100 × 512  — `AL · G` / `H · (L⊤·v_enc)` (decode-dense, ~78 % of BN-flat CPU time).
//!   100 × 1024 — `compute_products` / `verify_response` inner matvec
//!                (server-compute + verify, ~19 % combined on the sacs006 substep breakdown).
//!
//! Both kernels lane-decompose `a × b mod P` with `a = a_lo + a_hi · 2^32`
//! and `b = b_lo + b_hi · 2^32` into four u32×u32 → u64 widening products.
//! Under `P = 2^61 − 1`, `2^61 ≡ 1` so `2^64 ≡ 8`; the `acc_hh · 2^64` term
//! recombines as `acc_hh · 8` after reduction.
//!
//! NEON: 2-lane `vmull_u32`, three u128 accumulators with per-pair lane
//! extracts. AVX-512: 8-lane `_mm512_mul_epu32`, three u64x8 accumulators
//! with per-iter in-vector Mersenne fold (no scalar round-trip).
//!
//! Run with:
//!   cargo run --release --example mersenne_matvec_simd_bench -p scorer-bntm
//! For AVX-512 on sacs006 (or any x86_64 with the feature):
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release \
//!     --example mersenne_matvec_simd_bench -p scorer-bntm

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

const P: u64 = (1 << 61) - 1;
const P_U128: u128 = P as u128;

/// (m, n, label) — production-relevant shapes from BN flat / IVF.
const SHAPES: &[(usize, usize, &str)] = &[
    (100, 512, "AL·G / decode-dense"),
    (100, 1024, "compute_products / verify_response"),
];

const TIMING_CHUNKS: usize = 11; // odd → unambiguous median; report min/median/max.
const ITERS_PER_CHUNK: usize = 200; // 200 × 11 = 2 200 matvecs per (path, shape).
const WARMUP_ITERS: usize = 50;

#[inline]
fn mersenne_reduce(x: u128) -> u64 {
    let lo1 = x & P_U128;
    let hi1 = x >> 61;
    let s1 = lo1 + hi1;
    let lo2 = (s1 & P_U128) as u64;
    let hi2 = (s1 >> 61) as u64;
    let s = lo2 + hi2;
    if s >= P { s - P } else { s }
}

fn scalar_matvec(mat: &[u64], v: &[u64], out: &mut [u64], m: usize, n: usize) {
    debug_assert_eq!(mat.len(), m * n);
    debug_assert_eq!(v.len(), n);
    for i in 0..m {
        let row = &mat[i * n..(i + 1) * n];
        let mut acc: u128 = 0;
        for j in 0..n {
            acc += (row[j] as u128) * (v[j] as u128);
            if j & 31 == 31 {
                acc = mersenne_reduce(acc) as u128;
            }
        }
        out[i] = mersenne_reduce(acc);
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_matvec(mat: &[u64], v: &[u64], out: &mut [u64], m: usize, n: usize) {
    use std::arch::aarch64::*;
    debug_assert_eq!(mat.len(), m * n);
    debug_assert_eq!(v.len(), n);
    debug_assert_eq!(n % 2, 0);

    unsafe {
        for i in 0..m {
            let row = &mat[i * n..(i + 1) * n];

            let mut acc_ll: u128 = 0;
            let mut acc_cross: u128 = 0;
            let mut acc_hh: u128 = 0;

            let mut j = 0;
            while j < n {
                let a0 = row[j];
                let a1 = row[j + 1];
                let b0 = v[j];
                let b1 = v[j + 1];

                let a_lo_packed = (a0 & 0xFFFF_FFFF) | ((a1 & 0xFFFF_FFFF) << 32);
                let a_hi_packed = (a0 >> 32) | ((a1 >> 32) << 32);
                let b_lo_packed = (b0 & 0xFFFF_FFFF) | ((b1 & 0xFFFF_FFFF) << 32);
                let b_hi_packed = (b0 >> 32) | ((b1 >> 32) << 32);

                let a_lo_v = vcreate_u32(a_lo_packed);
                let a_hi_v = vcreate_u32(a_hi_packed);
                let b_lo_v = vcreate_u32(b_lo_packed);
                let b_hi_v = vcreate_u32(b_hi_packed);

                let prod_ll = vmull_u32(a_lo_v, b_lo_v);
                let prod_lh = vmull_u32(a_lo_v, b_hi_v);
                let prod_hl = vmull_u32(a_hi_v, b_lo_v);
                let prod_hh = vmull_u32(a_hi_v, b_hi_v);
                let prod_cross = vaddq_u64(prod_lh, prod_hl);

                acc_ll +=
                    vgetq_lane_u64::<0>(prod_ll) as u128 + vgetq_lane_u64::<1>(prod_ll) as u128;
                acc_cross += vgetq_lane_u64::<0>(prod_cross) as u128
                    + vgetq_lane_u64::<1>(prod_cross) as u128;
                acc_hh +=
                    vgetq_lane_u64::<0>(prod_hh) as u128 + vgetq_lane_u64::<1>(prod_hh) as u128;

                j += 2;
                if ((j / 2) & 15) == 15 {
                    acc_ll = mersenne_reduce(acc_ll) as u128;
                    acc_cross = mersenne_reduce(acc_cross) as u128;
                    acc_hh = mersenne_reduce(acc_hh) as u128;
                }
            }

            let r_ll = mersenne_reduce(acc_ll) as u128;
            let r_cross = mersenne_reduce(acc_cross) as u128;
            let r_hh = mersenne_reduce(acc_hh) as u128;

            let total = r_ll + (r_cross << 32) + r_hh * 8;
            out[i] = mersenne_reduce(total);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fold_avx512(
    v: std::arch::x86_64::__m512i,
    mask: std::arch::x86_64::__m512i,
) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    let lo = _mm512_and_si512(v, mask);
    let hi = _mm512_srli_epi64::<61>(v);
    _mm512_add_epi64(lo, hi)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn avx512_matvec(mat: &[u64], v: &[u64], out: &mut [u64], m: usize, n: usize) {
    use std::arch::x86_64::*;
    debug_assert_eq!(mat.len(), m * n);
    debug_assert_eq!(v.len(), n);
    debug_assert_eq!(n % 8, 0);

    let mask = _mm512_set1_epi64(P as i64);

    for i in 0..m {
        let row = &mat[i * n..(i + 1) * n];

        let mut acc_ll_v = _mm512_setzero_si512();
        let mut acc_cross_v = _mm512_setzero_si512();
        let mut acc_hh_v = _mm512_setzero_si512();

        let mut j = 0;
        while j < n {
            let a_v = _mm512_loadu_si512(row.as_ptr().add(j) as *const _);
            let b_v = _mm512_loadu_si512(v.as_ptr().add(j) as *const _);

            let a_hi_v = _mm512_srli_epi64::<32>(a_v);
            let b_hi_v = _mm512_srli_epi64::<32>(b_v);

            let prod_ll = _mm512_mul_epu32(a_v, b_v);
            let prod_lh = _mm512_mul_epu32(a_v, b_hi_v);
            let prod_hl = _mm512_mul_epu32(a_hi_v, b_v);
            let prod_hh = _mm512_mul_epu32(a_hi_v, b_hi_v);
            let prod_cross = _mm512_add_epi64(prod_lh, prod_hl);

            let prod_ll_f = fold_avx512(prod_ll, mask);
            let prod_cross_f = fold_avx512(prod_cross, mask);

            acc_ll_v = _mm512_add_epi64(acc_ll_v, prod_ll_f);
            acc_cross_v = _mm512_add_epi64(acc_cross_v, prod_cross_f);
            acc_hh_v = _mm512_add_epi64(acc_hh_v, prod_hh);

            acc_ll_v = fold_avx512(acc_ll_v, mask);
            acc_cross_v = fold_avx512(acc_cross_v, mask);
            j += 8;
            if (j / 8) % 8 == 0 {
                acc_hh_v = fold_avx512(acc_hh_v, mask);
            }
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

        let r_ll = mersenne_reduce(ll_sum) as u128;
        let r_cross = mersenne_reduce(cross_sum) as u128;
        let r_hh = mersenne_reduce(hh_sum) as u128;

        let total = r_ll + (r_cross << 32) + r_hh * 8;
        out[i] = mersenne_reduce(total);
    }
}

fn random_field_vec(rng: &mut ChaCha20Rng, n: usize) -> Vec<u64> {
    (0..n).map(|_| rng.random_range(0..P)).collect()
}

/// Time `work` over `chunks` chunks of `iters` iterations each. Returns
/// per-matvec wallclock (seconds) for each chunk — one f64 per chunk.
fn time_chunks<F: FnMut()>(mut work: F, chunks: usize, iters: usize) -> Vec<f64> {
    for _ in 0..WARMUP_ITERS {
        work();
    }
    let mut chunk_times = Vec::with_capacity(chunks);
    for _ in 0..chunks {
        let t = Instant::now();
        for _ in 0..iters {
            work();
        }
        chunk_times.push(t.elapsed().as_secs_f64() / iters as f64);
    }
    chunk_times
}

fn summarize(label: &str, times: &[f64], muls_per_matvec: f64) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let min = sorted[0];
    let median = sorted[n / 2];
    let max = sorted[n - 1];
    println!(
        "  {label:<6}: median {:>7.3} µs/matvec  min {:>7.3}  max {:>7.3}  ({:.0} Mmul/s @ median)",
        median * 1e6,
        min * 1e6,
        max * 1e6,
        muls_per_matvec / median / 1e6,
    );
    median
}

fn bench_shape(m: usize, n: usize, label: &str) {
    println!("\n=== shape m × n = {m} × {n} ({label}) ===");
    let mut rng = ChaCha20Rng::seed_from_u64(0xC0DE_BEEF ^ (m as u64) ^ ((n as u64) << 32));
    let mat = random_field_vec(&mut rng, m * n);
    let v = random_field_vec(&mut rng, n);

    let mut out_scalar = vec![0u64; m];
    let mut out_simd = vec![0u64; m];

    scalar_matvec(&mat, &v, &mut out_scalar, m, n);

    // SIMD correctness probe (one call) before timing.
    let simd_label: &'static str;
    let simd_ran: bool;

    #[cfg(target_arch = "aarch64")]
    {
        neon_matvec(&mat, &v, &mut out_simd, m, n);
        simd_label = "neon";
        simd_ran = true;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            unsafe { avx512_matvec(&mat, &v, &mut out_simd, m, n) };
            simd_label = "avx512";
            simd_ran = true;
        } else {
            simd_label = "none";
            simd_ran = false;
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = &out_simd;
        simd_label = "none";
        simd_ran = false;
    }

    if simd_ran {
        let mismatches = out_scalar
            .iter()
            .zip(out_simd.iter())
            .filter(|(a, b)| a != b)
            .count();
        if mismatches > 0 {
            eprintln!("CORRECTNESS FAIL ({simd_label}): {mismatches} of {m} rows differ");
            std::process::exit(1);
        }
        println!("  correctness ({simd_label}): bit-equal to scalar");
    } else {
        println!("  (no SIMD path available on this target)");
    }

    let muls = (m * n) as f64;

    let scalar_times = time_chunks(
        || scalar_matvec(&mat, &v, &mut out_scalar, m, n),
        TIMING_CHUNKS,
        ITERS_PER_CHUNK,
    );
    let scalar_median = summarize("scalar", &scalar_times, muls);

    if simd_ran {
        #[cfg(target_arch = "aarch64")]
        let simd_times = time_chunks(
            || neon_matvec(&mat, &v, &mut out_simd, m, n),
            TIMING_CHUNKS,
            ITERS_PER_CHUNK,
        );
        #[cfg(target_arch = "x86_64")]
        let simd_times = time_chunks(
            || unsafe { avx512_matvec(&mat, &v, &mut out_simd, m, n) },
            TIMING_CHUNKS,
            ITERS_PER_CHUNK,
        );
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let simd_times: Vec<f64> = vec![];

        let simd_median = summarize(simd_label, &simd_times, muls);
        println!(
            "  speedup (scalar / {simd_label}, at median): {:.3}×",
            scalar_median / simd_median,
        );
    }
}

fn main() {
    println!(
        "mersenne_matvec SIMD micro-bench — {TIMING_CHUNKS} chunks × {ITERS_PER_CHUNK} iters/chunk per (path, shape)",
    );
    for (m, n, l) in SHAPES {
        bench_shape(*m, *n, l);
    }
}
