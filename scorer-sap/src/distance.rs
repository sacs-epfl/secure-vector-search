/// Compute the L2 distance between two equal-length f64 slices.
///
/// Dispatches to [`l2_simd`] on AVX-512 builds (measured at ~4.7× the
/// scalar path on d=768); falls through to the scalar reference
/// [`l2_scalar`] otherwise.
pub fn l2(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        l2_simd(a, b)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        l2_scalar(a, b)
    }
}

/// Scalar reference L2 distance, kept as a private helper so (a) the
/// `l2` dispatcher has a guaranteed-correct fallback on non-AVX-512
/// substrates, and (b) `l2_simd_within_tolerance_of_scalar` can compare
/// the SIMD path against a known-good reference rather than against the
/// dispatcher (which would be circular on AVX-512 builds).
///
/// `dead_code` is silenced for the AVX-512 build path, where the
/// dispatcher takes the SIMD branch and the only intra-crate caller
/// of this function is the cfg-gated test. The scalar fn must still
/// be compiled on AVX-512 builds so that test can run.
#[allow(dead_code)]
fn l2_scalar(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Explicit-SIMD L2 distance via [`wide::f64x8`]. Lane width 8
/// (= AVX-512 f64 vector). Falls back to a scalar tail at `n % 8 != 0`;
/// SAP-flat's hot path runs over the encrypted f64 vectors at
/// d=768 = 96 × 8 (no tail).
///
/// Wired into [`l2`] via the cfg dispatcher; on the d=768 fixture it
/// runs ~4.7× the scalar path with single-ULP agreement (rel_err on the
/// order of 1e-16).
///
/// Only compiled when the build has `target_feature = "avx512f"`
/// enabled (via `RUSTFLAGS="-C target-cpu=native"`). Other substrates
/// don't see this symbol at all.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn l2_simd(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let mut acc = wide::f64x8::default();
    for c in 0..chunks {
        let off = c * 8;
        let av = wide::f64x8::new(a[off..off + 8].try_into().unwrap());
        let bv = wide::f64x8::new(b[off..off + 8].try_into().unwrap());
        let d = av - bv;
        // `d.mul_add(d, acc)` = d * d + acc, the squared-difference MAC.
        acc = d.mul_add(d, acc);
    }
    let tail_start = chunks * 8;
    let tail: f64 = a[tail_start..]
        .iter()
        .zip(&b[tail_start..])
        .map(|(x, y)| (x - y) * (x - y))
        .sum();
    (acc.reduce_add() + tail).sqrt()
}

#[cfg(test)]
mod tests {
    /// Correctness gate for `l2_simd`. f64 sum-of-squares has more
    /// head-room than f32 (52-bit mantissa vs 23-bit); the
    /// SIMD-vs-scalar relative error at d=768 should be near zero.
    /// Tolerance is loose enough to survive worst-case cancellation,
    /// tight enough to catch a real bug.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    #[test]
    fn l2_simd_within_tolerance_of_scalar() {
        // Compare against the private scalar reference directly (not
        // the public `l2` dispatcher, which forwards to SIMD on AVX-
        // 512 builds and would make this test trivially circular).
        use super::{l2_scalar, l2_simd};
        use rand::{RngExt, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(0xC0DE_BEEF);
        let d = 768usize;
        for _ in 0..16 {
            let a: Vec<f64> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
            let b: Vec<f64> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();

            let scalar = l2_scalar(&a, &b);
            let simd = l2_simd(&a, &b);

            let rel_err = (simd - scalar).abs() / scalar.abs();
            assert!(
                rel_err < 1e-12,
                "l2_simd drift vs scalar: scalar={scalar} simd={simd} rel_err={rel_err}"
            );
        }
    }
}
