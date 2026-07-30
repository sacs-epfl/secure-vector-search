/// Compute the L2 distance between two equal-length f32 slices.
///
/// Dispatches to [`l2_f32_simd`] on AVX-512 builds (measured at 10.64×
/// scalar on d=768 by `ivf-index/examples/l2_f32_simd_bench.rs`); falls
/// through to the scalar reference [`l2_f32_scalar`] otherwise.
pub fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        l2_f32_simd(a, b)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        l2_f32_scalar(a, b)
    }
}

/// Scalar reference L2 distance, preserved as a private helper so
/// (a) the `l2_f32` dispatcher has
/// a guaranteed-correct fallback on non-AVX-512 substrates, and (b)
/// `l2_f32_simd_within_tolerance_of_scalar` can compare the SIMD
/// path against a known-good reference rather than against the
/// dispatcher (which would be circular on AVX-512 builds).
///
/// `dead_code` is silenced for the AVX-512 build path, where the
/// dispatcher takes the SIMD branch and the only intra-crate caller
/// of this function is the cfg-gated test. The scalar fn must still
/// be compiled on AVX-512 builds so that test can run.
#[allow(dead_code)]
fn l2_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Explicit-SIMD L2 distance via [`wide::f32x16`]. Lane width 16
/// (= AVX-512 f32 vector). Falls back to scalar tail at `n % 16 != 0`;
/// for E5-base-v2 (d=768 = 48 × 16) the tail is empty at the hot-path
/// call site.
///
/// Wired into [`l2_f32`] via the cfg dispatcher after
/// `ivf-index/examples/l2_f32_simd_bench.rs` measured **10.64× scalar**
/// on the d=768 fixture.
///
/// Only compiled when the build has `target_feature = "avx512f"`
/// enabled (via `RUSTFLAGS="-C target-cpu=native"`). Other substrates
/// don't see this symbol at all.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn l2_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 16;
    let mut acc = wide::f32x16::default();
    for c in 0..chunks {
        let off = c * 16;
        let av = wide::f32x16::new(a[off..off + 16].try_into().unwrap());
        let bv = wide::f32x16::new(b[off..off + 16].try_into().unwrap());
        let d = av - bv;
        // `d.mul_add(d, acc)` = d * d + acc, the squared-difference MAC.
        acc = d.mul_add(d, acc);
    }
    let tail_start = chunks * 16;
    let tail: f32 = a[tail_start..]
        .iter()
        .zip(&b[tail_start..])
        .map(|(x, y)| (x - y) * (x - y))
        .sum();
    (acc.reduce_add() + tail).sqrt()
}

#[cfg(test)]
mod tests {
    /// Correctness gate for `l2_f32_simd`. Sum-of-squares reductions
    /// don't commute under IEEE-754 f32, so the SIMD-vs-scalar
    /// comparison uses a relative-error tolerance, not bit-equality. The
    /// bound is loose enough to survive worst-case cancellation at
    /// E5-base-v2's d=768, tight enough to catch a real bug.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    #[test]
    fn l2_f32_simd_within_tolerance_of_scalar() {
        // Compare against the private scalar reference directly (not
        // the public `l2_f32` dispatcher, which forwards to SIMD on
        // AVX-512 builds and would make this test trivially circular).
        use super::{l2_f32_scalar, l2_f32_simd};
        use rand::{RngExt, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(0xC0DE_BEEF);
        let d = 768usize;
        for _ in 0..16 {
            let a: Vec<f32> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
            let b: Vec<f32> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();

            let scalar = l2_f32_scalar(&a, &b);
            let simd = l2_f32_simd(&a, &b);

            let rel_err = (simd - scalar).abs() / scalar.abs();
            assert!(
                rel_err < 1e-5,
                "l2_f32_simd drift vs scalar: scalar={scalar} simd={simd} rel_err={rel_err}"
            );
        }
    }
}
