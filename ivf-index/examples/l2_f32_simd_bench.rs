//! Micro-bench for `l2_f32_simd` vs the auto-vec'd scalar `l2_f32` at
//! the E5-base-v2 dimension (d=768). Decision gate: ship the SIMD path
//! through a dispatcher in `l2_f32` only if it beats scalar by any
//! margin on this fixture (no churn for no win).
//!
//! Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release \
//!     --example l2_f32_simd_bench -p ivf-index

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn main() {
    use ivf_index::distance::{l2_f32, l2_f32_simd};
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    const D: usize = 768;
    const WARMUP: usize = 5_000;
    const TIMING_CHUNKS: usize = 11;
    const ITERS_PER_CHUNK: usize = 50_000;

    let mut rng = ChaCha20Rng::seed_from_u64(0xC0DE_BEEF);
    let a: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0_f32..1.0)).collect();
    let b: Vec<f32> = (0..D).map(|_| rng.random_range(-1.0_f32..1.0)).collect();

    println!("=== Plan 25 step 6 — l2_f32 scalar vs l2_f32_simd (d={D}) ===");
    let scalar_one = l2_f32(&a, &b);
    let simd_one = l2_f32_simd(&a, &b);
    let rel_err = (scalar_one - simd_one).abs() / scalar_one.abs();
    println!("correctness probe: scalar={scalar_one}  simd={simd_one}  rel_err={rel_err:.3e}");

    // Warmup so the cache + branch predictor are settled.
    let mut acc: f32 = 0.0;
    for _ in 0..WARMUP {
        acc += l2_f32(&a, &b);
        acc += l2_f32_simd(&a, &b);
    }
    std::hint::black_box(acc);

    let scalar_times = time_chunks(
        || {
            std::hint::black_box(l2_f32(&a, &b));
        },
        TIMING_CHUNKS,
        ITERS_PER_CHUNK,
    );
    let simd_times = time_chunks(
        || {
            std::hint::black_box(l2_f32_simd(&a, &b));
        },
        TIMING_CHUNKS,
        ITERS_PER_CHUNK,
    );

    let scalar_median = median(&scalar_times);
    let simd_median = median(&simd_times);
    let speedup = scalar_median / simd_median;

    println!(
        "scalar : median {:>8.2} ns/call  min {:>8.2}  max {:>8.2}",
        scalar_median * 1e9,
        min(&scalar_times) * 1e9,
        max(&scalar_times) * 1e9,
    );
    println!(
        "simd   : median {:>8.2} ns/call  min {:>8.2}  max {:>8.2}",
        simd_median * 1e9,
        min(&simd_times) * 1e9,
        max(&simd_times) * 1e9,
    );
    println!("speedup: {speedup:.3}×  (gate: any positive lift over scalar — Decision 3)");

    if speedup > 1.0 {
        println!(
            "RECOMMEND: wire l2_f32_simd into l2_f32 via cfg dispatcher (per Plan 25 step 6 + Decision 2)."
        );
        std::process::exit(0);
    } else {
        println!(
            "RECOMMEND: keep scalar l2_f32 (per Decision 3 — no churn for no win). SIMD code stays as dead-code under the cfg gate."
        );
        std::process::exit(1);
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
fn main() {
    eprintln!("This bench requires AVX-512F. Rebuild with `RUSTFLAGS=\"-C target-cpu=native\"`.");
    std::process::exit(2);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn time_chunks<F: FnMut()>(mut work: F, chunks: usize, iters: usize) -> Vec<f64> {
    let mut times = Vec::with_capacity(chunks);
    for _ in 0..chunks {
        let t = std::time::Instant::now();
        for _ in 0..iters {
            work();
        }
        times.push(t.elapsed().as_secs_f64() / iters as f64);
    }
    times
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn median(times: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = times.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted[sorted.len() / 2]
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn min(times: &[f64]) -> f64 {
    times.iter().copied().fold(f64::INFINITY, f64::min)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn max(times: &[f64]) -> f64 {
    times.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
