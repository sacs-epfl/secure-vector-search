//! Micro-bench for `l2_simd` vs the auto-vectorised scalar `l2`
//! (f64 sum-of-squares) at d=768. The SIMD path is worth wiring into
//! `l2` via a dispatcher only if it beats the scalar path by any margin.
//!
//! Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release \
//!     --example l2_simd_bench -p scorer-sap

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn main() {
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use scorer_sap::distance::{l2, l2_simd};

    const D: usize = 768;
    const WARMUP: usize = 5_000;
    const TIMING_CHUNKS: usize = 11;
    const ITERS_PER_CHUNK: usize = 30_000;

    let mut rng = ChaCha20Rng::seed_from_u64(0xC0DE_BEEF);
    let a: Vec<f64> = (0..D).map(|_| rng.random_range(-1.0_f64..1.0)).collect();
    let b: Vec<f64> = (0..D).map(|_| rng.random_range(-1.0_f64..1.0)).collect();

    println!("=== l2 scalar vs l2_simd (d={D}, f64) ===");
    let scalar_one = l2(&a, &b);
    let simd_one = l2_simd(&a, &b);
    let rel_err = (scalar_one - simd_one).abs() / scalar_one.abs();
    println!("correctness probe: scalar={scalar_one}  simd={simd_one}  rel_err={rel_err:.3e}");

    let mut acc: f64 = 0.0;
    for _ in 0..WARMUP {
        acc += l2(&a, &b);
        acc += l2_simd(&a, &b);
    }
    std::hint::black_box(acc);

    let scalar_times = time_chunks(
        || {
            std::hint::black_box(l2(&a, &b));
        },
        TIMING_CHUNKS,
        ITERS_PER_CHUNK,
    );
    let simd_times = time_chunks(
        || {
            std::hint::black_box(l2_simd(&a, &b));
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
            "RECOMMEND: wire l2_simd into l2 via cfg dispatcher."
        );
        std::process::exit(0);
    } else {
        println!(
            "RECOMMEND: keep scalar l2 (per Decision 3 — no churn for no win). SIMD code stays as dead-code under the cfg gate."
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
