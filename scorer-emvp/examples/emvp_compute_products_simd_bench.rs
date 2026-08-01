//! Micro-bench for EMVP `compute_products_avx512` vs scalar
//! `compute_products` at production cluster sizes. EMVP's inner extent
//! (b=17) doesn't fit a clean AVX-512 lane width; the SIMD kernel pays
//! a fixed per-(si, i) horizontal-sum cost over only two SIMD MAC
//! iterations. Whether that beats scalar at the m × s × ell0 shape
//! needs measuring.
//!
//! Decision gate: wire `compute_products_avx512` into the production
//! `compute_products` only if it clears a **1.355× scalar** speedup
//! floor. Below that, the SIMD code stays as dead-code under the cfg
//! gate and `compute_products` keeps the scalar fold.
//!
//! Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release \
//!     --example emvp_compute_products_simd_bench -p scorer-emvp

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn main() {
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use scorer_emvp::SEC128;
    use scorer_emvp::crypto::{compute_products, compute_products_avx512};

    // m=100 (the 100k-corpus cluster mean) for direct comparability
    // with the BN bench. Production clusters at 8.8M run m ≈ 2966 per
    // cluster (sqrt(8.8M)); m=100 isolates the per-call kernel cost
    // from rayon scheduling overhead.
    const M: usize = 100;
    const WARMUP: usize = 50;
    const TIMING_CHUNKS: usize = 11;
    const ITERS_PER_CHUNK: usize = 200;

    let params = &SEC128;
    let p: u64 = (1 << 61) - 1;
    let s = params.s;
    let b = params.b;
    let n = params.n;

    println!(
        "=== EMVP compute_products scalar vs avx512 ===\n  m={M}  s={s}  b={b}  n={n}  (SEC128)"
    );

    let mut rng = ChaCha20Rng::seed_from_u64(0xC0DE_BEEF ^ 0xE);
    let m_hat: Vec<u64> = (0..M * n).map(|_| rng.random_range(0..p)).collect();
    let q_hat: Vec<u64> = (0..n).map(|_| rng.random_range(0..p)).collect();

    // Correctness probe before timing.
    let out_scalar = compute_products(&m_hat, M, &q_hat);
    let out_simd = compute_products_avx512(&m_hat, M, &q_hat);
    let mismatches = out_scalar
        .iter()
        .zip(out_simd.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "correctness probe: scalar.len={}  simd.len={}  mismatches={mismatches}",
        out_scalar.len(),
        out_simd.len(),
    );
    assert_eq!(
        mismatches, 0,
        "scalar/SIMD output diverged — fix before measuring"
    );

    let mut acc: u64 = 0;
    for _ in 0..WARMUP {
        let r = compute_products(&m_hat, M, &q_hat);
        acc = acc.wrapping_add(r[0]);
        let r = compute_products_avx512(&m_hat, M, &q_hat);
        acc = acc.wrapping_add(r[0]);
    }
    std::hint::black_box(acc);

    let scalar_times = time_chunks(
        || {
            std::hint::black_box(compute_products(&m_hat, M, &q_hat));
        },
        TIMING_CHUNKS,
        ITERS_PER_CHUNK,
    );
    let simd_times = time_chunks(
        || {
            std::hint::black_box(compute_products_avx512(&m_hat, M, &q_hat));
        },
        TIMING_CHUNKS,
        ITERS_PER_CHUNK,
    );

    let scalar_median = median(&scalar_times);
    let simd_median = median(&simd_times);
    let speedup = scalar_median / simd_median;

    println!(
        "scalar : median {:>10.3} µs/call  min {:>10.3}  max {:>10.3}",
        scalar_median * 1e6,
        min(&scalar_times) * 1e6,
        max(&scalar_times) * 1e6,
    );
    println!(
        "simd   : median {:>10.3} µs/call  min {:>10.3}  max {:>10.3}",
        simd_median * 1e6,
        min(&simd_times) * 1e6,
        max(&simd_times) * 1e6,
    );
    println!(
        "speedup: {speedup:.3}×  (gate: ≥1.355× — operator-accepted Family A floor; below this site 5 stays scalar)"
    );

    if speedup >= 1.355 {
        println!(
            "RECOMMEND: wire compute_products_avx512 into compute_products via cfg dispatcher (clears the Family A floor)."
        );
        std::process::exit(0);
    } else {
        println!(
            "RECOMMEND: keep scalar compute_products. The SIMD site stays as dead-code under the cfg gate."
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
