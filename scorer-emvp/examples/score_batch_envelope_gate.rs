//! EMVP `score_batch` envelope-gate micro-bench.
//!
//! Measures aggregate throughput at `B=64` vs the sequential `score()`
//! default on a fixed EMVP-flat cluster. Pass criterion: ≥10×
//! aggregate-throughput gain. In practice the gate failed at both
//! `m=10_000` (1.50×) and `m=100_000` (1.23×) on a 64-core Xeon Gold
//! 6426Y — batching is memory-bandwidth-bound on `m_hat`, so the
//! projected speedup does not materialise. This bench is the
//! reproducer for both shapes — edit `M` to switch.
//!
//! Run with:
//!   cargo run --release --example score_batch_envelope_gate -p scorer-emvp

use std::time::{Duration, Instant};

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use scorer_core::{Device, Scorer, Vector};
use scorer_emvp::{EmvpConfig, EmvpScorer, SEC128};

const M: usize = 100_000;
const DIM: usize = 768;
const B: usize = 64;
const K: usize = 10;
const CHUNKS: usize = 3;
const SEED_CORPUS: u64 = 42;
const SEED_QUERIES: u64 = 7;

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vector> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim)
                .map(|_| rng.random_range(-1.0_f32..1.0_f32))
                .collect();
            Vector(v)
        })
        .collect()
}

fn median(times: &mut [Duration]) -> Duration {
    times.sort();
    times[times.len() / 2]
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("EMVP score_batch envelope-gate micro-bench");
    println!("  m={M}  dim={DIM}  B={B}  k={K}  chunks={CHUNKS}");
    println!(
        "  m_hat size  ≈ {} MB  (m × SEC128.n × 8 = {} × {} × 8)",
        (M * SEC128.n * 8) / (1024 * 1024),
        M,
        SEC128.n
    );
    println!();

    let scorer = EmvpScorer::new();
    let corpus = random_vectors(M, DIM, SEED_CORPUS);
    let queries = random_vectors(B, DIM, SEED_QUERIES);

    let cfg = EmvpConfig {
        key_seed: [0u8; 32],
        progress: None,
        device: Device::Cpu,
        vram_budget_bytes: None,
    };

    println!("Building EMVP cluster ...");
    let t_build = Instant::now();
    let (handle, _) = scorer
        .upload_cluster(&cfg, &corpus)
        .await
        .expect("upload_cluster");
    println!("  built in {:.2} s", t_build.elapsed().as_secs_f64());
    println!();

    // Warm-up: page in m_hat for both paths so the first measured chunk
    // doesn't bake the cold-cache cost into the sequential side.
    let _ = scorer.score(&handle, &queries[0], K).await.unwrap();
    let _ = scorer.score_batch(&handle, &queries, K).await.unwrap();

    let mut seq_times = Vec::with_capacity(CHUNKS);
    let mut bat_times = Vec::with_capacity(CHUNKS);

    println!("Per-chunk timings:");
    for chunk in 0..CHUNKS {
        let t0 = Instant::now();
        for q in &queries {
            let _ = scorer.score(&handle, q, K).await.unwrap();
        }
        let seq = t0.elapsed();

        let t0 = Instant::now();
        let _ = scorer.score_batch(&handle, &queries, K).await.unwrap();
        let bat = t0.elapsed();

        let seq_qps = B as f64 / seq.as_secs_f64();
        let bat_qps = B as f64 / bat.as_secs_f64();
        let ratio = seq.as_secs_f64() / bat.as_secs_f64();
        println!(
            "  chunk {chunk}: sequential = {:>8.2} ms ({:>7.1} q/s)   batch = {:>8.2} ms ({:>7.1} q/s)   ratio = {:>5.2}×",
            seq.as_secs_f64() * 1000.0,
            seq_qps,
            bat.as_secs_f64() * 1000.0,
            bat_qps,
            ratio
        );

        seq_times.push(seq);
        bat_times.push(bat);
    }

    let seq_med = median(&mut seq_times);
    let bat_med = median(&mut bat_times);
    let seq_qps = B as f64 / seq_med.as_secs_f64();
    let bat_qps = B as f64 / bat_med.as_secs_f64();
    let ratio = seq_med.as_secs_f64() / bat_med.as_secs_f64();

    println!();
    println!("Median over {CHUNKS} chunks:");
    println!(
        "  sequential : {:>8.2} ms / {B} queries   ({:>7.1} q/s aggregate)",
        seq_med.as_secs_f64() * 1000.0,
        seq_qps
    );
    println!(
        "  batched B  : {:>8.2} ms / {B} queries   ({:>7.1} q/s aggregate)",
        bat_med.as_secs_f64() * 1000.0,
        bat_qps
    );
    println!("  ratio      : {ratio:.2}×");
    println!();
    println!(
        "Step 7 gate (≥10× aggregate-throughput gain): {}",
        if ratio >= 10.0 { "PASS" } else { "FAIL" }
    );
}
