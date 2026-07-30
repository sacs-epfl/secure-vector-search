//! Decision-gate measurement.
//!
//! Compares per-query latency of `BnTmScorer::score` vs
//! `BnTmScorer::score_with_breakdown` (and same for BN+IVF). If the
//! breakdown overhead is < 2 % of `score`, the BN dispatch arms route
//! through `score_with_breakdown` and pull `verify_us` out of the
//! timing struct, dropping the proposed inherent
//! `score_with_verify_us` method and its test pair.
//!
//! Methodology: identical-config sweeps at `verification_enabled =
//! true` (the trigger config). Cache-warm corpus build (run after a
//! prior eval-bntm has populated the disk cache); first-Q wait time
//! is build-only and doesn't enter the per-query latency arithmetic.
//!
//! Run:
//!     cargo run --release --example bn_breakdown_overhead -- \
//!         --data-dir data/msmarco --queries 50 --reps 2

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use eval_harness::load_fvecs;
use scorer_bntm::{BnTmConfig, BnTmIvfConfig, BnTmIvfScorer, BnTmParams, BnTmScorer};
use scorer_core::{Device, Scorer, Vector};

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "data/msmarco")]
    data_dir: PathBuf,
    /// Query subset size — short enough to finish the gate in minutes.
    #[arg(long, default_value_t = 50)]
    queries: usize,
    #[arg(long, default_value_t = 2)]
    reps: usize,
    /// nprobe for BN+IVF (BN flat ignores this).
    #[arg(long, default_value_t = 32)]
    nprobe: usize,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 42)]
    train_seed: u64,
    #[arg(long, default_value_t = 25)]
    max_iter: usize,
}

fn fmt_pct(num: f64, denom: f64) -> String {
    if denom <= 0.0 {
        "n/a".into()
    } else {
        format!("{:+.2}%", 100.0 * (num - denom) / denom)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let passages_path = cli.data_dir.join("passages.fvecs");
    let queries_path = cli.data_dir.join("queries.fvecs");

    eprintln!("loading {}…", passages_path.display());
    let corpus_raw = load_fvecs(&passages_path)?;
    let corpus: Vec<Vector> = corpus_raw.into_iter().map(Vector).collect();

    eprintln!("loading {}…", queries_path.display());
    let queries_raw = load_fvecs(&queries_path)?;
    let queries: Vec<Vector> = queries_raw
        .into_iter()
        .take(cli.queries)
        .map(Vector)
        .collect();
    eprintln!(
        "  using first {} queries × {} reps for the gate",
        queries.len(),
        cli.reps
    );

    // -----------------------------------------------------------------
    // BN flat
    // -----------------------------------------------------------------
    eprintln!("\n=== BN flat (verification=on) ===");
    let flat = BnTmScorer::new();
    let flat_cfg = BnTmConfig {
        params: BnTmParams::Sec128,
        key_seed: [42u8; 32],
        verification_enabled: true,
        quantisation_q: 1u64 << 20,
        progress: None,
        device: Device::Cpu,
        vram_budget_bytes: None,
    };
    let t0 = Instant::now();
    let (flat_handle, _build) = flat.upload_cluster(&flat_cfg, &corpus).await?;
    eprintln!("  build: {:.2}s", t0.elapsed().as_secs_f64());
    let flat_handle = Arc::new(flat_handle);

    let mut score_us_total: u128 = 0;
    let mut breakdown_us_total: u128 = 0;
    let mut score_n: u64 = 0;
    let mut breakdown_n: u64 = 0;

    for rep in 0..cli.reps {
        eprintln!("  rep {}/{}: score()", rep + 1, cli.reps);
        for q in &queries {
            let t = Instant::now();
            let _ = flat.score(&flat_handle, q, cli.k).await?;
            score_us_total += t.elapsed().as_micros();
            score_n += 1;
        }
        eprintln!("  rep {}/{}: score_with_breakdown()", rep + 1, cli.reps);
        for q in &queries {
            let t = Instant::now();
            let _ = flat.score_with_breakdown(&flat_handle, q, cli.k).await?;
            breakdown_us_total += t.elapsed().as_micros();
            breakdown_n += 1;
        }
    }

    let flat_score_mean = score_us_total as f64 / score_n as f64;
    let flat_breakdown_mean = breakdown_us_total as f64 / breakdown_n as f64;
    println!(
        "BN flat:        score    mean = {:.0} us/query  (n={})",
        flat_score_mean, score_n
    );
    println!(
        "BN flat:        breakdown mean = {:.0} us/query  (n={})  Δ vs score = {}",
        flat_breakdown_mean,
        breakdown_n,
        fmt_pct(flat_breakdown_mean, flat_score_mean)
    );

    // -----------------------------------------------------------------
    // BN+IVF
    // -----------------------------------------------------------------
    eprintln!("\n=== BN+IVF (verification=on, nprobe={}) ===", cli.nprobe);
    let n = corpus.len();
    let n_centroids = (n as f64).sqrt().ceil() as usize;
    let ivf = BnTmIvfScorer::with_cache_dir(cli.data_dir.as_path());
    let ivf_cfg = BnTmIvfConfig {
        params: BnTmParams::Sec128,
        key_seed: [42u8; 32],
        n_centroids,
        nprobe: cli.nprobe,
        train_seed: cli.train_seed,
        max_iter: cli.max_iter,
        upload_seed: 7,
        verification_enabled: true,
        quantisation_q: 1u64 << 20,
        progress: None,
        device: Device::Cpu,
        vram_budget_bytes: None,
    };
    let t0 = Instant::now();
    let (ivf_handle, _build) = ivf.upload_cluster(&ivf_cfg, &corpus).await?;
    eprintln!("  build: {:.2}s", t0.elapsed().as_secs_f64());
    let ivf_handle = Arc::new(ivf_handle);

    let mut score_us_total: u128 = 0;
    let mut breakdown_us_total: u128 = 0;
    let mut score_n: u64 = 0;
    let mut breakdown_n: u64 = 0;

    for rep in 0..cli.reps {
        eprintln!("  rep {}/{}: score()", rep + 1, cli.reps);
        for q in &queries {
            let t = Instant::now();
            let _ = ivf.score(&ivf_handle, q, cli.k).await?;
            score_us_total += t.elapsed().as_micros();
            score_n += 1;
        }
        eprintln!("  rep {}/{}: score_with_breakdown()", rep + 1, cli.reps);
        for q in &queries {
            let t = Instant::now();
            let _ = ivf.score_with_breakdown(&ivf_handle, q, cli.k).await?;
            breakdown_us_total += t.elapsed().as_micros();
            breakdown_n += 1;
        }
    }

    let ivf_score_mean = score_us_total as f64 / score_n as f64;
    let ivf_breakdown_mean = breakdown_us_total as f64 / breakdown_n as f64;
    println!(
        "BN+IVF (np={}): score    mean = {:.0} us/query  (n={})",
        cli.nprobe, ivf_score_mean, score_n
    );
    println!(
        "BN+IVF (np={}): breakdown mean = {:.0} us/query  (n={})  Δ vs score = {}",
        cli.nprobe,
        ivf_breakdown_mean,
        breakdown_n,
        fmt_pct(ivf_breakdown_mean, ivf_score_mean)
    );

    println!();
    println!("Plan 18 Amendment 1 gate: < 2% on BOTH paths ⇒ route through breakdown.");
    Ok(())
}
