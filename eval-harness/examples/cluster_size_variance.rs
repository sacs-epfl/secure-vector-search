//! One-shot cluster-size variance measurement.
//!
//! Builds the shared IVF (k-means at the workspace defaults) on the
//! MS MARCO 100k corpus and reports `stddev(m_i) / mean(m_i)`. The
//! threshold for switching EMVP+IVF's `cluster_response_bytes` from
//! per-handle analytical to per-query logged is 20 %.
//!
//! Run:
//!     cargo run --release --example cluster_size_variance \
//!         -- --data-dir data/msmarco
//!
//! Reproducible by construction: deterministic given the IVF defaults
//! (n_centroids = ceil(sqrt(N)), train_seed = 42, max_iter = 25) and
//! the corpus file at data-dir/passages.fvecs.

use std::path::PathBuf;

use clap::Parser;
use eval_harness::load_fvecs;
use ivf_index::ivf::build_index;
use ivf_index::kmeans::train;

#[derive(Parser)]
struct Cli {
    /// Per-dataset directory (containing passages.fvecs).
    #[arg(long, default_value = "data/msmarco")]
    data_dir: PathBuf,
    /// IVF training seed (must match the workspace default).
    #[arg(long, default_value_t = 42)]
    train_seed: u64,
    /// IVF max iterations (must match the workspace default).
    #[arg(long, default_value_t = 25)]
    max_iter: usize,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let passages_path = cli.data_dir.join("passages.fvecs");

    println!("loading {}…", passages_path.display());
    let passages = load_fvecs(&passages_path)?;
    let n = passages.len();
    let dim = passages.first().map(|v| v.len()).unwrap_or(0);
    let k = (n as f64).sqrt().ceil() as usize;
    println!("  N = {n}, dim = {dim}, n_centroids = ceil(sqrt(N)) = {k}");

    println!(
        "training k-means (seed = {}, max_iter = {})…",
        cli.train_seed, cli.max_iter
    );
    let centroids = train(&passages, k, cli.train_seed, cli.max_iter, |_| {}, |_| {});
    let index = build_index(&passages, centroids, dim);

    let sizes: Vec<usize> = index.clusters.iter().map(|c| c.len()).collect();
    let mean = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
    let var = sizes
        .iter()
        .map(|&m| (m as f64 - mean).powi(2))
        .sum::<f64>()
        / sizes.len() as f64;
    let stddev = var.sqrt();
    let cv = stddev / mean;
    let min = *sizes.iter().min().unwrap();
    let max = *sizes.iter().max().unwrap();

    println!();
    println!("cluster-size statistics");
    println!("  n_clusters = {}", sizes.len());
    println!("  min        = {min}");
    println!("  max        = {max}");
    println!("  mean       = {mean:.2}");
    println!("  stddev     = {stddev:.2}");
    println!("  cv         = stddev/mean = {cv:.4} = {:.2}%", cv * 100.0);
    println!();
    if cv <= 0.20 {
        println!("DECISION: cv ≤ 20% — keep per-handle analytical bytes in EmvpIvfScorer.");
    } else {
        println!("DECISION: cv > 20% — switch cluster_response_bytes to per-query logged.");
    }

    Ok(())
}
