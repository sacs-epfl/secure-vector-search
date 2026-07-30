//! One-shot quantisation-scale sweep for BN+IVF.
//!
//! Builds the BN+IVF index over the 100k MS MARCO corpus at full probe
//! (nprobe = n_centroids = ceil(sqrt(N))), runs every test query, and
//! reports recall@10 vs ground truth for each `Q ∈ {2^16, 2^18, 2^20,
//! 2^22, 2^24}`. Verification is disabled — Protocol 2 is independent
//! of recall correctness, and turning it off makes the sweep finish in
//! a sane wall-clock.
//!
//! Reproducible by construction: deterministic given workspace IVF
//! defaults (`n_centroids = ceil(sqrt(N))`, `train_seed = 42`,
//! `max_iter = 25`), a fixed `key_seed` and `upload_seed`, and the
//! corpus / queries / ground-truth files at `data-dir/*`.
//!
//! Run:
//!     cargo run --release --example bntm_q_sweep \
//!         -- --data-dir data/msmarco

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use eval_harness::{load_fvecs, load_ivecs, recall_at_k};
use scorer_bntm::{BnTmIvfConfig, BnTmIvfScorer, BnTmParams};
use scorer_core::{Device, Scorer, Vector};

#[derive(Parser)]
struct Cli {
    /// Per-dataset directory (containing passages.fvecs, queries.fvecs,
    /// ground_truth.ivecs).
    #[arg(long, default_value = "data/msmarco")]
    data_dir: PathBuf,
    /// Comma-separated list of `q_bits` (Q = 2^q_bits) to sweep.
    /// Default brackets the EMVP-inherited 2^20 by ±4 bits.
    #[arg(long, value_delimiter = ',', default_value = "16,18,20,22,24")]
    q_bits: Vec<u32>,
    /// Recall cutoff k (matches `--k 10` in the standard eval).
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// IVF training seed (must match the workspace default).
    #[arg(long, default_value_t = 42)]
    train_seed: u64,
    /// IVF max iterations (must match the workspace default).
    #[arg(long, default_value_t = 25)]
    max_iter: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let passages_path = cli.data_dir.join("passages.fvecs");
    let queries_path = cli.data_dir.join("queries.fvecs");
    let gt_path = cli.data_dir.join("ground_truth.ivecs");

    eprintln!("loading {}…", passages_path.display());
    let corpus_raw = load_fvecs(&passages_path)?;
    let n = corpus_raw.len();
    let dim = corpus_raw.first().map(|v| v.len()).unwrap_or(0);
    let n_centroids = (n as f64).sqrt().ceil() as usize;
    eprintln!("  N = {n}, dim = {dim}, n_centroids = ceil(sqrt(N)) = {n_centroids}");
    let corpus: Vec<Vector> = corpus_raw.into_iter().map(Vector).collect();

    eprintln!("loading {}…", queries_path.display());
    let queries_raw = load_fvecs(&queries_path)?;
    eprintln!("  {} queries", queries_raw.len());
    let queries: Vec<Vector> = queries_raw.into_iter().map(Vector).collect();

    eprintln!("loading {}…", gt_path.display());
    let ground_truth: Vec<Vec<u32>> = load_ivecs(&gt_path)?;
    eprintln!("  {} ground-truth lists", ground_truth.len());

    println!();
    println!(
        "Plan 12 OQ §2 — BN+IVF Q-sweep at full probe (nprobe={n_centroids}, k={}, verification=off)",
        cli.k
    );
    println!(
        "{:<10} {:<14} {:<14} {:<14}",
        "q_bits", "Q", "recall@10", "build_secs"
    );
    println!("{:-<10} {:-<14} {:-<14} {:-<14}", "", "", "", "");

    for &q_bits in &cli.q_bits {
        // Fresh scorer per Q so the previous index releases its memory
        // before the next Q's encrypt pass starts (each ~1.5 GB at our
        // 100k corpus).
        let scorer = BnTmIvfScorer::new();
        let q_scale: u64 = 1u64 << q_bits;

        let cfg = BnTmIvfConfig {
            params: BnTmParams::Sec128,
            key_seed: [42u8; 32],
            n_centroids,
            nprobe: n_centroids,
            train_seed: cli.train_seed,
            max_iter: cli.max_iter,
            upload_seed: 7,
            verification_enabled: false,
            quantisation_q: q_scale,
            progress: None,
            device: Device::Cpu,
            vram_budget_bytes: None,
        };

        let t0 = Instant::now();
        let (handle, build) = scorer.upload_cluster(&cfg, &corpus).await?;
        let build_secs = t0.elapsed().as_secs_f64();
        let _ = build;

        let handle = Arc::new(handle);
        let mut total_recall = 0.0_f64;
        for (qi, query) in queries.iter().enumerate() {
            let hits = scorer.score(&handle, query, cli.k).await?;
            let returned: Vec<u32> = hits.iter().map(|h| h.id).collect();
            let r = recall_at_k(
                ground_truth.get(qi).map(Vec::as_slice).unwrap_or(&[]),
                &returned,
                cli.k,
            );
            total_recall += r;
        }
        let mean_recall = total_recall / queries.len() as f64;

        println!(
            "{:<10} 2^{:<11} {:<14.6} {:<14.2}",
            q_bits, q_bits, mean_recall, build_secs
        );
    }

    Ok(())
}
