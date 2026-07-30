use std::path::PathBuf;

use clap::Parser;
use eval_harness::{load_fvecs, save_ivecs};
use rayon::prelude::*;

#[derive(Parser)]
#[command(about = "Compute brute-force top-k ground truth and write to .ivecs")]
struct Args {
    /// Dataset directory containing passages.fvecs and queries.fvecs.
    /// Output is written to <data-dir>/ground_truth.ivecs.
    #[arg(long)]
    data_dir: PathBuf,

    #[arg(long, default_value = "100")]
    k: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let embeddings_path = args.data_dir.join("passages.fvecs");
    let queries_path = args.data_dir.join("queries.fvecs");
    let output_path = args.data_dir.join("ground_truth.ivecs");

    eprintln!("Loading corpus from {:?}…", embeddings_path);
    let corpus = load_fvecs(&embeddings_path)?;
    eprintln!(
        "  {} vectors, dim={}",
        corpus.len(),
        corpus.first().map_or(0, |v| v.len())
    );

    eprintln!("Loading queries from {:?}…", queries_path);
    let queries = load_fvecs(&queries_path)?;
    eprintln!("  {} queries", queries.len());

    eprintln!(
        "Computing brute-force top-{} ground truth on {} threads…",
        args.k,
        rayon::current_num_threads()
    );
    let ground_truth: Vec<Vec<u32>> = queries
        .par_iter()
        .map(|q| {
            let mut dists: Vec<(u32, f32)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = v
                        .iter()
                        .zip(q)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>()
                        .sqrt();
                    (i as u32, d)
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            dists[..args.k.min(dists.len())]
                .iter()
                .map(|(id, _)| *id)
                .collect()
        })
        .collect();

    save_ivecs(&output_path, &ground_truth)?;
    eprintln!("Wrote ground truth to {:?}", output_path);
    Ok(())
}
