//! Continuous (path-filtered) Rust ↔ Go cross-check on a small fixture.
//!
//! Runs the in-process Rust Tiptoe scorer and the Go reference (via
//! `eval-harness/bin/tiptoe_go_runner`) over the same synthetic
//! corpus, then asserts the per-query top-`k` ID sets overlap by at
//! least the validation-gate threshold (80%, matching
//! `analysis/tiptoe_diff.py`).
//!
//! ### Why `#[ignore]` rather than always-on
//!
//! Each invocation builds the Go reference (~30s first time, cached
//! after) and the eval-harness binary (~30-60s in cold release mode),
//! plus runs the full BFV pipeline twice (Rust + Go subprocess). This
//! is ~1-3 min wall-clock — too heavy to gate every PR. The
//! path-filtered workflow `.github/workflows/tiptoe-cross-check.yml`
//! runs `cargo test ... -- --ignored` only when a PR touches
//! `scorer-tiptoe/src/pir/`, `scorer-tiptoe/src/bfv/`, or the Go
//! runner driver, where port drift is plausible. Manual local
//! invocation:
//!     `cargo test -p scorer-tiptoe --test cross_check_continuous --release -- --ignored --nocapture`
//!
//! ### Fixture sizing
//!
//! - `N = 256`, `d = 64`, 5 queries — the smallest size where the
//!   LWE/BFV pipeline runs end-to-end (m_max = corpus size since
//!   `n_centroids = 1` here keeps routing trivial).
//! - `q = 4` (the Tiptoe paper's headline). The Go ref's defensive
//!   `2·4^q·d < P` check at `P = 2¹⁷` gives `2·256·64 = 32 768 < 131 072`,
//!   so q=4 is safe at d=64.
//! - Vectors are grid-aligned to the q=4 lattice
//!   (`{-1, -14/15, …, 14/15, 1}`), so quantisation is lossless. Any
//!   recall miss is attributable to LWE/BFV noise, port drift, or
//!   a wire-format bug — not data quantisation. The 80% threshold
//!   absorbs the negligible LWE failure rate.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use scorer_core::{Scorer, Vector};
use scorer_tiptoe::{TiptoeConfig, TiptoeScorer};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const N: usize = 256;
const DIM: usize = 64;
const N_QUERIES: usize = 5;
const K: usize = 10;
const Q_BITS: u8 = 4;
const N_CENTROIDS: usize = 1;
const OVERLAP_THRESHOLD: f64 = 0.80;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR must have a parent (the workspace root)")
        .to_path_buf()
}

fn check_environment() -> Result<(), String> {
    if Command::new("go").arg("version").output().is_err() {
        return Err("`go` not on PATH".into());
    }
    let root = workspace_root();
    let rev = root.join("tools/tiptoe-go-rev");
    let patch = root.join("tools/tiptoe-go.patch");
    if !rev.exists() {
        return Err(format!("missing {}", rev.display()));
    }
    if !patch.exists() {
        return Err(format!("missing {}", patch.display()));
    }
    Ok(())
}

fn random_grid_vectors(n: usize, d: usize, seed: u64, q_bits: u8) -> Vec<Vector> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mag = (1_i32 << q_bits) - 1; // = 2^q − 1
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..d)
                .map(|_| {
                    let q: i32 = rng.random_range(-mag..=mag);
                    q as f32 / mag as f32
                })
                .collect();
            Vector(v)
        })
        .collect()
}

fn write_fvecs(path: &Path, vectors: &[Vector]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for v in vectors {
        let dim = v.0.len() as u32;
        w.write_all(&dim.to_le_bytes())?;
        for x in &v.0 {
            w.write_all(&x.to_le_bytes())?;
        }
    }
    Ok(())
}

fn write_ivecs(path: &Path, ids: &[Vec<u32>]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for row in ids {
        let dim = row.len() as u32;
        w.write_all(&dim.to_le_bytes())?;
        for &id in row {
            w.write_all(&(id as i32).to_le_bytes())?;
        }
    }
    Ok(())
}

fn brute_force_top_k(vectors: &[Vector], query: &Vector, k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = v.0.iter().zip(&query.0).map(|(a, b)| a * b).sum();
            (i as u32, dot)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Find the most-recent `tiptoe-go` run-id under `results_dir/runs`,
/// returning the path to its `top_k.csv`. The runner writes to a
/// nested `runs/<machine-id>/<git-sha>/<run-id>/` path; we descend
/// the freshly-created tree once per test, so the deepest leaf is
/// the only run.
fn find_top_k(results_dir: &Path) -> PathBuf {
    let runs = results_dir.join("runs");
    let machine = std::fs::read_dir(&runs)
        .unwrap_or_else(|e| panic!("read {}: {e}", runs.display()))
        .filter_map(Result::ok)
        .find(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .expect("at least one machine-id dir under runs/")
        .path();
    let sha = std::fs::read_dir(&machine)
        .unwrap()
        .filter_map(Result::ok)
        .find(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .expect("at least one git-sha dir")
        .path();
    let run = std::fs::read_dir(&sha)
        .unwrap()
        .filter_map(Result::ok)
        .find(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .expect("at least one run-id dir")
        .path();
    run.join("top_k.csv")
}

/// Parse the Go runner's `top_k.csv` (`config-label, query-id, rank,
/// doc-id, score`) into per-query ID lists, sorted by ascending rank.
fn parse_top_k(path: &Path) -> HashMap<u32, Vec<u32>> {
    let body = std::fs::read_to_string(path).expect("read top_k.csv");
    let mut by_query: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    for line in body.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 5 {
            continue;
        }
        let qid: u32 = cols[1].parse().unwrap();
        let rank: u32 = cols[2].parse().unwrap();
        let doc_id: u32 = cols[3].parse().unwrap();
        by_query.entry(qid).or_default().push((rank, doc_id));
    }
    by_query
        .into_iter()
        .map(|(qid, mut entries)| {
            entries.sort_by_key(|&(r, _)| r);
            (qid, entries.into_iter().map(|(_, d)| d).collect())
        })
        .collect()
}

#[tokio::test]
#[ignore = "drives the Go subprocess + builds the Go ref; run via path-filtered CI or `--ignored`"]
async fn rust_and_go_top_k_overlap_on_small_fixture() {
    if let Err(reason) = check_environment() {
        // Skip cleanly when the Go toolchain is unavailable; the
        // path-filtered CI job ensures it *is* available where this
        // test must run.
        eprintln!("SKIP cross_check_continuous: {reason}");
        return;
    }

    let tmp = tempfile::tempdir().expect("tempdir");
    let data_dir = tmp.path().join("data");
    let results_dir = tmp.path().join("results");
    // Reuse a workspace-level `tmp/tiptoe-go` checkout across test
    // invocations (and across `make eval-tiptoe-go`); rebuilds are
    // incremental, fresh clones are not. The path-filtered CI job is
    // expected to either pre-clone here or point at a cached layer.
    let go_ref_dir = workspace_root().join("tmp/tiptoe-go");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&results_dir).unwrap();

    let corpus = random_grid_vectors(N, DIM, 0xC0_0F_FE_E0, Q_BITS);
    let queries = random_grid_vectors(N_QUERIES, DIM, 0xDE_AD_BE_EF, Q_BITS);
    let ground_truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| brute_force_top_k(&corpus, q, K))
        .collect();

    write_fvecs(&data_dir.join("passages.fvecs"), &corpus).unwrap();
    write_fvecs(&data_dir.join("queries.fvecs"), &queries).unwrap();
    write_ivecs(&data_dir.join("ground_truth.ivecs"), &ground_truth).unwrap();

    // Run the Go reference subprocess. `cargo run` from inside a test
    // re-enters cargo on the workspace; cargo serialises target/ access
    // via .cargo-lock so this is safe even when the parent test build
    // is running.
    let root = workspace_root();
    let status = Command::new(env!("CARGO"))
        .current_dir(&root)
        .args([
            "run",
            "--release",
            "--quiet",
            "-p",
            "eval-harness",
            "--bin",
            "tiptoe_go_runner",
            "--",
            "--data-dir",
        ])
        .arg(&data_dir)
        .args(["--k", &K.to_string()])
        .args(["--repetitions", "1"])
        .args(["--quantisation-bits", &Q_BITS.to_string()])
        .args(["--n-centroids", &N_CENTROIDS.to_string()])
        .arg("--results-dir")
        .arg(&results_dir)
        .arg("--go-ref-dir")
        .arg(&go_ref_dir)
        .args(["--gpu-kind", "none"])
        .status()
        .expect("spawn cargo run tiptoe_go_runner");
    assert!(status.success(), "tiptoe_go_runner exited {status:?}");

    let go_top_k_path = find_top_k(&results_dir);
    let go_top_k = parse_top_k(&go_top_k_path);

    // In-process Rust Tiptoe scorer at the same config.
    let scorer = TiptoeScorer::new();
    let cfg = TiptoeConfig {
        n_centroids: N_CENTROIDS,
        quantisation_bits: Q_BITS,
        ..Default::default()
    };
    let (handle, _build) = scorer.upload_cluster(&cfg, &corpus).await.expect("upload");

    let mut total_overlap = 0.0_f64;
    for (qid, q) in queries.iter().enumerate() {
        let rust_hits = scorer.score(&handle, q, K).await.expect("score");
        let rust_ids: std::collections::HashSet<u32> = rust_hits.iter().map(|h| h.id).collect();
        let go_ids_v = go_top_k
            .get(&(qid as u32))
            .unwrap_or_else(|| panic!("Go output missing query {qid}"));
        let go_ids: std::collections::HashSet<u32> = go_ids_v.iter().copied().collect();
        let overlap = rust_ids.intersection(&go_ids).count() as f64 / K as f64;
        total_overlap += overlap;
    }
    let mean_overlap = total_overlap / N_QUERIES as f64;
    assert!(
        mean_overlap >= OVERLAP_THRESHOLD,
        "Rust↔Go top-{K} mean overlap = {mean_overlap:.3}, expected ≥ {OVERLAP_THRESHOLD} \
         (q={Q_BITS}, n_centroids={N_CENTROIDS}, N={N}, dim={DIM})"
    );
}
