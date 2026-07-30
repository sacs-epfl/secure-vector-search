//! Driver for the Tiptoe Go reference's `paired-runner`.
//!
//! Runs alongside `bin/eval.rs` in the same `make eval` pipeline,
//! producing a standard run directory with `scheme = "tiptoe-go"`,
//! used as the validation gate for the Rust Tiptoe port.
//!
//! Flow:
//!   1. Build the Go paired-runner binary (clones+patches+`go build`).
//!   2. Load the corpus and run the Rust Tiptoe scorer's
//!      `upload_cluster` to derive the IVF + quantised cluster matrix.
//!   3. Stream the int8-quantised cluster matrix into `emb.csv` and
//!      the per-query routed cluster + int8-quantised query into
//!      `queries.csv`. The Rust side owns IVF training and routing;
//!      the Go side is just the crypto-stack oracle.
//!   4. Spawn `paired-runner --emb-csv ... --queries-csv ...`,
//!      consume `out.csv` (per-query top-k local idx + score) and
//!      `timing.csv` (per-query offline/online wall + bytes).
//!   5. Map local idx → global `VectorId` via the cluster_membership
//!      table; compute recall@k against ground truth; write the
//!      standard `raw.csv` and `run-metadata.toml`.
//!
//! `tiptoe_go_runner` is **not** a `Scorer` impl — the Rust process
//! never owns the wire protocol, only the harness around the Go
//! subprocess.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, anyhow, bail};
use clap::Parser;
use scorer_core::{Scorer, Vector};
use scorer_tiptoe::routing::route;
use scorer_tiptoe::{TiptoeConfig, TiptoeScorer};
use serde::Serialize;

use eval_harness::tiptoe_go_build::{GoRefBuild, ensure_paired_runner};
use eval_harness::{
    CsvRow, TopKRow, load_fvecs, load_ivecs, meta, recall_at_k, write_csv_header, write_csv_row,
    write_top_k_header, write_top_k_row,
};

#[derive(Parser)]
#[command(
    about = "Drive the Tiptoe Go reference's paired-runner over our IVF clustering, writing a Plan-08 raw.csv with scheme=tiptoe-go."
)]
struct Args {
    /// Dataset directory (passages.fvecs, queries.fvecs, ground_truth.ivecs).
    #[arg(long)]
    data_dir: PathBuf,

    #[arg(long, default_value = "10")]
    k: usize,

    /// Reserved for future per-query repetitions; v1 ignores >1.
    #[arg(long, default_value = "1")]
    repetitions: usize,

    /// Root of the results tree.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Local checkout of the ahenzinger/tiptoe Go reference. Patch is
    /// applied here at build time.
    #[arg(long, default_value = "tmp/tiptoe-go")]
    go_ref_dir: PathBuf,

    /// File containing the pinned commit ("ahenzinger/tiptoe <sha>").
    #[arg(long, default_value = "tools/tiptoe-go-rev")]
    rev_file: PathBuf,

    /// The vendor patch (paired-runner + clustered-CSV reader + go.mod fix).
    #[arg(long, default_value = "tools/tiptoe-go.patch")]
    patch_file: PathBuf,

    /// Magnitude bits per embedding component (= slot_bits − 1).
    /// Default 4 matches the Tiptoe paper. With our 768-dim corpus the
    /// Go ref's max-inner-product check `2 · 4^q · d < P` panics — use
    /// `--quantisation-bits 3` (or subsample d) for d=768 corpora.
    #[arg(long, default_value = "4")]
    quantisation_bits: u8,

    /// Override IVF centroid count. Default: ceil(sqrt(corpus_len)).
    #[arg(long)]
    n_centroids: Option<usize>,

    /// SimplePIR hint size in MB, passed to paired-runner. Affects the
    /// (M, L) DB shape; bigger hint → smaller M, fewer LWE rows.
    #[arg(long, default_value = "25")]
    hint_mb: u64,

    /// GPU/accelerator label for machines.csv.
    #[arg(long, default_value = "none")]
    gpu_kind: String,

    /// Campaign id. See `eval --help`'s `--campaign-id` for the
    /// schema; identical semantics here so the paired tiptoe-rust +
    /// tiptoe-go runs in the same sweep can share a campaign id.
    #[arg(long, env = "CAMPAIGN_ID")]
    campaign_id: Option<String>,

    /// Campaign title.
    #[arg(long, env = "CAMPAIGN_TITLE")]
    campaign_title: Option<String>,

    /// Campaign note (optional).
    #[arg(long, env = "CAMPAIGN_NOTE")]
    campaign_note: Option<String>,
}

const TRAIN_SEED: u64 = 42;
const MAX_ITER: usize = 25;

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct DatasetMeta {
    path: String,
    corpus_file: String,
    query_file: String,
    ground_truth: String,
    n_passages: usize,
    n_queries: usize,
    embedding_model: String,
    dimension: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct IvfMeta {
    n_centroids: usize,
    train_seed: u64,
    max_iter: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct RunMetadata {
    run_id: String,
    machine_id: String,
    git_sha: String,
    git_dirty: bool,
    git_branch: String,
    started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    finished_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_secs: Option<u64>,
    status: String,
    harness_version: String,
    rust_toolchain: String,
    /// Compile-time `target_feature` gates the Rust-side eval binary
    /// was built with. The Go paired-runner subprocess is unaffected
    /// (it's a separate binary), but we record the field on the Rust
    /// side anyway so cross-machine comparison tooling sees a uniform
    /// shape across schemes. Sorted; empty for substrates with no
    /// relevant SIMD features.
    target_features: Vec<String>,
    kernel_version: String,
    cpu_governor: String,
    notes: String,
    /// Realised Go pool size for the paired-runner subprocess
    /// (`runtime.GOMAXPROCS(0)` equivalent — read from the `GOMAXPROCS`
    /// env var the parent Makefile sets, falling back to OS logical
    /// cores). Same TOML key as the Rust harness writes so figure 07
    /// joins on a single threading column.
    parallel_threads: usize,
    /// Active numactl binding string, sourced verbatim from
    /// `NUMACTL_BINDING`. `"none"` if unset.
    numactl_binding: String,
    /// Cgroup v2 CPU quota in vCPU equivalents, `None` when unconstrained
    /// or off-Linux. Same shape and semantics as the Rust harness writes.
    #[serde(skip_serializing_if = "Option::is_none")]
    cgroup_cpu_quota: Option<f64>,
    /// Cgroup v2 memory limit in bytes, `None` when unconstrained.
    #[serde(skip_serializing_if = "Option::is_none")]
    cgroup_memory_bytes: Option<u64>,
    /// Substrate. Always `"cpu"` for the Go runner; Tiptoe-GPU is not
    /// implemented.
    device: String,
    /// `[campaign]` block. Absent when the run was launched without
    /// campaign flags. Positioned ahead of the other table sections
    /// so `[campaign]` reads adjacent to where `[bulk]` will land.
    #[serde(skip_serializing_if = "Option::is_none")]
    campaign: Option<eval_harness::meta::Campaign>,
    ivf: IvfMeta,
    scheme_config: toml::Table,
    dataset: DatasetMeta,
}

fn write_metadata(path: &Path, m: &RunMetadata) -> anyhow::Result<()> {
    let s = toml::to_string(m).context("serialising run-metadata.toml")?;
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))
}

fn write_metadata_atomic(path: &Path, m: &RunMetadata) -> anyhow::Result<()> {
    let s = toml::to_string(m).context("serialising run-metadata.toml")?;
    let tmp = path.with_extension("toml.tmp");
    std::fs::write(&tmp, s).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))
}

fn log_machine(machines_path: &Path, info: &meta::MachineInfo) -> anyhow::Result<()> {
    if machines_path.exists() {
        let content = std::fs::read_to_string(machines_path)?;
        if content.lines().any(|l| l.starts_with(&info.id)) {
            return Ok(());
        }
    }
    if let Some(p) = machines_path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let is_new = !machines_path.exists() || File::open(machines_path)?.metadata()?.len() == 0;
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(machines_path)?;
    let mut w = BufWriter::new(file);
    if is_new {
        writeln!(
            w,
            "machine-id,machine-name,cpu-model,cores,ram-bytes,os,kernel-version,cpu-governor,gpu-kind"
        )?;
    }
    writeln!(
        w,
        "{},{},{},{},{},{},{},{},{}",
        info.id,
        info.hostname,
        info.cpu_model,
        info.cores,
        info.ram_bytes,
        info.os,
        info.kernel_version,
        info.cpu_governor,
        info.gpu_kind,
    )?;
    w.flush()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_index(
    index_path: &Path,
    run_id: &str,
    machine_id: &str,
    git_sha: &str,
    git_dirty: bool,
    scheme: &str,
    dataset: &str,
    started_at: &str,
    duration_secs: u64,
    status: &str,
    rel_path: &str,
) -> anyhow::Result<()> {
    let is_new = !index_path.exists() || File::open(index_path)?.metadata()?.len() == 0;
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(index_path)?;
    let mut w = BufWriter::new(file);
    if is_new {
        writeln!(
            w,
            "run-id,machine-id,git-sha,git-dirty,scheme,dataset,started-at,duration-secs,status,path"
        )?;
    }
    writeln!(
        w,
        "{run_id},{machine_id},{git_sha},{git_dirty},{scheme},{dataset},{started_at},{duration_secs},{status},{rel_path}"
    )?;
    w.flush()?;
    Ok(())
}

// -- emb.csv / queries.csv generation ---------------------------------------

/// Map a `Z_p` matrix entry back to its signed slot value, using the
/// upper-half-as-negative convention. Result fits in `i32` for any
/// reasonable `q ≤ 30`.
#[inline]
fn z_p_to_signed(z: u64, p: u64) -> i32 {
    let half = p / 2;
    if z >= half {
        (z as i64 - p as i64) as i32
    } else {
        z as i32
    }
}

/// Quantise an `f32` to its signed slot value. Mirrors
/// `scorer_tiptoe::encoding::quantise_component` *before* the final
/// `rem_euclid` step: the integer that the Go side will see in its
/// CSV column.
#[inline]
fn quantise_signed(x: f32, q: u8) -> i32 {
    let mag = ((1i32 << q) - 1) as f32;
    let scaled = (x * mag).round() as i32;
    scaled.clamp(-(mag as i32), mag as i32)
}

/// Write `emb.csv` in the format consumed by the Go ref's
/// `corpus.ReadEmbeddingsClusteredCsv`:
///
///   numDocs
///   embeddingSlots
///   slotBits     (= quantisation_bits + 1; the Go ref uses sign+magnitude)
///   numClusters
///   <cluster_id>,<int>,<int>,...  (one row per doc, sorted by cluster_id)
fn write_emb_csv(path: &Path, handle: &scorer_tiptoe::TiptoeHandle, p: u64) -> anyhow::Result<()> {
    let enc = handle.encoded();
    let n_centroids = enc.n_centroids();
    let d = enc.d();
    let m_max = enc.m_max();
    let width = enc.width();
    let q = handle.quantisation_bits();
    let slot_bits = q + 1;

    let n_docs: usize = (0..n_centroids)
        .map(|c| enc.cluster_membership(c).len())
        .sum();

    let f = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(f);
    writeln!(w, "{n_docs}")?;
    writeln!(w, "{d}")?;
    writeln!(w, "{slot_bits}")?;
    writeln!(w, "{n_centroids}")?;

    // Walk clusters in order; for each cluster c, the c-th d-block of
    // rows [0, len(c)) holds the encoded vectors. Translate each Z_p
    // entry back to its signed slot value for the CSV.
    let matrix = enc.matrix();
    for c in 0..n_centroids {
        let col_start = enc.cluster_col_start(c);
        let len_c = enc.cluster_membership(c).len();
        for r in 0..len_c {
            let row_offset = r * width + col_start;
            // Build "<c>,<v0>,<v1>,...,<v_{d-1}>".
            write!(w, "{c}")?;
            for j in 0..d {
                let z = matrix[row_offset + j];
                let s = z_p_to_signed(z, p);
                write!(w, ",{s}")?;
            }
            writeln!(w)?;
        }
        debug_assert!(
            (0..(m_max - len_c))
                .all(|r| { (0..d).all(|j| matrix[((len_c + r) * width) + col_start + j] == 0) }),
            "padding rows must be zero"
        );
    }

    w.flush()?;
    Ok(())
}

/// Write `queries.csv`: per-query routed cluster + int8-quantised query.
///
///   <query_id>,<cluster_id>,<int>,<int>,...,<int>
///
/// Returns the routed cluster index per query for downstream mapping.
fn write_queries_csv(
    path: &Path,
    queries: &[Vector],
    handle: &scorer_tiptoe::TiptoeHandle,
) -> anyhow::Result<Vec<usize>> {
    let centroids = handle.centroids();
    let q = handle.quantisation_bits();

    let f = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(f);
    let mut routed = Vec::with_capacity(queries.len());
    for (qid, query) in queries.iter().enumerate() {
        let c = route(&query.0, centroids);
        routed.push(c);
        write!(w, "{qid},{c}")?;
        for &x in &query.0 {
            let s = quantise_signed(x, q);
            write!(w, ",{s}")?;
        }
        writeln!(w)?;
    }
    w.flush()?;
    Ok(routed)
}

// -- subprocess + parsing ---------------------------------------------------

#[derive(Debug)]
struct Timing {
    offline_us: u64,
    online_us: u64,
    offline_up: u64,
    offline_down: u64,
    online_up: u64,
    online_down: u64,
}

fn parse_timing_csv(path: &Path) -> anyhow::Result<Vec<Timing>> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::new(f);
    let mut header = String::new();
    r.read_line(&mut header)?; // discard header
    let mut out = Vec::new();
    for (lineno, line) in r.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let _qid: u64 = parts
            .next()
            .ok_or_else(|| anyhow!("timing.csv line {} missing query_id", lineno + 2))?
            .parse()?;
        let offline_us: u64 = parts.next().context("offline_us")?.parse()?;
        let online_us: u64 = parts.next().context("online_us")?.parse()?;
        let offline_up: u64 = parts.next().context("offline_up_bytes")?.parse()?;
        let offline_down: u64 = parts.next().context("offline_down_bytes")?.parse()?;
        let online_up: u64 = parts.next().context("online_up_bytes")?.parse()?;
        let online_down: u64 = parts.next().context("online_down_bytes")?.parse()?;
        out.push(Timing {
            offline_us,
            online_us,
            offline_up,
            offline_down,
            online_up,
            online_down,
        });
    }
    Ok(out)
}

/// Per-query top-k as `(doc_local_idx, score)` lists indexed by
/// query_id (which appears in arbitrary order in out.csv but recovers
/// to dense `0..n_queries` after grouping).
fn parse_out_csv(path: &Path, n_queries: usize, k: usize) -> anyhow::Result<Vec<Vec<(u32, i64)>>> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::new(f);
    let mut header = String::new();
    r.read_line(&mut header)?;
    let mut hits: Vec<Vec<(u32, i64)>> = vec![Vec::with_capacity(k); n_queries];
    for line in r.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let qid: usize = parts.next().context("out.csv query_id")?.parse()?;
        let _rank: usize = parts.next().context("out.csv rank")?.parse()?;
        let local_idx: u32 = parts.next().context("out.csv doc_local_idx")?.parse()?;
        let score: i64 = parts.next().context("out.csv score")?.parse()?;
        if qid >= n_queries {
            bail!("out.csv: query_id {qid} ≥ n_queries {n_queries}");
        }
        hits[qid].push((local_idx, score));
    }
    Ok(hits)
}

// -- main -------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Validate the campaign tuple before any compute starts.
    let campaign = meta::Campaign::try_new(
        args.campaign_id.clone(),
        args.campaign_title.clone(),
        args.campaign_note.clone(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let git = meta::collect_git_state();
    let machine = meta::collect_machine_info(&args.gpu_kind);
    let rust_toolchain = meta::collect_rust_toolchain();

    let started_secs = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let started_at = meta::unix_secs_to_iso8601(started_secs);
    let run_id = format!("{started_secs}");

    if git.dirty {
        eprintln!(
            "warning: working tree has uncommitted changes (git-dirty = true) — \
             SHA {} may not be reproducible from git history alone",
            &git.sha[..git.sha.len().min(8)],
        );
    }

    let machines_path = args.results_dir.join("machines.csv");
    log_machine(&machines_path, &machine)?;

    // Capture threading axis. The Go subprocess inherits GOMAXPROCS
    // from this process by default; mirroring the Rust harness's
    // eprintln keeps the operator-visible signal symmetric.
    let parallel_threads = meta::capture_gomaxprocs();
    let numactl_binding = meta::capture_numactl_binding();
    let cgroup_cpu_quota = meta::capture_cgroup_cpu_quota();
    let cgroup_memory_bytes = meta::capture_cgroup_memory_bytes();
    eprintln!("Threading: parallel-threads={parallel_threads}, numactl-binding={numactl_binding}");

    // 1. Build the Go binary.
    let go_build: GoRefBuild =
        ensure_paired_runner(&args.go_ref_dir, &args.rev_file, &args.patch_file)?;
    eprintln!(
        "Go ref: commit={} toolchain={}",
        &go_build.commit[..go_build.commit.len().min(8)],
        go_build.toolchain
    );

    // 2. Load data.
    let embeddings_path = args.data_dir.join("passages.fvecs");
    let queries_path = args.data_dir.join("queries.fvecs");
    let ground_truth_path = args.data_dir.join("ground_truth.ivecs");
    if !ground_truth_path.exists() {
        bail!(
            "ground-truth file not found: {}\nRun `make ground-truth DATASET=<name>` first.",
            ground_truth_path.display()
        );
    }

    eprintln!("Loading corpus from {:?}", embeddings_path);
    let raw_corpus = load_fvecs(&embeddings_path)?;
    let corpus: Vec<Vector> = raw_corpus.into_iter().map(Vector).collect();
    eprintln!("  {} vectors", corpus.len());

    eprintln!("Loading queries from {:?}", queries_path);
    let raw_queries = load_fvecs(&queries_path)?;
    let queries: Vec<Vector> = raw_queries.into_iter().map(Vector).collect();
    eprintln!("  {} queries", queries.len());

    eprintln!("Loading ground truth from {:?}", ground_truth_path);
    let ground_truth = load_ivecs(&ground_truth_path)?;

    let dim = corpus.first().map_or(0, |v| v.0.len());
    let n_centroids = args
        .n_centroids
        .unwrap_or_else(|| (corpus.len() as f64).sqrt().ceil() as usize);
    eprintln!(
        "IVF: n_centroids={n_centroids} (={}) train_seed={TRAIN_SEED} max_iter={MAX_ITER}",
        if args.n_centroids.is_some() {
            "manual"
        } else {
            "ceil(sqrt(N))"
        }
    );

    // Run directory: <results-dir>/runs/<machine-id>/<git-sha>/<run-id>/
    let run_dir = args
        .results_dir
        .join("runs")
        .join(&machine.id)
        .join(&git.sha)
        .join(&run_id);
    std::fs::create_dir_all(&run_dir)?;

    // 3. Build the Tiptoe handle. Its IVF + encoded matrix is the
    //    canonical input we feed to the Go ref.
    let scorer = TiptoeScorer::with_cache_dir(args.data_dir.as_path());
    let cfg = TiptoeConfig {
        n_centroids,
        train_seed: TRAIN_SEED,
        max_iter: MAX_ITER,
        quantisation_bits: args.quantisation_bits,
        progress: None,
    };
    eprintln!("Building Rust Tiptoe IVF + encoded matrix");
    let (handle, _build) = scorer
        .upload_cluster(&cfg, &corpus)
        .await
        .map_err(|e| anyhow!("upload_cluster failed: {e}"))?;
    let p = scorer_tiptoe::pir::lwe::Params::tiptoe_text().p();
    eprintln!(
        "  m_max={} d={} n_centroids={} q={} (slot_bits={}) p={p}",
        handle.encoded().m_max(),
        handle.encoded().d(),
        handle.encoded().n_centroids(),
        handle.quantisation_bits(),
        handle.quantisation_bits() + 1,
    );

    // 4. Write the CSVs the Go binary consumes.
    let emb_csv = run_dir.join("emb.csv");
    let queries_csv = run_dir.join("queries.csv");
    eprintln!("Writing {}", emb_csv.display());
    write_emb_csv(&emb_csv, &handle, p)?;
    eprintln!("Writing {}", queries_csv.display());
    let routed = write_queries_csv(&queries_csv, &queries, &handle)?;

    // 5. Run paired-runner.
    let out_csv = run_dir.join("paired-out.csv");
    let timing_csv = run_dir.join("paired-timing.csv");
    eprintln!("Spawning paired-runner");
    let t_run_start = Instant::now();
    let status = Command::new(&go_build.binary_path)
        .args([
            "--emb-csv",
            emb_csv.to_str().unwrap(),
            "--queries-csv",
            queries_csv.to_str().unwrap(),
            "--k",
            &args.k.to_string(),
            "--hint-mb",
            &args.hint_mb.to_string(),
            "--out-csv",
            out_csv.to_str().unwrap(),
            "--timing-csv",
            timing_csv.to_str().unwrap(),
        ])
        .status()
        .with_context(|| format!("spawning {}", go_build.binary_path.display()))?;
    if !status.success() {
        bail!(
            "paired-runner exited with {} — see stderr above",
            status.code().unwrap_or(-1)
        );
    }
    eprintln!("paired-runner done in {:?}", t_run_start.elapsed());

    // 6. Parse outputs.
    let timing = parse_timing_csv(&timing_csv)?;
    let hits_per_q = parse_out_csv(&out_csv, queries.len(), args.k)?;
    if timing.len() != queries.len() {
        bail!(
            "timing.csv has {} rows, expected {}",
            timing.len(),
            queries.len()
        );
    }

    // 7. Map local idx → global VectorId, compute recall, write raw.csv + top_k.csv.
    let raw_csv_path = run_dir.join("raw.csv");
    let csv_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&raw_csv_path)?;
    let mut w = BufWriter::new(csv_file);
    if File::open(&raw_csv_path)?.metadata()?.len() == 0 {
        write_csv_header(&mut w)?;
    }

    let top_k_csv_path = run_dir.join("top_k.csv");
    let top_k_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&top_k_csv_path)?;
    let mut top_k_w = BufWriter::new(top_k_file);
    if File::open(&top_k_csv_path)?.metadata()?.len() == 0 {
        write_top_k_header(&mut top_k_w)?;
    }

    // The Tiptoe-go runner runs at a single (quantisation-bits, hint-mb)
    // tuple per invocation; reuse the same config-label tiptoe.rs
    // would write so the diff script can match by label.
    let config_label = format!("quantisation-bits={}", args.quantisation_bits);

    // GPU bandwidth proxy for Tiptoe is the SimplePIR matrix the
    // kernel touches per query, m_max × n_lwe × 8. Tiptoe-GPU itself
    // is not implemented, so every Go-runner row writes
    // `device = "cpu"` and the proxy via the analytical formula. The
    // value is constant per run.
    let lwe_params = scorer_tiptoe::pir::lwe::Params::tiptoe_text();
    let effective_bytes_per_query = (handle.encoded().m_max() * lwe_params.n * 8) as u64;

    for qid in 0..queries.len() {
        let cluster = routed[qid];
        let membership = handle.encoded().cluster_membership(cluster);
        let local_hits = &hits_per_q[qid];
        let global_ids_with_score: Vec<(u32, i64)> = local_hits
            .iter()
            .map(|&(local, score)| {
                let id = membership.get(local as usize).copied().unwrap_or_else(|| {
                    eprintln!(
                        "warning: qid={qid} local_idx={local} ≥ membership.len()={} (padding row hit?)",
                        membership.len()
                    );
                    u32::MAX
                });
                (id, score)
            })
            .collect();
        let global_ids: Vec<u32> = global_ids_with_score.iter().map(|&(id, _)| id).collect();

        let recall = recall_at_k(
            ground_truth.get(qid).map(Vec::as_slice).unwrap_or(&[]),
            &global_ids,
            args.k,
        );
        let t = &timing[qid];

        write_csv_row(
            &mut w,
            &CsvRow {
                run_id: run_id.clone(),
                scheme: "tiptoe-go".into(),
                quality_param_name: "quantisation-bits".into(),
                quality_param: args.quantisation_bits as f64,
                config_label: config_label.clone(),
                k: args.k as u32,
                query_id: qid as u32,
                recall_at_k: recall,
                latency_us: Some(t.online_us + t.offline_us),
                query_bytes: t.online_up,
                response_bytes: t.online_down,
                cluster_response_bytes: t.online_down,
                // Directional split of the offline byte counts:
                // - `pre_query_offline_up_bytes`   = BFV token
                //   upload (`t.offline_up`).
                // - `pre_query_offline_down_bytes` = apply-hint
                //   result (`t.offline_down`).
                // - `setup_bytes` counts the one-time SimplePIR hint
                //   download as the configured `--hint-mb` budget;
                //   the Go reference targets that size by construction
                //   (`--hint-mb` controls the (M, L) split). Realised
                //   size may differ by O(<1 MB) packing overhead.
                setup_bytes: args.hint_mb * 1024 * 1024,
                pre_query_offline_up_bytes: t.offline_up,
                pre_query_offline_down_bytes: t.offline_down,
                // BN-only column; Tiptoe has no Protocol 2
                // verification step.
                verification_overhead_us: 0,
                machine_id: machine.id.clone(),
                // Tiptoe-GPU not implemented — runner is always CPU.
                device: "cpu".into(),
                effective_bytes_per_query,
                // tiptoe_go_runner runs single-query (the Go ref is
                // sequential by construction; Tiptoe is excluded from
                // batched scope). Every row is B=1 here, and
                // wallclock/amortised collapse to latency_us.
                batch_size: 1,
                wallclock_us: t.online_us + t.offline_us,
                amortised_latency_us: t.online_us + t.offline_us,
            },
        )?;

        for (rank, &(doc_id, score)) in global_ids_with_score.iter().enumerate() {
            write_top_k_row(
                &mut top_k_w,
                &TopKRow {
                    config_label: &config_label,
                    query_id: qid as u32,
                    rank: rank as u32,
                    doc_id,
                    score: score as f32,
                },
            )?;
        }
    }
    w.flush()?;
    top_k_w.flush()?;

    // 8. run-metadata.toml.
    let mut scheme_cfg = toml::Table::new();
    scheme_cfg.insert("scheme".into(), toml::Value::String("tiptoe-go".into()));
    scheme_cfg.insert(
        "go-commit".into(),
        toml::Value::String(go_build.commit.clone()),
    );
    scheme_cfg.insert(
        "go-toolchain".into(),
        toml::Value::String(go_build.toolchain.clone()),
    );
    scheme_cfg.insert(
        "ivf-source".into(),
        toml::Value::String("matched-with-tiptoe".into()),
    );
    scheme_cfg.insert(
        "quantisation-bits".into(),
        toml::Value::Integer(args.quantisation_bits as i64),
    );
    scheme_cfg.insert("hint-mb".into(), toml::Value::Integer(args.hint_mb as i64));

    let dataset_meta = DatasetMeta {
        path: args.data_dir.display().to_string(),
        corpus_file: "passages.fvecs".into(),
        query_file: "queries.fvecs".into(),
        ground_truth: "ground_truth.ivecs".into(),
        n_passages: corpus.len(),
        n_queries: queries.len(),
        embedding_model: "intfloat/e5-base-v2".into(),
        dimension: dim,
    };

    let finished_secs = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_meta = RunMetadata {
        run_id: run_id.clone(),
        machine_id: machine.id.clone(),
        git_sha: git.sha.clone(),
        git_dirty: git.dirty,
        git_branch: git.branch.clone(),
        started_at: started_at.clone(),
        finished_at: Some(meta::unix_secs_to_iso8601(finished_secs)),
        duration_secs: Some(finished_secs.saturating_sub(started_secs)),
        status: "complete".into(),
        harness_version: env!("CARGO_PKG_VERSION").to_string(),
        rust_toolchain,
        target_features: meta::collect_target_features(),
        kernel_version: machine.kernel_version.clone(),
        cpu_governor: machine.cpu_governor.clone(),
        notes: String::new(),
        parallel_threads,
        numactl_binding,
        cgroup_cpu_quota,
        cgroup_memory_bytes,
        device: "cpu".into(),
        campaign: campaign.clone(),
        ivf: IvfMeta {
            n_centroids,
            train_seed: TRAIN_SEED,
            max_iter: MAX_ITER,
        },
        scheme_config: scheme_cfg,
        dataset: dataset_meta,
    };
    let toml_path = run_dir.join("run-metadata.toml");
    write_metadata(&toml_path, &run_meta)?;
    write_metadata_atomic(&toml_path, &run_meta)?;

    // 9. index.csv append.
    let index_path = args.results_dir.join("index.csv");
    let rel_path = run_dir
        .strip_prefix(&args.results_dir)
        .unwrap_or(&run_dir)
        .display()
        .to_string();
    append_index(
        &index_path,
        &run_id,
        &machine.id,
        &git.sha,
        git.dirty,
        "tiptoe-go",
        &args.data_dir.display().to_string(),
        &started_at,
        run_meta.duration_secs.unwrap_or(0),
        "complete",
        &rel_path,
    )?;

    eprintln!("Results written to {}", run_dir.display());
    Ok(())
}
