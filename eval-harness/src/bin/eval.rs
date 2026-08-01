use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, LineWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, bail};
use async_trait::async_trait;
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use scorer_bntm::{
    BnTmConfig, BnTmError, BnTmHandle, BnTmIvfConfig, BnTmIvfError, BnTmIvfHandle, BnTmIvfScorer,
    BnTmParams, BnTmScorer,
};
use scorer_core::{BuildOutcome, Device, Hit, Scorer, Vector};
use scorer_emvp::{
    EmvpConfig, EmvpError, EmvpHandle, EmvpIvfConfig, EmvpIvfError, EmvpIvfHandle, EmvpIvfScorer,
    EmvpScorer,
};
use scorer_plaintext::{PlaintextConfig, PlaintextError, PlaintextHandle, PlaintextScorer};
use scorer_sap::{
    SapClusterHandle, SapConfig, SapError, SapIvfConfig, SapIvfError, SapIvfHandle, SapIvfScorer,
    SapScorer, keygen,
};
use scorer_tiptoe::{TiptoeConfig, TiptoeError, TiptoeHandle, TiptoeScorer};
use serde::Serialize;

use eval_harness::progress::IndicatifProgress;
use eval_harness::{
    CANONICAL_SUBSTEPS, CsvRow, SubstepRow, TopKRow, load_fvecs, load_ivecs, meta, recall_at_k,
    write_csv_header, write_csv_row, write_substep_breakdown_header, write_substep_breakdown_row,
    write_top_k_header, write_top_k_row,
};

#[derive(Parser)]
#[command(about = "Run a scorer sweep and emit one CSV row per (query, config, repetition)")]
struct Args {
    #[arg(long, value_enum)]
    scorer: SchemeArg,

    /// Dataset directory containing passages.fvecs, queries.fvecs, and ground_truth.ivecs.
    #[arg(long)]
    data_dir: PathBuf,

    #[arg(long, default_value = "10")]
    k: usize,

    #[arg(long, default_value = "1")]
    repetitions: usize,

    /// Optional cap on the number of queries to run from queries.fvecs.
    /// Smoke-test convenience: `--queries 50` keeps short loops short
    /// without subsetting the corpus. Defaults to all queries.
    #[arg(long)]
    queries: Option<usize>,

    /// Root of the results tree. Writes to <results-dir>/runs/<machine-id>/<git-sha>/<run-id>/.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Removed — use --results-dir instead.
    #[arg(long, hide = true)]
    output: Option<PathBuf>,

    /// Comma-separated nprobe values (plaintext, sap-ivf nprobe sweep). Default: 1,2,4,8,16,32,64,128
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16,32,64,128")]
    nprobe: Vec<u64>,

    /// Comma-separated beta values (sap, sap-ivf beta sweep). Default: 0.0,0.1,0.5,1.0
    #[arg(long, value_delimiter = ',', default_value = "0.0,0.1,0.5,1.0")]
    beta: Vec<f64>,

    /// Comma-separated quantisation-bits values (tiptoe sweep).
    /// Default: 3,4. At dim=768, q=4 exceeds the Tiptoe paper's
    /// inner-product wrap budget but stays within Z_p (signed mod-p);
    /// q=3 fits the budget. Recall comparison across q values exposes
    /// the precision/throughput trade-off.
    #[arg(long, value_delimiter = ',', default_value = "3,4")]
    quantisation_bits: Vec<u8>,

    /// Comma-separated batch sizes B for the score_batch sweep.
    /// Default "1" gives the per-query path (`score_with_realised_cost`);
    /// larger values enable `score_batch` chunks. Within a single
    /// invocation the harness loops over
    /// (batch_sizes × quality_params × repetitions); the build/index
    /// is constructed once per quality_param and reused across all
    /// batch sizes. For each B>1 chunk, queries are chunked into
    /// groups of exactly B (final partial chunk dropped to keep
    /// per-row cost uniform) and `score_batch(handle, &chunk, k)` is
    /// invoked once; the harness writes B raw.csv rows per chunk with
    /// `latency-us` empty and `wallclock-us` / `amortised-latency-us`
    /// populated.
    ///
    /// B>1 rows use `communication_cost(&handle, k)` analytically —
    /// score_batch has no realised cost analog. B=1 rows continue to
    /// use realised cost.
    ///
    /// BN dispatch: when `batch-sizes != [1]`, BN routes through
    /// `run_eval` (loses the per-row `verification-overhead-us`
    /// signal that `run_eval_with_verify_us` would have populated).
    /// Run BN twice if both the B=1 verify timing and the B>1
    /// throughput are needed.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    batch_sizes: Vec<usize>,

    /// Whether BN scorers run Protocol 2 (Freivalds) verification
    /// on each query. Default `false` so every scheme is evaluated
    /// under the honest-but-curious threat model; pass
    /// `--bntm-verification true` to capture the malicious-server
    /// detection cost. `ArgAction::Set` so the value is explicit
    /// (`--bntm-verification true|false`) rather than a presence-only
    /// flag — keeps the Makefile invocation
    /// `--bntm-verification $(BNTM_VERIFICATION)` parseable.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    bntm_verification: bool,

    /// f32 → F_p quantisation scale Q for BN scorers, expressed as the
    /// power-of-two exponent (Q = 2^bntm_q_bits). Default 20 inherits
    /// the EMVP scale; a sweep over this exponent confirms
    /// recall@10 = 1.0 holds at our chosen Q.
    #[arg(long, default_value_t = 20)]
    bntm_q_bits: u32,

    /// Fixed beta for sap-ivf nprobe sweep (ignored for other scorers).
    #[arg(long, default_value = "0.0")]
    beta_fixed: f64,

    /// Fixed nprobe for sap-ivf beta sweep; if set, sap-ivf sweeps beta instead of nprobe.
    #[arg(long)]
    nprobe_fixed: Option<usize>,

    /// GPU/accelerator present on this machine.
    #[arg(long, value_enum, default_value = "none")]
    gpu_kind: GpuKind,

    /// Substrate this run measures against. `cpu` (default) for the
    /// host paths; `gpu` uses cuVS / custom CUDA kernels and is
    /// rejected at the scorer layer until the corresponding `gpu`
    /// Cargo feature lands. The chosen value is recorded both as the
    /// top-level `device` field of `run-metadata.toml` and the per-row
    /// `device` column of `raw.csv`.
    #[arg(long, value_enum, default_value = "cpu")]
    device: DeviceArg,

    /// GPU SKU this run targets. Free-form string so we don't churn
    /// the CLI for every new card. Today's recognised values:
    /// `rtx-5000-ada` (workstation), `v100` (datacenter-class Volta,
    /// the substrate the headline numbers run on), `h100` (Hopper,
    /// kept as a forward-compatible recognised SKU). Required iff
    /// `--device gpu`; ignored for CPU runs (left as `None`).
    #[arg(long)]
    gpu_sku: Option<String>,

    /// `local` (operator-controlled box, on-prem or long-running pod)
    /// or `cloud` (per-rental instance). Required iff `--device gpu`.
    /// Drives whether the `[gpu.cloud]` sub-block in
    /// `run-metadata.toml` is required.
    #[arg(long, value_enum)]
    gpu_location: Option<GpuLocationArg>,

    /// Cloud provider when `--gpu-location cloud`. Required in that
    /// case (rented-instance provenance pinning is enforced, not
    /// optional).
    #[arg(long)]
    cloud_provider: Option<String>,

    /// Instance type, e.g. `p5.48xlarge`. Required when
    /// `--gpu-location cloud`.
    #[arg(long)]
    cloud_instance_type: Option<String>,

    /// Region, e.g. `us-east-1`. Required when `--gpu-location cloud`.
    #[arg(long)]
    cloud_region: Option<String>,

    /// Driver version reported by `nvidia-smi`. Required when
    /// `--gpu-location cloud`.
    #[arg(long)]
    cloud_driver_version: Option<String>,

    /// CUDA toolkit version. Required when `--gpu-location cloud`.
    #[arg(long)]
    cloud_cuda_version: Option<String>,

    /// Hard VRAM cap for the streaming GPU cluster store, in bytes.
    /// Omit to auto-detect 80 % of free VRAM via NVML at handle
    /// build. Ignored for `--device cpu`.
    ///
    /// On a 32 GB RTX 5000 Ada, the NVML auto-detect lands around
    /// ~25 GB depending on the host's other-process pressure; pin
    /// `--gpu-vram-budget-bytes` if cross-run reproducibility matters
    /// (e.g. before publishing a figure that compares two cards).
    #[arg(long)]
    gpu_vram_budget_bytes: Option<u64>,

    /// Wipe scheme-specific cache files in --data-dir before running.
    /// Forces a cold rebuild; `[index].cache-hit` in run-metadata.toml
    /// then reads `false`. Complementary to BuildOutcome's runtime
    /// signal — `--no-cache` records intent, BuildOutcome records
    /// outcome.
    #[arg(long, default_value_t = false)]
    no_cache: bool,

    /// Run substep-instrumented scoring and emit
    /// `substep-breakdown.csv`. Long-format: one row
    /// per `(scheme, config-label, query-id, substep)` with the seven
    /// canonical substeps `route, encode, server-compute, verify,
    /// decompress, decode, merge`. Substeps that don't apply to a
    /// scheme write `us = 0` (uniform schema across schemes). Slower
    /// than the throughput path — per-query `Instant::now()` adds
    /// nanosecond-scale overhead per substep, so do not use these
    /// numbers for `latency-us` / throughput claims (figure 02 still
    /// reads `raw.csv`). In breakdown mode `raw.csv` is created with
    /// only a header; `top_k.csv` is still populated for ID overlap.
    #[arg(long, default_value_t = false)]
    breakdown: bool,

    /// Campaign id. Stable, machine-parseable
    /// identifier grouping this run with related ones (planned sweep,
    /// figure-anchoring batch, exploratory). Convention (not enforced):
    /// `<topic-or-fig>-<hardware-tier>-<ISO-date>`, e.g.
    /// `decode-gpu-2026-05-14`. Allowed chars: ASCII
    /// alphanumerics, `-`, `_`, `.`, `:`. Max 128 chars.
    ///
    /// Setting `--campaign-id` requires `--campaign-title` and vice
    /// versa; the harness refuses to start with a partial set.
    #[arg(long, env = "CAMPAIGN_ID")]
    campaign_id: Option<String>,

    /// Campaign title (human-readable). Free-form string. Required iff
    /// `--campaign-id` is set.
    #[arg(long, env = "CAMPAIGN_TITLE")]
    campaign_title: Option<String>,

    /// Campaign note (optional free-form annotation). Ignored unless
    /// `--campaign-id` and `--campaign-title` are both set.
    #[arg(long, env = "CAMPAIGN_NOTE")]
    campaign_note: Option<String>,

    /// Pin a canonical machine-id, overriding the computed
    /// FNV-1a hash of (cpu_model, cores, ram_bytes). Needed when the same
    /// physical box reports slightly different sysinfo across contexts
    /// (e.g. bare-metal vs a memory-capped container) and would otherwise
    /// fragment results across distinct ids. Must be 8 lowercase hex
    /// chars to match the on-disk `runs/<machine-id>/` convention.
    #[arg(long, env = "MACHINE_ID")]
    machine_id: Option<String>,
}

#[derive(Clone, PartialEq, Eq, clap::ValueEnum)]
enum SchemeArg {
    Plaintext,
    Sap,
    SapIvf,
    Emvp,
    EmvpIvf,
    Tiptoe,
    Bntm,
    BntmIvf,
}

/// Shared IVF defaults used by every IVF-using scorer in a run.
/// All IVF scorers must read from a single source so cross-scheme
/// comparisons aren't confounded by drift in `n_centroids`,
/// `train_seed`, or `max_iter`. Logged at startup and embedded in
/// `run-metadata.toml [ivf]`.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "kebab-case")]
struct IvfDefaults {
    n_centroids: usize,
    train_seed: u64,
    max_iter: usize,
}

impl IvfDefaults {
    fn for_corpus(corpus_len: usize) -> Self {
        Self {
            n_centroids: (corpus_len as f64).sqrt().ceil() as usize,
            train_seed: 42,
            max_iter: 25,
        }
    }
}

/// Filename prefixes for the on-disk cache that each scheme writes
/// into `data_dir`. Empty slice = no disk cache (flat SAP, flat BN).
/// Plaintext uses `.ivf-cache-` (the IVF index built from a plaintext
/// k-means; see scorer-plaintext/src/lib.rs).
fn scheme_cache_prefixes(scheme: &SchemeArg) -> &'static [&'static str] {
    match scheme {
        SchemeArg::Plaintext => &[".ivf-cache-"],
        SchemeArg::Sap => &[],
        SchemeArg::SapIvf => &[".sap-ivf-cache-"],
        SchemeArg::Emvp => &[".emvp-cache-"],
        SchemeArg::EmvpIvf => &[".emvp-ivf-cache-"],
        SchemeArg::Tiptoe => &[".tiptoe-cache-"],
        SchemeArg::Bntm => &[],
        SchemeArg::BntmIvf => &[".bntm-ivf-cache-"],
    }
}

/// Delete `.{prefix}*.bin` cache files for `scheme` under `data_dir`.
/// Returns the number of files removed. Missing dir is treated as
/// "nothing to wipe" (not an error).
fn wipe_caches(scheme: &SchemeArg, data_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let prefixes = scheme_cache_prefixes(scheme);
    let mut wiped = Vec::new();
    if prefixes.is_empty() || !data_dir.is_dir() {
        return Ok(wiped);
    }
    for entry in
        std::fs::read_dir(data_dir).with_context(|| format!("reading {}", data_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        let s = name.to_string_lossy();
        if !s.ends_with(".bin") {
            continue;
        }
        if prefixes.iter().any(|p| s.starts_with(p)) {
            std::fs::remove_file(entry.path())
                .with_context(|| format!("removing {}", entry.path().display()))?;
            wiped.push(entry.path());
        }
    }
    Ok(wiped)
}

#[derive(Clone, clap::ValueEnum)]
enum GpuKind {
    None,
    Mps,
    Cuda,
}

impl fmt::Display for GpuKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuKind::None => write!(f, "none"),
            GpuKind::Mps => write!(f, "mps"),
            GpuKind::Cuda => write!(f, "cuda"),
        }
    }
}

/// `--device` CLI value. Mirrors `scorer_core::Device`; kept as a
/// separate clap-friendly type so the CLI surface and the trait
/// surface evolve independently.
#[derive(Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum DeviceArg {
    Cpu,
    Gpu,
}

impl DeviceArg {
    fn as_str(&self) -> &'static str {
        match self {
            DeviceArg::Cpu => "cpu",
            DeviceArg::Gpu => "gpu",
        }
    }
}

impl From<DeviceArg> for Device {
    fn from(d: DeviceArg) -> Self {
        match d {
            DeviceArg::Cpu => Device::Cpu,
            DeviceArg::Gpu => Device::Gpu,
        }
    }
}

#[derive(Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum GpuLocationArg {
    Local,
    Cloud,
}

impl GpuLocationArg {
    fn as_str(&self) -> &'static str {
        match self {
            GpuLocationArg::Local => "local",
            GpuLocationArg::Cloud => "cloud",
        }
    }
}

// ---------------------------------------------------------------------------
// run-metadata.toml
// ---------------------------------------------------------------------------

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

/// Per-run index-build snapshot. Populated from the
/// first `upload_cluster` call's `BuildOutcome` plus the scheme's
/// known cluster shape. For multi-config sweeps, the first config's
/// build is the one that actually does work; subsequent configs hit
/// the in-memory cache and would all read `cache-hit = true` with
/// sub-millisecond `build-duration-secs`.
#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct IndexBlock {
    cache_hit: bool,
    build_duration_secs: f64,
    cluster_count: usize,
    m_total: usize,
}

/// GPU provenance. Populated only when
/// `--device gpu`; absent on CPU-only runs (the `Option<GpuBlock>`
/// wrapper on `RunMetadata` plus `skip_serializing_if` in the
/// surrounding struct keeps `[gpu]` out of CPU TOML entirely).
#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct GpuBlock {
    sku: String,
    location: String,
    /// Total GPU memory in bytes, captured via `nvidia-smi` at run
    /// time. Skipped when `nvidia-smi` is absent or returns
    /// unparseable output (e.g. CPU-only host with `--device gpu`
    /// rejected by the harness anyway, or Apple Silicon dev box).
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_bytes: Option<u64>,
    /// Resolved streaming-store VRAM budget for this run. `Some(b)`
    /// when `--device gpu` and the budget resolver ran (either
    /// `--gpu-vram-budget-bytes` passthrough or NVML 80 % auto-detect).
    /// Absent on `--device cpu` or when the resolver errored (NVML
    /// unavailable + no explicit override).
    #[serde(skip_serializing_if = "Option::is_none")]
    vram_budget_bytes: Option<u64>,
    /// NVML-sampled high-water mark over the run. Absent when NVML
    /// wasn't available at the sampler's `start()` call or when
    /// `--device cpu`.
    #[serde(skip_serializing_if = "Option::is_none")]
    peak_vram_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cloud: Option<GpuCloudBlock>,
}

/// Memory-pressure telemetry. CPU page-fault deltas
/// over the scoring loop (`/proc/self/stat::{minflt, majflt}`), plus
/// process-wide `ClusterStore` aggregate counters on GPU runs. All
/// fields are absent rather than zero when their source is unavailable
/// so a `--device cpu` run on a non-Linux host emits no block at all.
#[derive(Serialize, Default)]
#[serde(rename_all = "kebab-case")]
struct MemoryBlock {
    /// Minor page faults during the scoring loop (page mapped without
    /// disk I/O — page-table population, COW splits). Mostly background
    /// noise but cheap to capture.
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu_minor_faults: Option<u64>,
    /// Major page faults during the scoring loop (page brought in from
    /// disk). The signal that matters when mmap-backed cache files
    /// exceed RAM (the 8.8 M EMVP / BN caches are 30–70 GB; on a
    /// 64 GB box the OS page cache can spill).
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu_major_faults: Option<u64>,
    /// `ClusterStore::get_or_upload` calls (cache hits + misses).
    /// Sum across every probed cluster across every query in the run.
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_get_count: Option<u64>,
    /// Misses that ran the upload closure successfully. `uploads /
    /// gets` is the miss rate; `gets - uploads` is the hit count.
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_upload_count: Option<u64>,
    /// LRU tail pops driven by budget pressure during scoring. Zero =
    /// the budget held the whole working set; non-zero with
    /// `gpu_upload_count > cluster_count` = thrashing.
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_eviction_count: Option<u64>,
    /// Sum of `DeviceCluster::device_bytes()` across every successful
    /// upload — total H2D bandwidth the streaming pattern paid. A
    /// useful caption denominator: `bytes_uploaded / cluster_count /
    /// per_cluster_bytes` is the per-cluster re-upload multiplier.
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_bytes_uploaded_total: Option<u64>,
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct GpuCloudBlock {
    provider: String,
    instance_type: String,
    region: String,
    driver_version: String,
    cuda_version: String,
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
    /// Compile-time `target_feature` gates the
    /// binary was built with, captured from
    /// `eval_harness::meta::collect_target_features`. Distinguishes
    /// scalar-baseline (`make eval`, baseline x86-64) from
    /// SIMD-enabled (`make eval-native`, AVX-512 etc.) runs on the
    /// same git-sha + machine — without this field the two are
    /// indistinguishable in run-metadata.toml. Sorted; empty for
    /// substrates with no relevant SIMD features detected.
    target_features: Vec<String>,
    kernel_version: String,
    cpu_governor: String,
    notes: String,
    /// Whether `--no-cache` was passed on the command line.
    /// Records *intent* — `[index].cache-hit` records the realised
    /// outcome from BuildOutcome. Both signals are useful: --no-cache
    /// forces a cold run, and cache-hit confirms it (or, on a run
    /// without --no-cache, reports whether warm caches were available).
    no_cache: bool,
    /// Whether `--breakdown` was passed. In breakdown mode the run
    /// emits `substep-breakdown.csv` (per-query × per-substep timings,
    /// long format) instead of `raw.csv`. preprocess.py reads this
    /// flag to decide which CSV is the data source for the run.
    breakdown: bool,
    /// Realised rayon pool size (`rayon::current_num_threads()`
    /// after one trivial use locks the lazy initialiser). Drives the
    /// figure-07 threading axis. All rows in a run share this value, so
    /// it lives here and not on `raw.csv`.
    parallel_threads: usize,
    /// Active numactl binding string for this run, sourced verbatim
    /// from the `NUMACTL_BINDING` env var that the `eval-scaling`
    /// Makefile loop sets per step. `"none"` if unset. Figure 07
    /// splits the N=16 column by this field.
    numactl_binding: String,
    /// Batch sizes B swept in this run. Mirrors the `--batch-sizes`
    /// CLI flag (default `[1]`). Top-level because the batch sweep
    /// applies across schemes, so it lives next to `parallel-threads`
    /// and `device`, not under `[scheme-config]`. Figure 14's TSV
    /// emitter cross-checks against this field.
    batch_sizes: Vec<usize>,
    /// Cgroup v2 CPU quota in vCPU equivalents (`Some(8.0)` for an
    /// enforced `--cpu 8`), or `None` when the run is unconstrained
    /// or not on Linux. Disagreement with `parallel_threads` ×
    /// `MachineInfo::cores` quantifies how soft a RunAI submission's
    /// `--cpu N` is in practice.
    #[serde(skip_serializing_if = "Option::is_none")]
    cgroup_cpu_quota: Option<f64>,
    /// Cgroup v2 memory limit in bytes, `None` when unconstrained.
    #[serde(skip_serializing_if = "Option::is_none")]
    cgroup_memory_bytes: Option<u64>,
    /// Process-wide peak resident set size in bytes, captured from
    /// `/proc/self/status::VmHWM` at end of run. Linux-only; absent on
    /// other platforms or when `/proc/self/status` is unreadable.
    /// Records the actual memory ceiling each scheme's build + sweep
    /// hit, so post-hoc OOM analysis doesn't need operator-side `ps` /
    /// `top` snapshots. Absent on partial runs (the field is set just
    /// before the final atomic write at end of `main`).
    #[serde(skip_serializing_if = "Option::is_none")]
    peak_rss_bytes: Option<u64>,
    /// Substrate this run measured against. `"cpu"` (default) or
    /// `"gpu"`. Mirrors the per-row CSV `device` column.
    device: String,
    // Table sections must come last for valid TOML serialization.
    /// `[campaign]` block. Absent when the run was launched without
    /// `--campaign-id` / `--campaign-title` (legacy runs and one-off
    /// invocations); present with id, title, and optional note
    /// otherwise. Positioned ahead of the other table sections so
    /// `[campaign]` and `[bulk]` (the adjacent grouping/storage
    /// blocks) read together.
    #[serde(skip_serializing_if = "Option::is_none")]
    campaign: Option<eval_harness::meta::Campaign>,
    ivf: IvfDefaults,
    /// Populated before the final atomic write; absent on partial
    /// runs (status="partial") because the first upload hasn't
    /// happened yet at initial-write time.
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<IndexBlock>,
    /// Populated only when `device = "gpu"`. CPU runs omit the entire
    /// `[gpu]` block (and its `[gpu.cloud]` sub-block) so legacy
    /// parsers don't see noisy empty tables.
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu: Option<GpuBlock>,
    /// CPU page-fault deltas + GPU `ClusterStore` aggregates over the
    /// scoring loop. Absent on non-Linux hosts (no `/proc/self/stat`)
    /// and when no fields are populated. Captioned in figures
    /// explaining "fits in cache" vs "thrashing" regimes.
    #[serde(skip_serializing_if = "Option::is_none")]
    memory: Option<MemoryBlock>,
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
    // Atomic on Linux/macOS (POSIX rename(2)).
    // On Windows, rename overwrites but is not guaranteed atomic; project is Linux-first.
    std::fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))
}

// ---------------------------------------------------------------------------
// machines.csv
// ---------------------------------------------------------------------------

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
    // machine-name defaults to the hostname; edit by hand if a friendlier label is wanted
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
    eprintln!(
        "Machine: {} ({}, {} cores, {} bytes RAM, os={}, governor={}, gpu={})",
        info.id,
        info.cpu_model,
        info.cores,
        info.ram_bytes,
        info.os,
        info.cpu_governor,
        info.gpu_kind,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// index.csv
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// GPU block construction
// ---------------------------------------------------------------------------

/// Validate the `--device` / `--gpu-*` / `--cloud-*` matrix and assemble
/// a `[gpu]` block when applicable. Returns `Ok(None)` for CPU runs;
/// returns an error if a `--device gpu` invocation is missing required
/// SKU / location / cloud-provenance fields. Cloud-flag absence with
/// `--gpu-location cloud` is rejected — rented-instance provenance
/// pinning is enforced, not optional.
fn build_gpu_block(args: &Args) -> anyhow::Result<Option<GpuBlock>> {
    if args.device == DeviceArg::Cpu {
        if args.gpu_sku.is_some() || args.gpu_location.is_some() {
            bail!(
                "--gpu-sku / --gpu-location require --device gpu;\n       \
                 the harness rejects them on CPU runs to prevent stale GPU\n       \
                 metadata from landing in run-metadata.toml"
            );
        }
        return Ok(None);
    }

    let sku = args.gpu_sku.clone().ok_or_else(|| {
        anyhow::anyhow!("--device gpu requires --gpu-sku <SKU> (e.g. rtx-5000-ada, v100, h100)")
    })?;
    let location = args
        .gpu_location
        .ok_or_else(|| anyhow::anyhow!("--device gpu requires --gpu-location {{local|cloud}}"))?;

    let cloud = if location == GpuLocationArg::Cloud {
        let provider = args
            .cloud_provider
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--gpu-location cloud requires --cloud-provider"))?;
        let instance_type = args.cloud_instance_type.clone().ok_or_else(|| {
            anyhow::anyhow!("--gpu-location cloud requires --cloud-instance-type")
        })?;
        let region = args
            .cloud_region
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--gpu-location cloud requires --cloud-region"))?;
        let driver_version = args.cloud_driver_version.clone().ok_or_else(|| {
            anyhow::anyhow!("--gpu-location cloud requires --cloud-driver-version")
        })?;
        let cuda_version = args
            .cloud_cuda_version
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--gpu-location cloud requires --cloud-cuda-version"))?;
        Some(GpuCloudBlock {
            provider,
            instance_type,
            region,
            driver_version,
            cuda_version,
        })
    } else {
        // local — cloud-* flags must NOT be set.
        if args.cloud_provider.is_some()
            || args.cloud_instance_type.is_some()
            || args.cloud_region.is_some()
            || args.cloud_driver_version.is_some()
            || args.cloud_cuda_version.is_some()
        {
            bail!(
                "--cloud-* flags require --gpu-location cloud;\n       \
                 they're rejected for local runs so a misfire doesn't bury\n       \
                 stale provider metadata in a workstation run's metadata"
            );
        }
        None
    };

    Ok(Some(GpuBlock {
        sku,
        location: location.as_str().to_string(),
        memory_bytes: meta::capture_gpu_memory_bytes(),
        // Filled in after the streaming budget is resolved + the
        // sampler runs over the score loop; see main() below.
        vram_budget_bytes: None,
        peak_vram_bytes: None,
        cloud,
    }))
}

// ---------------------------------------------------------------------------
// scheme_config table
// ---------------------------------------------------------------------------

fn scheme_config_plaintext(
    n_centroids: usize,
    nprobe_values: &[u64],
    max_iter: usize,
) -> toml::Table {
    let mut t = toml::Table::new();
    t.insert("scheme".into(), toml::Value::String("plaintext".into()));
    t.insert(
        "n-centroids".into(),
        toml::Value::Integer(n_centroids as i64),
    );
    t.insert("max-iter".into(), toml::Value::Integer(max_iter as i64));
    t.insert("train-seed".into(), toml::Value::Integer(42));
    t.insert(
        "nprobe-values".into(),
        toml::Value::Array(
            nprobe_values
                .iter()
                .map(|&n| toml::Value::Integer(n as i64))
                .collect(),
        ),
    );
    t
}

fn scheme_config_sap(beta_values: &[f64]) -> toml::Table {
    let mut t = toml::Table::new();
    t.insert("scheme".into(), toml::Value::String("sap".into()));
    t.insert("upload-seed".into(), toml::Value::Integer(42));
    t.insert(
        "beta-values".into(),
        toml::Value::Array(beta_values.iter().map(|&b| toml::Value::Float(b)).collect()),
    );
    t
}

fn scheme_config_sap_ivf(
    n_centroids: usize,
    max_iter: usize,
    nprobe_fixed: Option<usize>,
    beta_fixed: f64,
    nprobe_values: &[u64],
    beta_values: &[f64],
) -> toml::Table {
    let mut t = toml::Table::new();
    t.insert("scheme".into(), toml::Value::String("sap-ivf".into()));
    t.insert(
        "n-centroids".into(),
        toml::Value::Integer(n_centroids as i64),
    );
    t.insert("max-iter".into(), toml::Value::Integer(max_iter as i64));
    t.insert("train-seed".into(), toml::Value::Integer(42));
    t.insert("upload-seed".into(), toml::Value::Integer(42));
    match nprobe_fixed {
        None => {
            t.insert("sweep".into(), toml::Value::String("nprobe".into()));
            t.insert("beta-fixed".into(), toml::Value::Float(beta_fixed));
            t.insert(
                "nprobe-values".into(),
                toml::Value::Array(
                    nprobe_values
                        .iter()
                        .map(|&n| toml::Value::Integer(n as i64))
                        .collect(),
                ),
            );
        }
        Some(np) => {
            t.insert("sweep".into(), toml::Value::String("beta".into()));
            t.insert("nprobe-fixed".into(), toml::Value::Integer(np as i64));
            t.insert(
                "beta-values".into(),
                toml::Value::Array(beta_values.iter().map(|&b| toml::Value::Float(b)).collect()),
            );
        }
    }
    t
}

fn scheme_config_emvp() -> toml::Table {
    use scorer_emvp::params::{Q, SEC128};
    let p = &SEC128;
    let mut t = toml::Table::new();
    t.insert("scheme".into(), toml::Value::String("emvp".into()));
    t.insert("params".into(), toml::Value::String("Sec128".into()));
    t.insert("n".into(), toml::Value::Integer(p.n as i64));
    t.insert("k".into(), toml::Value::Integer(p.k as i64));
    t.insert("s".into(), toml::Value::Integer(p.s as i64));
    t.insert("b".into(), toml::Value::Integer(p.b as i64));
    t.insert("ell0".into(), toml::Value::Integer(p.ell0 as i64));
    t.insert("quantisation-q".into(), toml::Value::Integer(Q as i64));
    t.insert(
        "trapdoor".into(),
        toml::Value::String("option-b-placeholder-r".into()),
    );
    t.insert("key-seed".into(), toml::Value::String("42*32".into()));
    t
}

fn scheme_config_emvp_ivf(
    n_centroids: usize,
    max_iter: usize,
    nprobe_values: &[u64],
) -> toml::Table {
    use scorer_emvp::params::{Q, SEC128};
    let p = &SEC128;
    let mut t = toml::Table::new();
    t.insert("scheme".into(), toml::Value::String("emvp-ivf".into()));
    t.insert("params".into(), toml::Value::String("Sec128".into()));
    t.insert("n".into(), toml::Value::Integer(p.n as i64));
    t.insert("k".into(), toml::Value::Integer(p.k as i64));
    t.insert("s".into(), toml::Value::Integer(p.s as i64));
    t.insert("b".into(), toml::Value::Integer(p.b as i64));
    t.insert("ell0".into(), toml::Value::Integer(p.ell0 as i64));
    t.insert("quantisation-q".into(), toml::Value::Integer(Q as i64));
    t.insert(
        "trapdoor".into(),
        toml::Value::String("option-b-placeholder-r".into()),
    );
    t.insert("key-seed".into(), toml::Value::String("42*32".into()));
    t.insert(
        "n-centroids".into(),
        toml::Value::Integer(n_centroids as i64),
    );
    t.insert("max-iter".into(), toml::Value::Integer(max_iter as i64));
    t.insert("train-seed".into(), toml::Value::Integer(42));
    t.insert("upload-seed".into(), toml::Value::Integer(7));
    t.insert(
        "nprobe-values".into(),
        toml::Value::Array(
            nprobe_values
                .iter()
                .map(|&n| toml::Value::Integer(n as i64))
                .collect(),
        ),
    );
    t
}

fn scheme_config_tiptoe(
    n_centroids: usize,
    max_iter: usize,
    quantisation_bits_values: &[u8],
) -> toml::Table {
    let mut t = toml::Table::new();
    t.insert("scheme".into(), toml::Value::String("tiptoe".into()));
    t.insert(
        "n-centroids".into(),
        toml::Value::Integer(n_centroids as i64),
    );
    t.insert("max-iter".into(), toml::Value::Integer(max_iter as i64));
    t.insert("train-seed".into(), toml::Value::Integer(42));
    t.insert(
        "lwe-params".into(),
        toml::Value::String("tiptoe-text".into()),
    );
    t.insert(
        "bfv-params".into(),
        toml::Value::String("tiptoe-text".into()),
    );
    t.insert(
        "quantisation-bits-values".into(),
        toml::Value::Array(
            quantisation_bits_values
                .iter()
                .map(|&q| toml::Value::Integer(q as i64))
                .collect(),
        ),
    );
    t
}

fn scheme_config_bntm(verification_enabled: bool, q_bits: u32) -> toml::Table {
    let p = BnTmParams::Sec128;
    let mut t = toml::Table::new();
    t.insert("scheme".into(), toml::Value::String("bntm".into()));
    t.insert("params".into(), toml::Value::String("Sec128".into()));
    t.insert("n".into(), toml::Value::Integer(p.n() as i64));
    t.insert("n-1".into(), toml::Value::Integer(p.n1() as i64));
    t.insert("delta".into(), toml::Value::Float(p.delta()));
    t.insert("epsilon".into(), toml::Value::Float(p.epsilon()));
    t.insert("mu".into(), toml::Value::Float(p.mu()));
    t.insert(
        "lambda-prime".into(),
        toml::Value::Integer(p.verification_trials() as i64),
    );
    t.insert(
        "verification-enabled".into(),
        toml::Value::Boolean(verification_enabled),
    );
    t.insert(
        "trapdoor".into(),
        toml::Value::String("option-b-placeholder-hl-s".into()),
    );
    t.insert("key-seed".into(), toml::Value::String("42*32".into()));
    t.insert(
        "quantisation-q-bits".into(),
        toml::Value::Integer(q_bits as i64),
    );
    t.insert(
        "quantisation-q".into(),
        toml::Value::Integer((1u64 << q_bits) as i64),
    );
    t
}

fn scheme_config_bntm_ivf(
    n_centroids: usize,
    max_iter: usize,
    nprobe_values: &[u64],
    verification_enabled: bool,
    q_bits: u32,
) -> toml::Table {
    let mut t = scheme_config_bntm(verification_enabled, q_bits);
    t.insert("scheme".into(), toml::Value::String("bntm-ivf".into()));
    t.insert(
        "n-centroids".into(),
        toml::Value::Integer(n_centroids as i64),
    );
    t.insert("max-iter".into(), toml::Value::Integer(max_iter as i64));
    t.insert("train-seed".into(), toml::Value::Integer(42));
    t.insert("upload-seed".into(), toml::Value::Integer(7));
    t.insert(
        "nprobe-values".into(),
        toml::Value::Array(
            nprobe_values
                .iter()
                .map(|&n| toml::Value::Integer(n as i64))
                .collect(),
        ),
    );
    t
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.output.is_some() {
        bail!(
            "--output has been removed; use --results-dir <PATH> instead\n       \
             (the harness now writes to <results-dir>/runs/<machine-id>/<git-sha>/<run-id>/raw.csv)"
        );
    }

    // Validate the campaign tuple before any compute starts. Surfaces
    // missing `--campaign-title` (or vice versa) as a clap-time error
    // instead of mid-run.
    let campaign = meta::Campaign::try_new(
        args.campaign_id.clone(),
        args.campaign_title.clone(),
        args.campaign_note.clone(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Collect provenance once at startup.
    let git = meta::collect_git_state();
    let mut machine = meta::collect_machine_info(&args.gpu_kind.to_string());
    if let Some(ref id) = args.machine_id {
        if id.len() != 8
            || !id
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            bail!("--machine-id must be 8 lowercase hex chars, got {id:?}");
        }
        machine.id = id.clone();
    }
    let rust_toolchain = meta::collect_rust_toolchain();

    let started_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let started_at = meta::unix_secs_to_iso8601(started_secs);
    let run_id = format!("{started_secs}");

    // Baseline page-fault snapshot for the [memory] block. Sample
    // again at end-of-run; the delta lands in run-metadata.toml.
    // ClusterStore stats use process-wide atomics
    // (see scorer_gpu_common::global_stats), so we don't need to
    // snapshot them here — they start at zero at process startup.
    let memory_baseline_faults = meta::capture_proc_faults();

    if git.dirty {
        eprintln!(
            "warning: working tree has uncommitted changes (git-dirty = true)\n         \
             results recorded under SHA {} may not be reproducible from\n         \
             git history alone",
            &git.sha[..git.sha.len().min(8)],
        );
    }

    let results_dir = &args.results_dir;
    let machines_path = results_dir.join("machines.csv");
    log_machine(&machines_path, &machine)?;

    let embeddings_path = args.data_dir.join("passages.fvecs");
    let queries_path = args.data_dir.join("queries.fvecs");
    let ground_truth_path = args.data_dir.join("ground_truth.ivecs");

    if !ground_truth_path.exists() {
        bail!(
            "ground-truth file not found: {}\n       \
             Run `make ground-truth DATASET=<name>` first.",
            ground_truth_path.display()
        );
    }

    if args.no_cache {
        let wiped = wipe_caches(&args.scorer, &args.data_dir)?;
        if wiped.is_empty() {
            eprintln!(
                "--no-cache: no scheme-specific cache files to wipe in {:?}",
                args.data_dir
            );
        } else {
            eprintln!("--no-cache: wiped {} cache file(s):", wiped.len());
            for p in &wiped {
                eprintln!("  {}", p.display());
            }
        }
    }

    eprintln!("Loading corpus from {:?}…", embeddings_path);
    let raw_corpus =
        load_fvecs(&embeddings_path).with_context(|| format!("reading {:?}", embeddings_path))?;
    let corpus: Vec<Vector> = raw_corpus.into_iter().map(Vector).collect();
    eprintln!("  {} vectors", corpus.len());

    eprintln!("Loading queries from {:?}…", queries_path);
    let raw_queries =
        load_fvecs(&queries_path).with_context(|| format!("reading {:?}", queries_path))?;
    let mut queries: Vec<Vector> = raw_queries.into_iter().map(Vector).collect();
    if let Some(n) = args.queries {
        let n = n.min(queries.len());
        queries.truncate(n);
        eprintln!("  --queries {n}: capping to first {n} queries");
    }
    eprintln!("  {} queries", queries.len());

    eprintln!("Loading ground truth from {:?}…", ground_truth_path);
    let ground_truth = load_ivecs(&ground_truth_path)
        .with_context(|| format!("reading {:?}", ground_truth_path))?;

    // Build run directory: <results-dir>/runs/<machine-id>/<git-sha>/<run-id>/
    let run_dir = results_dir
        .join("runs")
        .join(&machine.id)
        .join(&git.sha)
        .join(&run_id);
    std::fs::create_dir_all(&run_dir)?;

    let dim = corpus.first().map_or(0, |v| v.0.len());
    let ivf = IvfDefaults::for_corpus(corpus.len());
    let n_centroids = ivf.n_centroids;
    let max_iter = ivf.max_iter;
    eprintln!(
        "IVF defaults for this run: n_centroids={}, train_seed={}, max_iter={}",
        ivf.n_centroids, ivf.train_seed, ivf.max_iter
    );

    // Capture the threading axis before any scorer work runs.
    // `capture_parallel_threads` issues one trivial rayon use to lock
    // the pool size, so the value matches what the eval will actually
    // use rather than the configured-but-not-yet-built default.
    let parallel_threads = meta::capture_parallel_threads();
    let numactl_binding = meta::capture_numactl_binding();
    let cgroup_cpu_quota = meta::capture_cgroup_cpu_quota();
    let cgroup_memory_bytes = meta::capture_cgroup_memory_bytes();
    eprintln!("Threading: parallel-threads={parallel_threads}, numactl-binding={numactl_binding}");
    if let Some(q) = cgroup_cpu_quota {
        eprintln!("Cgroup: cpu-quota={q:.2} vCPU");
    }
    if let Some(b) = cgroup_memory_bytes {
        eprintln!("Cgroup: memory-bytes={b}");
    }

    // Validate the GPU CLI surface and assemble the optional `[gpu]`
    // block. CPU runs leave `gpu_block` as `None` so the `[gpu]` table
    // never lands in their `run-metadata.toml`.
    let gpu_block = build_gpu_block(&args)?;
    let device_str = args.device.as_str();
    // Tiptoe-GPU is not implemented (analytical proxy only), so reject
    // `--device gpu` for tiptoe here — otherwise we'd emit `device=gpu`
    // rows that were actually produced by the CPU code path. Every
    // other scheme accepts `--device gpu`.
    //
    // Note on layered guards: requesting `--device gpu` against a
    // binary built without `--features gpu` passes this CLI check and
    // then surfaces as the scorer crate's `GpuFeatureNotEnabled`
    // variant. That's intentional — a feature-off build is a
    // build-time choice, not a CLI argument error.
    if args.device == DeviceArg::Gpu && matches!(args.scorer, SchemeArg::Tiptoe) {
        bail!(
            "--device gpu is not supported for --scorer tiptoe — Tiptoe-GPU is\n       \
             deferred (analytical proxy only). The `tiptoe-go`\n       \
             paired runner is its own binary (`bin/tiptoe_go_runner`) and pins\n       \
             device = \"cpu\" by construction. All other measured schemes\n       \
             (plaintext, sap, sap-ivf, emvp, emvp-ivf, bntm, bntm-ivf) accept\n       \
             --device gpu."
        );
    }

    // Resolve the streaming VRAM budget once per run and start the
    // NVML peak sampler. Both are gated on `--device gpu`
    // AND `feature = "gpu"`; CPU builds and CPU runs leave them at
    // `None` and the `[gpu].vram-budget-bytes` / `[gpu].peak-vram-bytes`
    // fields stay absent from `run-metadata.toml`.
    //
    // Budget resolution happens up-front so every per-quality-param
    // handle the scheme dispatch builds below shares the same value,
    // recorded in `[gpu].vram-budget-bytes` for cross-run
    // reproducibility checks. The sampler runs as a background
    // thread; reading `peak_bytes()` at run end gives the high-water
    // mark across the entire sweep.
    #[cfg(feature = "gpu")]
    let (gpu_vram_budget_bytes, gpu_peak_sampler) = if args.device == DeviceArg::Gpu {
        // NVML doesn't respect CUDA_VISIBLE_DEVICES; the scorer-side
        // helper resolves the physical index for cuda(0) so the
        // budget + sampler watch the same card cudarc / cuvs land
        // on. With two-card eval (`CUDA_VISIBLE_DEVICES=0` vs `=1`)
        // the sampler otherwise sits on the idle card and reports a
        // spuriously low peak.
        let nvml_idx = scorer_gpu_common::nvml_index_for_cuda_device(0);
        let budget = scorer_gpu_common::resolve_budget(args.gpu_vram_budget_bytes, nvml_idx)
            .map_err(|e| anyhow::anyhow!("VRAM budget resolver: {e}"))?;
        let sampler = scorer_gpu_common::PeakVramSampler::start_default(nvml_idx).ok();
        (Some(budget), sampler)
    } else {
        (None, None)
    };
    #[cfg(not(feature = "gpu"))]
    let gpu_vram_budget_bytes: Option<u64> = None;

    // Determine scheme name and config table for metadata.
    let scheme_name = match args.scorer {
        SchemeArg::Plaintext => "plaintext",
        SchemeArg::Sap => "sap",
        SchemeArg::SapIvf => "sap-ivf",
        SchemeArg::Emvp => "emvp",
        SchemeArg::EmvpIvf => "emvp-ivf",
        SchemeArg::Tiptoe => "tiptoe",
        SchemeArg::Bntm => "bntm",
        SchemeArg::BntmIvf => "bntm-ivf",
    };
    let scheme_cfg = match args.scorer {
        SchemeArg::Plaintext => scheme_config_plaintext(n_centroids, &args.nprobe, max_iter),
        SchemeArg::Sap => scheme_config_sap(&args.beta),
        SchemeArg::SapIvf => scheme_config_sap_ivf(
            n_centroids,
            max_iter,
            args.nprobe_fixed,
            args.beta_fixed,
            &args.nprobe,
            &args.beta,
        ),
        SchemeArg::Emvp => scheme_config_emvp(),
        SchemeArg::EmvpIvf => scheme_config_emvp_ivf(n_centroids, max_iter, &args.nprobe),
        SchemeArg::Tiptoe => scheme_config_tiptoe(n_centroids, max_iter, &args.quantisation_bits),
        SchemeArg::Bntm => scheme_config_bntm(args.bntm_verification, args.bntm_q_bits),
        SchemeArg::BntmIvf => scheme_config_bntm_ivf(
            n_centroids,
            max_iter,
            &args.nprobe,
            args.bntm_verification,
            args.bntm_q_bits,
        ),
    };

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

    // Write initial run-metadata.toml with status="partial".
    let toml_path = run_dir.join("run-metadata.toml");
    let mut run_meta = RunMetadata {
        run_id: run_id.clone(),
        machine_id: machine.id.clone(),
        git_sha: git.sha.clone(),
        git_dirty: git.dirty,
        git_branch: git.branch.clone(),
        started_at: started_at.clone(),
        finished_at: None,
        duration_secs: None,
        status: "partial".into(),
        harness_version: env!("CARGO_PKG_VERSION").to_string(),
        rust_toolchain,
        target_features: meta::collect_target_features(),
        kernel_version: machine.kernel_version.clone(),
        cpu_governor: machine.cpu_governor.clone(),
        notes: String::new(),
        no_cache: args.no_cache,
        breakdown: args.breakdown,
        parallel_threads,
        numactl_binding,
        batch_sizes: args.batch_sizes.clone(),
        cgroup_cpu_quota,
        cgroup_memory_bytes,
        peak_rss_bytes: None,
        device: device_str.to_string(),
        campaign: campaign.clone(),
        ivf,
        index: None,
        gpu: gpu_block,
        memory: None,
        scheme_config: scheme_cfg,
        dataset: dataset_meta,
    };
    write_metadata(&toml_path, &run_meta)?;

    // Open raw.csv (append; write header only if new).
    let raw_csv_path = run_dir.join("raw.csv");
    let csv_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&raw_csv_path)?;
    // LineWriter instead of BufWriter so each row's trailing `\n`
    // flushes to the kernel. Default BufWriter (8 KB) holds ~40
    // ~200-byte rows; at ~14 s/query on emvp-ivf nprobe=2967 that
    // hides the last ~9 min of writes from concurrent log readers.
    let mut w = LineWriter::new(csv_file);
    {
        if File::open(&raw_csv_path)?.metadata()?.len() == 0 {
            write_csv_header(&mut w)?;
        }
    }

    // Open top_k.csv (per-query top-k IDs; consumed by
    // analysis/tiptoe_diff.py for the validation gate's ID overlap).
    let top_k_csv_path = run_dir.join("top_k.csv");
    let top_k_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&top_k_csv_path)?;
    let mut top_k_w = LineWriter::new(top_k_file);
    {
        if File::open(&top_k_csv_path)?.metadata()?.len() == 0 {
            write_top_k_header(&mut top_k_w)?;
        }
    }

    // Open substep-breakdown.csv only in breakdown mode. The canonical
    // taxonomy emits 7 long-format rows per (config, query)
    // — see eval_harness::CANONICAL_SUBSTEPS.
    let mut breakdown_w: Option<LineWriter<File>> = if args.breakdown {
        let path = run_dir.join("substep-breakdown.csv");
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let mut w = LineWriter::new(file);
        if File::open(&path)?.metadata()?.len() == 0 {
            write_substep_breakdown_header(&mut w)?;
        }
        Some(w)
    } else {
        None
    };

    let mp = MultiProgress::new();

    let build_outcome: BuildOutcome = match args.scorer {
        SchemeArg::Plaintext => {
            eprintln!(
                "PlaintextScorer: dim={dim}, n_centroids={n_centroids} (sqrt({}))",
                corpus.len()
            );
            let cache_dir = args.data_dir.as_path();
            let scorer = PlaintextScorer::with_cache_dir(cache_dir);
            let nprobes: Vec<f64> = args.nprobe.iter().map(|&n| n as f64).collect();

            let km_total = (n_centroids + max_iter) as u64;
            let km_pb = mp.add(ProgressBar::new(km_total));
            km_pb.set_style(
                ProgressStyle::with_template(
                    "  building index [{bar:30}] {pos}/{len}  {elapsed_precise}",
                )
                .unwrap()
                .progress_chars("=> "),
            );
            if args.breakdown {
                let bw = breakdown_w
                    .as_mut()
                    .expect("breakdown writer present when --breakdown is set");
                run_eval_breakdown(
                    &scorer,
                    "plaintext",
                    "nprobe",
                    &nprobes,
                    |&nprobe| PlaintextConfig {
                        n_centroids,
                        nprobe: nprobe as usize,
                        train_seed: 42,
                        max_iter,
                        progress: Some(Arc::new(IndicatifProgress::new(km_pb.clone()))),
                        device: args.device.into(),
                        vram_budget_bytes: gpu_vram_budget_bytes,
                    },
                    |qp| format!("nprobe={}", qp as usize),
                    &corpus,
                    &queries,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    bw,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            } else {
                run_eval(
                    &scorer,
                    "plaintext",
                    "nprobe",
                    &nprobes,
                    |&nprobe| PlaintextConfig {
                        n_centroids,
                        nprobe: nprobe as usize,
                        train_seed: 42,
                        max_iter,
                        progress: Some(Arc::new(IndicatifProgress::new(km_pb.clone()))),
                        device: args.device.into(),
                        vram_budget_bytes: gpu_vram_budget_bytes,
                    },
                    |qp| format!("nprobe={}", qp as usize),
                    &corpus,
                    &queries,
                    &ground_truth,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    device_str,
                    &args.batch_sizes,
                    &mut w,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            }
        }
        SchemeArg::Sap => {
            eprintln!("SapScorer: dim={dim}");
            let scorer = SapScorer;
            if args.breakdown {
                let bw = breakdown_w
                    .as_mut()
                    .expect("breakdown writer present when --breakdown is set");
                run_eval_breakdown(
                    &scorer,
                    "sap",
                    "beta",
                    &args.beta,
                    |&beta| SapConfig {
                        key: keygen(1.0, 2.0, 42),
                        beta,
                        upload_seed: 42,
                        progress: None,
                        device: args.device.into(),
                        vram_budget_bytes: gpu_vram_budget_bytes,
                    },
                    |qp| format!("beta={qp:.4}"),
                    &corpus,
                    &queries,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    bw,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            } else {
                run_eval(
                    &scorer,
                    "sap",
                    "beta",
                    &args.beta,
                    |&beta| SapConfig {
                        key: keygen(1.0, 2.0, 42),
                        beta,
                        upload_seed: 42,
                        progress: None,
                        device: args.device.into(),
                        vram_budget_bytes: gpu_vram_budget_bytes,
                    },
                    |qp| format!("beta={qp:.4}"),
                    &corpus,
                    &queries,
                    &ground_truth,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    device_str,
                    &args.batch_sizes,
                    &mut w,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            }
        }
        SchemeArg::SapIvf => {
            eprintln!(
                "SapIvfScorer: dim={dim}, n_centroids={n_centroids} (sqrt({}))",
                corpus.len()
            );
            let scorer = SapIvfScorer::with_cache_dir(args.data_dir.as_path());
            // SAP-IVF fires `on_encrypt` per-row (one event per encrypted
            // vector), unlike EMVP-IVF which is per-cluster. The total
            // must include the corpus length or the bar overshoots its
            // `len` once encryption starts.
            let km_total = (n_centroids + max_iter + corpus.len()) as u64;

            match args.nprobe_fixed {
                None => {
                    let beta_fixed = args.beta_fixed;
                    let nprobes: Vec<f64> = args.nprobe.iter().map(|&n| n as f64).collect();
                    let mp_c = mp.clone();
                    let make_config = move |&nprobe: &f64| {
                        let km_pb = mp_c.add(ProgressBar::new(km_total));
                        km_pb.set_style(
                            ProgressStyle::with_template(
                                "  building index [{bar:30}] {pos}/{len}  {elapsed_precise}",
                            )
                            .unwrap()
                            .progress_chars("=> "),
                        );
                        SapIvfConfig {
                            key: keygen(1.0, 2.0, 42),
                            beta: beta_fixed,
                            upload_seed: 42,
                            n_centroids,
                            nprobe: nprobe as usize,
                            train_seed: 42,
                            max_iter,
                            progress: Some(Arc::new(IndicatifProgress::new(km_pb))),
                            device: args.device.into(),
                            vram_budget_bytes: gpu_vram_budget_bytes,
                        }
                    };
                    let make_label =
                        move |qp: f64| format!("beta={beta_fixed:.4}|nprobe={}", qp as usize);
                    if args.breakdown {
                        let bw = breakdown_w
                            .as_mut()
                            .expect("breakdown writer present when --breakdown is set");
                        run_eval_breakdown(
                            &scorer,
                            "sap-ivf",
                            "nprobe",
                            &nprobes,
                            make_config,
                            make_label,
                            &corpus,
                            &queries,
                            args.k,
                            args.repetitions,
                            &run_id,
                            &machine.id,
                            bw,
                            &mut top_k_w,
                            &mp,
                        )
                        .await?
                    } else {
                        run_eval(
                            &scorer,
                            "sap-ivf",
                            "nprobe",
                            &nprobes,
                            make_config,
                            make_label,
                            &corpus,
                            &queries,
                            &ground_truth,
                            args.k,
                            args.repetitions,
                            &run_id,
                            &machine.id,
                            device_str,
                            &args.batch_sizes,
                            &mut w,
                            &mut top_k_w,
                            &mp,
                        )
                        .await?
                    }
                }
                Some(nprobe_fixed) => {
                    let mp_c = mp.clone();
                    let make_config = move |&beta: &f64| {
                        let km_pb = mp_c.add(ProgressBar::new(km_total));
                        km_pb.set_style(
                            ProgressStyle::with_template(
                                "  building index [{bar:30}] {pos}/{len}  {elapsed_precise}",
                            )
                            .unwrap()
                            .progress_chars("=> "),
                        );
                        SapIvfConfig {
                            key: keygen(1.0, 2.0, 42),
                            beta,
                            upload_seed: 42,
                            n_centroids,
                            nprobe: nprobe_fixed,
                            train_seed: 42,
                            max_iter,
                            progress: Some(Arc::new(IndicatifProgress::new(km_pb))),
                            device: args.device.into(),
                            vram_budget_bytes: gpu_vram_budget_bytes,
                        }
                    };
                    let make_label = move |qp: f64| format!("beta={qp:.4}|nprobe={nprobe_fixed}");
                    if args.breakdown {
                        let bw = breakdown_w
                            .as_mut()
                            .expect("breakdown writer present when --breakdown is set");
                        run_eval_breakdown(
                            &scorer,
                            "sap-ivf",
                            "beta",
                            &args.beta,
                            make_config,
                            make_label,
                            &corpus,
                            &queries,
                            args.k,
                            args.repetitions,
                            &run_id,
                            &machine.id,
                            bw,
                            &mut top_k_w,
                            &mp,
                        )
                        .await?
                    } else {
                        run_eval(
                            &scorer,
                            "sap-ivf",
                            "beta",
                            &args.beta,
                            make_config,
                            make_label,
                            &corpus,
                            &queries,
                            &ground_truth,
                            args.k,
                            args.repetitions,
                            &run_id,
                            &machine.id,
                            device_str,
                            &args.batch_sizes,
                            &mut w,
                            &mut top_k_w,
                            &mp,
                        )
                        .await?
                    }
                }
            }
        }
        SchemeArg::Emvp => {
            eprintln!("EmvpScorer: dim={dim}, Sec128 params (n=1292, s=76, b=17, ell0=1024)");
            let scorer = EmvpScorer::with_cache_dir(args.data_dir.as_path());
            let key_seed = [42u8; 32];

            let enc_pb = mp.add(ProgressBar::new(corpus.len() as u64));
            enc_pb.set_style(
                ProgressStyle::with_template(
                    "  encrypting [{bar:30}] {pos}/{len} rows  {elapsed_precise}",
                )
                .unwrap()
                .progress_chars("=> "),
            );
            if args.breakdown {
                let bw = breakdown_w
                    .as_mut()
                    .expect("breakdown writer present when --breakdown is set");
                run_eval_breakdown(
                    &scorer,
                    "emvp",
                    "none",
                    &[0.0f64],
                    move |_| EmvpConfig {
                        key_seed,
                        progress: Some(Arc::new(IndicatifProgress::new(enc_pb.clone()))),
                        device: args.device.into(),
                        vram_budget_bytes: gpu_vram_budget_bytes,
                    },
                    |_| "emvp".to_string(),
                    &corpus,
                    &queries,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    bw,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            } else {
                run_eval(
                    &scorer,
                    "emvp",
                    "none",
                    &[0.0f64],
                    move |_| EmvpConfig {
                        key_seed,
                        progress: Some(Arc::new(IndicatifProgress::new(enc_pb.clone()))),
                        device: args.device.into(),
                        vram_budget_bytes: gpu_vram_budget_bytes,
                    },
                    |_| "emvp".to_string(),
                    &corpus,
                    &queries,
                    &ground_truth,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    device_str,
                    &args.batch_sizes,
                    &mut w,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            }
        }
        SchemeArg::EmvpIvf => {
            eprintln!(
                "EmvpIvfScorer: dim={dim}, n_centroids={n_centroids} (sqrt({})), Sec128 params (n=1292, s=76, b=17, ell0=1024)",
                corpus.len()
            );
            let scorer = EmvpIvfScorer::with_cache_dir(args.data_dir.as_path());
            let key_seed = [42u8; 32];
            let nprobes: Vec<f64> = args.nprobe.iter().map(|&n| n as f64).collect();
            let mp_c = mp.clone();
            let train_seed = ivf.train_seed;
            let make_config = move |&nprobe: &f64| {
                let total = (n_centroids + max_iter + n_centroids) as u64;
                let pb = mp_c.add(ProgressBar::new(total));
                pb.set_style(
                    ProgressStyle::with_template(
                        "  building index [{bar:30}] {pos}/{len}  {elapsed_precise}",
                    )
                    .unwrap()
                    .progress_chars("=> "),
                );
                EmvpIvfConfig {
                    key_seed,
                    n_centroids,
                    nprobe: nprobe as usize,
                    train_seed,
                    max_iter,
                    upload_seed: 7,
                    progress: Some(Arc::new(IndicatifProgress::new(pb))),
                    device: args.device.into(),
                    vram_budget_bytes: gpu_vram_budget_bytes,
                }
            };
            let make_label = |qp: f64| format!("nprobe={}", qp as usize);
            if args.breakdown {
                let bw = breakdown_w
                    .as_mut()
                    .expect("breakdown writer present when --breakdown is set");
                run_eval_breakdown(
                    &scorer,
                    "emvp-ivf",
                    "nprobe",
                    &nprobes,
                    make_config,
                    make_label,
                    &corpus,
                    &queries,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    bw,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            } else {
                run_eval(
                    &scorer,
                    "emvp-ivf",
                    "nprobe",
                    &nprobes,
                    make_config,
                    make_label,
                    &corpus,
                    &queries,
                    &ground_truth,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    device_str,
                    &args.batch_sizes,
                    &mut w,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            }
        }
        SchemeArg::Tiptoe => {
            eprintln!(
                "TiptoeScorer: dim={dim}, n_centroids={n_centroids} (sqrt({})), tiptoe-text LWE+BFV params",
                corpus.len()
            );
            let scorer = TiptoeScorer::with_cache_dir(args.data_dir.as_path());
            let qbits: Vec<f64> = args.quantisation_bits.iter().map(|&q| q as f64).collect();
            let mp_c = mp.clone();
            let train_seed = ivf.train_seed;
            let make_config = move |&q: &f64| {
                let total = (n_centroids + max_iter) as u64;
                let pb = mp_c.add(ProgressBar::new(total));
                pb.set_style(
                    ProgressStyle::with_template(
                        "  build (kmeans+SimplePIR) [{bar:30}] {pos}/{len}  {elapsed_precise}",
                    )
                    .unwrap()
                    .progress_chars("=> "),
                );
                TiptoeConfig {
                    n_centroids,
                    train_seed,
                    max_iter,
                    quantisation_bits: q as u8,
                    progress: Some(Arc::new(IndicatifProgress::new(pb))),
                }
            };
            let make_label = |qp: f64| format!("quantisation-bits={}", qp as u8);
            if args.breakdown {
                let bw = breakdown_w
                    .as_mut()
                    .expect("breakdown writer present when --breakdown is set");
                run_eval_breakdown(
                    &scorer,
                    "tiptoe",
                    "quantisation-bits",
                    &qbits,
                    make_config,
                    make_label,
                    &corpus,
                    &queries,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    bw,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            } else {
                run_eval(
                    &scorer,
                    "tiptoe",
                    "quantisation-bits",
                    &qbits,
                    make_config,
                    make_label,
                    &corpus,
                    &queries,
                    &ground_truth,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    device_str,
                    &args.batch_sizes,
                    &mut w,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            }
        }
        SchemeArg::Bntm => {
            eprintln!(
                "BnTmScorer (flat): dim={dim}, n={}, n_1={}, δ={}, ε={}, μ={:.4}, λ'={}, verification={}",
                BnTmParams::Sec128.n(),
                BnTmParams::Sec128.n1(),
                BnTmParams::Sec128.delta(),
                BnTmParams::Sec128.epsilon(),
                BnTmParams::Sec128.mu(),
                BnTmParams::Sec128.verification_trials(),
                args.bntm_verification,
            );
            let scorer = BnTmScorer::new();
            let key_seed = [42u8; 32];
            let verification_enabled = args.bntm_verification;
            let quantisation_q: u64 = 1u64 << args.bntm_q_bits;
            let label_suffix = if verification_enabled { "on" } else { "off" };
            let mp_c = mp.clone();
            let make_config = move |_: &f64| {
                let pb = mp_c.add(ProgressBar::new(1));
                pb.set_style(
                    ProgressStyle::with_template(
                        "  encrypting cluster [{bar:30}] {pos}/{len}  {elapsed_precise}",
                    )
                    .unwrap()
                    .progress_chars("=> "),
                );
                BnTmConfig {
                    params: BnTmParams::Sec128,
                    key_seed,
                    verification_enabled,
                    quantisation_q,
                    progress: Some(Arc::new(IndicatifProgress::new(pb))),
                    device: args.device.into(),
                    vram_budget_bytes: gpu_vram_budget_bytes,
                }
            };
            let make_label = move |_: f64| format!("verification={label_suffix}");
            let qps = [verification_enabled as u64 as f64];
            if args.breakdown {
                let bw = breakdown_w
                    .as_mut()
                    .expect("breakdown writer present when --breakdown is set");
                run_eval_breakdown(
                    &scorer,
                    "bntm",
                    "verification",
                    &qps,
                    make_config,
                    make_label,
                    &corpus,
                    &queries,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    bw,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            } else if args.batch_sizes.as_slice() == [1] {
                // B=1 only: route through Breakdownable so
                // canonical[3] (verify_us) populates
                // verification_overhead_us per query.
                run_eval_with_verify_us(
                    &scorer,
                    "bntm",
                    "verification",
                    &qps,
                    make_config,
                    make_label,
                    &corpus,
                    &queries,
                    &ground_truth,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    device_str,
                    &mut w,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            } else {
                // User requested batched scoring. Route through
                // `run_eval` (no per-row verify_us — the verify-timing
                // figure needs a separate B=1 run). Batched throughput
                // takes priority for this dispatch.
                run_eval(
                    &scorer,
                    "bntm",
                    "verification",
                    &qps,
                    make_config,
                    make_label,
                    &corpus,
                    &queries,
                    &ground_truth,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    device_str,
                    &args.batch_sizes,
                    &mut w,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            }
        }
        SchemeArg::BntmIvf => {
            eprintln!(
                "BnTmIvfScorer: dim={dim}, n_centroids={n_centroids} (sqrt({})), n={}, verification={}",
                corpus.len(),
                BnTmParams::Sec128.n(),
                args.bntm_verification,
            );
            let scorer = BnTmIvfScorer::with_cache_dir(args.data_dir.as_path());
            let key_seed = [42u8; 32];
            let verification_enabled = args.bntm_verification;
            let quantisation_q: u64 = 1u64 << args.bntm_q_bits;
            let nprobes: Vec<f64> = args.nprobe.iter().map(|&n| n as f64).collect();
            let mp_c = mp.clone();
            let train_seed = ivf.train_seed;
            let make_config = move |&nprobe: &f64| {
                let total = (n_centroids + max_iter + n_centroids) as u64;
                let pb = mp_c.add(ProgressBar::new(total));
                pb.set_style(
                    ProgressStyle::with_template(
                        "  building index [{bar:30}] {pos}/{len}  {elapsed_precise}",
                    )
                    .unwrap()
                    .progress_chars("=> "),
                );
                BnTmIvfConfig {
                    params: BnTmParams::Sec128,
                    key_seed,
                    n_centroids,
                    nprobe: nprobe as usize,
                    train_seed,
                    max_iter,
                    upload_seed: 7,
                    verification_enabled,
                    quantisation_q,
                    progress: Some(Arc::new(IndicatifProgress::new(pb))),
                    device: args.device.into(),
                    vram_budget_bytes: gpu_vram_budget_bytes,
                }
            };
            let make_label = |qp: f64| format!("nprobe={}", qp as usize);
            if args.breakdown {
                let bw = breakdown_w
                    .as_mut()
                    .expect("breakdown writer present when --breakdown is set");
                run_eval_breakdown(
                    &scorer,
                    "bntm-ivf",
                    "nprobe",
                    &nprobes,
                    make_config,
                    make_label,
                    &corpus,
                    &queries,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    bw,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            } else if args.batch_sizes.as_slice() == [1] {
                // B=1 only: route through Breakdownable so
                // canonical[3] (verify_us) populates
                // verification_overhead_us per query.
                run_eval_with_verify_us(
                    &scorer,
                    "bntm-ivf",
                    "nprobe",
                    &nprobes,
                    make_config,
                    make_label,
                    &corpus,
                    &queries,
                    &ground_truth,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    device_str,
                    &mut w,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            } else {
                // Batched scoring requested. Route through `run_eval`
                // (no per-row verify_us — the verify-timing figure
                // needs a separate B=1 run).
                run_eval(
                    &scorer,
                    "bntm-ivf",
                    "nprobe",
                    &nprobes,
                    make_config,
                    make_label,
                    &corpus,
                    &queries,
                    &ground_truth,
                    args.k,
                    args.repetitions,
                    &run_id,
                    &machine.id,
                    device_str,
                    &args.batch_sizes,
                    &mut w,
                    &mut top_k_w,
                    &mp,
                )
                .await?
            }
        }
    };

    w.flush()?;
    top_k_w.flush()?;
    if let Some(bw) = breakdown_w.as_mut() {
        bw.flush()?;
    }

    // Populate the [index] block from the first BuildOutcome and the
    // scheme's known cluster shape. cluster-count is 1 for flat
    // scorers, n_centroids for IVF; m-total is the corpus size (sum
    // across all clusters).
    let cluster_count = match args.scorer {
        SchemeArg::Sap | SchemeArg::Bntm | SchemeArg::Emvp => 1,
        SchemeArg::Plaintext
        | SchemeArg::SapIvf
        | SchemeArg::EmvpIvf
        | SchemeArg::Tiptoe
        | SchemeArg::BntmIvf => n_centroids,
    };
    run_meta.index = Some(IndexBlock {
        cache_hit: build_outcome.cache_hit,
        build_duration_secs: build_outcome.build_duration.as_secs_f64(),
        cluster_count,
        m_total: corpus.len(),
    });

    // Finalize run-metadata.toml atomically.
    let finished_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    run_meta.finished_at = Some(meta::unix_secs_to_iso8601(finished_secs));
    run_meta.duration_secs = Some(finished_secs.saturating_sub(started_secs));
    run_meta.status = "complete".into();
    run_meta.peak_rss_bytes = meta::capture_peak_rss_bytes();

    // Record the resolved streaming-store budget and the NVML
    // high-water mark on the `[gpu]` block. Both stay `None` on CPU
    // runs / unavailable NVML, keeping the fields absent in the TOML
    // on those paths.
    #[cfg(feature = "gpu")]
    if let Some(ref mut g) = run_meta.gpu {
        g.vram_budget_bytes = gpu_vram_budget_bytes;
        g.peak_vram_bytes = gpu_peak_sampler.as_ref().map(|s| s.peak_bytes());
    }
    // Stopping the sampler explicitly: the Drop runs at end-of-fn,
    // but reading `peak_bytes` before that is fine — the sampler
    // updates the atomic on a 50 ms cadence and we already passed
    // the score loop. Drop here makes the lifetime explicit.
    #[cfg(feature = "gpu")]
    drop(gpu_peak_sampler);

    // [memory] block: CPU page-fault delta over the scoring loop +
    // GPU ClusterStore aggregates. Both fields skip-on-None so a
    // `--device cpu` run on a non-Linux host emits no block at all.
    let cpu_fault_delta = meta::capture_proc_faults()
        .zip(memory_baseline_faults)
        .map(|(end, base)| end.delta(base));
    #[cfg(feature = "gpu")]
    let gpu_store_stats = if args.device == DeviceArg::Gpu {
        Some(scorer_gpu_common::global_stats())
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    let gpu_store_stats: Option<()> = None;

    let memory_block = MemoryBlock {
        cpu_minor_faults: cpu_fault_delta.map(|d| d.minor),
        cpu_major_faults: cpu_fault_delta.map(|d| d.major),
        #[cfg(feature = "gpu")]
        gpu_get_count: gpu_store_stats.as_ref().map(|s| s.gets),
        #[cfg(feature = "gpu")]
        gpu_upload_count: gpu_store_stats.as_ref().map(|s| s.uploads),
        #[cfg(feature = "gpu")]
        gpu_eviction_count: gpu_store_stats.as_ref().map(|s| s.evictions),
        #[cfg(feature = "gpu")]
        gpu_bytes_uploaded_total: gpu_store_stats.as_ref().map(|s| s.bytes_uploaded),
        #[cfg(not(feature = "gpu"))]
        gpu_get_count: None,
        #[cfg(not(feature = "gpu"))]
        gpu_upload_count: None,
        #[cfg(not(feature = "gpu"))]
        gpu_eviction_count: None,
        #[cfg(not(feature = "gpu"))]
        gpu_bytes_uploaded_total: None,
    };
    let any_populated = memory_block.cpu_minor_faults.is_some()
        || memory_block.cpu_major_faults.is_some()
        || memory_block.gpu_get_count.is_some();
    run_meta.memory = if any_populated {
        Some(memory_block)
    } else {
        None
    };
    let _ = gpu_store_stats; // silence unused warning on cpu-only builds

    write_metadata_atomic(&toml_path, &run_meta)?;

    // Append to index.csv.
    let index_path = results_dir.join("index.csv");
    let rel_path = run_dir
        .strip_prefix(results_dir)
        .unwrap_or(&run_dir)
        .display()
        .to_string();
    append_index(
        &index_path,
        &run_id,
        &machine.id,
        &git.sha,
        git.dirty,
        scheme_name,
        &args.data_dir.display().to_string(),
        &started_at,
        run_meta.duration_secs.unwrap_or(0),
        "complete",
        &rel_path,
    )?;

    eprintln!("Results written to {}", run_dir.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// run_eval
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn run_eval<S, F, L>(
    scorer: &S,
    scheme: &str,
    quality_param_name: &str,
    quality_params: &[f64],
    make_config: F,
    make_label: L,
    corpus: &[Vector],
    queries: &[Vector],
    ground_truth: &[Vec<u32>],
    k: usize,
    repetitions: usize,
    run_id: &str,
    machine_id: &str,
    device: &str,
    batch_sizes: &[usize],
    w: &mut impl Write,
    top_k_w: &mut impl Write,
    mp: &MultiProgress,
) -> anyhow::Result<BuildOutcome>
where
    S: Scorer,
    S::Error: std::fmt::Display,
    F: Fn(&f64) -> S::Config,
    L: Fn(f64) -> String,
{
    let param_style = ProgressStyle::with_template(
        "{prefix} [{bar:40}] {pos}/{len} configs  elapsed {elapsed_precise}  {msg}",
    )
    .unwrap()
    .progress_chars("=> ");
    let query_style = ProgressStyle::with_template(
        "  {prefix} [{bar:40}] {pos}/{len} queries  {per_sec}  ETA {eta}",
    )
    .unwrap()
    .progress_chars("=> ");

    let param_pb = mp.add(ProgressBar::new(quality_params.len() as u64));
    param_pb.set_style(param_style);
    param_pb.set_prefix(format!("{scheme} {quality_param_name} sweep"));

    if quality_params.is_empty() {
        anyhow::bail!("scheme {scheme} produced an empty quality_params sweep");
    }
    if batch_sizes.is_empty() {
        anyhow::bail!("--batch-sizes must contain at least one value");
    }
    if batch_sizes.contains(&0) {
        anyhow::bail!("--batch-sizes values must be > 0; got {batch_sizes:?}");
    }

    let mut first_build: Option<BuildOutcome> = None;
    for &qp in quality_params {
        param_pb.set_message(format!("{quality_param_name}={qp}: building index…"));
        let config = make_config(&qp);
        let label = make_label(qp);
        let (handle, build) = scorer
            .upload_cluster(&config, corpus)
            .await
            .map_err(|e| anyhow::anyhow!("upload_cluster failed: {e}"))?;
        if first_build.is_none() {
            first_build = Some(build);
        }

        // Nested sweep over batch sizes × reps. The handle is built
        // once per qp and reused — the batch sweep is orthogonal to
        // the per-cluster work the upload paid for. top_k.csv is
        // written on the (rep == 0, batch_size == 1) path only — the
        // B>1 throughput rows don't need per-query top-k (the
        // tiptoe_diff validation gate runs separately, B=1).
        for &big_b in batch_sizes {
            for rep in 0..repetitions {
                if big_b == 1 {
                    // B=1: per-query path. Realised cost via
                    // score_with_realised_cost. latency_us populated;
                    // wallclock/amortised collapse to it.
                    let query_pb = mp.add(ProgressBar::new(queries.len() as u64));
                    query_pb.set_style(query_style.clone());
                    query_pb.set_prefix(format!(
                        "{quality_param_name}={qp} B=1 rep {}/{}",
                        rep + 1,
                        repetitions
                    ));

                    for (qi, query) in queries.iter().enumerate() {
                        let t0 = Instant::now();
                        let (hits, cost) = scorer
                            .score_with_realised_cost(&handle, query, k)
                            .await
                            .map_err(|e| anyhow::anyhow!("score_with_realised_cost failed: {e}"))?;
                        let latency_us = t0.elapsed().as_micros() as u64;

                        let returned: Vec<u32> = hits.iter().map(|h| h.id).collect();
                        let r = recall_at_k(
                            ground_truth.get(qi).map(Vec::as_slice).unwrap_or(&[]),
                            &returned,
                            k,
                        );

                        write_csv_row(
                            w,
                            &CsvRow {
                                run_id: run_id.to_string(),
                                scheme: scheme.to_string(),
                                quality_param_name: quality_param_name.to_string(),
                                quality_param: qp,
                                config_label: label.clone(),
                                k: k as u32,
                                query_id: qi as u32,
                                recall_at_k: r,
                                latency_us: Some(latency_us),
                                query_bytes: cost.query_bytes,
                                response_bytes: cost.response_bytes,
                                cluster_response_bytes: cost.cluster_response_bytes,
                                setup_bytes: cost.setup_bytes,
                                pre_query_offline_up_bytes: cost.pre_query_offline_up_bytes,
                                pre_query_offline_down_bytes: cost.pre_query_offline_down_bytes,
                                verification_overhead_us: 0,
                                machine_id: machine_id.to_string(),
                                device: device.to_string(),
                                effective_bytes_per_query: cost.effective_bytes_per_query,
                                batch_size: 1,
                                wallclock_us: latency_us,
                                amortised_latency_us: latency_us,
                            },
                        )?;

                        if rep == 0 {
                            for (rank, hit) in hits.iter().enumerate() {
                                write_top_k_row(
                                    top_k_w,
                                    &TopKRow {
                                        config_label: &label,
                                        query_id: qi as u32,
                                        rank: rank as u32,
                                        doc_id: hit.id,
                                        score: hit.score,
                                    },
                                )?;
                            }
                        }
                        query_pb.inc(1);
                    }
                    query_pb.finish_and_clear();
                } else {
                    // B>1: chunk queries into groups of exactly B
                    // (final partial chunk dropped to keep per-row cost
                    // uniform). score_batch produces B Vec<Hit>s per
                    // chunk; the harness writes B raw.csv rows per
                    // chunk with latency_us = None (the per-query
                    // latency inside a batch isn't separately
                    // observable) and wallclock_us /
                    // amortised_latency_us populated.
                    //
                    // Cost accounting: communication_cost(&handle, k)
                    // analytical only — score_batch has no realised
                    // cost analog. For IVF scorers this means
                    // cluster_response_bytes reflects the *handle mean*
                    // probe set, not the realised probe set; figure 14
                    // ignores cost columns and reads latency/throughput,
                    // so this doesn't affect the figure.
                    let chunks = queries.chunks_exact(big_b);
                    let n_chunks = chunks.len();
                    let chunk_pb = mp.add(ProgressBar::new(n_chunks as u64));
                    chunk_pb.set_style(query_style.clone());
                    chunk_pb.set_prefix(format!(
                        "{quality_param_name}={qp} B={big_b} rep {}/{}",
                        rep + 1,
                        repetitions
                    ));

                    let cost = scorer.communication_cost(&handle, k);

                    for (chunk_idx, chunk) in chunks.enumerate() {
                        let t0 = Instant::now();
                        let hits_batch = scorer
                            .score_batch(&handle, chunk, k)
                            .await
                            .map_err(|e| anyhow::anyhow!("score_batch failed: {e}"))?;
                        let wallclock_us = t0.elapsed().as_micros() as u64;
                        let amortised_latency_us = wallclock_us / big_b as u64;

                        for (local_i, hits) in hits_batch.iter().enumerate() {
                            let qi = chunk_idx * big_b + local_i;
                            let returned: Vec<u32> = hits.iter().map(|h| h.id).collect();
                            let r = recall_at_k(
                                ground_truth.get(qi).map(Vec::as_slice).unwrap_or(&[]),
                                &returned,
                                k,
                            );

                            write_csv_row(
                                w,
                                &CsvRow {
                                    run_id: run_id.to_string(),
                                    scheme: scheme.to_string(),
                                    quality_param_name: quality_param_name.to_string(),
                                    quality_param: qp,
                                    config_label: label.clone(),
                                    k: k as u32,
                                    query_id: qi as u32,
                                    recall_at_k: r,
                                    latency_us: None,
                                    query_bytes: cost.query_bytes,
                                    response_bytes: cost.response_bytes,
                                    cluster_response_bytes: cost.cluster_response_bytes,
                                    setup_bytes: cost.setup_bytes,
                                    pre_query_offline_up_bytes: cost.pre_query_offline_up_bytes,
                                    pre_query_offline_down_bytes: cost.pre_query_offline_down_bytes,
                                    verification_overhead_us: 0,
                                    machine_id: machine_id.to_string(),
                                    device: device.to_string(),
                                    effective_bytes_per_query: cost.effective_bytes_per_query,
                                    batch_size: big_b as u32,
                                    wallclock_us,
                                    amortised_latency_us,
                                },
                            )?;
                        }
                        chunk_pb.inc(1);
                    }
                    chunk_pb.finish_and_clear();
                }
            }
        }
        param_pb.inc(1);
    }
    param_pb.finish_with_message("done");
    // SAFETY: quality_params.is_empty() is rejected above, and the
    // first iteration unconditionally populates first_build.
    Ok(first_build.expect("at least one upload_cluster call occurred"))
}

// `run_eval` variant that surfaces per-query
// `verification-overhead-us` for schemes that have a verification
// substep (today only BN). Routes through `Breakdownable::score_canonical`
// instead of `Scorer::score` so the existing per-impl
// `score_with_breakdown` inherent method is the single source of
// truth for verify timing — no parallel "inherent score_with_verify_us"
// API on each BN scorer (measurement showed breakdown adds ≪ 2% over
// score on BN).
//
// Non-BN scorers' `Breakdownable` impl maps 0 into canonical[3], so
// callers that route a non-BN scorer here would also write 0 to the
// CSV column — preserving the uniform-row invariant preprocess.py
// relies on. Today only the BN dispatch arms call this path.
#[allow(clippy::too_many_arguments)]
async fn run_eval_with_verify_us<S, F, L>(
    scorer: &S,
    scheme: &str,
    quality_param_name: &str,
    quality_params: &[f64],
    make_config: F,
    make_label: L,
    corpus: &[Vector],
    queries: &[Vector],
    ground_truth: &[Vec<u32>],
    k: usize,
    repetitions: usize,
    run_id: &str,
    machine_id: &str,
    device: &str,
    w: &mut impl Write,
    top_k_w: &mut impl Write,
    mp: &MultiProgress,
) -> anyhow::Result<BuildOutcome>
where
    S: Breakdownable,
    S::Error: std::fmt::Display,
    F: Fn(&f64) -> S::Config,
    L: Fn(f64) -> String,
{
    let param_style = ProgressStyle::with_template(
        "{prefix} [{bar:40}] {pos}/{len} configs  elapsed {elapsed_precise}  {msg}",
    )
    .unwrap()
    .progress_chars("=> ");
    let query_style = ProgressStyle::with_template(
        "  {prefix} [{bar:40}] {pos}/{len} queries  {per_sec}  ETA {eta}",
    )
    .unwrap()
    .progress_chars("=> ");

    let param_pb = mp.add(ProgressBar::new(quality_params.len() as u64));
    param_pb.set_style(param_style);
    param_pb.set_prefix(format!("{scheme} {quality_param_name} sweep"));

    if quality_params.is_empty() {
        anyhow::bail!("scheme {scheme} produced an empty quality_params sweep");
    }

    let mut first_build: Option<BuildOutcome> = None;
    for &qp in quality_params {
        param_pb.set_message(format!("{quality_param_name}={qp}: building index…"));
        let config = make_config(&qp);
        let label = make_label(qp);
        let (handle, build) = scorer
            .upload_cluster(&config, corpus)
            .await
            .map_err(|e| anyhow::anyhow!("upload_cluster failed: {e}"))?;
        if first_build.is_none() {
            first_build = Some(build);
        }

        for rep in 0..repetitions {
            let query_pb = mp.add(ProgressBar::new(queries.len() as u64));
            query_pb.set_style(query_style.clone());
            query_pb.set_prefix(format!(
                "{quality_param_name}={qp} rep {}/{}",
                rep + 1,
                repetitions
            ));

            for (qi, query) in queries.iter().enumerate() {
                let t0 = Instant::now();
                let (hits, canonical) = scorer
                    .score_canonical(&handle, query, k)
                    .await
                    .map_err(|e| anyhow::anyhow!("score_canonical failed: {e}"))?;
                let latency_us = t0.elapsed().as_micros() as u64;
                // canonical = [route, encode, server-compute, verify,
                // decompress, decode, merge]. Slot 3 is verify, which
                // is non-zero only for BN with verification_enabled.
                let verification_overhead_us = canonical[3];

                // Realised cost via an untimed second call.
                // The breakdown path doesn't expose the probe set, so
                // we can't reuse routing across the two calls — this
                // doubles per-query work on the BN sweep, traded for
                // honest per-query response_bytes in raw.csv. Future
                // work: thread the probe set through `score_canonical`
                // so one call covers both.
                let (_, cost) = scorer
                    .score_with_realised_cost(&handle, query, k)
                    .await
                    .map_err(|e| anyhow::anyhow!("score_with_realised_cost failed: {e}"))?;

                let returned: Vec<u32> = hits.iter().map(|h| h.id).collect();
                let r = recall_at_k(
                    ground_truth.get(qi).map(Vec::as_slice).unwrap_or(&[]),
                    &returned,
                    k,
                );

                write_csv_row(
                    w,
                    &CsvRow {
                        run_id: run_id.to_string(),
                        scheme: scheme.to_string(),
                        quality_param_name: quality_param_name.to_string(),
                        quality_param: qp,
                        config_label: label.clone(),
                        k: k as u32,
                        query_id: qi as u32,
                        recall_at_k: r,
                        latency_us: Some(latency_us),
                        query_bytes: cost.query_bytes,
                        response_bytes: cost.response_bytes,
                        cluster_response_bytes: cost.cluster_response_bytes,
                        setup_bytes: cost.setup_bytes,
                        pre_query_offline_up_bytes: cost.pre_query_offline_up_bytes,
                        pre_query_offline_down_bytes: cost.pre_query_offline_down_bytes,
                        verification_overhead_us,
                        machine_id: machine_id.to_string(),
                        device: device.to_string(),
                        effective_bytes_per_query: cost.effective_bytes_per_query,
                        // run_eval_with_verify_us is the BN-only B=1
                        // per-query path (per-query verification timing
                        // is what motivates this function); batched BN
                        // goes through the separate batched producer
                        // added with --batch-sizes.
                        batch_size: 1,
                        wallclock_us: latency_us,
                        amortised_latency_us: latency_us,
                    },
                )?;

                if rep == 0 {
                    for (rank, hit) in hits.iter().enumerate() {
                        write_top_k_row(
                            top_k_w,
                            &TopKRow {
                                config_label: &label,
                                query_id: qi as u32,
                                rank: rank as u32,
                                doc_id: hit.id,
                                score: hit.score,
                            },
                        )?;
                    }
                }

                query_pb.inc(1);
            }
            query_pb.finish_and_clear();
        }
        param_pb.inc(1);
    }
    param_pb.finish_with_message("done");
    Ok(first_build.expect("at least one upload_cluster call occurred"))
}

// ---------------------------------------------------------------------------
// Breakdownable trait + run_eval_breakdown
// ---------------------------------------------------------------------------
//
// `Breakdownable` is a private trait that adapts each scorer's
// inherent `score_with_breakdown` method into the canonical 7-substep
// taxonomy (`route, encode, server-compute, verify, decompress,
// decode, merge`) that figure 09 stacks. Per-impl `*Timing` structs
// have scheme-specific field sets; the impl below maps them onto
// `[u64; 7]` so `run_eval_breakdown` can stay generic over `S`.
//
// The trait is local to the eval binary — no public surface, no
// addition to scorer-core's `Scorer`.

#[async_trait]
trait Breakdownable: Scorer {
    async fn score_canonical(
        &self,
        handle: &Self::ClusterHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, [u64; 7]), Self::Error>;
}

// Each impl only differs in its scorer/handle/error types and how the
// scheme's `*Timing` fields map onto the canonical 7-slot array
// [route, encode, server-compute, verify, decompress, decode, merge].
// The macro captures exactly that variation so a new scorer is one
// line, and the mapping stays visible at each call site.
macro_rules! impl_breakdownable {
    ($scorer:ty, $handle:ty, $err:ty, |$t:ident| $canonical:expr $(,)?) => {
        #[async_trait]
        impl Breakdownable for $scorer {
            async fn score_canonical(
                &self,
                handle: &$handle,
                query: &Vector,
                k: usize,
            ) -> Result<(Vec<Hit>, [u64; 7]), $err> {
                let (hits, $t) = self.score_with_breakdown(handle, query, k).await?;
                Ok((hits, $canonical))
            }
        }
    };
}

impl_breakdownable!(PlaintextScorer, PlaintextHandle, PlaintextError, |t| [
    t.route_us,
    0,
    t.distance_us,
    0,
    0,
    0,
    t.merge_us
]);

// Flat: no routing phase; top-k folds into the decode slot.
impl_breakdownable!(SapScorer, SapClusterHandle, SapError, |t| [
    0,
    t.encode_us,
    t.server_us,
    0,
    0,
    t.merge_us,
    0
]);

impl_breakdownable!(SapIvfScorer, SapIvfHandle, SapIvfError, |t| [
    t.route_us,
    t.encode_us,
    t.distance_us,
    0,
    0,
    0,
    t.merge_us
]);

// Flat: no routing phase; the impl folds top-k into decode_us.
impl_breakdownable!(EmvpScorer, EmvpHandle, EmvpError, |t| [
    0,
    t.encode_us,
    t.server_us,
    0,
    0,
    t.decode_us,
    0
]);

impl_breakdownable!(EmvpIvfScorer, EmvpIvfHandle, EmvpIvfError, |t| [
    t.route_us,
    t.encode_us,
    t.server_us,
    0,
    0,
    t.decode_us,
    t.merge_us
]);

impl_breakdownable!(BnTmScorer, BnTmHandle, BnTmError, |t| [
    0,
    t.encode_us,
    t.server_us,
    t.verify_us,
    0,
    t.decode_us,
    0
]);

impl_breakdownable!(BnTmIvfScorer, BnTmIvfHandle, BnTmIvfError, |t| [
    t.route_us,
    t.encode_us,
    t.server_us,
    t.verify_us,
    0,
    t.decode_us,
    t.merge_us
]);

// Tiptoe is single-cluster: no merge. `bfv_decompress` slots into the
// decompress phase.
impl_breakdownable!(TiptoeScorer, TiptoeHandle, TiptoeError, |t| [
    t.route_us,
    t.lwe_encrypt_us,
    t.server_us,
    0,
    t.bfv_decompress_us,
    t.decode_us,
    0
]);

#[allow(clippy::too_many_arguments)]
async fn run_eval_breakdown<S, F, L>(
    scorer: &S,
    scheme: &str,
    quality_param_name: &str,
    quality_params: &[f64],
    make_config: F,
    make_label: L,
    corpus: &[Vector],
    queries: &[Vector],
    k: usize,
    repetitions: usize,
    run_id: &str,
    machine_id: &str,
    breakdown_w: &mut impl Write,
    top_k_w: &mut impl Write,
    mp: &MultiProgress,
) -> anyhow::Result<BuildOutcome>
where
    S: Breakdownable,
    S::Error: std::fmt::Display,
    F: Fn(&f64) -> S::Config,
    L: Fn(f64) -> String,
{
    let param_style = ProgressStyle::with_template(
        "{prefix} [{bar:40}] {pos}/{len} configs  elapsed {elapsed_precise}  {msg}",
    )
    .unwrap()
    .progress_chars("=> ");
    let query_style = ProgressStyle::with_template(
        "  {prefix} [{bar:40}] {pos}/{len} queries  {per_sec}  ETA {eta}",
    )
    .unwrap()
    .progress_chars("=> ");

    let param_pb = mp.add(ProgressBar::new(quality_params.len() as u64));
    param_pb.set_style(param_style);
    param_pb.set_prefix(format!("{scheme} {quality_param_name} sweep (breakdown)"));

    if quality_params.is_empty() {
        anyhow::bail!("scheme {scheme} produced an empty quality_params sweep");
    }

    let mut first_build: Option<BuildOutcome> = None;
    for &qp in quality_params {
        param_pb.set_message(format!("{quality_param_name}={qp}: building index…"));
        let config = make_config(&qp);
        let label = make_label(qp);
        let (handle, build) = scorer
            .upload_cluster(&config, corpus)
            .await
            .map_err(|e| anyhow::anyhow!("upload_cluster failed: {e}"))?;
        if first_build.is_none() {
            first_build = Some(build);
        }

        for rep in 0..repetitions {
            let query_pb = mp.add(ProgressBar::new(queries.len() as u64));
            query_pb.set_style(query_style.clone());
            query_pb.set_prefix(format!(
                "{quality_param_name}={qp} rep {}/{} (breakdown)",
                rep + 1,
                repetitions
            ));

            for (qi, query) in queries.iter().enumerate() {
                let (hits, canonical) = scorer
                    .score_canonical(&handle, query, k)
                    .await
                    .map_err(|e| anyhow::anyhow!("score_with_breakdown failed: {e}"))?;

                // Substep rows are emitted on rep=0 only — substep
                // variance across reps is out of scope, matching
                // top_k.csv's per-query semantics.
                if rep == 0 {
                    for (substep, &us) in CANONICAL_SUBSTEPS.iter().zip(canonical.iter()) {
                        write_substep_breakdown_row(
                            breakdown_w,
                            &SubstepRow {
                                run_id,
                                scheme,
                                quality_param_name,
                                quality_param: qp,
                                config_label: &label,
                                query_id: qi as u32,
                                substep,
                                us,
                                machine_id,
                            },
                        )?;
                    }
                    for (rank, hit) in hits.iter().enumerate() {
                        write_top_k_row(
                            top_k_w,
                            &TopKRow {
                                config_label: &label,
                                query_id: qi as u32,
                                rank: rank as u32,
                                doc_id: hit.id,
                                score: hit.score,
                            },
                        )?;
                    }
                }

                query_pb.inc(1);
            }
            query_pb.finish_and_clear();
        }
        param_pb.inc(1);
    }
    param_pb.finish_with_message("done");
    Ok(first_build.expect("at least one upload_cluster call occurred"))
}
