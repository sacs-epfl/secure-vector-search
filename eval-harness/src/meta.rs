use serde::Serialize;
use sysinfo::System;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Campaign
// ---------------------------------------------------------------------------

/// `[campaign]` block. Groups runs by intent (planned sweep,
/// figure-anchoring batch, exploratory). `id` and `title` are
/// required; `note` is optional. Absent block means the run is a
/// singleton by downstream tooling.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub struct Campaign {
    pub id: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Campaign {
    /// Construct from optional flag/env values:
    ///
    /// - All three absent (or empty after trim) → `Ok(None)`.
    /// - `id` AND `title` present → `Ok(Some(Campaign{...}))`.
    /// - Any other combination → `Err(CampaignError::...)`.
    ///
    /// Empty / whitespace-only strings are treated as absent so an
    /// unset env var (which clap delivers as `Some("".into())`) is
    /// equivalent to no flag.
    pub fn try_new(
        id: Option<String>,
        title: Option<String>,
        note: Option<String>,
    ) -> Result<Option<Self>, CampaignError> {
        let id = id.and_then(|s| {
            let t = s.trim();
            (!t.is_empty()).then(|| t.to_string())
        });
        let title = title.and_then(|s| {
            let t = s.trim();
            (!t.is_empty()).then(|| t.to_string())
        });
        let note = note.and_then(|s| {
            let t = s.trim();
            (!t.is_empty()).then(|| t.to_string())
        });
        match (id, title, note) {
            (None, None, None) => Ok(None),
            (Some(id), Some(title), note) => {
                validate_campaign_id(&id)?;
                Ok(Some(Self { id, title, note }))
            }
            (Some(_), None, _) => Err(CampaignError::TitleMissing),
            (None, Some(_), _) => Err(CampaignError::IdMissing),
            (None, None, Some(_)) => Err(CampaignError::NoteWithoutIdTitle),
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CampaignError {
    #[error(
        "campaign id is required when title or note is set; pass --campaign-id or set CAMPAIGN_ID"
    )]
    IdMissing,
    #[error(
        "campaign title is required when id is set; pass --campaign-title or set CAMPAIGN_TITLE"
    )]
    TitleMissing,
    #[error(
        "campaign note requires both id and title; pass --campaign-id and --campaign-title (or unset the note)"
    )]
    NoteWithoutIdTitle,
    #[error("campaign id length {0} exceeds 128 chars")]
    IdTooLong(usize),
    #[error("campaign id may not contain whitespace")]
    IdHasWhitespace,
    #[error("campaign id may not contain path separators (`/` or `\\`)")]
    IdHasPathSeparator,
    #[error("campaign id may not contain quotes (`'` or `\"`)")]
    IdHasQuote,
    #[error("campaign id may not contain control characters")]
    IdHasControlChar,
    #[error(
        "campaign id may not contain `{0}`; allowed: ASCII alphanumerics plus `-`, `_`, `.`, `:`"
    )]
    IdHasInvalidChar(char),
}

fn validate_campaign_id(id: &str) -> Result<(), CampaignError> {
    if id.len() > 128 {
        return Err(CampaignError::IdTooLong(id.len()));
    }
    for ch in id.chars() {
        if ch.is_whitespace() {
            return Err(CampaignError::IdHasWhitespace);
        }
        if ch == '/' || ch == '\\' {
            return Err(CampaignError::IdHasPathSeparator);
        }
        if ch == '\'' || ch == '"' {
            return Err(CampaignError::IdHasQuote);
        }
        if ch.is_control() {
            return Err(CampaignError::IdHasControlChar);
        }
        let allowed = ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | ':');
        if !allowed {
            return Err(CampaignError::IdHasInvalidChar(ch));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_cmd(prog: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(prog)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

/// Convert Unix timestamp (seconds) to ISO 8601 UTC string without external deps.
pub fn unix_secs_to_iso8601(secs: u64) -> String {
    let secs_of_day = secs % 86400;
    let days = secs / 86400;
    let h = secs_of_day / 3600;
    let m = (secs_of_day % 3600) / 60;
    let s = secs_of_day % 60;
    // Gregorian calendar from days since Unix epoch, via civil_from_days algorithm.
    let z = days as i64 + 719468;
    let era = if z >= 0 {
        z / 146097
    } else {
        (z - 146096) / 146097
    };
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

// ---------------------------------------------------------------------------
// Git
// ---------------------------------------------------------------------------

pub struct GitState {
    pub sha: String,
    pub dirty: bool,
    pub branch: String,
}

pub fn collect_git_state() -> GitState {
    let sha = run_cmd("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    // dirty iff porcelain output is non-empty; treat git failure as clean
    let dirty = run_cmd("git", &["status", "--porcelain"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let branch = run_cmd("git", &["rev-parse", "--abbrev-ref", "HEAD"])
        .unwrap_or_else(|| "unknown".to_string());
    GitState { sha, dirty, branch }
}

// ---------------------------------------------------------------------------
// Machine
// ---------------------------------------------------------------------------

pub struct MachineInfo {
    /// 8-char FNV-1a fingerprint of "{cpu_model}\n{cores}\n{ram_bytes}".
    pub id: String,
    /// Hostname as reported by the OS; used as the default `machine-name` in
    /// `results/machines.csv` (editable by hand afterwards).
    pub hostname: String,
    pub cpu_model: String,
    /// Logical core count.
    pub cores: usize,
    /// Total RAM in bytes.
    pub ram_bytes: u64,
    /// "{os_type} {os_version}", e.g. "Linux Ubuntu 24.04" or "Mac OS X 15.4".
    pub os: String,
    /// Output of `uname -r`; "unknown" on non-Linux.
    pub kernel_version: String,
    /// Contents of /sys/.../scaling_governor; "unknown" on non-Linux or missing.
    pub cpu_governor: String,
    pub gpu_kind: String,
}

pub fn collect_machine_info(gpu_kind: &str) -> MachineInfo {
    let mut sys = System::new();
    sys.refresh_cpu_all();
    sys.refresh_memory();

    let cpu_model = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_default();
    let cores = sys.cpus().len();
    let ram_bytes = sys.total_memory();

    let os_data = os_info::get();
    let os = format!("{} {}", os_data.os_type(), os_data.version());

    #[cfg(target_os = "linux")]
    let kernel_version = run_cmd("uname", &["-r"]).unwrap_or_else(|| "unknown".to_string());
    #[cfg(not(target_os = "linux"))]
    let kernel_version = "unknown".to_string();

    #[cfg(target_os = "linux")]
    let cpu_governor =
        std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());
    #[cfg(not(target_os = "linux"))]
    let cpu_governor = "unknown".to_string();

    // FNV-1a 64-bit hash of "{cpu_model}\n{cores}\n{ram_bytes}" → first 8 hex chars.
    let key = format!("{cpu_model}\n{cores}\n{ram_bytes}");
    let mut h: u64 = 14695981039346656037;
    for b in key.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    let id = format!("{h:016x}")[..8].to_string();

    let hostname = System::host_name().unwrap_or_else(|| "unknown".to_string());

    MachineInfo {
        id,
        hostname,
        cpu_model,
        cores,
        ram_bytes,
        os,
        kernel_version,
        cpu_governor,
        gpu_kind: gpu_kind.to_string(),
    }
}

pub fn collect_rust_toolchain() -> String {
    run_cmd("rustc", &["--version"]).unwrap_or_else(|| {
        std::env::var("RUSTUP_TOOLCHAIN").unwrap_or_else(|_| "unknown".to_string())
    })
}

/// Record which compile-time `target_feature` gates the eval binary
/// was built with. The SIMD paths in `scorer-bntm` (BN AVX-512
/// matvec) and `ivf-index` / `scorer-sap` (L2 sum-of-squares) gate on
/// `target_feature = "avx512f"`; this lets the SIMD code be silently
/// disabled at compile time without leaving a trace in `raw.csv`.
/// Recording the active features in `run-metadata.toml` makes
/// scalar-vs-SIMD runs distinguishable post-hoc.
///
/// We probe a curated, hand-picked list of features that gate
/// project-internal SIMD paths or that materially affect LLVM
/// auto-vectorisation. The list is not exhaustive — adding a new
/// `cfg(target_feature = "X")` site to the workspace should be
/// accompanied by a check here so the field still reflects reality.
/// Output is sorted alphabetically for stable diffs across runs.
pub fn collect_target_features() -> Vec<String> {
    // Each cfg!() is a compile-time constant; the dead arms are
    // eliminated by codegen. Using cfg!() in an array literal (rather
    // than #[cfg]-gated `.push()` calls) keeps clippy's
    // `vec_init_then_push` lint quiet — same semantics, one
    // expression.
    let probes: [(bool, &'static str); 14] = [
        // x86_64 SIMD families.
        (cfg!(target_feature = "sse2"), "sse2"),
        (cfg!(target_feature = "sse4.2"), "sse4.2"),
        (cfg!(target_feature = "avx"), "avx"),
        (cfg!(target_feature = "avx2"), "avx2"),
        (cfg!(target_feature = "fma"), "fma"),
        (cfg!(target_feature = "bmi1"), "bmi1"),
        (cfg!(target_feature = "bmi2"), "bmi2"),
        (cfg!(target_feature = "avx512f"), "avx512f"),
        (cfg!(target_feature = "avx512dq"), "avx512dq"),
        (cfg!(target_feature = "avx512bw"), "avx512bw"),
        (cfg!(target_feature = "avx512vl"), "avx512vl"),
        // aarch64 SIMD.
        (cfg!(target_feature = "neon"), "neon"),
        (cfg!(target_feature = "sve"), "sve"),
        (cfg!(target_feature = "sve2"), "sve2"),
    ];
    let mut features: Vec<&'static str> = probes
        .iter()
        .filter_map(|(on, name)| on.then_some(*name))
        .collect();
    features.sort_unstable();
    features.into_iter().map(String::from).collect()
}

// ---------------------------------------------------------------------------
// Parallelism / NUMA binding
// ---------------------------------------------------------------------------

/// Realised rayon pool size for this process. Trips the lazy
/// initialiser once (so the answer reflects the pool that's actually
/// going to run, not the as-yet-unbuilt configured size) before
/// reading `rayon::current_num_threads()`.
pub fn capture_parallel_threads() -> usize {
    use rayon::prelude::*;
    (0..1).into_par_iter().for_each(|_| {});
    rayon::current_num_threads()
}

/// Realised Go pool size. The Go-runner driver inherits `GOMAXPROCS`
/// from its parent shell (set by the `eval-scaling-tiptoe` Makefile
/// loop) and forwards it to the child via the default env inheritance.
/// We compute the same value the Go runtime would: the env var if it
/// parses to a positive integer, otherwise the OS logical-core count.
pub fn capture_gomaxprocs() -> usize {
    if let Ok(s) = std::env::var("GOMAXPROCS")
        && let Ok(n) = s.parse::<usize>()
        && n >= 1
    {
        return n;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Active numactl binding for this run, sourced from the
/// `NUMACTL_BINDING` env var that the `eval-scaling` Makefile loop
/// sets per step. Empty / unset → `"none"`. The string is stored
/// verbatim so future bindings need no harness change; figure 07
/// joins on it as an opaque label.
pub fn capture_numactl_binding() -> String {
    match std::env::var("NUMACTL_BINDING") {
        Ok(s) if !s.is_empty() => s,
        _ => "none".to_string(),
    }
}

/// Cgroup v2 CPU quota in vCPU equivalents, or `None` if no limit is
/// set, the cgroup files aren't readable, or we're not on Linux.
///
/// Reads `/sys/fs/cgroup/cpu.max` whose payload is `"<quota_us>
/// <period_us>"` or `"max <period_us>"`. The quota / period ratio
/// gives the effective CPU cap (e.g. `quota=800000 period=100000` →
/// 8.0 vCPU equivalents). Tells us whether a RunAI pod's `--cpu N`
/// was enforced as a hard cap or as a request only — soft requests
/// surface as `None` while the host's logical core count is what
/// rayon actually uses.
pub fn capture_cgroup_cpu_quota() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/sys/fs/cgroup/cpu.max").ok()?;
        let mut parts = s.split_whitespace();
        let quota = parts.next()?;
        let period = parts.next()?;
        if quota == "max" {
            return None;
        }
        let q: f64 = quota.parse().ok()?;
        let p: f64 = period.parse().ok()?;
        if p > 0.0 { Some(q / p) } else { None }
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Total GPU memory in bytes for the primary CUDA device, or `None`
/// if `nvidia-smi` isn't available, returns nothing parseable, or
/// the host has no NVIDIA GPU. Used to populate
/// `[gpu].memory-bytes` in `run-metadata.toml`.
///
/// Shells out to `nvidia-smi --query-gpu=memory.total
/// --format=csv,noheader,nounits` (returns MiB), multiplies by
/// 1024² to land bytes. Picks the first GPU when multiple are
/// present — fine for our single-GPU pods and workstations.
pub fn capture_gpu_memory_bytes() -> Option<u64> {
    let raw = run_cmd(
        "nvidia-smi",
        &["--query-gpu=memory.total", "--format=csv,noheader,nounits"],
    )?;
    let first = raw.lines().next()?.trim();
    let mib: u64 = first.parse().ok()?;
    Some(mib.saturating_mul(1024 * 1024))
}

/// Cgroup v2 memory limit in bytes, or `None` if no limit is set, the
/// cgroup file isn't readable, or we're not on Linux.
///
/// Reads `/sys/fs/cgroup/memory.max` which is either a byte count or
/// the literal `"max"`. Disagreement with `MachineInfo::ram_bytes`
/// flags a cgroup-shaped run.
pub fn capture_cgroup_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/sys/fs/cgroup/memory.max").ok()?;
        let s = s.trim();
        if s == "max" {
            return None;
        }
        s.parse().ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Capture process-wide peak resident set size (high-water mark) in
/// bytes via `/proc/self/status::VmHWM`. Linux-only; returns `None`
/// on other platforms or if the file is unreadable. Recorded in
/// `run-metadata.toml` as `peak-rss-bytes` so post-hoc memory analysis
/// doesn't need operator-side `ps` / `top` snapshots.
pub fn capture_peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let v = rest.split_whitespace().next()?;
                let kb: u64 = v.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Snapshot of process-wide page-fault counters from `/proc/self/stat`.
/// Minor faults map a page already in memory into the address space
/// (cheap); major faults read a page from disk (the signal we care
/// about when mmap-backed cache files exceed RAM). Sample twice and
/// subtract for a scoring-loop delta — see
/// `eval-harness/src/bin/eval.rs` for the call sites.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcFaults {
    pub minor: u64,
    pub major: u64,
}

impl ProcFaults {
    pub fn delta(self, base: Self) -> Self {
        ProcFaults {
            minor: self.minor.saturating_sub(base.minor),
            major: self.major.saturating_sub(base.major),
        }
    }
}

/// Read `/proc/self/stat` and extract (minflt, majflt). Linux-only;
/// returns `None` on other platforms or if parsing fails. The `comm`
/// field can contain spaces and parentheses, so we slice from the last
/// `)` (closing the comm field) to skip past it cleanly.
pub fn capture_proc_faults() -> Option<ProcFaults> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/stat").ok()?;
        let close_paren = s.rfind(')')?;
        let after_comm = s[close_paren + 1..].trim();
        let fields: Vec<&str> = after_comm.split_ascii_whitespace().collect();
        // /proc/self/stat 1-indexed: 1 pid, 2 comm, 3 state, 4 ppid,
        // 5 pgrp, 6 session, 7 tty_nr, 8 tpgid, 9 flags, 10 minflt,
        // 11 cminflt, 12 majflt. After slicing past `)`, fields[0] is
        // field 3 (state), so minflt = fields[7], majflt = fields[9].
        let minor: u64 = fields.get(7)?.parse().ok()?;
        let major: u64 = fields.get(9)?.parse().ok()?;
        Some(ProcFaults { minor, major })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env-var manipulation is process-global; serialise the two tests
    // that touch NUMACTL_BINDING so they don't race within this binary.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn meta_captures_parallel_threads() {
        let n = capture_parallel_threads();
        assert!(n >= 1, "parallel_threads must be ≥ 1, got {n}");
        // Capture function must agree with rayon's view *after* one
        // trivial use has locked the pool size.
        assert_eq!(n, rayon::current_num_threads());
    }

    /// `collect_target_features` must (a) be pure / non-panicking,
    /// (b) return a sorted list, and (c) on the build we're testing
    /// under, contain `"avx512f"` exactly when the build target
    /// enables it. The cfg-gated assertion at the end pins the
    /// contract: a `make eval-native` build records `target-features`
    /// containing `avx512f`, distinguishing it from a baseline
    /// `make eval` build which omits the gate.
    #[test]
    fn meta_target_features_is_sorted_and_records_avx512_when_enabled() {
        let features = collect_target_features();
        let sorted = {
            let mut copy = features.clone();
            copy.sort();
            copy
        };
        assert_eq!(features, sorted, "target features must be sorted");

        #[cfg(target_feature = "avx512f")]
        assert!(
            features.iter().any(|f| f == "avx512f"),
            "avx512f cfg is on but the collector didn't record it: {features:?}"
        );
        #[cfg(not(target_feature = "avx512f"))]
        assert!(
            !features.iter().any(|f| f == "avx512f"),
            "avx512f cfg is off but the collector recorded it: {features:?}"
        );
    }

    #[test]
    fn meta_peak_rss_capture_does_not_panic() {
        // Returns Some on Linux with the live VmHWM, None on other
        // platforms. Either way the call must not panic, and on Linux
        // the value should be plausibly > 0 (any process has touched
        // some memory by the time the test runs).
        let v = capture_peak_rss_bytes();
        #[cfg(target_os = "linux")]
        assert!(v.is_some_and(|b| b > 0));
        #[cfg(not(target_os = "linux"))]
        assert!(v.is_none());
    }

    #[test]
    fn meta_cgroup_capture_does_not_panic() {
        // Returns None on non-Linux; on Linux returns Some/None based on
        // whether cgroup v2 files are present and whether a limit is set.
        // We can't predict which, so just confirm both calls run cleanly
        // — the parse paths never panic on bad input.
        let _ = capture_cgroup_cpu_quota();
        let _ = capture_cgroup_memory_bytes();
    }

    #[test]
    fn meta_captures_numactl_binding() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: env mutation in tests; serialised via ENV_LOCK and
        // the only reader within this test binary is capture_numactl_binding.
        unsafe { std::env::remove_var("NUMACTL_BINDING") };
        assert_eq!(capture_numactl_binding(), "none");

        unsafe { std::env::set_var("NUMACTL_BINDING", "physcpubind=0-15,membind=0") };
        assert_eq!(capture_numactl_binding(), "physcpubind=0-15,membind=0");

        unsafe { std::env::set_var("NUMACTL_BINDING", "") };
        assert_eq!(capture_numactl_binding(), "none");

        unsafe { std::env::remove_var("NUMACTL_BINDING") };
    }

    // -----------------------------------------------------------------
    // Campaign — behaviour matrix
    // -----------------------------------------------------------------

    #[test]
    fn campaign_all_absent_returns_none() {
        assert_eq!(Campaign::try_new(None, None, None), Ok(None));
        // Empty strings are treated as absent (env-var-from-clap path).
        assert_eq!(
            Campaign::try_new(Some("".into()), Some("".into()), Some("".into())),
            Ok(None)
        );
        assert_eq!(Campaign::try_new(Some("   ".into()), None, None), Ok(None));
    }

    #[test]
    fn campaign_id_and_title_present_emits_block() {
        let c = Campaign::try_new(
            Some("validation-2026-05-12".into()),
            Some("bulk-store e2e".into()),
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(c.id, "validation-2026-05-12");
        assert_eq!(c.title, "bulk-store e2e");
        assert_eq!(c.note, None);
    }

    #[test]
    fn campaign_with_note_preserves_note() {
        let c = Campaign::try_new(
            Some("fig07-mini-2026-05-12".into()),
            Some("Figure 7 mini sweep".into()),
            Some("rep=1, nprobe=32".into()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(c.note.as_deref(), Some("rep=1, nprobe=32"));
    }

    #[test]
    fn campaign_id_without_title_errors() {
        assert_eq!(
            Campaign::try_new(Some("foo".into()), None, None),
            Err(CampaignError::TitleMissing)
        );
    }

    #[test]
    fn campaign_title_without_id_errors() {
        assert_eq!(
            Campaign::try_new(None, Some("foo".into()), None),
            Err(CampaignError::IdMissing)
        );
    }

    #[test]
    fn campaign_note_without_id_or_title_errors() {
        assert_eq!(
            Campaign::try_new(None, None, Some("orphan note".into())),
            Err(CampaignError::NoteWithoutIdTitle)
        );
    }

    #[test]
    fn campaign_id_rejects_whitespace() {
        assert_eq!(
            Campaign::try_new(Some("has space".into()), Some("t".into()), None),
            Err(CampaignError::IdHasWhitespace)
        );
    }

    #[test]
    fn campaign_id_rejects_path_separator() {
        assert_eq!(
            Campaign::try_new(Some("a/b".into()), Some("t".into()), None),
            Err(CampaignError::IdHasPathSeparator)
        );
        assert_eq!(
            Campaign::try_new(Some("a\\b".into()), Some("t".into()), None),
            Err(CampaignError::IdHasPathSeparator)
        );
    }

    #[test]
    fn campaign_id_rejects_quotes() {
        assert_eq!(
            Campaign::try_new(Some("a'b".into()), Some("t".into()), None),
            Err(CampaignError::IdHasQuote)
        );
        assert_eq!(
            Campaign::try_new(Some("a\"b".into()), Some("t".into()), None),
            Err(CampaignError::IdHasQuote)
        );
    }

    #[test]
    fn campaign_id_rejects_overlong() {
        let long = "a".repeat(129);
        assert_eq!(
            Campaign::try_new(Some(long), Some("t".into()), None),
            Err(CampaignError::IdTooLong(129))
        );
    }

    #[test]
    fn campaign_id_accepts_allowed_chars() {
        // Convention example for an allowed campaign id.
        let allowed = "decode-gpu.2026-05-14:rep1";
        let c = Campaign::try_new(Some(allowed.into()), Some("t".into()), None)
            .unwrap()
            .unwrap();
        assert_eq!(c.id, allowed);
    }

    #[test]
    fn campaign_id_rejects_invalid_punct() {
        assert_eq!(
            Campaign::try_new(Some("foo!".into()), Some("t".into()), None),
            Err(CampaignError::IdHasInvalidChar('!'))
        );
    }

    #[test]
    fn campaign_serialises_to_kebab_case_toml() {
        let c = Campaign::try_new(
            Some("acceptance-2026-05-12".into()),
            Some("bulk-store acceptance".into()),
            Some("hello".into()),
        )
        .unwrap()
        .unwrap();
        let s = toml::to_string(&c).unwrap();
        assert!(s.contains(r#"id = "acceptance-2026-05-12""#));
        assert!(s.contains(r#"title = "bulk-store acceptance""#));
        assert!(s.contains(r#"note = "hello""#));
    }

    #[test]
    fn campaign_note_omitted_when_absent() {
        let c = Campaign::try_new(Some("k".into()), Some("t".into()), None)
            .unwrap()
            .unwrap();
        let s = toml::to_string(&c).unwrap();
        assert!(!s.contains("note"), "note key must be absent: {s:?}");
    }
}
