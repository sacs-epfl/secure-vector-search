//! Build the Tiptoe Go reference's paired-runner binary, applying our
//! vendor patch on top of the pinned commit.
//!
//! Used by `eval-harness/bin/tiptoe_go_runner.rs` to ensure the Go-side
//! validation-gate driver exists before the Rust runner shells out to
//! it.
//!
//! Conventions:
//! - The pinned commit lives in `tools/tiptoe-go-rev`.
//! - The vendor patch lives in `tools/tiptoe-go.patch`.
//! - The Go reference is expected at a path the caller supplies
//!   (`--go-ref-dir`, default `tmp/tiptoe-go`). Auto-clone is not
//!   supported; if the dir is missing we error out with clone
//!   instructions.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, anyhow, bail};

pub struct GoRefBuild {
    /// Absolute path to the built `paired-runner` binary.
    pub binary_path: PathBuf,
    /// Pinned commit from `tools/tiptoe-go-rev` (`ahenzinger/tiptoe` row).
    pub commit: String,
    /// `go version` output captured at build time, e.g.
    /// `"go version go1.21.9 linux/amd64"`.
    pub toolchain: String,
}

/// Read the `ahenzinger/tiptoe` pinned commit from a `tools/tiptoe-go-rev`
/// file. Lines starting with `#` are comments; body lines are
/// `<repo> <commit>` whitespace-separated.
fn parse_pinned_commit(rev_file: &Path) -> anyhow::Result<String> {
    let s = std::fs::read_to_string(rev_file)
        .with_context(|| format!("reading {}", rev_file.display()))?;
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let repo = parts.next().unwrap_or("");
        let commit = parts.next().unwrap_or("");
        if repo == "ahenzinger/tiptoe" && !commit.is_empty() {
            return Ok(commit.to_string());
        }
    }
    Err(anyhow!(
        "no `ahenzinger/tiptoe <commit>` line in {}",
        rev_file.display()
    ))
}

fn run_in(dir: &Path, prog: &str, args: &[&str]) -> anyhow::Result<String> {
    let out = Command::new(prog)
        .args(args)
        .current_dir(dir)
        .output()
        .with_context(|| format!("spawning {prog} in {}", dir.display()))?;
    if !out.status.success() {
        bail!(
            "{prog} {} failed in {}\nstdout:\n{}\nstderr:\n{}",
            args.join(" "),
            dir.display(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn run_capturing(prog: &str, args: &[&str]) -> anyhow::Result<String> {
    let out = Command::new(prog)
        .args(args)
        .output()
        .with_context(|| format!("spawning {prog}"))?;
    if !out.status.success() {
        bail!(
            "{prog} {} failed\nstderr:\n{}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr),
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Ensure the paired-runner binary is built at `<go_ref_dir>/paired-runner`.
///
/// Steps (idempotent):
/// 1. Read the pinned commit from `rev_file`.
/// 2. Confirm `go_ref_dir` exists and its HEAD matches the pin (warn,
///    don't fail, if it differs — patch may already have been applied
///    on top of it).
/// 3. If `cmd/paired-runner/main.go` is missing, apply `patch_file`.
/// 4. Run `go mod download` and `go build ./cmd/paired-runner/`.
/// 5. Capture `go version` and return the build info.
pub fn ensure_paired_runner(
    go_ref_dir: &Path,
    rev_file: &Path,
    patch_file: &Path,
) -> anyhow::Result<GoRefBuild> {
    let pinned_commit = parse_pinned_commit(rev_file)?;

    if !go_ref_dir.exists() {
        bail!(
            "Go reference checkout missing: {}\n\
             To set up:\n  \
             git clone https://github.com/ahenzinger/tiptoe.git {}\n  \
             cd {} && git checkout {}\n  \
             (the patch at {} will be applied automatically by the runner)",
            go_ref_dir.display(),
            go_ref_dir.display(),
            go_ref_dir.display(),
            pinned_commit,
            patch_file.display(),
        );
    }

    let head = run_in(go_ref_dir, "git", &["rev-parse", "HEAD"])?;
    if head != pinned_commit {
        eprintln!(
            "warning: Go ref at {} is at {}, expected pinned {} (per {})",
            go_ref_dir.display(),
            &head[..head.len().min(8)],
            &pinned_commit[..pinned_commit.len().min(8)],
            rev_file.display(),
        );
    }

    // Apply the vendor patch if the marker file (added by the patch)
    // isn't there yet.
    let patch_marker = go_ref_dir.join("cmd/paired-runner/main.go");
    if !patch_marker.exists() {
        let patch_abs = patch_file
            .canonicalize()
            .with_context(|| format!("canonicalising {}", patch_file.display()))?;
        eprintln!(
            "Applying {} to {}",
            patch_abs.display(),
            go_ref_dir.display()
        );
        run_in(
            go_ref_dir,
            "git",
            &["apply", patch_abs.to_str().expect("non-UTF-8 patch path")],
        )
        .with_context(|| format!("git apply {}", patch_abs.display()))?;
    }

    eprintln!("Running `go mod download` in {}", go_ref_dir.display());
    run_in(go_ref_dir, "go", &["mod", "download"])?;

    eprintln!("Building paired-runner");
    run_in(
        go_ref_dir,
        "go",
        &["build", "-o", "paired-runner", "./cmd/paired-runner/"],
    )?;

    let binary_path = go_ref_dir.join("paired-runner").canonicalize()?;
    let toolchain = run_capturing("go", &["version"])?;

    Ok(GoRefBuild {
        binary_path,
        commit: pinned_commit,
        toolchain,
    })
}
