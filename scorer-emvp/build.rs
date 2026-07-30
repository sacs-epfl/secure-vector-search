//! Compile `src/gpu/kernels.cu` to PTX with `nvcc` when the `gpu`
//! Cargo feature is on.
//!
//! Cargo invokes this for every build; we short-circuit unless the
//! feature is enabled (cargo sets `CARGO_FEATURE_GPU` for that case).
//! The generated PTX lands in `OUT_DIR/emvp_kernels.ptx` and is
//! pulled into the crate via `include_str!` from the gpu module.
//!
//! ## PTX target architecture
//!
//! We target `compute_70` — the V100 dev box's compute capability —
//! because PTX is forward-compatible: PTX targeting a lower SM is
//! JIT-compiled by the driver to whatever SM the running GPU has
//! (RTX 5000 Ada / sm_89, H100 / sm_90, etc.). Pinning the target
//! at compute_70 means the same shipped binary works across the
//! workstation, dev box, and cloud-rented H100 — at the cost of one
//! extra JIT step the driver caches anyway.
//!
//! ## When this build script fails
//!
//! If `--features gpu` is set but `nvcc` is not on PATH, this build
//! script bails with a clear error telling the operator to put nvcc
//! on PATH. Don't try to fall back silently — a feature-on build that
//! produced no kernel would surface as a confusing runtime panic
//! later.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Always tell cargo to re-run if these change so the kernel.ptx
    // stays in sync with its source. (Cargo only re-invokes the
    // build script when an explicit `rerun-if-*` directive lists the
    // file or when build.rs itself changes; declare both here.)
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/gpu/kernels.cu");

    // Short-circuit on CPU-only builds. Cargo sets `CARGO_FEATURE_GPU`
    // exactly when `--features gpu` is active for this crate.
    if env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));

    let kernel_src = manifest_dir.join("src/gpu/kernels.cu");
    let kernel_ptx = out_dir.join("emvp_kernels.ptx");

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    let status = Command::new(&nvcc)
        .arg("--ptx")
        .arg("-O3")
        // Pin the PTX virtual arch at compute_70 (V100). PTX targeting
        // compute_70 is forward-compatible: the CUDA driver JIT-
        // compiles it to whatever SM the host GPU has at first
        // launch. Same shipped PTX runs on V100 / RTX 5000 Ada / H100.
        .arg("--gpu-architecture=compute_70")
        .arg("-o")
        .arg(&kernel_ptx)
        .arg(&kernel_src)
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            panic!(
                "nvcc failed (exit {}) compiling {}.\n\
                 Confirm the rapids conda env is active (see docs/envs/README.md)\n\
                 and that nvcc is on PATH; current NVCC = {nvcc}.",
                s.code().unwrap_or(-1),
                kernel_src.display()
            );
        }
        Err(e) => {
            panic!(
                "Could not invoke nvcc ({nvcc}): {e}.\n\
                 The `gpu` feature requires nvcc on PATH; activate the\n\
                 rapids conda env (see docs/envs/README.md) before building."
            );
        }
    }
}
