//! Compile `src/gpu/kernels.cu` to PTX with `nvcc` when the `gpu`
//! Cargo feature is on.
//!
//! Mirrors `scorer-emvp/build.rs` exactly — same nvcc invocation,
//! same `compute_70` PTX target, same `CARGO_FEATURE_GPU`-gated
//! short-circuit. The output PTX lives in `OUT_DIR/bntm_kernels.ptx`
//! and is pulled in by `src/gpu/mod.rs` via `include_str!`.
//!
//! This crate ships its own kernel rather than sharing EMVP's: BN's
//! `compute_products` is a single full m × n GEMV, while EMVP's writes
//! a block-major s × m partials matrix. The Mersenne arithmetic is
//! identical between the two but the kernel shape isn't, so a clean
//! field-generic share would mean either (a) collapsing EMVP's S=76,
//! B=17 to S=1, B=N=1024 at runtime (loses `#pragma unroll` on B) or
//! (b) duplicating the shape logic. The crate-local kernel here is
//! ~30 lines and shares the Mersenne `fp_mul` / `fp_add` definitions
//! inline.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/gpu/kernels.cu");

    if env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));

    let kernel_src = manifest_dir.join("src/gpu/kernels.cu");
    let kernel_ptx = out_dir.join("bntm_kernels.ptx");

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    let status = Command::new(&nvcc)
        .arg("--ptx")
        .arg("-O3")
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
                 Confirm the rapids conda env is active\n\
                 and that nvcc is on PATH; current NVCC = {nvcc}.",
                s.code().unwrap_or(-1),
                kernel_src.display()
            );
        }
        Err(e) => {
            panic!(
                "Could not invoke nvcc ({nvcc}): {e}.\n\
                 The `gpu` feature requires nvcc on PATH; activate the\n\
                 rapids conda env before building."
            );
        }
    }
}
