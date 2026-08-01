# Secure Vector Search

Research prototype comparing five vector similarity search backends
(Plaintext, SAP/DCPE, EMVP, Tiptoe, BN) behind a shared `Scorer` trait.

The extended version of the paper, with appendices, is at
[`extended-version.pdf`](extended-version.pdf).

## Crates

| Crate | Role |
|---|---|
| `scorer-core` | `Scorer` trait and shared types |
| `ivf-index` | Shared IVF/k-means building block |
| `scorer-plaintext` | IVF baseline, no encryption |
| `scorer-sap` | SAP/DCPE — scale-and-perturb; flat scan + IVF variant |
| `scorer-emvp` | EMVP — encrypted matrix-vector products; `EmvpScorer` (flat scan) + `EmvpIvfScorer` (IVF), Sec128 params |
| `scorer-tiptoe` | Tiptoe — LWE-based private ANN; in-house LWE + SimplePIR-LHE primitives + `fhe`-based BFV layer with §6.2 limb-decomposition glue. Correctness gated against the Go reference via `analysis/tiptoe_diff.py` |
| `scorer-bntm` | Braverman–Newman trapdoored matrices — `BnTmScorer` (flat) + `BnTmIvfScorer` (IVF) over F_p with p = 2^61 − 1, Sec128 params; Protocol 2 (iterated Freivalds) verification for malicious-server detection |
| `eval-harness` | Corpus loading, ground-truth computation, benchmark runner; `bin/eval` for in-process scorers and `bin/tiptoe_go_runner` for the Go-reference paired-run validation gate |

## Running the tests

```bash
cargo test --workspace
```

This runs all unit and integration tests, including a behavioural agreement test that
asserts SAP at β=0 achieves recall@10 = 1.0 against the plaintext scorer on synthetic data.

## Corpus setup and evaluation

Requires Python ≥ 3.10. Install Python deps once:

```bash
pip install -r scripts/requirements.txt
```

Then use `make` to drive the full pipeline. Each step is skipped automatically if
its output file already exists:

```bash
make all                       # download → subsample → embed → ground-truth → eval
make embed DEVICE=mps          # use Apple Silicon GPU for embedding (also: cuda)
make analysis                  # preprocess CSVs → TSVs → compile pgfplots figures
make report                    # one PDF report per machine under results/runs/ (default)
make report RUN=<run-dir>      # target only the machine that owns that run directory
make clean                     # delete run directories and analysis outputs
make distclean                 # also delete derived data files (re-embed is slow)
```

### Run every evaluation needed for the paper

`make eval` covers the throughput / recall / communication figures (01–06, 10),
but several figures need extra diagnostic passes — cold-build (08), per-substep
timing (09a/09b), BN verification compare (13), and parallel scaling (07). Run
each block from a clean tree (or accept that old runs are retained alongside):

```bash
# 1. Throughput sweeps — figures 01, 02, 03, 04, 05, 06, 10
make eval                      # plaintext, SAP, SAP-IVF, EMVP, EMVP-IVF,
                               #   Tiptoe (Rust + Go ref), BN, BN-IVF
make tiptoe-diff               # validation gate: diff latest tiptoe vs tiptoe-go runs

# 2. Cold-build pass — figure 08 (build-time per scheme)
make eval-cold                 # re-runs every bin/eval scheme with --no-cache so
                               #   `[index].cache-hit = false` on at least one run
                               #   per scheme; tiptoe-go always rebuilds anyway

# 3. Per-substep timing — figures 09a, 09b
make eval-breakdown            # one --breakdown run per scheme at a single
                               #   representative config (nprobe=32, β=0, q=3,
                               #   verification on); raw.csv is header-only in
                               #   this mode, so these don't double as throughput
                               #   runs

# 4. BN verification on/off — figure 13
#    `make eval` already runs BN with verification ON (the default). Add the
#    OFF pass for both flat and IVF so the figure has both halves:
make eval-bntm     BNTM_VERIFICATION=false
make eval-bntm-ivf BNTM_VERIFICATION=false

# 5. Parallel scaling — figure 07 (optional, ~8 h on sacs006)
make eval-scaling              # eval-scaling-no-tiptoe (~1 h) + eval-scaling-tiptoe (~7 h)
                               #   Requires passwordless sudo for
                               #   `/usr/bin/tee /proc/sys/vm/drop_caches`; verify
                               #   with `sudo -n tee /proc/sys/vm/drop_caches < /dev/null`.

# 6. Compile figures and PDF report
make analysis
make report
```

### Per-scheme and diagnostic targets

```bash
make eval-no-tiptoe            # like `make eval` but skips Tiptoe (Rust + Go ref) — fast subset
make eval-plaintext            # run only the plaintext sweep
make eval-sap                  # run only the SAP flat-scan sweep
make eval-sap-ivf              # run both SAP-IVF sub-sweeps (nprobe + beta)
make eval-sap-ivf-nprobe       # SAP-IVF: sweep nprobe at fixed beta=0
make eval-sap-ivf-beta         # SAP-IVF: sweep beta at fixed nprobe=32
make eval-emvp                 # run EMVP flat-scan sweep (Sec128 params)
make eval-emvp-ivf             # run EMVP+IVF nprobe sweep
make eval-tiptoe               # run Tiptoe (Rust port) quantisation-bits sweep
make eval-tiptoe-go            # run Tiptoe (Go reference) at matched parameters
make eval-bntm                 # run BN flat-scan (verification on by default; set BNTM_VERIFICATION=false to compare)
make eval-bntm-ivf             # run BN+IVF nprobe sweep

# GPU passes. Activate the rapids conda env first:
#   conda activate rapids   # provides cuvs, nvcc, libcuda
#   export CMAKE_PREFIX_PATH="$CONDA_PREFIX" LIBCLANG_PATH="$CONDA_PREFIX/lib" …
# Tiptoe is excluded from the GPU targets because Tiptoe-GPU is deferred
# and reported via the analytical proxy column only.
make eval-gpu-workstation      # eval-no-tiptoe with --device gpu --gpu-sku rtx-5000-ada
                               #   --gpu-location local. Runs the consumer-class
                               #   substrate locally on workstation hardware.
make eval-gpu-cloud            # eval-no-tiptoe with --device gpu --gpu-sku v100
                               #   --gpu-location cloud and the --cloud-* provenance
                               #   flags. Override PROVIDER/INSTANCE/REGION/
                               #   DRIVER_VERSION/CUDA_VERSION/GPU_SKU_CLOUD on the
                               #   command line to match the actual rental — defaults
                               #   document a representative AWS V100 instance pin.

# Diagnostic passes. Each maps to a bin/eval flag:
#   --no-cache   wipes scheme-specific caches at startup so BuildOutcome reports
#                a true cold build — source for figure 08
#   --breakdown  switches dispatch to per-scorer score_with_breakdown methods and
#                emits substep-breakdown.csv (route, encode, server-compute,
#                verify, decompress, decode, merge); raw.csv stays header-only
#                — source for figures 09a / 09b. Slower than the throughput
#                path; do not use these timings for latency claims.
make eval-cold                 # `make eval` with --no-cache (figure 08)
make eval-breakdown            # one --breakdown run per scheme at a fixed config
make eval-breakdown-no-tiptoe  # same, skipping Tiptoe (faster)
make eval-breakdown-plaintext  # individual --breakdown targets exist for every scheme:
                               #   plaintext, sap, sap-ivf, emvp, emvp-ivf,
                               #   tiptoe, bntm, bntm-ivf

# Parallel-scaling sub-targets:
make eval-scaling-no-tiptoe    # thread-count sweep over plaintext/SAP+IVF/
                               #   EMVP+IVF/BN+IVF (~1 h on sacs006)
make eval-scaling-tiptoe       # same sweep for Tiptoe (Rust + Go ref). ~7 h.
make eval-scaling              # both passes; runs ~8 h total
```

> **Cache format.** Disk caches use hashed filenames
> (`.<scorer>-cache-<16hex>.bin`) with self-verifying headers. Old caches
> from prior versions (`.ivf-cache-{n_centroids}-{seed}-...`,
> `.sap-ivf-cache-...`, `.emvp-cache-...`) are not auto-deleted; remove
> them once with `rm data/*/.*-cache-*.bin` to reclaim disk. Rebuild is
> ~30 s for plaintext IVF, a few minutes for SAP-IVF, ~10–20 min for EMVP.

Key variables (override on the command line):

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `auto` | Embedding device: `auto` (detect MPS/CUDA), `cpu`, `mps`, `cuda` |
| `DATASET` | `msmarco` | Dataset name; data lives in `data/$(DATASET)/` |
| `GPU_KIND` | _(auto)_ | GPU label written to `machines.csv`: auto-detected from the venv's torch; override with `mps`, `cuda`, or `none` |
| `GT_K` | `100` | Neighbours stored in ground truth |
| `EVAL_K` | `10` | Neighbours retrieved during eval |
| `REPS` | `3` | Repetitions per configuration |
| `NPROBE` | `1,2,4,8,16,32,64,128` | IVF probe counts |
| `BETA` | `0.0,0.1,0.5,1.0` | SAP perturbation levels |
| `SAP_IVF_BETA` | `0.0,0.5` | Beta values for the SAP-IVF beta sweep |
| `SAP_IVF_NPROBE_FIXED` | `32` | Fixed nprobe for the SAP-IVF beta sweep |
| `RUN` | _(auto)_ | Run directory passed to `make report`; determines target machine |

Each `make eval-*` invocation writes to a new timestamped directory:

```
results/
  index.csv                        # one-row-per-run registry
  machines.csv                     # hardware specs keyed by machine-id
  runs/
    <machine-id>/
      <git-sha>/
        <run-id>/
          raw.csv                  # one row per query × config × repetition
                                   #   (header-only when --breakdown is set)
          top_k.csv                # per-query top-k IDs (rep=0)
          substep-breakdown.csv    # per-(scheme, query, substep) timings —
                                   #   only present when --breakdown is set
          run-metadata.toml        # full provenance (git state, hardware, config, timing,
                                   #   [index] block with cold-build outcome,
                                   #   no-cache / breakdown intent flags)
      reports/
        <timestamp>/
          report.pdf               # compiled PDF (all schemes for this machine)
          report.tex               # LaTeX source
          style.tex                # copy of analysis/style.tex (self-contained)
          data/                    # per-figure TSVs used by this report
          figures/                 # .tex sources for each figure (input'd by report.tex)
```

Old runs are never overwritten; delete `results/runs/` and `results/index.csv` to start fresh.
