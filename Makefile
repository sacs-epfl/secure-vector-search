# ── Configurable ──────────────────────────────────────────────────────────────
# Embed device: auto | cpu | mps | cuda  (auto = detect at embed time)
DEVICE   ?= auto
# Dataset name; data lives in data/$(DATASET)/
DATASET  ?= msmarco
# Number of passages to subsample from the source corpus and
# embed. `msmarco-full` overrides to the full MS MARCO passage count.
ifeq ($(DATASET),msmarco-full)
N_PASSAGES ?= 8800000
else
N_PASSAGES ?= 100000
endif
# GPU on the eval machine: none | mps | cuda  (auto-detected if not set)
ifndef GPU_KIND
GPU_KIND := $(shell venv/bin/python -c \
  "import torch; print('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'none')" \
  2>/dev/null || echo none)
endif
# Neighbours stored in ground truth
GT_K     ?= 100
# Neighbours retrieved during eval
EVAL_K   ?= 10
REPS     ?= 3
NPROBE            ?= 1,2,4,8,16,32,64,128
BETA              ?= 0.0,0.1,0.5,1.0
SAP_IVF_BETA      ?= 0.0,0.5
SAP_IVF_NPROBE_FIXED ?= 32
TIPTOE_QUANTISATION_BITS ?= 3,4
# `eval-batch` sweeps these batch sizes via the `--batch-sizes`
# flag (default `1` preserves single-query semantics in every other
# `eval-*` target). Override to a different sweep on a per-host basis;
# the harness drops the final partial chunk per query budget so any
# divisor of `N_QUERIES` works cleanly.
BATCH_SIZES ?= 1,8,64,256
# Single q-bits value for the Go-ref paired runner. The Go ref's
# defensive max-IP check requires this to fit `2 · 4^q · d < P`; at
# d=768 that bounds q ≤ 3.
TIPTOE_GO_QUANTISATION_BITS ?= 3
TIPTOE_GO_HINT_MB ?= 25
# Extra flags appended to every `bin/eval` invocation. Empty by default;
# `eval-cold` sets `--no-cache` so figure 9 sees one cold build per
# scheme. Threaded only through bin/eval recipes — `tiptoe_go_runner`
# does not accept this flag.
EVAL_FLAGS ?=
# Comma-separated list of cargo features to enable on
# bin/eval. Empty (default) builds CPU-only; `gpu` enables the cuvs +
# cudarc paths for plaintext / sap / emvp / bntm. The `eval-gpu-*`
# Makefile targets set this automatically; manual `make eval-no-tiptoe
# CARGO_FEATURES=gpu …` invocations also work.
CARGO_FEATURES ?=
CARGO_FEATURES_ARG := $(if $(CARGO_FEATURES),--features $(CARGO_FEATURES))

PYTHON  ?= venv/bin/python
CARGO   := cargo
DATA    := data/$(DATASET)
RESULTS := results
SCRIPTS := scripts

.PHONY: all download subsample embed ground-truth eval eval-cold eval-native eval-no-tiptoe eval-plaintext eval-sap eval-sap-ivf eval-sap-ivf-nprobe eval-sap-ivf-beta eval-emvp eval-emvp-ivf eval-tiptoe eval-tiptoe-go eval-bntm eval-bntm-ivf eval-batch eval-breakdown eval-breakdown-no-tiptoe eval-breakdown-plaintext eval-breakdown-sap eval-breakdown-sap-ivf eval-breakdown-emvp eval-breakdown-emvp-ivf eval-breakdown-tiptoe eval-breakdown-bntm eval-breakdown-bntm-ivf eval-scaling eval-scaling-no-tiptoe eval-scaling-tiptoe eval-gpu-workstation eval-gpu-workstation-2card _eval-gpu-card0 _eval-gpu-card1 eval-gpu-cloud tiptoe-diff analysis figures preprocess report finalize build clean distclean

# ── top-level ──────────────────────────────────────────────────────────────────

all: eval

eval: eval-plaintext eval-sap eval-sap-ivf eval-emvp eval-emvp-ivf eval-tiptoe eval-tiptoe-go eval-bntm eval-bntm-ivf

# Cold-build pass: re-runs every bin/eval scorer with --no-cache so
# `[index].cache-hit = false` on at least one run per scheme.
# Required for figure 9 (build time) to show all schemes —
# write_build_time_summary filters cache-hit=true rows out.
# `eval-tiptoe-go` is included verbatim because the Go ref builds its
# index every invocation regardless; `tiptoe_go_runner` doesn't take
# `--no-cache`, so EVAL_FLAGS is scoped to bin/eval recipes only.
eval-cold:
	$(MAKE) eval EVAL_FLAGS=--no-cache

# Opt-in `target-cpu=native`. Re-runs `eval`
# with RUSTFLAGS exported for the recursive make, so every cargo
# invocation in that subprocess tree (compile of bin/eval, the
# scorer crates, anything cargo-build pulls along the way) picks
# up the per-host codegen lift (FMA fusion on M4, AVX-512 on
# Xeon). Persisted alternative: copy `.cargo/config.toml.example`
# to `.cargo/config.toml` (gitignored) on the producer host.
#
# Results recorded under this target carry machine-specific
# binary differences; the cross-machine-comparison filter is
# what keeps them from being compared against non-native runs.
eval-native:
	RUSTFLAGS="-C target-cpu=native" $(MAKE) eval

# Subset of `eval` that skips Tiptoe (Rust port + Go reference). Useful
# for iterating on the other scorers — Tiptoe is the slowest scheme in
# the lineup (per-query LWE+BFV is ~5 s on the 100k corpus, vs <1 s
# for the others). Run `make eval-tiptoe eval-tiptoe-go tiptoe-diff`
# separately when you actually want Tiptoe data.
eval-no-tiptoe: eval-plaintext eval-sap eval-sap-ivf eval-emvp eval-emvp-ivf eval-bntm eval-bntm-ivf

# Figure 14 production sweep. Reuses every non-Tiptoe
# eval-* target with `--batch-sizes $(BATCH_SIZES)` threaded through
# EVAL_FLAGS so each bin/eval invocation nests batch-size × quality-
# param × repetitions inside its sweep loop. Tiptoe is excluded by
# construction (per-query cost floor): its `score_batch`
# falls back to the sequential default, so batched throughput would
# trace a flat line on figure 14.
#
# BN dispatch caveat: when BATCH_SIZES != [1] the BN scheme
# routes through `run_eval` (not `run_eval_with_verify_us`), so the
# per-row `verification-overhead-us` column is 0 on every emitted
# row. Figure 13 (BN verify on/off) consumes a separate B=1 run from
# `eval-bntm` / `eval-bntm-ivf`; figure 14 doesn't read that column.
# To gather both signals on one machine, run `eval` (or `eval-no-
# tiptoe`) first for figure 13, then `eval-batch` for figure 14.
eval-batch:
	$(MAKE) eval-no-tiptoe EVAL_FLAGS='--batch-sizes $(BATCH_SIZES)'

# ── data pipeline ─────────────────────────────────────────────────────────────

$(DATA)/collection.tsv:
	bash $(SCRIPTS)/download_corpus.sh --data-dir $(DATA)

download: $(DATA)/collection.tsv

$(DATA)/passages_$(N_PASSAGES).jsonl: $(DATA)/collection.tsv
	$(PYTHON) $(SCRIPTS)/subsample_corpus.py --data-dir $(DATA) --n $(N_PASSAGES)

subsample: $(DATA)/passages_$(N_PASSAGES).jsonl

# embed_corpus.py builds both fvecs files in one pass; chain so they run in
# sequence and the script's own skip-if-exists logic handles the second call.
$(DATA)/passages.fvecs: $(DATA)/passages_$(N_PASSAGES).jsonl
	$(PYTHON) $(SCRIPTS)/embed_corpus.py --device $(DEVICE) --data-dir $(DATA) \
	    --input-jsonl passages_$(N_PASSAGES).jsonl

$(DATA)/queries.fvecs: $(DATA)/passages.fvecs
	$(PYTHON) $(SCRIPTS)/embed_corpus.py --device $(DEVICE) --data-dir $(DATA) \
	    --input-jsonl passages_$(N_PASSAGES).jsonl

embed: $(DATA)/passages.fvecs $(DATA)/queries.fvecs

$(DATA)/ground_truth.ivecs: $(DATA)/passages.fvecs $(DATA)/queries.fvecs
	$(CARGO) run --release --bin ground_truth -- \
	    --data-dir $(DATA) \
	    --k        $(GT_K)

ground-truth: $(DATA)/ground_truth.ivecs

# ── eval sweeps ───────────────────────────────────────────────────────────────
# Eval targets are phony: each invocation writes to a new timestamped run
# directory under results/runs/. The IVF disk cache handles index reuse.
# Run `make eval-plaintext` to add a new run; old runs are preserved.

eval-plaintext: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer      plaintext \
	    --data-dir    $(DATA) \
	    --k           $(EVAL_K) \
	    --nprobe      $(NPROBE) \
	    --repetitions $(REPS) \
	    --gpu-kind    $(GPU_KIND) \
	    --results-dir $(RESULTS) \
	    $(EVAL_FLAGS)

eval-sap: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer      sap \
	    --data-dir    $(DATA) \
	    --k           $(EVAL_K) \
	    --beta        $(BETA) \
	    --repetitions $(REPS) \
	    --gpu-kind    $(GPU_KIND) \
	    --results-dir $(RESULTS) \
	    $(EVAL_FLAGS)

eval-sap-ivf-nprobe: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer      sap-ivf \
	    --data-dir    $(DATA) \
	    --k           $(EVAL_K) \
	    --nprobe      $(NPROBE) \
	    --beta-fixed  0.0 \
	    --repetitions $(REPS) \
	    --gpu-kind    $(GPU_KIND) \
	    --results-dir $(RESULTS) \
	    $(EVAL_FLAGS)

eval-sap-ivf-beta: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer       sap-ivf \
	    --data-dir     $(DATA) \
	    --k            $(EVAL_K) \
	    --beta         $(SAP_IVF_BETA) \
	    --nprobe-fixed $(SAP_IVF_NPROBE_FIXED) \
	    --repetitions  $(REPS) \
	    --gpu-kind     $(GPU_KIND) \
	    --results-dir  $(RESULTS) \
	    $(EVAL_FLAGS)

eval-sap-ivf: eval-sap-ivf-nprobe eval-sap-ivf-beta

eval-emvp: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer      emvp \
	    --data-dir    $(DATA) \
	    --k           $(EVAL_K) \
	    --repetitions $(REPS) \
	    --gpu-kind    $(GPU_KIND) \
	    --results-dir $(RESULTS) \
	    $(EVAL_FLAGS)

eval-emvp-ivf: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer      emvp-ivf \
	    --data-dir    $(DATA) \
	    --k           $(EVAL_K) \
	    --nprobe      $(NPROBE) \
	    --repetitions $(REPS) \
	    --gpu-kind    $(GPU_KIND) \
	    --results-dir $(RESULTS) \
	    $(EVAL_FLAGS)

eval-tiptoe: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer            tiptoe \
	    --data-dir          $(DATA) \
	    --k                 $(EVAL_K) \
	    --quantisation-bits $(TIPTOE_QUANTISATION_BITS) \
	    --repetitions       $(REPS) \
	    --gpu-kind          $(GPU_KIND) \
	    --results-dir       $(RESULTS) \
	    $(EVAL_FLAGS)

# Braverman–Newman trapdoored matrices. Flat scorer sweeps
# verification on/off via two invocations (BNTM_VERIFICATION). The IVF
# wrapper sweeps nprobe with verification fixed. BNTM_Q_BITS sets the
# f32 → F_p quantisation scale Q = 2^BNTM_Q_BITS (sweep variable;
# default 20 inherits the EMVP scale).
#
# Default is `false` — paper evaluates all schemes
# under HbC threat model alignment; pass `BNTM_VERIFICATION=true` to
# regenerate figure 13's verification-on data.
BNTM_VERIFICATION ?= false
BNTM_Q_BITS ?= 20

eval-bntm: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer            bntm \
	    --data-dir          $(DATA) \
	    --k                 $(EVAL_K) \
	    --bntm-verification $(BNTM_VERIFICATION) \
	    --bntm-q-bits       $(BNTM_Q_BITS) \
	    --repetitions       $(REPS) \
	    --gpu-kind          $(GPU_KIND) \
	    --results-dir       $(RESULTS) \
	    $(EVAL_FLAGS)

eval-bntm-ivf: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer            bntm-ivf \
	    --data-dir          $(DATA) \
	    --k                 $(EVAL_K) \
	    --nprobe            $(NPROBE) \
	    --bntm-verification $(BNTM_VERIFICATION) \
	    --bntm-q-bits       $(BNTM_Q_BITS) \
	    --repetitions       $(REPS) \
	    --gpu-kind          $(GPU_KIND) \
	    --results-dir       $(RESULTS) \
	    $(EVAL_FLAGS)

# ── breakdown sweeps ──────────────────────────────────────────────────────────
# Pass --breakdown to bin/eval to capture per-substep timings into
# substep-breakdown.csv (figure 09a/b). raw.csv is header-only in this
# mode, so a breakdown run cannot serve double-duty as a throughput run
# — keep these targets separate from `eval-*`.
#
# `analysis/preprocess.py::write_substep_breakdown` averages every
# (scheme, substep) row in the loaded run, mixing operating points when
# multiple configs are present. Each target below pins a single
# representative config so the figure-09 bars stay meaningful; override
# via the BREAKDOWN_* vars or BNTM_VERIFICATION if you want a different
# operating point.
BREAKDOWN_NPROBE                  ?= 32
BREAKDOWN_BETA                    ?= 0.0
BREAKDOWN_TIPTOE_QUANTISATION_BITS ?= 3
BREAKDOWN_REPS                    ?= 1

eval-breakdown: eval-breakdown-plaintext eval-breakdown-sap eval-breakdown-sap-ivf eval-breakdown-emvp eval-breakdown-emvp-ivf eval-breakdown-tiptoe eval-breakdown-bntm eval-breakdown-bntm-ivf

eval-breakdown-no-tiptoe: eval-breakdown-plaintext eval-breakdown-sap eval-breakdown-sap-ivf eval-breakdown-emvp eval-breakdown-emvp-ivf eval-breakdown-bntm eval-breakdown-bntm-ivf

eval-breakdown-plaintext: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer      plaintext \
	    --data-dir    $(DATA) \
	    --k           $(EVAL_K) \
	    --nprobe      $(BREAKDOWN_NPROBE) \
	    --repetitions $(BREAKDOWN_REPS) \
	    --gpu-kind    $(GPU_KIND) \
	    --results-dir $(RESULTS) \
	    --breakdown \
	    $(EVAL_FLAGS)

eval-breakdown-sap: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer      sap \
	    --data-dir    $(DATA) \
	    --k           $(EVAL_K) \
	    --beta        $(BREAKDOWN_BETA) \
	    --repetitions $(BREAKDOWN_REPS) \
	    --gpu-kind    $(GPU_KIND) \
	    --results-dir $(RESULTS) \
	    --breakdown \
	    $(EVAL_FLAGS)

eval-breakdown-sap-ivf: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer      sap-ivf \
	    --data-dir    $(DATA) \
	    --k           $(EVAL_K) \
	    --nprobe      $(BREAKDOWN_NPROBE) \
	    --beta-fixed  $(BREAKDOWN_BETA) \
	    --repetitions $(BREAKDOWN_REPS) \
	    --gpu-kind    $(GPU_KIND) \
	    --results-dir $(RESULTS) \
	    --breakdown \
	    $(EVAL_FLAGS)

eval-breakdown-emvp: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer      emvp \
	    --data-dir    $(DATA) \
	    --k           $(EVAL_K) \
	    --repetitions $(BREAKDOWN_REPS) \
	    --gpu-kind    $(GPU_KIND) \
	    --results-dir $(RESULTS) \
	    --breakdown \
	    $(EVAL_FLAGS)

eval-breakdown-emvp-ivf: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer      emvp-ivf \
	    --data-dir    $(DATA) \
	    --k           $(EVAL_K) \
	    --nprobe      $(BREAKDOWN_NPROBE) \
	    --repetitions $(BREAKDOWN_REPS) \
	    --gpu-kind    $(GPU_KIND) \
	    --results-dir $(RESULTS) \
	    --breakdown \
	    $(EVAL_FLAGS)

eval-breakdown-tiptoe: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer            tiptoe \
	    --data-dir          $(DATA) \
	    --k                 $(EVAL_K) \
	    --quantisation-bits $(BREAKDOWN_TIPTOE_QUANTISATION_BITS) \
	    --repetitions       $(BREAKDOWN_REPS) \
	    --gpu-kind          $(GPU_KIND) \
	    --results-dir       $(RESULTS) \
	    --breakdown

eval-breakdown-bntm: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer            bntm \
	    --data-dir          $(DATA) \
	    --k                 $(EVAL_K) \
	    --bntm-verification $(BNTM_VERIFICATION) \
	    --repetitions       $(BREAKDOWN_REPS) \
	    --gpu-kind          $(GPU_KIND) \
	    --results-dir       $(RESULTS) \
	    --breakdown \
	    $(EVAL_FLAGS)

eval-breakdown-bntm-ivf: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	$(CARGO) run --release $(CARGO_FEATURES_ARG) --bin eval -- \
	    --scorer            bntm-ivf \
	    --data-dir          $(DATA) \
	    --k                 $(EVAL_K) \
	    --nprobe            $(BREAKDOWN_NPROBE) \
	    --bntm-verification $(BNTM_VERIFICATION) \
	    --repetitions       $(BREAKDOWN_REPS) \
	    --gpu-kind          $(GPU_KIND) \
	    --results-dir       $(RESULTS) \
	    --breakdown \
	    $(EVAL_FLAGS)

# ── parallel scaling sweep ────────────────────────────────────────────────────
# Sweeps RAYON_NUM_THREADS (Rust) and GOMAXPROCS (Go) over the sacs006
# topology: dual-socket Xeon Gold 6426Y, 2 × 16 physical cores + SMT
# for 64 logical. Three regimes split the sweep:
#   N ∈ [1, 16]  — within one socket, one NUMA node (pinned pass)
#   N ∈ [16, 32] — cross-socket, second NUMA node engaged (unpinned)
#   N ∈ [32, 64] — SMT-fed, no new physical cores (unpinned)
# N=16 appears in both passes as a matched anchor so the figure can
# distinguish pinning cost from cross-socket cost.
#
# The drop-caches step requires passwordless sudo for
# `/usr/bin/tee /proc/sys/vm/drop_caches`. Verify with
# `sudo -n tee /proc/sys/vm/drop_caches < /dev/null` before kicking
# off the sweep.
SCALING_THREADS_PINNED          ?= 1 2 4 8 16
SCALING_THREADS_UNPINNED        ?= 16 32 64
SCALING_THREADS_TIPTOE_PINNED   ?= 1 8 16
SCALING_THREADS_TIPTOE_UNPINNED ?= 16 64

# physcpubind=0-15 forces socket-0's 16 *physical* cores; verified
# against `lscpu --extended` on sacs006 — adjust on a different box.
NUMACTL_PINNED ?= numactl --physcpubind=0-15 --membind=0
DROP_CACHES    := sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

# Single representative config per scheme: nprobe=32 for IVF, q=3 for
# Tiptoe, verification=on for BN. The threading axis is the variable;
# the quality knob is fixed.
eval-scaling-no-tiptoe: $(DATA)/ground_truth.ivecs
	mkdir -p $(RESULTS)
	@for N in $(SCALING_THREADS_PINNED); do \
	    echo "=== threads=$$N pinned (physcpubind=0-15 membind=0) ==="; \
	    $(DROP_CACHES); \
	    NUMACTL_BINDING="physcpubind=0-15,membind=0" \
	      RAYON_NUM_THREADS=$$N $(NUMACTL_PINNED) $(MAKE) eval-plaintext NPROBE=32 REPS=1; \
	    $(DROP_CACHES); \
	    NUMACTL_BINDING="physcpubind=0-15,membind=0" \
	      RAYON_NUM_THREADS=$$N $(NUMACTL_PINNED) $(MAKE) eval-sap-ivf-nprobe NPROBE=32 REPS=1; \
	    $(DROP_CACHES); \
	    NUMACTL_BINDING="physcpubind=0-15,membind=0" \
	      RAYON_NUM_THREADS=$$N $(NUMACTL_PINNED) $(MAKE) eval-emvp-ivf NPROBE=32 REPS=1; \
	    $(DROP_CACHES); \
	    NUMACTL_BINDING="physcpubind=0-15,membind=0" \
	      RAYON_NUM_THREADS=$$N $(NUMACTL_PINNED) $(MAKE) eval-bntm-ivf NPROBE=32 REPS=1; \
	done
	@for N in $(SCALING_THREADS_UNPINNED); do \
	    echo "=== threads=$$N unpinned ==="; \
	    $(DROP_CACHES); \
	    NUMACTL_BINDING="none" RAYON_NUM_THREADS=$$N $(MAKE) eval-plaintext NPROBE=32 REPS=1; \
	    $(DROP_CACHES); \
	    NUMACTL_BINDING="none" RAYON_NUM_THREADS=$$N $(MAKE) eval-sap-ivf-nprobe NPROBE=32 REPS=1; \
	    $(DROP_CACHES); \
	    NUMACTL_BINDING="none" RAYON_NUM_THREADS=$$N $(MAKE) eval-emvp-ivf NPROBE=32 REPS=1; \
	    $(DROP_CACHES); \
	    NUMACTL_BINDING="none" RAYON_NUM_THREADS=$$N $(MAKE) eval-bntm-ivf NPROBE=32 REPS=1; \
	done

# Tiptoe scaling, separated because each step is ~85 min.
eval-scaling-tiptoe: $(DATA)/ground_truth.ivecs tools/tiptoe-go-rev tools/tiptoe-go.patch
	mkdir -p $(RESULTS)
	@for N in $(SCALING_THREADS_TIPTOE_PINNED); do \
	    echo "=== threads=$$N pinned (RAYON_NUM_THREADS, GOMAXPROCS) ==="; \
	    $(DROP_CACHES); \
	    NUMACTL_BINDING="physcpubind=0-15,membind=0" \
	      RAYON_NUM_THREADS=$$N $(NUMACTL_PINNED) $(MAKE) eval-tiptoe TIPTOE_QUANTISATION_BITS=3 REPS=1; \
	    $(DROP_CACHES); \
	    NUMACTL_BINDING="physcpubind=0-15,membind=0" \
	      GOMAXPROCS=$$N $(NUMACTL_PINNED) $(MAKE) eval-tiptoe-go TIPTOE_GO_QUANTISATION_BITS=3; \
	done
	@for N in $(SCALING_THREADS_TIPTOE_UNPINNED); do \
	    echo "=== threads=$$N unpinned ==="; \
	    $(DROP_CACHES); \
	    NUMACTL_BINDING="none" RAYON_NUM_THREADS=$$N $(MAKE) eval-tiptoe TIPTOE_QUANTISATION_BITS=3 REPS=1; \
	    $(DROP_CACHES); \
	    NUMACTL_BINDING="none" GOMAXPROCS=$$N $(MAKE) eval-tiptoe-go TIPTOE_GO_QUANTISATION_BITS=3; \
	done

eval-scaling: eval-scaling-no-tiptoe eval-scaling-tiptoe

# ── GPU eval ──────────────────────────────────────────────────────────────────
# `eval-gpu-workstation` and `eval-gpu-cloud` run the eval suite with
# `--device gpu` on every measured scheme that supports it (plaintext,
# sap, sap-ivf, emvp, emvp-ivf, bntm, bntm-ivf — i.e. `eval-no-tiptoe`).
# Tiptoe and tiptoe-go are excluded by construction: Tiptoe-GPU is
# deferred and reported via the analytical proxy
# (`effective_bytes_per_query`) only.
#
# The conda-resident RAPIDS env must be active
# so cuvs / nvcc are on PATH; without it the build fails with a clear
# error pointing at the env recipe. The harness's `--gpu-sku` /
# `--gpu-location` / `--cloud-*` flags map onto Make variables so an
# operator can override per-run.
GPU_SKU_WORKSTATION   ?= rtx-5000-ada
GPU_SKU_CLOUD         ?= v100
# Cloud provenance: required iff `--gpu-location cloud`. The
# operator running the cloud sweep should override these to match
# the actual rental — provider / instance type vary by SKU
# (e.g. AWS `p3.2xlarge` for V100, `p5.48xlarge` for H100).
PROVIDER              ?= aws
INSTANCE              ?= p3.2xlarge
REGION                ?= us-east-1
DRIVER_VERSION        ?= 580.82.07
CUDA_VERSION          ?= 12.4

eval-gpu-workstation:
	$(MAKE) eval-no-tiptoe CARGO_FEATURES=gpu EVAL_FLAGS="--device gpu --gpu-sku $(GPU_SKU_WORKSTATION) --gpu-location local"

# Two-card parallel variant: splits the seven `eval-no-tiptoe` schemes
# across two GPUs pinned via CUDA_VISIBLE_DEVICES. Pre-builds the eval
# binary serially first so the parallel `cargo run` invocations don't
# race on the target-dir build lock. Card-0 group ends up CPU-light
# (plaintext + the EMVP pair); card-1 carries the BN pair plus the two
# SAP variants. Outputs from the two halves interleave on stdout.
# Use this on sacs006 (or any host with two RTX cards visible at
# indices 0 and 1); the single-card `eval-gpu-workstation` above is
# the fallback.
eval-gpu-workstation-2card:
	$(CARGO) build --release --features gpu --bin eval
	$(MAKE) -j2 _eval-gpu-card0 _eval-gpu-card1

_eval-gpu-card0:
	CUDA_VISIBLE_DEVICES=0 $(MAKE) eval-plaintext eval-emvp eval-emvp-ivf \
	    CARGO_FEATURES=gpu \
	    EVAL_FLAGS="--device gpu --gpu-sku $(GPU_SKU_WORKSTATION) --gpu-location local"

_eval-gpu-card1:
	CUDA_VISIBLE_DEVICES=1 $(MAKE) eval-sap eval-sap-ivf eval-bntm eval-bntm-ivf \
	    CARGO_FEATURES=gpu \
	    EVAL_FLAGS="--device gpu --gpu-sku $(GPU_SKU_WORKSTATION) --gpu-location local"

eval-gpu-cloud:
	$(MAKE) eval-no-tiptoe CARGO_FEATURES=gpu EVAL_FLAGS="--device gpu --gpu-sku $(GPU_SKU_CLOUD) --gpu-location cloud --cloud-provider $(PROVIDER) --cloud-instance-type $(INSTANCE) --cloud-region $(REGION) --cloud-driver-version $(DRIVER_VERSION) --cloud-cuda-version $(CUDA_VERSION)"

# Drives the Go reference's paired-runner. Single q-bits value per
# invocation (the Go subprocess preprocess is per-corpus, not per-config);
# repeat with different TIPTOE_GO_QUANTISATION_BITS to sweep.
eval-tiptoe-go: $(DATA)/ground_truth.ivecs tools/tiptoe-go-rev tools/tiptoe-go.patch
	mkdir -p $(RESULTS)
	$(CARGO) run --release --bin tiptoe_go_runner -- \
	    --data-dir          $(DATA) \
	    --k                 $(EVAL_K) \
	    --quantisation-bits $(TIPTOE_GO_QUANTISATION_BITS) \
	    --hint-mb           $(TIPTOE_GO_HINT_MB) \
	    --gpu-kind          $(GPU_KIND) \
	    --results-dir       $(RESULTS)

# Validation gate: compares the most-recent tiptoe and tiptoe-go runs
# on this machine. Pass: recall@10 within 2pp AND mean top-10 ID
# overlap ≥ 80%. Non-zero exit on fail.
#
# --require-data: this target is for an operator on a HYDRATED checkout, so
# "no paired data" is a hard failure (exit 1), not a skip — the gate must
# actually run a comparison here. (CI runs without the flag, since its
# checkout has no bulk-stored CSVs and would otherwise be permanently red;
# CI surfaces the skip as a loud warning instead.)
tiptoe-diff:
	$(PYTHON) analysis/tiptoe_diff.py --results-dir $(RESULTS) --require-data

# ── analysis ──────────────────────────────────────────────────────────────────

analysis:
	$(MAKE) -C analysis figures $(if $(MACHINE),MACHINE=$(MACHINE),) $(if $(N_PASSAGES),N_PASSAGES=$(N_PASSAGES),)

# `figures` is the canonical name (analysis/Makefile uses it too).
# `analysis` is kept as a back-compat alias for the older muscle
# memory + scripts that call `make analysis`.
figures:
	$(MAKE) -C analysis figures $(if $(MACHINE),MACHINE=$(MACHINE),) $(if $(N_PASSAGES),N_PASSAGES=$(N_PASSAGES),)

# Mirror of `analysis:` for the prerequisite step — emits the canonical
# results/aggregated/<machine-id>/*.tsv TSVs from results/runs/<machine-id>
# raw.csv files. Required precondition for `scripts/upload_bulk.py`.
preprocess:
	$(MAKE) -C analysis preprocess $(if $(MACHINE),MACHINE=$(MACHINE),) $(if $(N_PASSAGES),N_PASSAGES=$(N_PASSAGES),)

report:
	$(MAKE) -C analysis report \
		$(if $(RUN),RUN=$(RUN),) \
		$(if $(MACHINE),MACHINE=$(MACHINE),)

# End-of-run convenience: preprocess → upload_bulk (per run dir) →
# figures → per-machine report.pdf, in that order. Campaign backfill
# is intentionally NOT bundled — campaign-id is operator-decided per
# sweep, so the operator runs scripts/backfill_campaign.py
# separately when needed before this target.
#
# Usage:
#   make finalize MACHINE=<machine-id>
#
# Fail-loud on missing MACHINE so a typo doesn't silently fall
# through to analysis/Makefile's `ls -1 results/runs | head -1`
# default (which is fine inside analysis/ on a single-machine box,
# but here we're explicitly per-machine).
finalize:
	@if [ -z "$(MACHINE)" ]; then \
	    echo "usage: make finalize MACHINE=<machine-id>" >&2; \
	    echo "  pipeline: preprocess → upload_bulk → figures → report" >&2; \
	    exit 2; \
	fi
	@echo "===> finalize $(MACHINE) :: step 1/4 preprocess"
	$(MAKE) preprocess MACHINE=$(MACHINE) $(if $(N_PASSAGES),N_PASSAGES=$(N_PASSAGES),)
	@echo "===> finalize $(MACHINE) :: step 2/4 upload_bulk per run-dir"
	@found=0; for d in results/runs/$(MACHINE)/*/*/ ; do \
	    if [ ! -f "$$d/run-metadata.toml" ]; then continue; fi; \
	    found=$$((found + 1)); \
	    echo "  ===> $$d"; \
	    scripts/upload_bulk.py "$$d" || exit $$?; \
	done; \
	if [ "$$found" -eq 0 ]; then \
	    echo "  warning: no run dirs found under results/runs/$(MACHINE)/" >&2; \
	fi
	@echo "===> finalize $(MACHINE) :: step 3/4 figures"
	$(MAKE) figures MACHINE=$(MACHINE) $(if $(N_PASSAGES),N_PASSAGES=$(N_PASSAGES),)
	@echo "===> finalize $(MACHINE) :: step 4/4 report"
	$(PYTHON) analysis/report.py --results $(RESULTS) --machine $(MACHINE)
	@echo "===> finalize $(MACHINE) :: complete"

# End-to-end paper-eval suite: every phase needed to fill out the
# per-machine report.pdf, plus the report rebuild itself. Slack-
# notified at start / each phase failure / the (optional) sudo
# pause / completion. Webhook URL resolved by deploy/notify.sh
# (env > .slack-webhook > $HOME/.config/secure-vector-search/).
# `SKIP_SCALING=1` skips eval-scaling on hosts without the
# drop_caches sudoers entry; `SKIP_REPORT=1` skips the closing
# `make report`.
eval-suite:
	scripts/eval-suite.sh

# Exploratory cross-machine PDF dossier (NOT paper figures). Walks every
# results/runs/**/raw.csv, joins with machines.csv + per-run
# run-metadata.toml, emits per-(figure × series) TSVs + auto-generated
# tikz fragments, then assembles them into one cross_machine.pdf via
# pgfplots/latexmk. Output: analysis/cross_machine/output/cross_machine.pdf.
cross_machine:
	$(MAKE) -C analysis/cross_machine INVOCATION_DIR=$(CURDIR)

# ── utilities ─────────────────────────────────────────────────────────────────

build:
	$(CARGO) build --release

# Remove run directories and analysis outputs (fast to regenerate).
# Legacy CSVs in results/legacy/ are preserved.
clean:
	rm -rf $(RESULTS)/runs $(RESULTS)/index.csv $(RESULTS)/machines.csv
	$(MAKE) -C analysis clean

# Also remove derived data (requires re-embedding; slow).
distclean: clean
	rm -rf $(DATA)
