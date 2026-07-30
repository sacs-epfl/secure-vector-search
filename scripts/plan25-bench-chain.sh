#!/usr/bin/env bash
#
# Plan 25 step 5 / 6 / 7 chain — bench + re-measurement.
#
# Run after the W4 BN sweep's SIGTERM watcher fires (i.e., after the
# `target/release/eval --scorer bntm-ivf` process exits and the
# `raw.csv.recovered` streamer closes). Sequence:
#
#   0. Preflight: confirm W4 BN eval is no longer running; confirm
#      `raw.csv.recovered` exists and has expected rows.
#   1. Cleanup: move recovered raw.csv / top_k.csv over the bogus
#      header-only visible files; drop the stale [bulk] block from
#      run-metadata.toml so a future re-upload won't skip; flip
#      `status = "partial"` → `"complete"` so the upload_bulk guard
#      (commit c095ff5) doesn't block.
#   2. Step 6 L2 benches: l2_f32_simd vs scalar, l2_simd vs scalar.
#      Each bench prints a wire-in recommendation; the script does
#      NOT auto-edit source — the operator reviews and (optionally)
#      hand-wires the dispatcher in `ivf-index/src/distance.rs` /
#      `scorer-sap/src/distance.rs`.
#   3. Step 5 EMVP bench: compute_products_avx512 vs scalar. Same
#      review-then-wire workflow.
#   4. Step 7 end-to-end re-measurement: rebuild with `make eval-
#      native`, re-run bntm-ivf at `nprobe=64,256` (matches the
#      in-flight scalar baseline you'll have just cleaned up), and
#      print per-config mean-latency deltas vs that baseline.
#
# Usage:
#   scripts/plan25-bench-chain.sh
#
# Env (defaults shown):
#   DATASET=msmarco-full
#   MACHINE_ID=d860be76
#   BASELINE_RUN_DIR=results/runs/d860be76/3c4a3aacf48747518a3b8337423e7e48ceaef9d7/1779263883
#   NPROBE=64,256
#   SKIP_END_TO_END=0  (set to 1 to bench only, skip step 7)
#
# Cosmetic / non-blocking: the W4 driver's run_phase wrapper around
# `make eval-bntm-ivf` Slack-notified a `:x:` failure when the SIGTERM
# fired — that's expected and unrelated.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET="${DATASET:-msmarco-full}"
MACHINE_ID="${MACHINE_ID:-d860be76}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-results/runs/d860be76/3c4a3aacf48747518a3b8337423e7e48ceaef9d7/1779263883}"
NPROBE="${NPROBE:-64,256}"
SKIP_END_TO_END="${SKIP_END_TO_END:-0}"
LOG_DIR="logs/plan25-bench-chain-$(date +%Y%m%dT%H%M%S)"

mkdir -p "$LOG_DIR"
echo "[chain] logs at $LOG_DIR"

NOTIFY=""
if [ -r deploy/notify.sh ]; then
    # shellcheck source=deploy/notify.sh
    source deploy/notify.sh
    NOTIFY=1
fi
notify_or_echo() {
    if [ -n "$NOTIFY" ]; then notify "$@"; else echo "[notify] $*"; fi
}

# ---------------------------------------------------------------------------
# Step 0: preflight
# ---------------------------------------------------------------------------
step_preflight() {
    echo
    echo "===> step 0 — preflight"
    # Match the binary name exactly (`pgrep -x eval`), not anywhere in
    # the cmdline — otherwise unrelated shell processes whose `eval`
    # argument happens to contain the pattern (e.g. debugging shells
    # from earlier in the session) would false-positive.
    if pgrep -x eval > /dev/null; then
        echo "[chain] ERROR: an `eval` binary is still running. PIDs:"
        pgrep -ax eval
        echo "[chain] Wait for the SIGTERM watcher to fire, then re-run."
        exit 1
    fi
    if [ ! -f "$BASELINE_RUN_DIR/raw.csv.recovered" ]; then
        echo "[chain] ERROR: $BASELINE_RUN_DIR/raw.csv.recovered missing — no recovered data to clean up."
        exit 1
    fi
    local rec_rows
    rec_rows=$(wc -l < "$BASELINE_RUN_DIR/raw.csv.recovered")
    echo "[chain] recovered raw.csv has $rec_rows lines (incl. header)"
    if [ "$rec_rows" -lt 1000 ]; then
        echo "[chain] WARNING: recovered raw.csv has fewer rows than expected — proceeding anyway"
    fi
}

# ---------------------------------------------------------------------------
# Step 1: cleanup
# ---------------------------------------------------------------------------
step_cleanup() {
    echo
    echo "===> step 1 — cleanup W4 partial baseline"
    # 1a. Replace visible raw.csv / top_k.csv with the recovered streamer
    #     output (the visible files are the bogus header-only inodes left
    #     behind by the upload_bulk.py unlink during the morning's
    #     finalize; see commit c095ff5 for the root cause).
    if [ -f "$BASELINE_RUN_DIR/raw.csv.recovered" ]; then
        mv -v "$BASELINE_RUN_DIR/raw.csv" "$BASELINE_RUN_DIR/raw.csv.bogus-header-only.bak"
        mv -v "$BASELINE_RUN_DIR/raw.csv.recovered" "$BASELINE_RUN_DIR/raw.csv"
    fi
    if [ -f "$BASELINE_RUN_DIR/top_k.csv.recovered" ]; then
        mv -v "$BASELINE_RUN_DIR/top_k.csv" "$BASELINE_RUN_DIR/top_k.csv.bogus-header-only.bak"
        mv -v "$BASELINE_RUN_DIR/top_k.csv.recovered" "$BASELINE_RUN_DIR/top_k.csv"
    fi
    # 1b. Drop the [bulk] block + flip status. Run-metadata is TOML;
    #     do a careful in-place edit. The block is contiguous and ends
    #     before the next `[section]` header.
    local meta="$BASELINE_RUN_DIR/run-metadata.toml"
    if [ ! -f "$meta" ]; then
        echo "[chain] ERROR: missing $meta"
        exit 1
    fi
    venv/bin/python - "$meta" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
text = p.read_text()
# Drop the entire [bulk] block (and any [bulk.subsection] under it).
new = re.sub(
    r"\n\[bulk\][^\[]*(?:\[\[bulk\.[^\]]+\]\][^\[]*)*",
    "\n",
    text,
)
# Flip status = "partial" → "complete".
new = re.sub(r'^status\s*=\s*"partial"', 'status = "complete"', new, flags=re.M)
if new == text:
    sys.stderr.write("no [bulk] block or partial status found — nothing changed\n")
p.write_text(new)
PY
    echo "[chain] cleanup complete: $BASELINE_RUN_DIR"
    notify_or_echo ":broom: Plan 25 chain: cleanup done on \`${BASELINE_RUN_DIR}\`"
}

# ---------------------------------------------------------------------------
# Step 6: L2 benches (Family B). Records exit codes; does NOT auto-wire.
# ---------------------------------------------------------------------------
step_l2_benches() {
    echo
    echo "===> step 6a — l2_f32 (ivf-index) bench"
    local rc
    RUSTFLAGS="-C target-cpu=native" cargo run --release \
        --example l2_f32_simd_bench -p ivf-index 2>&1 | tee "$LOG_DIR/l2_f32_simd_bench.log"
    rc=${PIPESTATUS[0]}
    echo "[chain] l2_f32 bench rc=$rc (0 = wire SIMD in, 1 = keep scalar, 2 = no AVX-512)"

    echo
    echo "===> step 6b — l2 (scorer-sap) bench"
    RUSTFLAGS="-C target-cpu=native" cargo run --release \
        --example l2_simd_bench -p scorer-sap 2>&1 | tee "$LOG_DIR/l2_simd_bench.log"
    rc=${PIPESTATUS[0]}
    echo "[chain] l2 bench rc=$rc (0 = wire SIMD in, 1 = keep scalar, 2 = no AVX-512)"

    notify_or_echo ":chart_with_upwards_trend: Plan 25 step 6 L2 benches done — see \`${LOG_DIR}/l2*.log\` for the wire-in recommendation"
}

# ---------------------------------------------------------------------------
# Step 5: EMVP bench (Family A site 5). Records exit; does NOT auto-wire.
# ---------------------------------------------------------------------------
step_emvp_bench() {
    echo
    echo "===> step 5 — EMVP compute_products bench"
    local rc
    RUSTFLAGS="-C target-cpu=native" cargo run --release \
        --example emvp_compute_products_simd_bench -p scorer-emvp 2>&1 \
        | tee "$LOG_DIR/emvp_compute_products_simd_bench.log"
    rc=${PIPESTATUS[0]}
    echo "[chain] emvp bench rc=$rc (0 = wire SIMD in @ ≥1.355×, 1 = keep scalar, 2 = no AVX-512)"

    notify_or_echo ":chart_with_upwards_trend: Plan 25 step 5 EMVP bench done — see \`${LOG_DIR}/emvp_compute_products_simd_bench.log\`"
}

# ---------------------------------------------------------------------------
# Step 7: end-to-end re-measurement. Re-runs bntm-ivf at NPROBE on a
#         native build; compares per-config mean latency to the
#         cleaned-up scalar baseline above.
# ---------------------------------------------------------------------------
step_end_to_end() {
    if [ "$SKIP_END_TO_END" = "1" ]; then
        echo "[chain] SKIP_END_TO_END=1 — skipping step 7"
        return 0
    fi
    echo
    echo "===> step 7 — end-to-end re-measurement (native build, NPROBE=${NPROBE})"
    notify_or_echo ":hourglass: Plan 25 step 7: end-to-end bntm-ivf re-run starting (NPROBE=${NPROBE})"

    # Rebuild + re-run with native flags. The eval harness writes a
    # new run dir under results/runs/<machine>/<sha>/<run-id>/.
    local before_run_ids
    before_run_ids=$(ls "results/runs/${MACHINE_ID}/"*/[0-9]*/ 2>/dev/null | sort)
    if ! RUSTFLAGS="-C target-cpu=native" make eval-bntm-ivf \
            DATASET="$DATASET" \
            NPROBE="$NPROBE" \
            BNTM_VERIFICATION=false \
            2>&1 | tee "$LOG_DIR/step7-eval.log"; then
        notify_or_echo ":x: Plan 25 step 7 failed during eval — see \`${LOG_DIR}/step7-eval.log\`"
        exit 1
    fi

    # Locate the new run dir (the latest under MACHINE_ID).
    local new_run_dir
    new_run_dir=$(ls -dt "results/runs/${MACHINE_ID}/"*/[0-9]*/ 2>/dev/null | head -1)
    if [ -z "$new_run_dir" ] || [ ! -f "${new_run_dir}/raw.csv" ]; then
        notify_or_echo ":x: Plan 25 step 7: cannot locate new run dir (or its raw.csv)"
        exit 1
    fi
    echo "[chain] new run dir: $new_run_dir"

    # Compare per-(nprobe, rep) wallclock against the baseline.
    venv/bin/python - "$BASELINE_RUN_DIR/raw.csv" "${new_run_dir}/raw.csv" \
        > "$LOG_DIR/step7-comparison.txt" <<'PY'
import csv, statistics, sys
baseline_path, simd_path = sys.argv[1], sys.argv[2]

def per_config_lat(path):
    by_label = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            by_label.setdefault(row["config-label"], []).append(float(row["latency-us"]))
    return {k: statistics.fmean(v) for k, v in by_label.items()}

base = per_config_lat(baseline_path)
simd = per_config_lat(simd_path)
shared = sorted(set(base.keys()) & set(simd.keys()))
print(f"=== Plan 25 step 7 — per-config wallclock (mean latency-us, lower is better) ===")
print(f"  baseline scalar : {baseline_path}")
print(f"  SIMD native     : {simd_path}")
print()
print(f"  {'config':<24} {'scalar µs':>12} {'simd µs':>12} {'speedup':>10}")
for label in shared:
    sp = base[label] / simd[label] if simd[label] > 0 else float("nan")
    print(f"  {label:<24} {base[label]:>12.1f} {simd[label]:>12.1f} {sp:>9.3f}×")
PY
    cat "$LOG_DIR/step7-comparison.txt"
    notify_or_echo ":white_check_mark: Plan 25 step 7 done — see \`${LOG_DIR}/step7-comparison.txt\` for the per-config speedup table"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
notify_or_echo ":arrow_forward: Plan 25 chain starting (DATASET=${DATASET}, NPROBE=${NPROBE})"
step_preflight
step_cleanup
step_l2_benches
step_emvp_bench
step_end_to_end
notify_or_echo ":white_check_mark: Plan 25 chain complete — logs at \`${LOG_DIR}\`"
echo
echo "[chain] complete. Inspect:"
echo "  $LOG_DIR/l2_f32_simd_bench.log         (step 6a recommendation)"
echo "  $LOG_DIR/l2_simd_bench.log             (step 6b recommendation)"
echo "  $LOG_DIR/emvp_compute_products_simd_bench.log  (step 5 recommendation)"
echo "  $LOG_DIR/step7-comparison.txt          (step 7 per-config speedup)"
