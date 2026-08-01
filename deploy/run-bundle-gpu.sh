#!/usr/bin/env bash
# Runs the GPU paper-eval bundle on a GPU pod (the subset of evals that
# exercise the cuVS / custom-CUDA paths) and
# Slack-posts at start, per phase start/finish, on each phase failure,
# and on overall completion. Designed to be the argument to
# deploy/runai-submit-gpu-cloud.sh (or deploy/runai-submit.sh directly).
#
# Phases (default order):
#   1. make eval-gpu-cloud                             — figs 01-04, 10 (GPU paths)
#   2. make eval-breakdown-no-tiptoe EVAL_FLAGS=…      — figs 09a/09b (GPU substep)
#
# Skip-phase env-vars (non-empty to skip):
#   SKIP_BREAKDOWN  — skip phase 2 (`eval-breakdown-no-tiptoe` on GPU)
#
# Cloud-provenance env-vars (defaults match the project's RCP V100
# workspace; override at submit time for a different SKU / cluster):
#   GPU_SKU_CLOUD    SKU passed to --gpu-sku                  (default: v100)
#   PROVIDER         --cloud-provider                          (default: epfl-rcp)
#   INSTANCE         --cloud-instance-type                     (default: rcp-runai-v100)
#   REGION           --cloud-region                            (default: epfl)
#   DRIVER_VERSION   --cloud-driver-version                    (default: auto-detect via nvidia-smi)
#   CUDA_VERSION     --cloud-cuda-version                      (default: auto-detect via nvcc)
#
# ── GPU 2h-idle reaper mitigation ──────────────────────────────────────
# RCP's non-interactive (Train) workloads reclaim the pod after 2h of
# zero GPU utilisation. Most eval phases hit the GPU per query (well
# under 2h between touches), but the initial `cargo build --release`
# + k-means cluster build on a cold cache is CPU-bound and can stretch
# past 2h on a fresh pod. A background keep-alive sidecar does a tiny
# torch.cuda op every 25 min — enough to register utilisation above
# the reaper's threshold without perturbing the eval. Killed on script
# exit via trap.
#
# Slack notifications fire only when SLACK_WEBHOOK is set (forwarded
# into the pod by deploy/runai-submit-gpu-cloud.sh when the env var
# is exported on the host).

set -uo pipefail

export PYTHON="${PYTHON:-python3}"

# shellcheck source=deploy/notify.sh
source "$(dirname "${BASH_SOURCE[0]}")/notify.sh"

JOB_NAME="${RUNAI_JOB_NAME:-eval-bundle-gpu}"
START_TS=$(date +%s)

# ── GPU keep-alive sidecar ─────────────────────────────────────────────
# Touches the GPU every 25 min via a 1024-element torch.cuda.sum() —
# microseconds of actual work, but registers SM utilisation > 0 on the
# RCP reaper's polling window. The `2>/dev/null || true` swallows
# transient errors (e.g. driver mid-restart) so a single failed touch
# can't kill the loop. Killed on script exit via trap.
KEEPALIVE_PID=""
start_keepalive() {
  (
    while true; do
      $PYTHON - <<'PY' 2>/dev/null || true
import torch
_ = torch.randn(1024, device='cuda').sum().item()
PY
      sleep 1500
    done
  ) &
  KEEPALIVE_PID=$!
  echo "===> $(date -Iseconds) GPU keep-alive started (PID $KEEPALIVE_PID, 25 min interval)"
}

stop_keepalive() {
  if [ -n "${KEEPALIVE_PID:-}" ] && kill -0 "$KEEPALIVE_PID" 2>/dev/null; then
    kill "$KEEPALIVE_PID" 2>/dev/null || true
    echo "===> $(date -Iseconds) GPU keep-alive stopped (PID $KEEPALIVE_PID)"
  fi
}
trap stop_keepalive EXIT

start_keepalive

# ── cloud-provenance defaults ──────────────────────────────────────────
GPU_SKU_CLOUD="${GPU_SKU_CLOUD:-v100}"
PROVIDER="${PROVIDER:-epfl-rcp}"
INSTANCE="${INSTANCE:-rcp-runai-v100}"
REGION="${REGION:-epfl}"

if [ -z "${DRIVER_VERSION:-}" ]; then
  DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  DRIVER_VERSION="${DRIVER_VERSION:-unknown}"
fi
if [ -z "${CUDA_VERSION:-}" ]; then
  CUDA_VERSION=$(nvcc --version 2>/dev/null | awk -F', ' '/release/ {sub(/release /,"",$2); print $2}' | head -1)
  CUDA_VERSION="${CUDA_VERSION:-unknown}"
fi
echo "===> $(date -Iseconds) GPU SKU=${GPU_SKU_CLOUD} provider=${PROVIDER} instance=${INSTANCE} region=${REGION} driver=${DRIVER_VERSION} cuda=${CUDA_VERSION}"

# ── tiptoe-go pin (mirror run-bundle.sh) ───────────────────────────────
# Not strictly required by the GPU bundle today (eval-gpu-cloud routes
# through eval-no-tiptoe and skips Tiptoe entirely),
# but kept here so a future operator who adds a Tiptoe-GPU path
# doesn't trip the harness's missing-clone error mid-bundle.
ensure_tiptoe_go() {
  local rev_file="tools/tiptoe-go-rev"
  local repo_dir="tmp/tiptoe-go"
  local sha
  sha=$(awk '/^ahenzinger\/tiptoe[[:space:]]+/{print $2; exit}' "$rev_file")
  if [ -z "$sha" ]; then
    echo "===> $(date -Iseconds) tiptoe-go SHA not pinned in $rev_file; skipping ref clone"
    return 0
  fi
  if [ ! -d "$repo_dir/.git" ]; then
    echo "===> $(date -Iseconds) cloning ahenzinger/tiptoe into $repo_dir"
    mkdir -p "$(dirname "$repo_dir")"
    git clone https://github.com/ahenzinger/tiptoe.git "$repo_dir" || return $?
    git -C "$repo_dir" checkout --quiet "$sha" || return $?
  elif [ "$(git -C "$repo_dir" rev-parse HEAD)" != "$sha" ]; then
    git -C "$repo_dir" fetch --quiet || return $?
    git -C "$repo_dir" checkout --quiet "$sha" || return $?
  fi
}
ensure_tiptoe_go || true

# ── phase list ─────────────────────────────────────────────────────────
phase_labels=()
phase_argvs=()

add_phase() {
  local label="$1"; shift
  phase_labels+=("$label")
  local argv="$1"; shift
  for arg in "$@"; do
    argv+=$'\t'"$arg"
  done
  phase_argvs+=("$argv")
}

# Phase 1 — eval-gpu-cloud: existing Makefile target. Reads
# GPU_SKU_CLOUD, PROVIDER, INSTANCE, REGION, DRIVER_VERSION,
# CUDA_VERSION as Makefile variables (Makefile § eval-gpu-cloud).
add_phase "make eval-gpu-cloud (V100, epfl-rcp)" \
          make eval-gpu-cloud \
          "GPU_SKU_CLOUD=${GPU_SKU_CLOUD}" \
          "PROVIDER=${PROVIDER}" \
          "INSTANCE=${INSTANCE}" \
          "REGION=${REGION}" \
          "DRIVER_VERSION=${DRIVER_VERSION}" \
          "CUDA_VERSION=${CUDA_VERSION}"

# Phase 2 — substep breakdown on GPU substrate. Threads
# CARGO_FEATURES=gpu and full --device gpu + cloud-* fields through
# EVAL_FLAGS (the breakdown sub-targets accept EVAL_FLAGS).
if [ -z "${SKIP_BREAKDOWN:-}" ]; then
  GPU_EVAL_FLAGS="--device gpu --gpu-sku ${GPU_SKU_CLOUD} --gpu-location cloud --cloud-provider ${PROVIDER} --cloud-instance-type ${INSTANCE} --cloud-region ${REGION} --cloud-driver-version ${DRIVER_VERSION} --cloud-cuda-version ${CUDA_VERSION}"
  add_phase "make eval-breakdown-no-tiptoe (GPU)" \
            make eval-breakdown-no-tiptoe \
            CARGO_FEATURES=gpu \
            "EVAL_FLAGS=${GPU_EVAL_FLAGS}"
fi

total_phases=${#phase_labels[@]}

fmt_elapsed() {
  local secs=$1
  printf '%dh %02dm' $((secs / 3600)) $(((secs % 3600) / 60))
}

run_phase() {
  local idx=$1 label=$2 argv_packed=$3
  local tag="[${idx}/${total_phases}]"
  local phase_start
  phase_start=$(date +%s)
  local suite_elapsed=$((phase_start - START_TS))
  echo
  echo "===> $(date -Iseconds) ${tag} ${label}"
  notify ":arrow_forward: \`${JOB_NAME}\` ${tag} starting \`${label}\` (suite elapsed $(fmt_elapsed $suite_elapsed))"
  local -a argv=()
  IFS=$'\t' read -r -a argv <<< "$argv_packed"
  if ! "${argv[@]}"; then
    local elapsed=$(( $(date +%s) - START_TS ))
    notify ":x: \`${JOB_NAME}\` ${tag} failed at \`${label}\` (after $(fmt_elapsed $elapsed))"
    return 1
  fi
  local phase_elapsed=$(( $(date +%s) - phase_start ))
  local total_elapsed=$(( $(date +%s) - START_TS ))
  notify ":white_check_mark: \`${JOB_NAME}\` ${tag} \`${label}\` done in $(fmt_elapsed $phase_elapsed) (suite elapsed $(fmt_elapsed $total_elapsed))"
}

notify ":hourglass: \`${JOB_NAME}\` starting GPU paper-eval bundle (${total_phases} phases, SKU=${GPU_SKU_CLOUD}, keep-alive armed)"

failed_step=""
exit_status=0
for i in "${!phase_labels[@]}"; do
  if ! run_phase $((i + 1)) "${phase_labels[$i]}" "${phase_argvs[$i]}"; then
    exit_status=$?
    failed_step="${phase_labels[$i]}"
    break
  fi
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

if [ -z "$failed_step" ]; then
  notify ":white_check_mark: \`${JOB_NAME}\` complete (GPU bundle, ${total_phases} phases, $(fmt_elapsed $ELAPSED))"
else
  notify ":x: \`${JOB_NAME}\` failed at \`${failed_step}\` (exit ${exit_status}, after $(fmt_elapsed $ELAPSED))"
fi

# Keep-alive killed by EXIT trap.
exit $exit_status
