#!/usr/bin/env bash
# Runs the full paper-eval bundle on a CPU pod (every pass listed in the
# README under "Run every evaluation needed for the paper" + the Plan 23
# W3 batch sweep) and Slack-posts at start, per phase start/finish, on
# each phase failure, and on overall completion. Designed to be the
# argument to deploy/runai-submit-cpu.sh — the cd into $SCRATCH and the
# UID/GID context are already set up by RunAI.
#
# Mirrors scripts/eval-suite.sh's `run_phase` shape (numbered [N/T]
# progress markers + Slack notify per phase), minus the eval-scaling +
# sudo gate (pods are non-interactive and skip eval-scaling per the
# bundle's "optional ~8h, needs passwordless sudo" framing).
#
# Phases (default order):
#   1. make eval                                       — figs 01-06, 10
#   2. make eval-cold                                  — fig 08
#   3. make eval-breakdown                             — figs 09a/09b
#   4. make eval-bntm  BNTM_VERIFICATION=false         — fig 13 (verify-off B=1)
#   5. make eval-bntm-ivf BNTM_VERIFICATION=false      — fig 13 (verify-off B=1)
#   6. make tiptoe-diff                                — Tiptoe Rust vs Go gate
#   7. make eval-batch                                 — Plan 23 W3 (fig 14, verify-on)
#   8. make eval-bntm BNTM_VERIFICATION=false
#         EVAL_FLAGS='--batch-sizes 1,8,64,256'        — fig 14 BN-no-verify line
#   9. make eval-bntm-ivf BNTM_VERIFICATION=false
#         EVAL_FLAGS='--batch-sizes 1,8,64,256'        — fig 14 BN-IVF-no-verify line
#
# Skip-phase env-vars (set to non-empty to skip):
#   SKIP_COLD       — skip phase 2 (`make eval-cold`)
#   SKIP_BREAKDOWN  — skip phase 3 (`make eval-breakdown`)
#   SKIP_BATCH      — skip phases 7-9 (Plan 23 W3 batched sweep)
#   SKIP_TIPTOE     — skip phase 6 (`make tiptoe-diff`)
#
# Slack notifications fire only when SLACK_WEBHOOK is set (forwarded
# into the pod by deploy/runai-submit-cpu.sh when the env var is
# exported on the host).

set -uo pipefail

# In the toolchain image Python is installed system-wide; there is no ./venv.
# The Makefile defaults to venv/bin/python via `?=`, so exporting overrides it.
export PYTHON="${PYTHON:-python3}"

# Resolve SLACK_WEBHOOK + provide notify(); env > .slack-webhook > $HOME.
# shellcheck source=deploy/notify.sh
source "$(dirname "${BASH_SOURCE[0]}")/notify.sh"

JOB_NAME="${RUNAI_JOB_NAME:-eval-bundle}"
START_TS=$(date +%s)

# Pinned tiptoe-go SHA check (the harness errors out with clone
# instructions if the ref repo is missing; on the cluster the bundle
# should self-heal so a single submission completes end-to-end).
ensure_tiptoe_go() {
  local rev_file="tools/tiptoe-go-rev"
  local repo_dir="tmp/tiptoe-go"
  local sha
  sha=$(awk '/^ahenzinger\/tiptoe[[:space:]]+/{print $2; exit}' "$rev_file")
  if [ -z "$sha" ]; then
    echo "fatal: pinned ahenzinger/tiptoe SHA missing from $rev_file" >&2
    return 2
  fi
  if [ ! -d "$repo_dir/.git" ]; then
    echo "===> $(date -Iseconds) cloning ahenzinger/tiptoe into $repo_dir"
    mkdir -p "$(dirname "$repo_dir")"
    git clone https://github.com/ahenzinger/tiptoe.git "$repo_dir" || return $?
    git -C "$repo_dir" checkout --quiet "$sha" || return $?
  elif [ "$(git -C "$repo_dir" rev-parse HEAD)" != "$sha" ]; then
    echo "===> $(date -Iseconds) updating tiptoe-go pin to $sha"
    git -C "$repo_dir" fetch --quiet || return $?
    git -C "$repo_dir" checkout --quiet "$sha" || return $?
  fi
}

ensure_tiptoe_go || {
  status=$?
  echo "fatal: ensure_tiptoe_go failed (exit $status); aborting bundle" >&2
  exit "$status"
}

# Build the phase list. Parallel arrays: labels (one string per phase
# for Slack) and commands (one tab-separated argv string per phase).
# `add_phase "label" cmd args…` packs argv with TABs, which `run_phase`
# unpacks via `IFS=$'\t' read -r -a argv` — safe because Makefile
# targets never contain TABs in their arg values.
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

add_phase "make eval"                                       make eval
[ -z "${SKIP_COLD:-}" ] && \
  add_phase "make eval-cold"                                make eval-cold
[ -z "${SKIP_BREAKDOWN:-}" ] && \
  add_phase "make eval-breakdown"                           make eval-breakdown
add_phase "make eval-bntm BNTM_VERIFICATION=false"          make eval-bntm BNTM_VERIFICATION=false
add_phase "make eval-bntm-ivf BNTM_VERIFICATION=false"      make eval-bntm-ivf BNTM_VERIFICATION=false
[ -z "${SKIP_TIPTOE:-}" ] && \
  add_phase "make tiptoe-diff"                              make tiptoe-diff
if [ -z "${SKIP_BATCH:-}" ]; then
  add_phase "make eval-batch" \
            make eval-batch
  add_phase "make eval-bntm BNTM_VERIFICATION=false EVAL_FLAGS='--batch-sizes 1,8,64,256'" \
            make eval-bntm BNTM_VERIFICATION=false 'EVAL_FLAGS=--batch-sizes 1,8,64,256'
  add_phase "make eval-bntm-ivf BNTM_VERIFICATION=false EVAL_FLAGS='--batch-sizes 1,8,64,256'" \
            make eval-bntm-ivf BNTM_VERIFICATION=false 'EVAL_FLAGS=--batch-sizes 1,8,64,256'
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

notify ":hourglass: \`${JOB_NAME}\` starting CPU paper-eval bundle (${total_phases} phases)"

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
  notify ":white_check_mark: \`${JOB_NAME}\` complete (CPU bundle, ${total_phases} phases, $(fmt_elapsed $ELAPSED))"
else
  notify ":x: \`${JOB_NAME}\` failed at \`${failed_step}\` (exit ${exit_status}, after $(fmt_elapsed $ELAPSED))"
fi

exit $exit_status
