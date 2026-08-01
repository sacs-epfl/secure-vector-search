#!/usr/bin/env bash
# Submit a RunAI batch job on EPFL RCP/CaaS that runs the given command inside
# the secure-vector-search toolchain image.
#
# Storage model: stage the repo + data on NAS3 once via the jumphost — see
# deploy/Dockerfile header — and have RunAI mount the corresponding PVC into
# every pod. The PVC `sacs-scratch` (bound to `sacs-scratch-rpereira`) maps
# to `/mnt/sacs/scratch` inside the pod by default.
#
# Usage:
#   deploy/runai-submit.sh make eval
#   deploy/runai-submit.sh bash -lc 'make eval && make report'
#   GPU=2 NAME=svs-bigeval deploy/runai-submit.sh make eval
#
# Env-var overrides:
#   IMAGE          full image reference
#                  (default: registry.rcp.epfl.ch/sacs/rpereira/secure-vector-search:latest)
#   RUN_AS_UID     pod UID; must match the on-NAS3 owner    (default: 224954)
#   RUN_AS_GID     pod primary GID                          (default: 11259)
#   SCRATCH_CLAIM  PVC name for NAS3 scratch                (default: sacs-scratch)
#   SCRATCH_MOUNT  in-pod mount point for the PVC           (default: /mnt/sacs/scratch)
#   SCRATCH        in-pod path of the staged repo           (default: $SCRATCH_MOUNT/secure-vector-search)
#   HF_CACHE       HuggingFace model cache path             (default: $SCRATCH/.hf-cache)
#   GPU         GPUs requested            (default: 1)
#   CPU         CPU cores requested       (default: 8)
#   MEMORY      memory request            (default: 32G)
#   NAME        job name                  (default: svs-eval-<unix-ts>)
#   NODE_POOL   --node-pool value         (optional; RunAI auto-selects if unset)
#   SLACK_WEBHOOK  if set, forwarded into the pod so deploy/run-bundle.sh
#                  can post a completion notification
#   EXTRA_ARGS     extra flags passed verbatim to `runai submit`
#   DRY_RUN        if non-empty, print the runai command and exit
#
# RunAI job names must be lowercase DNS-compatible (alphanumeric + hyphen,
# <= 63 chars). The default name satisfies this.

set -euo pipefail

# Resolve SLACK_WEBHOOK from env / repo-local / $HOME so a submitted
# pod gets the same notification it would on bare metal — no need
# to remember exporting the URL each session. notify.sh `set -e`s
# nothing, so sourcing it can't break this script's own pipefail
# semantics.
# shellcheck source=deploy/notify.sh
source "$(dirname "${BASH_SOURCE[0]}")/notify.sh"

IMAGE="${IMAGE:-registry.rcp.epfl.ch/sacs/rpereira/secure-vector-search:latest}"
RUN_AS_UID="${RUN_AS_UID:-224954}"
RUN_AS_GID="${RUN_AS_GID:-11259}"
SCRATCH_CLAIM="${SCRATCH_CLAIM:-sacs-scratch}"
SCRATCH_MOUNT="${SCRATCH_MOUNT:-/mnt/sacs/scratch}"
SCRATCH="${SCRATCH:-${SCRATCH_MOUNT}/secure-vector-search}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/.hf-cache}"
GPU="${GPU:-1}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-32G}"
NAME="${NAME:-svs-eval-$(date +%s)}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  echo "       (e.g. $0 make eval)" >&2
  exit 2
fi

cmd=(
  runai submit "$NAME"
  --image "$IMAGE"
  --gpu "$GPU"
  --cpu "$CPU"
  --memory "$MEMORY"
  --existing-pvc "claimname=${SCRATCH_CLAIM},path=${SCRATCH_MOUNT}"
  --working-dir "$SCRATCH"
  --environment "HF_HOME=${HF_CACHE}"
  --environment "RUNAI_JOB_NAME=${NAME}"
  --environment "USER=rpereira"
  --environment "HOME=/tmp"
  --run-as-uid "$RUN_AS_UID"
  --run-as-gid "$RUN_AS_GID"
)

if [ -n "${SLACK_WEBHOOK:-}" ]; then
  cmd+=(--environment "SLACK_WEBHOOK=${SLACK_WEBHOOK}")
fi

# Campaign tagging. Forward each CAMPAIGN_*
# var if set in the operator's env at submit time — `bin/eval`'s clap
# Args reads them via `env = "CAMPAIGN_ID"` so no per-Makefile-target
# wiring is needed. CAMPAIGN_ID + CAMPAIGN_TITLE are a required pair
# (clap rejects a partial set); CAMPAIGN_NOTE is optional.
for var in CAMPAIGN_ID CAMPAIGN_TITLE CAMPAIGN_NOTE; do
  if [ -n "${!var:-}" ]; then
    cmd+=(--environment "${var}=${!var}")
  fi
done

if [ -n "${NODE_POOL:-}" ]; then
  cmd+=(--node-pool "$NODE_POOL")
fi

if [ -n "${EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  cmd+=(${EXTRA_ARGS})
fi

cmd+=(--command -- "$@")

printf '+ %q ' "${cmd[@]}"; echo

if [ -n "${DRY_RUN:-}" ]; then
  exit 0
fi

"${cmd[@]}"

echo
echo "Submitted: $NAME"
echo "Logs:      runai logs $NAME -f"
echo "Status:    runai describe job $NAME"
echo "Cancel:    runai delete job $NAME"
