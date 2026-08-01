#!/usr/bin/env bash
# GPU (cloud-tier) variant of deploy/runai-submit.sh.
#
# Symmetric wrapper to deploy/runai-submit-cpu.sh — sets the GPU count
# explicitly + a recognisable job-name prefix so an operator scanning
# `runai list jobs` can distinguish CPU-only from GPU runs at a glance.
# Defaults match the project's RCP V100 workspace (V100 is the
# cloud-tier default).
#
# Usage:
#   deploy/runai-submit-gpu-cloud.sh deploy/run-bundle-gpu.sh
#   GPU=2 deploy/runai-submit-gpu-cloud.sh deploy/run-bundle-gpu.sh
#   GPU_SKU_CLOUD=a100 deploy/runai-submit-gpu-cloud.sh deploy/run-bundle-gpu.sh

set -euo pipefail

export GPU="${GPU:-1}"
export NAME="${NAME:-svs-gpu-$(date +%s)}"

exec "$(dirname "$0")/runai-submit.sh" "$@"
