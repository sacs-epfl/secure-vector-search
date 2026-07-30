#!/usr/bin/env bash
# CPU-only variant of deploy/runai-submit.sh.
#
# Use this for jobs that don't actually exercise the GPU — e.g. the Rust
# scorer eval (`make eval`) which is CPU-bound after embeddings are cached.
# Submitting these as `--gpu 0` avoids the project's TRAIN IDLE LIMIT
# reclaiming the workload after 2h of idle GPU.
#
# Usage:
#   deploy/runai-submit-cpu.sh make eval
#   CPU=16 MEMORY=64G deploy/runai-submit-cpu.sh deploy/run-bundle.sh

set -euo pipefail

export GPU="${GPU:-0}"
export NAME="${NAME:-svs-cpu-$(date +%s)}"

exec "$(dirname "$0")/runai-submit.sh" "$@"
