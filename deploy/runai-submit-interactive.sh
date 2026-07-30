#!/usr/bin/env bash
# Interactive variant of deploy/runai-submit.sh.
#
# Interactive workloads have a 12h INT limit but are not subject to the
# TRAIN IDLE LIMIT (so they won't get reclaimed for low GPU utilisation).
# Use for runs that mix CPU-heavy eval phases with sporadic GPU work and
# would otherwise trip the train-idle reclaim on a regular Train job.
#
# Set PREEMPTIBLE=1 to allow scheduling above the project's deserved quota
# (job may be reclaimed at any time). Off by default — non-preemptible
# interactive jobs are guaranteed to keep their share for the 12h INT limit.
#
# Usage:
#   deploy/runai-submit-interactive.sh deploy/run-bundle.sh
#   PREEMPTIBLE=1 deploy/runai-submit-interactive.sh make eval

set -euo pipefail

export NAME="${NAME:-svs-int-$(date +%s)}"

extra="--interactive"
if [ -n "${PREEMPTIBLE:-}" ]; then
  extra+=" --preemptible"
fi
export EXTRA_ARGS="${extra}${EXTRA_ARGS:+ ${EXTRA_ARGS}}"

exec "$(dirname "$0")/runai-submit.sh" "$@"
