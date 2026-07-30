#!/usr/bin/env bash
# Interactive dev variant of deploy/runai-submit.sh.
#
# Spawns a long-lived interactive pod with sshd + tmux + claude-code +
# Rust/Go/Python toolchains baked into the image. Two ways to connect:
#
#   1. `runai bash <job>` — terminal shell inside the pod, no port needed.
#   2. SSH via kubectl port-forward (the project policy blocks NodePort,
#      so we don't expose 2222 as a Service). From your laptop:
#
#        kubectl get pods -n runai-sacs-rpereira | grep <job>
#        kubectl port-forward -n runai-sacs-rpereira pod/<pod-name> 2222:2222 &
#        ssh -p 2222 rpereira@localhost
#
#      sshd is bound on container port 2222 with public-key auth.
#
# Defaults are sized for Plan-17 GPU dev: 1 GPU (RTX 5000 / H100 depending
# on what RunAI assigns), 16 cores, 64 GiB RAM. Override per-launch via
# env vars (see deploy/runai-submit.sh header).
#
# HOME is set to a per-user NAS3 path so claude-code login state, dotfiles,
# and shell history persist across pod restarts.
#
# Usage:
#   deploy/runai-submit-dev.sh
#   GPU=2 deploy/runai-submit-dev.sh

set -euo pipefail

export NAME="${NAME:-svs-dev-$(date +%s)}"
export GPU="${GPU:-1}"
export CPU="${CPU:-16}"
export MEMORY="${MEMORY:-64G}"

# Dev pod works over a separate NAS3 checkout so it doesn't collide with
# the pristine eval staging tree at .../secure-vector-search. Stage with:
#   ssh jumphost.rcp.epfl.ch
#   cd /mnt/sacs/scratch && git clone <repo> secure-vector-search-dev
export SCRATCH="${SCRATCH:-/mnt/sacs/scratch/secure-vector-search-dev}"

# Persistent HOME on NAS3. Per-user path so multiple devs in the same
# project don't collide on ~/.claude/ etc.
DEV_HOME_DEFAULT="/mnt/sacs/scratch/.dev-home/rpereira"
DEV_HOME="${DEV_HOME:-$DEV_HOME_DEFAULT}"

# Override HOME by re-injecting it via EXTRA_ARGS as an --environment flag.
# (deploy/runai-submit.sh sets HOME=/tmp unconditionally; the *last*
# --environment HOME=… on the runai command wins.)
export EXTRA_ARGS="--environment HOME=${DEV_HOME}${EXTRA_ARGS:+ ${EXTRA_ARGS}}"

exec "$(dirname "$0")/runai-submit-interactive.sh" /usr/local/bin/dev-entrypoint.sh
