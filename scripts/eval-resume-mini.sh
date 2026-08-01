#!/usr/bin/env bash
# Resume the eval-suite on mini. Skips
# `eval-tiptoe-go` (and consequently `tiptoe-diff`) per
# `results/runs/d81ba182/NOTES.md` — Go 1.26 rejects the pinned
# simplepir's CGO type aliases. Validation falls back to the
# bit-equality unit test plus the existing sacs006 tiptoe-diff.
#
# Phases run in this order (matches scripts/eval-suite.sh minus
# tiptoe-go + tiptoe-diff):
#
#   1. eval-bntm + eval-bntm-ivf            — warm-cache, pending from suite kill
#   2. eval-{plaintext,sap,sap-ivf,emvp,
#           emvp-ivf,tiptoe,bntm,bntm-ivf}  — cold-cache (figure 08)
#   3. eval-breakdown                       — substep timings (figs 9a/9b)
#   4. eval-bntm BNTM_VERIFICATION=false    — fig 13 off-row (flat)
#   5. eval-bntm-ivf BNTM_VERIFICATION=false — fig 13 off-row (IVF)
#   6. make report                          — assemble PDFs
#
# Total wall-clock ~7-9 h on mini (M4, 10 cores). Slack-notified at
# start, on each phase failure, and on completion.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# shellcheck source=deploy/notify.sh
source deploy/notify.sh

JOB_NAME="${JOB_NAME:-eval-resume-mini-$(date +%s)}"
START_TS=$(date +%s)

fmt_elapsed() {
    local secs=$1
    printf '%dh %02dm' $((secs / 3600)) $(((secs % 3600) / 60))
}

run_phase() {
    local label="$1"
    shift
    echo
    echo "===> $(date -Iseconds) $label"
    if ! "$@"; then
        local elapsed=$(( $(date +%s) - START_TS ))
        notify ":x: \`${JOB_NAME}\` failed at \`${label}\` (after $(fmt_elapsed $elapsed))"
        exit 1
    fi
}

notify ":hourglass: \`${JOB_NAME}\` starting resume (eval-tiptoe-go skipped per mini NOTES.md)"

# Phase 1: pending warm bntm
run_phase "make eval-bntm (warm)" make eval-bntm
run_phase "make eval-bntm-ivf (warm)" make eval-bntm-ivf

# Phase 2: cold-cache equivalent of eval-cold, scheme-by-scheme
# (can't use `make eval-cold` because that depends on eval-tiptoe-go)
for s in plaintext sap sap-ivf emvp emvp-ivf tiptoe bntm bntm-ivf; do
    run_phase "make eval-$s EVAL_FLAGS=--no-cache" make "eval-$s" EVAL_FLAGS=--no-cache
done

# Phase 3: substep breakdown (eval-breakdown has no tiptoe-go in deps)
run_phase "make eval-breakdown" make eval-breakdown

# Phase 4 & 5: BN without verification
run_phase "make eval-bntm BNTM_VERIFICATION=false" \
    make eval-bntm BNTM_VERIFICATION=false
run_phase "make eval-bntm-ivf BNTM_VERIFICATION=false" \
    make eval-bntm-ivf BNTM_VERIFICATION=false

# Phase 6: report
run_phase "make report" make report

ELAPSED=$(( $(date +%s) - START_TS ))
notify ":white_check_mark: \`${JOB_NAME}\` complete (resume incl. cold + breakdown + BN-no-verify + report, $(fmt_elapsed $ELAPSED))"
