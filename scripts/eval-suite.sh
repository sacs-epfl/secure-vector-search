#!/usr/bin/env bash
# Run the full paper-eval suite end-to-end on bare metal (sacs006 etc.).
# One command, Slack-notified at start / each phase start+finish /
# each phase failure / sudo pause / completion. Webhook URL is
# resolved by deploy/notify.sh — no need to put it in git.
#
# Phases:
#   1. make eval                       — figs 01-06, 10
#   2. make eval-cold                  — fig 08
#   3. make eval-breakdown             — figs 09a/9b
#   4. make eval-bntm     BNTM_VERIFICATION=false   — fig 13
#   5. make eval-bntm-ivf BNTM_VERIFICATION=false   — fig 13
#   6. make tiptoe-diff                — validation gate
#   7. (pause if sudo not cached) make eval-scaling — fig 07
#   8. make report                     — assemble PDFs
#
# eval-scaling needs passwordless sudo for `tee /proc/sys/vm/drop_caches`
# (see CLAUDE.md). If `sudo -n true` succeeds (NOPASSWD entry in place
# OR cached creds), the script proceeds straight through. If it fails,
# the script Slack-notifies, prompts for `sudo -v` interactively, and
# resumes — designed to run unattended on a tmux pane and only
# require human attention at the sudo gate.
#
# Skip the scaling phase entirely with `SKIP_SCALING=1` (e.g. on a
# host without the drop_caches sudoers entry). Skip the report
# rebuild with `SKIP_REPORT=1`.
#
# On Linux the script refuses to start unless cpu0's frequency
# governor is `performance`. `schedutil` (Ubuntu's default) depresses
# the single-thread baseline on idle hardware, which flatters
# figure 07's parallel-efficiency curve into looking like the
# scheme doesn't scale. Override with
# `ALLOW_NONPERFORMANCE_GOVERNOR=1` only when you accept that
# implication.
#
# Usage:
#   scripts/eval-suite.sh
#   SKIP_SCALING=1 scripts/eval-suite.sh
#   JOB_NAME=eval-suite-laptop scripts/eval-suite.sh

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# shellcheck source=deploy/notify.sh
source deploy/notify.sh

JOB_NAME="${JOB_NAME:-eval-suite-$(hostname -s)}"
START_TS=$(date +%s)

# Total phase count for `[N/T]` progress markers. Fixed phases:
# eval, eval-cold, eval-breakdown, eval-bntm, eval-bntm-ivf,
# tiptoe-diff (=6). eval-scaling and report add one each unless
# their SKIP_* flag is set.
TOTAL_PHASES=6
[ -z "${SKIP_SCALING:-}" ] && TOTAL_PHASES=$((TOTAL_PHASES + 1))
[ -z "${SKIP_REPORT:-}" ]  && TOTAL_PHASES=$((TOTAL_PHASES + 1))
PHASE_IDX=0

fmt_elapsed() {
    local secs=$1
    printf '%dh %02dm' $((secs / 3600)) $(((secs % 3600) / 60))
}

run_phase() {
    local label="$1"
    shift
    PHASE_IDX=$((PHASE_IDX + 1))
    local tag="[${PHASE_IDX}/${TOTAL_PHASES}]"
    local phase_start
    phase_start=$(date +%s)
    local suite_elapsed=$(( phase_start - START_TS ))
    echo
    echo "===> $(date -Iseconds) ${tag} ${label}"
    notify ":arrow_forward: \`${JOB_NAME}\` ${tag} starting \`${label}\` (suite elapsed $(fmt_elapsed $suite_elapsed))"
    if ! "$@"; then
        local elapsed=$(( $(date +%s) - START_TS ))
        notify ":x: \`${JOB_NAME}\` ${tag} failed at \`${label}\` (after $(fmt_elapsed $elapsed))"
        exit 1
    fi
    local phase_elapsed=$(( $(date +%s) - phase_start ))
    local total_elapsed=$(( $(date +%s) - START_TS ))
    notify ":white_check_mark: \`${JOB_NAME}\` ${tag} \`${label}\` done in $(fmt_elapsed $phase_elapsed) (suite elapsed $(fmt_elapsed $total_elapsed))"
}

assert_performance_governor() {
    # Same file the eval harness records in run-metadata.toml's
    # `cpu-governor`. Anything other than `performance` makes timing
    # non-reproducible on idle hardware and corrupts figure 07.
    local gov_file="/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if [ ! -r "$gov_file" ]; then
        # macOS, containers without cpufreq, or otherwise non-Linux.
        # The harness writes cpu-governor = "unknown" in this case;
        # there's nothing for us to assert.
        return 0
    fi
    local gov
    gov=$(cat "$gov_file")
    if [ "$gov" = "performance" ]; then
        return 0
    fi
    if [ -n "${ALLOW_NONPERFORMANCE_GOVERNOR:-}" ]; then
        notify ":warning: \`${JOB_NAME}\` cpu governor is \`${gov}\` (not \`performance\`) — proceeding because ALLOW_NONPERFORMANCE_GOVERNOR is set"
        echo "[suite] WARNING: cpu governor is '$gov', not 'performance' — proceeding anyway"
        return 0
    fi
    notify ":x: \`${JOB_NAME}\` aborted — cpu governor is \`${gov}\`, not \`performance\`. Run \`sudo cpupower frequency-set -g performance\` (or set \`ALLOW_NONPERFORMANCE_GOVERNOR=1\` to override)."
    echo
    echo "[suite] cpu governor is '$gov', not 'performance'."
    echo "[suite] schedutil/ondemand/powersave under-clock single-threaded"
    echo "[suite] work on idle hardware, which destroys figure 07's"
    echo "[suite] parallel-efficiency baseline. Fix with:"
    echo "[suite]   sudo cpupower frequency-set -g performance"
    echo "[suite] Or, to record runs anyway (e.g. you're debugging the"
    echo "[suite] harness, not measuring), re-run with"
    echo "[suite]   ALLOW_NONPERFORMANCE_GOVERNOR=1 scripts/eval-suite.sh"
    echo "[suite] Background: docs/operations/measurement-hygiene.md"
    exit 1
}

assert_performance_governor

notify ":hourglass: \`${JOB_NAME}\` starting full eval suite (${TOTAL_PHASES} phases)"

# Pre-scaling phases — order mirrors deploy/run-bundle.sh so the pod
# and bare-metal runs feed the same set of figures.
run_phase "make eval"                       make eval
run_phase "make eval-cold"                  make eval-cold
run_phase "make eval-breakdown"             make eval-breakdown
run_phase "make eval-bntm BNTM_VERIFICATION=false" \
    make eval-bntm BNTM_VERIFICATION=false
run_phase "make eval-bntm-ivf BNTM_VERIFICATION=false" \
    make eval-bntm-ivf BNTM_VERIFICATION=false
run_phase "make tiptoe-diff"                make tiptoe-diff

if [ -n "${SKIP_SCALING:-}" ]; then
    notify ":fast_forward: \`${JOB_NAME}\` skipping eval-scaling (SKIP_SCALING set)"
else
    # The only sudo invocation in `make eval-scaling` is
    # `tee /proc/sys/vm/drop_caches`. Probe it specifically with
    # `sudo -ln <cmd>` rather than `sudo -n true` — the canonical
    # production setup is a NOPASSWD entry scoped to that exact
    # command, in which case `sudo -n true` *fails* (because `true`
    # isn't NOPASSWD'd) even though scaling would run cleanly.
    # `sudo -ln <cmd>` exits 0 iff the matched sudoers entry is
    # NOPASSWD or creds are cached; otherwise it prints
    # "a password is required" and exits 1.
    DROP_CACHES_CMD="/usr/bin/tee /proc/sys/vm/drop_caches"
    sudo_probe() { sudo -ln $DROP_CACHES_CMD >/dev/null 2>&1; }

    if ! sudo_probe; then
        notify ":raised_hand: \`${JOB_NAME}\` paused before \`make eval-scaling\` — \`sudo ${DROP_CACHES_CMD}\` would prompt for a password. Either add a NOPASSWD sudoers entry for it, or run \`sudo -v\` in the same shell, then press Enter."
        echo
        echo "[suite] sudo probe failed for: $DROP_CACHES_CMD"
        echo "[suite] Either:"
        echo "[suite]   • Add to /etc/sudoers.d/secure-vector-search:"
        echo "[suite]       $(whoami) ALL=(root) NOPASSWD: $DROP_CACHES_CMD"
        echo "[suite]   • Or run 'sudo -v' in another tab now."
        echo "[suite] Then press Enter here to retry the probe and resume."
        read -r _
        if ! sudo_probe; then
            notify ":x: \`${JOB_NAME}\` aborted — sudo probe still failing after pause"
            exit 1
        fi
    fi
    run_phase "make eval-scaling"           make eval-scaling
fi

if [ -z "${SKIP_REPORT:-}" ]; then
    run_phase "make report"                 make report
fi

ELAPSED=$(( $(date +%s) - START_TS ))
notify ":white_check_mark: \`${JOB_NAME}\` complete (full suite, $(fmt_elapsed $ELAPSED))"
