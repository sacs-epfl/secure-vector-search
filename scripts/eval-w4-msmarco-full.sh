#!/usr/bin/env bash
# W4 driver — full 8.8 M MS MARCO sweep on sacs006 CPU with
# a post-plaintext recall-checkpoint pause. Runs four phases
# sequentially:
#
#   1. make eval-plaintext (~30 min)
#        -> parse recall@10 per nprobe from the run's raw.csv
#        -> Slack-notify the summary
#        -> block on /tmp/sentinel-w4-continue (operator touches it
#           to resume; Ctrl-C to abort and revisit n_centroids)
#   2. make eval-sap-ivf  (nprobe sweep + beta sweep)
#   3. make eval-emvp-ivf
#   4. make eval-bntm-ivf BNTM_VERIFICATION=false
#
# The Slack webhook URL is resolved by deploy/notify.sh.
# eval-suite.sh is left untouched per the W4 design discussion.
#
# Usage:
#   scripts/eval-w4-msmarco-full.sh
#   JOB_NAME=w4-msmarco-full-sacs006 scripts/eval-w4-msmarco-full.sh

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# shellcheck source=deploy/notify.sh
source deploy/notify.sh

JOB_NAME="${JOB_NAME:-eval-w4-msmarco-full-$(hostname -s)}"
SENTINEL="/tmp/sentinel-w4-continue"
START_TS=$(date +%s)

export CAMPAIGN_ID="${CAMPAIGN_ID:-w4-msmarco-full}"
export CAMPAIGN_TITLE="${CAMPAIGN_TITLE:-W4: 8.8M sacs006 CPU}"

PYTHON_BIN="${PYTHON:-venv/bin/python}"
DATASET="${DATASET:-msmarco-full}"
NPROBE="${NPROBE:-1,8,64,256,1024,2967}"
SAP_IVF_NPROBE_FIXED="${SAP_IVF_NPROBE_FIXED:-64}"

FROM_PHASE="plaintext"
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-plaintext)
            # Back-compat alias for --from-phase sap-ivf.
            FROM_PHASE="sap-ivf"
            shift
            ;;
        --from-phase)
            FROM_PHASE="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'USAGE'
Usage: eval-w4-msmarco-full.sh [--from-phase PHASE | --skip-plaintext]

  --from-phase PHASE   Start at the named phase, skipping earlier ones.
                       Valid values: plaintext, sap-ivf, emvp-ivf, bntm-ivf.
                       Default: plaintext (runs all four phases). Use to
                       recover from a mid-run termination — e.g.
                       `--from-phase emvp-ivf` if sap-ivf already wrote
                       a clean raw.csv in this campaign and only emvp-ivf
                       and bntm-ivf still need to run.
  --skip-plaintext     Back-compat alias for `--from-phase sap-ivf`.

Env vars (defaults shown):
  JOB_NAME                eval-w4-msmarco-full-<host>
  CAMPAIGN_ID             w4-msmarco-full
  CAMPAIGN_TITLE          W4: 8.8M sacs006 CPU
  DATASET                 msmarco-full
  NPROBE                  1,8,64,256,1024,2967
  SAP_IVF_NPROBE_FIXED    64
USAGE
            exit 0
            ;;
        *)
            echo "[w4] unknown argument: $1 (use --help)" >&2
            exit 2
            ;;
    esac
done

# Validate phase + compute start index.
case "$FROM_PHASE" in
    plaintext) START_IDX=0 ;;
    sap-ivf)   START_IDX=1 ;;
    emvp-ivf)  START_IDX=2 ;;
    bntm-ivf)  START_IDX=3 ;;
    *)
        echo "[w4] invalid --from-phase: $FROM_PHASE (must be plaintext|sap-ivf|emvp-ivf|bntm-ivf)" >&2
        exit 2
        ;;
esac

should_run_phase() {
    local idx=$1
    [ "$idx" -ge "$START_IDX" ]
}

fmt_elapsed() {
    local secs=$1
    printf '%dh %02dm' $((secs / 3600)) $(((secs % 3600) / 60))
}

# Stripped-down copy of eval-suite.sh::assert_performance_governor.
# Kept in-script (not refactored into a shared helper) so the W4
# driver stays self-contained per the design decision to leave
# eval-suite.sh untouched.
assert_performance_governor() {
    local gov_file="/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if [ ! -r "$gov_file" ]; then
        return 0
    fi
    local gov
    gov=$(cat "$gov_file")
    if [ "$gov" = "performance" ]; then
        return 0
    fi
    if [ -n "${ALLOW_NONPERFORMANCE_GOVERNOR:-}" ]; then
        notify ":warning: \`${JOB_NAME}\` cpu governor is \`${gov}\` (not \`performance\`) — proceeding because ALLOW_NONPERFORMANCE_GOVERNOR is set"
        return 0
    fi
    notify ":x: \`${JOB_NAME}\` aborted — cpu governor is \`${gov}\`, not \`performance\`. Run \`sudo cpupower frequency-set -g performance\` (or set \`ALLOW_NONPERFORMANCE_GOVERNOR=1\`)."
    echo "[w4] cpu governor is '$gov', not 'performance'. Fix and retry." >&2
    exit 1
}

run_phase() {
    local label="$1"
    shift
    local phase_start
    phase_start=$(date +%s)
    local suite_elapsed=$(( phase_start - START_TS ))
    echo
    echo "===> $(date -Iseconds) ${label}"
    notify ":arrow_forward: \`${JOB_NAME}\` starting \`${label}\` (suite elapsed $(fmt_elapsed $suite_elapsed))"
    if ! "$@"; then
        local elapsed=$(( $(date +%s) - START_TS ))
        notify ":x: \`${JOB_NAME}\` failed at \`${label}\` (after $(fmt_elapsed $elapsed))"
        exit 1
    fi
    local phase_elapsed=$(( $(date +%s) - phase_start ))
    local total_elapsed=$(( $(date +%s) - START_TS ))
    notify ":white_check_mark: \`${JOB_NAME}\` \`${label}\` done in $(fmt_elapsed $phase_elapsed) (suite elapsed $(fmt_elapsed $total_elapsed))"
}

# Recall summary helper. Finds the freshest plaintext run dir whose
# run-id (a Unix timestamp) is >= $1 and prints one line
# per (config-label) with mean recall@10 over (query x rep).
recall_summary_since() {
    local since_ts="$1"
    "$PYTHON_BIN" - "$since_ts" <<'PY'
import csv, glob, os, statistics, sys, tomllib

since_ts = int(sys.argv[1])

candidates = []
for raw in glob.glob("results/runs/*/*/[0-9]*/raw.csv"):
    run_dir = os.path.dirname(raw)
    try:
        run_id = int(os.path.basename(run_dir))
    except ValueError:
        continue
    if run_id < since_ts:
        continue
    meta = os.path.join(run_dir, "run-metadata.toml")
    if not os.path.exists(meta):
        continue
    try:
        with open(meta, "rb") as fh:
            md = tomllib.load(fh)
    except Exception:
        continue
    # The eval harness nests scheme under [scheme-config]; the top-level
    # `scheme` key does not exist. (Bug discovered post-launch on 2026-05-18
    # when the W4 plaintext checkpoint Slack-posted ":warning: recall summary
    # failed" despite a clean raw.csv being on disk.)
    if md.get("scheme-config", {}).get("scheme") != "plaintext":
        continue
    campaign = md.get("campaign") or {}
    if campaign.get("id") != os.environ.get("CAMPAIGN_ID", "w4-msmarco-full"):
        continue
    candidates.append((run_id, raw))

if not candidates:
    sys.stderr.write("recall_summary: no matching plaintext run found\n")
    sys.exit(1)

candidates.sort(reverse=True)
_, raw = candidates[0]

per_label = {}
with open(raw, newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        label = row["config-label"]
        per_label.setdefault(label, []).append(float(row["recall-at-k"]))

print(f"run-dir: {os.path.dirname(raw)}")
for label in sorted(per_label, key=lambda s: (len(s), s)):
    values = per_label[label]
    mean = statistics.fmean(values)
    print(f"  {label}: recall@10 = {mean:.4f}  (n={len(values)})")
PY
}

block_on_sentinel() {
    if [ -e "$SENTINEL" ]; then
        echo "[w4] removing pre-existing $SENTINEL so a stale file can't auto-resume"
        rm -f "$SENTINEL"
    fi
    notify ":raised_hand: \`${JOB_NAME}\` paused at post-plaintext checkpoint — operator: inspect recall above, then \`touch ${SENTINEL}\` to resume, or Ctrl-C to abort."
    echo
    echo "[w4] checkpoint: touch ${SENTINEL} to resume; Ctrl-C to abort."
    until [ -e "$SENTINEL" ]; do
        sleep 30
    done
    rm -f "$SENTINEL"
    notify ":arrow_forward: \`${JOB_NAME}\` sentinel observed; resuming with sap-ivf / emvp-ivf / bntm-ivf phases"
}

assert_performance_governor

# Pre-flight: clear any stale sentinel before phase 1 too, so the
# wait loop can't trip immediately on a leftover from a previous run.
if [ -e "$SENTINEL" ]; then
    echo "[w4] pre-flight: removing stale $SENTINEL"
    rm -f "$SENTINEL"
fi

notify ":hourglass: \`${JOB_NAME}\` starting W4 sweep (CAMPAIGN_ID=${CAMPAIGN_ID}, from-phase=${FROM_PHASE})"

if should_run_phase 0; then
    PHASE1_SINCE=$(date +%s)
    run_phase "make eval-plaintext DATASET=${DATASET} NPROBE=${NPROBE}" \
        make eval-plaintext DATASET="${DATASET}" NPROBE="${NPROBE}"

    echo
    echo "[w4] post-plaintext recall summary:"
    SUMMARY=$(recall_summary_since "$PHASE1_SINCE" 2>&1)
    RC=$?
    echo "$SUMMARY"
    if [ $RC -eq 0 ]; then
        notify ":bar_chart: \`${JOB_NAME}\` plaintext recall@10 per config:\n\`\`\`\n${SUMMARY}\n\`\`\`"
    else
        notify ":warning: \`${JOB_NAME}\` recall summary failed (rc=${RC}); pausing for operator inspection regardless"
    fi

    block_on_sentinel
else
    notify ":fast_forward: \`${JOB_NAME}\` skipping plaintext (--from-phase ${FROM_PHASE})"
fi

if should_run_phase 1; then
    run_phase "make eval-sap-ivf DATASET=${DATASET} NPROBE=${NPROBE} SAP_IVF_NPROBE_FIXED=${SAP_IVF_NPROBE_FIXED}" \
        make eval-sap-ivf DATASET="${DATASET}" NPROBE="${NPROBE}" SAP_IVF_NPROBE_FIXED="${SAP_IVF_NPROBE_FIXED}"
else
    notify ":fast_forward: \`${JOB_NAME}\` skipping sap-ivf (--from-phase ${FROM_PHASE})"
fi

if should_run_phase 2; then
    run_phase "make eval-emvp-ivf DATASET=${DATASET} NPROBE=${NPROBE}" \
        make eval-emvp-ivf DATASET="${DATASET}" NPROBE="${NPROBE}"
else
    notify ":fast_forward: \`${JOB_NAME}\` skipping emvp-ivf (--from-phase ${FROM_PHASE})"
fi

if should_run_phase 3; then
    run_phase "make eval-bntm-ivf DATASET=${DATASET} NPROBE=${NPROBE} BNTM_VERIFICATION=false" \
        make eval-bntm-ivf DATASET="${DATASET}" NPROBE="${NPROBE}" BNTM_VERIFICATION=false
else
    notify ":fast_forward: \`${JOB_NAME}\` skipping bntm-ivf (--from-phase ${FROM_PHASE})"
fi

ELAPSED=$(( $(date +%s) - START_TS ))
notify ":white_check_mark: \`${JOB_NAME}\` W4 sweep complete (total $(fmt_elapsed $ELAPSED)). Bulk-store upload + W4-GPU phase are separate operator actions."
