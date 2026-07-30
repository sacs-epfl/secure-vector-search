#!/usr/bin/env bash
# Smoke-test the Slack webhook from inside an RCP pod, before committing to a
# multi-hour run. SLACK_WEBHOOK must be set; deploy/runai-submit.sh forwards
# it from the host environment.

set -uo pipefail

JOB_NAME="${RUNAI_JOB_NAME:-slack-test}"

if [ -z "${SLACK_WEBHOOK:-}" ]; then
  echo "ERROR: SLACK_WEBHOOK is not set in the pod environment"
  exit 1
fi

payload=$(printf '{"text":":wave: webhook smoke test from RCP job `%s` — if you see this, the bundle run-end notification will work too"}' "$JOB_NAME")

echo "Posting to Slack..."
curl -fsS -X POST -H "Content-Type: application/json" --data "$payload" "$SLACK_WEBHOOK"
echo
echo "Done."
