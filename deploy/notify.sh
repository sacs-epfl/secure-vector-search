#!/usr/bin/env bash
# Slack notification helper, sourced by deploy/run-bundle.sh,
# deploy/runai-submit.sh, and scripts/eval-suite.sh.
#
# Resolves the webhook URL (in priority order) from:
#   1. The `SLACK_WEBHOOK` env var (already exported by the caller).
#   2. A repo-local `.slack-webhook` file (sibling to this script's
#      parent dir, i.e. <repo-root>/.slack-webhook). Gitignored so
#      committing your webhook URL is impossible by accident.
#   3. `$HOME/.config/secure-vector-search/slack-webhook` — preferred
#      home for cross-machine setup; symlink it from anywhere.
#
# Then provides:
#   - `SLACK_WEBHOOK` exported (when found) or unset.
#   - `notify "<message>"` function: prints to stdout always; POSTs
#     to Slack when SLACK_WEBHOOK is set; never errors out (failed
#     POST falls through with a single warning).
#
# All callers `set -uo pipefail` (no `-e`) so an early `notify`
# failure can't kill an in-progress sweep.

resolve_slack_webhook() {
    if [ -n "${SLACK_WEBHOOK:-}" ]; then
        return
    fi
    local script_dir repo_root
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(dirname "$script_dir")"
    local candidates=(
        "$repo_root/.slack-webhook"
        "$HOME/.config/secure-vector-search/slack-webhook"
    )
    local f
    for f in "${candidates[@]}"; do
        if [ -f "$f" ] && [ -r "$f" ]; then
            local val
            val="$(head -n1 "$f" | tr -d '[:space:]')"
            if [ -n "$val" ]; then
                SLACK_WEBHOOK="$val"
                export SLACK_WEBHOOK
                return
            fi
        fi
    done
}

notify() {
    local msg="$1"
    printf '[notify] %s\n' "$msg"
    if [ -n "${SLACK_WEBHOOK:-}" ]; then
        local payload
        # Escape backslashes and double-quotes so the JSON parses
        # cleanly even when the message carries paths or quoted
        # config strings. Newlines stay as `\n`.
        local escaped="${msg//\\/\\\\}"
        escaped="${escaped//\"/\\\"}"
        escaped="${escaped//$'\n'/\\n}"
        payload=$(printf '{"text":"%s"}' "$escaped")
        curl -fsS -X POST -H "Content-Type: application/json" \
            --data "$payload" "$SLACK_WEBHOOK" >/dev/null 2>&1 \
            || printf '[notify] warning: Slack POST failed\n' >&2
    fi
}

resolve_slack_webhook
