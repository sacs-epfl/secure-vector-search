#!/usr/bin/env bash
# Entrypoint for the interactive dev pod. Brings up sshd in the foreground
# so the container stays alive; `runai bash` continues to work too. The
# eval workflow does not use this entrypoint — it runs `deploy/run-bundle.sh`
# directly via `deploy/runai-submit*.sh` wrappers.
#
# Expected env (set by deploy/runai-submit-dev.sh):
#   USER=rpereira
#   HOME=/mnt/sacs/scratch/.dev-home/rpereira  (persists claude-code state,
#                                               dotfiles, shell history)

set -euo pipefail

mkdir -p /tmp/sshd
chmod 700 /tmp/sshd

# Generate host keys on first start. Idempotent across pod restarts within
# the same container; rotates each time a new pod comes up (acceptable for
# a single-user dev box — known_hosts gets reset client-side).
for type in ed25519 rsa; do
  key="/tmp/sshd/ssh_host_${type}_key"
  if [ ! -f "$key" ]; then
    ssh-keygen -t "$type" -f "$key" -N "" -q
  fi
done

# /etc/passwd's HOME for rpereira is /mnt/sacs/scratch/.dev-home/rpereira
# (baked at image-build time). Make sure the dir exists with the right
# ownership before sshd starts — first-boot on a fresh NAS3 will need it.
PERSIST_HOME="${HOME:-/mnt/sacs/scratch/.dev-home/rpereira}"
mkdir -p "$PERSIST_HOME"

# Drop a profile snippet so an SSH login inherits sane env (sshd doesn't
# propagate the pod's environment by default).
cat > /tmp/dev-profile.sh <<EOF
export USER=rpereira
export HOME=${PERSIST_HOME}
export PATH=/usr/local/go/bin:/opt/go/bin:/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CARGO_HOME=/opt/cargo
export RUSTUP_HOME=/opt/rustup
export GOPATH=/opt/go
EOF
chmod 644 /tmp/dev-profile.sh

# Make /etc/profile.d-style sourcing automatic for both interactive and
# non-interactive ssh sessions.
if ! grep -q 'dev-profile.sh' "$PERSIST_HOME/.bashrc" 2>/dev/null; then
  echo '[ -r /tmp/dev-profile.sh ] && . /tmp/dev-profile.sh' >> "$PERSIST_HOME/.bashrc"
fi
if ! grep -q 'dev-profile.sh' "$PERSIST_HOME/.profile" 2>/dev/null; then
  echo '[ -r /tmp/dev-profile.sh ] && . /tmp/dev-profile.sh' >> "$PERSIST_HOME/.profile"
fi

echo "[dev-entrypoint] HOME=$PERSIST_HOME"
echo "[dev-entrypoint] sshd listening on :2222 (auth: keys baked from GitHub)"
echo "[dev-entrypoint] connect with: runai bash <job>   OR   ssh -p <node-port> rpereira@<node>"

exec /usr/sbin/sshd -D -e -f /etc/ssh/sshd_config_dev
