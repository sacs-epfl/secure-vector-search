#!/usr/bin/env bash
# enable-gpu.sh — bring the 2x RTX 5000 Ada online and install the CUDA
# toolkit so the secure-vector-search `gpu` Cargo feature can build + run.
#
# This host (Ubuntu 24.04) already has the DKMS NVIDIA driver 595.71.05
# installed but the kernel modules are NOT loaded (no /dev/nvidia0, nvidia-smi
# fails) and there is no CUDA toolkit (no nvcc). Both need root.
#
# Run:   sudo bash enable-gpu.sh
# Safe to re-run (idempotent). Installs the TOOLKIT only (cuda-toolkit-13-0),
# never a driver package, so the working 595.71.05 DKMS driver is untouched.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "error: run as root —  sudo bash $0" >&2
  exit 1
fi

echo "==> 1/3  Load NVIDIA kernel modules + create device nodes"

# Clear any half-loaded nvidia stack from a previous failed attempt.
modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null || true

# The cards are bound to the open-source 'nouveau' driver, so the nvidia
# module gets ENODEV ('No such device'). A plain 'modprobe -r nouveau' does
# not stick (udev re-probes the PCI device and reloads it). So we: (1) hard-
# block nouveau auto-load for this boot, (2) PCI-unbind it from each GPU,
# (3) remove the module. This frees the cards without a reboot. The unbind
# only affects the compute GPUs (99/ab), not the ASPEED BMC console (02), so
# the SSH session is unaffected.
echo "  freeing the GPUs from nouveau..."
# (1) stronger than 'blacklist': prevents any auto-reload during this session
echo 'install nouveau /bin/true' > /etc/modprobe.d/zz-disable-nouveau.conf
# persist a real blacklist for future boots too
cat > /etc/modprobe.d/blacklist-nouveau.conf <<'NOUVEAU'
blacklist nouveau
options nouveau modeset=0
NOUVEAU
update-initramfs -u 2>/dev/null || true

# (2) unbind nouveau from each NVIDIA display/3D controller
for slot in $(lspci -D -d 10de: | grep -iE 'VGA|3D controller' | awk '{print $1}'); do
  drv_link="/sys/bus/pci/devices/$slot/driver"
  if [ -e "$drv_link" ] && [ "$(basename "$(readlink "$drv_link")")" = nouveau ]; then
    echo "    unbinding $slot from nouveau"
    echo "$slot" > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || true
  fi
done
# (3) remove the module now that no device holds it
modprobe -r nouveau 2>/tmp/rmmod-nouveau.err || {
  echo "  warn: 'modprobe -r nouveau' still failed:" >&2
  cat /tmp/rmmod-nouveau.err >&2
  echo "  -> reboot (blacklist is now in place) and re-run this script." >&2
}

if ! modprobe nvidia 2>/tmp/modprobe-nvidia.err; then
  echo "error: 'modprobe nvidia' failed:" >&2
  cat /tmp/modprobe-nvidia.err >&2
  echo "  - 'No such device' after this point usually means nouveau is still" >&2
  echo "    loaded (check 'lsmod | grep nouveau'); reboot to apply the" >&2
  echo "    blacklist, then re-run this script." >&2
  echo "  - A key/signature error means Secure Boot is blocking the unsigned" >&2
  echo "    DKMS module ('mokutil --sb-state')." >&2
  exit 1
fi
modprobe nvidia-uvm
modprobe nvidia-modeset 2>/dev/null || true

# nvidia-smi run as root normally creates /dev/nvidia* on first call.
nvidia-smi -L >/dev/null 2>&1 || true

# Belt-and-suspenders: create device nodes from the majors in /proc/devices
# if nvidia-smi did not.
n_gpu=$(lspci | grep -ciE 'NVIDIA.*(VGA|3D)')
if [ ! -e /dev/nvidia0 ]; then
  major=$(awk '/nvidia-frontend|^[0-9]+ nvidia$/{print $1; exit}' /proc/devices || true)
  if [ -n "${major:-}" ]; then
    [ -e /dev/nvidiactl ] || mknod -m 666 /dev/nvidiactl c "$major" 255
    for i in $(seq 0 $((n_gpu - 1))); do
      [ -e "/dev/nvidia$i" ] || mknod -m 666 "/dev/nvidia$i" c "$major" "$i"
    done
  fi
fi
if [ ! -e /dev/nvidia-uvm ]; then
  umajor=$(awk '/nvidia-uvm$/{print $1; exit}' /proc/devices || true)
  if [ -n "${umajor:-}" ]; then
    mknod -m 666 /dev/nvidia-uvm c "$umajor" 0 2>/dev/null || true
    mknod -m 666 /dev/nvidia-uvm-tools c "$umajor" 1 2>/dev/null || true
  fi
fi

# Keep the driver initialised across processes (avoids re-init latency / nodes
# vanishing). Non-fatal if persistenced is absent.
nvidia-persistenced --persistence-mode 2>/dev/null || true

echo "--- nvidia-smi ---"
nvidia-smi

echo "==> 2/3  Install CUDA toolkit 13.0 (nvcc) — matches driver 595 / torch cu130"
if command -v nvcc >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/nvcc ]; then
  echo "    nvcc already present, skipping toolkit install"
else
  cd /tmp
  KEYRING=cuda-keyring_1.1-1_all.deb
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/${KEYRING}"
  dpkg -i "${KEYRING}"
  apt-get update
  # TOOLKIT ONLY — `cuda-toolkit-13-0` has no driver dependency, so the
  # existing 595.71.05 DKMS driver is left alone. (Do NOT install `cuda`.)
  apt-get install -y cuda-toolkit-13-0
fi
echo "--- nvcc ---"
( /usr/local/cuda/bin/nvcc --version 2>/dev/null || nvcc --version ) | tail -3

echo "==> 3/3  Done. In your (non-root) shell, put CUDA on PATH before building:"
echo "      export PATH=/usr/local/cuda/bin:\$PATH"
echo "      export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}"
echo
echo "This unblocks the emvp/bntm GPU build (cudarc + nvcc-compiled kernels)."
echo "plaintext/sap GPU additionally need cuVS (RAPIDS), which this script does"
echo "NOT install — tell Claude and we'll handle cuVS as a separate step."
