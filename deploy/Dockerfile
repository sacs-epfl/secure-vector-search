# syntax=docker/dockerfile:1.7
#
# Toolchain-only image for the EPFL RCP/CaaS cluster. Source tree, dataset,
# and outputs all live on NAS3, which RunAI mounts into the pod via the
# `sacs-scratch` PVC (see deploy/runai-submit.sh). Nothing is baked into the
# image.
#
# Build context is the repo root, not deploy/, so that scripts/requirements.txt
# is reachable via COPY and the .dockerignore at the repo root applies.
#
# Workflow:
#
#   # 1. Build & push from your laptop
#   docker build --platform=linux/amd64 \
#     -f deploy/Dockerfile \
#     -t registry.rcp.epfl.ch/sacs/rpereira/secure-vector-search:latest .
#   docker login registry.rcp.epfl.ch -u rpereira
#   docker push registry.rcp.epfl.ch/sacs/rpereira/secure-vector-search:latest
#
#   # 2. Stage the repo on NAS3 via the jumphost (one-time, then `git pull` to update)
#   ssh jumphost.rcp.epfl.ch
#   cd /mnt/sacs/scratch
#   git clone <repo-url> secure-vector-search
#
#   # 3. Submit a job from your laptop
#   deploy/runai-submit.sh make eval
#
# Image is rootless-friendly: $CARGO_HOME, $RUSTUP_HOME, $GOPATH are
# world-writable so the UID RunAI injects into the pod can write toolchain
# caches. Outputs from the eval pipeline land on NAS3 via the working-dir
# (/mnt/sacs/scratch/secure-vector-search), which is owned by the runtime UID.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    GOPATH=/opt/go \
    PATH=/usr/local/go/bin:/opt/go/bin:/opt/cargo/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        jq \
        less \
        make \
        numactl \
        openssh-server \
        pkg-config \
        libssl-dev \
        python3 \
        python3-pip \
        python3-venv \
        tmux \
        vim \
    && rm -rf /var/lib/apt/lists/*

# Microsoft SEAL (pinned to 4.1.1, the version the Tiptoe Go reference's
# underhood/rlwe expects). Not in Ubuntu
# apt; build from source, install to /usr/local/{include,lib}. The Go
# reference's `go build` step compiles `underhood/rlwe`'s C++ glue
# which `#include`s `<seal/seal.h>` — without this the
# `make eval-tiptoe-go` step bombs ~10h into the bundle (after every
# other scheme already ran). SEAL_USE_INTRIN=ON enables AVX2/AVX512
# intrinsics; SEAL_BUILD_DEPS=ON pulls zlib+zstd via FetchContent so
# we don't need to install them separately.
RUN git clone --depth 1 --branch v4.1.1 https://github.com/microsoft/SEAL.git /tmp/seal \
    && cmake -S /tmp/seal -B /tmp/seal/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DSEAL_BUILD_DEPS=ON \
        -DSEAL_USE_INTRIN=ON \
        -DBUILD_SHARED_LIBS=ON \
    && cmake --build /tmp/seal/build -j"$(nproc)" \
    && cmake --install /tmp/seal/build \
    && ldconfig \
    && rm -rf /tmp/seal

# Node.js LTS + Claude Code (interactive dev tooling).
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code \
    && npm cache clean --force

# Register UID 224954 (the cluster identity owning NAS3) so sshd can resolve
# it and `pwd.getpwuid()` stops KeyError-ing. HOME is set to the NAS3 dev
# path so sshd logins land in the persistent location directly — no
# symlink dance needed at runtime, and /home/ doesn't need to be writable.
# The existing eval workflow's `USER=rpereira HOME=/tmp` env override still
# wins for getpass.getuser() / shells via the env-var fast path.
RUN echo 'rpereira:x:224954:11259:rpereira,,,:/mnt/sacs/scratch/.dev-home/rpereira:/bin/bash' >> /etc/passwd \
    && echo 'rpereira:x:11259:' >> /etc/group

# Bake authorized_keys from a GitHub user's published keys. Override at
# build time with `--build-arg GITHUB_USER=<username>`. Keys live at a
# fixed path so the entrypoint can install them without depending on
# /etc/passwd lookups.
ARG GITHUB_USER=rafaelppires
RUN mkdir -p /etc/ssh-keys \
    && curl -fsSL "https://github.com/${GITHUB_USER}.keys" -o /etc/ssh-keys/authorized_keys \
    && chmod 644 /etc/ssh-keys/authorized_keys \
    && test -s /etc/ssh-keys/authorized_keys

# sshd config for non-root operation: high port, writable host-key + pid
# locations, public-key auth only. Host keys are generated at startup by
# the entrypoint into /tmp so the image stays reproducible.
RUN printf '%s\n' \
        'Port 2222' \
        'HostKey /tmp/sshd/ssh_host_ed25519_key' \
        'HostKey /tmp/sshd/ssh_host_rsa_key' \
        'PidFile /tmp/sshd/sshd.pid' \
        'UsePAM no' \
        'PasswordAuthentication no' \
        'PermitRootLogin no' \
        'PubkeyAuthentication yes' \
        'AuthorizedKeysFile /etc/ssh-keys/authorized_keys' \
        'X11Forwarding no' \
        'PrintMotd no' \
        'AcceptEnv LANG LC_*' \
        'Subsystem sftp /usr/lib/openssh/sftp-server' \
        > /etc/ssh/sshd_config_dev

COPY deploy/dev-entrypoint.sh /usr/local/bin/dev-entrypoint.sh
RUN chmod +x /usr/local/bin/dev-entrypoint.sh

# Surface the CUDA toolchain (nvcc, ptxas, cuda-gdb, …) on PATH. The base
# image installs them at /usr/local/cuda/bin but its own ENV is shadowed
# by ours up top. Kept as a trailing ENV layer so editing it doesn't
# invalidate the heavy apt / rust / go / pip layers above.
ENV PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

ARG RUST_VERSION=1.95.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path \
            --default-toolchain ${RUST_VERSION} \
            --profile minimal \
            --component rustfmt --component clippy \
    && chmod -R a+rwX ${CARGO_HOME} ${RUSTUP_HOME}

ARG GO_VERSION=1.22.6
RUN curl -fsSL https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz \
      | tar -C /usr/local -xz \
    && mkdir -p ${GOPATH} && chmod -R a+rwX ${GOPATH}

COPY scripts/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Ubuntu 22.04's apt cmake is 3.22.1; cuvs 26.4 (workspace dep) needs
# >= 3.30.4. pip's cmake wheel lands a wrapper at /usr/local/bin/cmake
# which the standard PATH already prefers over /usr/bin/cmake.
# Defensive fallback only — the rapids conda env below ships cmake
# 4.3.2 and is what cuvs-sys's build.rs actually uses via PATH /
# CMAKE_PREFIX_PATH. Keeping this layer is harmless (~30 MB) and
# serves as a safety net if the conda env's PATH ever fails to
# take effect inside a pod.
RUN pip3 install --no-cache-dir --upgrade 'cmake>=3.30.4'

# RAPIDS / cuVS toolchain via micromamba (single-binary mamba).
# The conda env spec COPYd below is the source of truth — it pins
# cuvs 26.04.00, libcuvs-headers, cuda-cudart-dev, libclang,
# cmake 4.3.2, plus the transitive RAPIDS + RAFT + RMM stack
# cuvs-sys's `find_package(cuvs)` expects. Without this layer the
# `--features gpu` build of scorer-plaintext / scorer-sap fails at
# CMake's `cuvsConfig.cmake not found` step.
#
# The env vars below mirror that env's activation so the
# pod-runtime `cargo build --features gpu` inside `make eval-gpu-cloud`
# picks up the conda-installed libcuvs without needing a manual
# `conda activate rapids` step — pods never run an interactive shell.
ENV CONDA_DIR=/opt/conda \
    MAMBA_ROOT_PREFIX=/opt/conda

RUN mkdir -p ${CONDA_DIR}/bin \
    && curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C ${CONDA_DIR} bin/micromamba \
    && ln -sf ${CONDA_DIR}/bin/micromamba /usr/local/bin/micromamba

COPY docs/envs/rapids.yml /tmp/rapids.yml
RUN micromamba create --root-prefix ${CONDA_DIR} --yes --name rapids \
        --file /tmp/rapids.yml \
    && rm /tmp/rapids.yml \
    && find ${CONDA_DIR}/envs/rapids \( -name '*.pyc' -o -name '__pycache__' \) \
        -exec rm -rf {} + 2>/dev/null || true \
    && chmod -R a+rwX ${CONDA_DIR}

ENV RAPIDS_PREFIX=${CONDA_DIR}/envs/rapids \
    CMAKE_PREFIX_PATH=${CONDA_DIR}/envs/rapids \
    LIBCLANG_PATH=${CONDA_DIR}/envs/rapids/lib \
    BINDGEN_EXTRA_CLANG_ARGS="-I${CONDA_DIR}/envs/rapids/targets/x86_64-linux/include -I/usr/local/cuda/include" \
    RUSTFLAGS="-L ${CONDA_DIR}/envs/rapids/lib -L /usr/local/cuda/lib64" \
    LD_LIBRARY_PATH=${CONDA_DIR}/envs/rapids/lib:${LD_LIBRARY_PATH} \
    PATH=${CONDA_DIR}/envs/rapids/bin:${PATH}

# SEAL 4.1.1's `cmake --install` lays headers under a version-suffixed
# path (/usr/local/include/SEAL-4.1/seal/seal.h, NOT /usr/local/include/
# seal/seal.h). On the library side it follows the standard POSIX
# convention:
#   /usr/local/lib/libseal.so        (linker name, import symlink)
#   /usr/local/lib/libseal.so.4.1    (SONAME symlink)
#   /usr/local/lib/libseal.so.4.1.1  (actual shared object)
# No `libseal-4.1.so` exists.
#
# The Tiptoe Go reference's underhood/rlwe (pinned at v0.0.0-
# 20230922182337-f053a81c6385 by tools/tiptoe-go-rev) has a
# hardcoded in-source directive in `rlwe/rlwe.go`:
#
#   // #cgo CXXFLAGS: -std=c++20 -I/usr/local/include/SEAL-4.1 …
#   // #cgo LDFLAGS:  -lstdc++ -lseal-4.1
#
# The CXXFLAGS clause finds the headers (the SEAL-4.1 path is
# embedded right there), so the include side works without our
# CGO_CXXFLAGS being load-bearing. The LDFLAGS clause looks for
# `libseal-4.1.so` though — which doesn't exist. The bundle's
# `make eval-tiptoe-go` step bombs at the link with
# `cannot find -lseal-4.1` (svs-cpu-1778657260 ran this into
# the ground over ~2h before failing at the link).
#
# Two cleaner fixes don't apply:
# - patching underhood's source via tools/tiptoe-go.patch would
#   land outside the project's vendor patch's existing scope
#   (paired-runner addition + go.mod timestamp fix);
# - CGO_LDFLAGS additions are additive but can't override the
#   in-source `-lseal-4.1` — that flag still gets passed to ld
#   regardless of what the env adds.
#
# So: symlink `libseal-4.1.so` to the real SONAME-versioned file.
# ld follows the chain at link-time and runtime; `-lseal-4.1`
# resolves through the symlink to libseal.so.4.1 → libseal.so.4.1.1.
# Same single library, just an extra import name.
RUN ln -sf libseal.so.4.1 /usr/local/lib/libseal-4.1.so

# CGO env vars kept as defensive overlap. CGO_CXXFLAGS is redundant
# with underhood's in-source CXXFLAGS but harmless. CGO_LDFLAGS's
# `-lseal` (operator fix 788202a) gives a redundant second link
# reference to the same library — ld dedupes; no harm.
ENV CGO_CXXFLAGS="-I/usr/local/include/SEAL-4.1" \
    CGO_LDFLAGS="-L/usr/local/lib -lseal"

CMD ["/bin/bash"]
