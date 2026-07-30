#!/usr/bin/env bash
# SEAL link smoke test — runs inside a one-shot pod to confirm the image
# at registry:latest can actually compile + link against libseal. Use
# whenever a `make eval-tiptoe-go` failure makes you suspect the image
# is stale relative to the most recent Dockerfile fix.
#
# Submit from the jumphost:
#   deploy/runai-submit-cpu.sh scripts/smoke_seal_link.sh
#
# Fires in 1–2 min; status is the script's exit code (0 = link works,
# non-zero = link fails — paste the runai logs into chat).

set -u

echo "================ smoke_seal_link ================"
echo "$(date -Iseconds) on $(uname -n) (pod image probably at registry:latest)"
echo

echo "--- image-level state -------------------------------"
echo "## /usr/local/lib SEAL files"
ls -la /usr/local/lib/libseal* 2>&1 || echo "(no libseal* — install missing?)"
echo
echo "## /usr/local/include SEAL headers (top-level and subdirs)"
ls -d /usr/local/include/seal /usr/local/include/SEAL-4.1 2>&1 || true
echo
echo "## env vars baked into the image"
echo "CGO_CXXFLAGS=${CGO_CXXFLAGS:-(unset)}"
echo "CGO_LDFLAGS=${CGO_LDFLAGS:-(unset)}"
echo "LIBRARY_PATH=${LIBRARY_PATH:-(unset)}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(unset)}"
echo
echo "## which ld / g++ — toolchain origin matters"
which ld g++ 2>&1
echo

echo "--- minimal compile + link with image's CGO_* -------"
TMP=$(mktemp -d)
cat > "$TMP/seal_smoke.cpp" <<'CPP'
#include <seal/seal.h>
int main() {
    seal::EncryptionParameters p(seal::scheme_type::bfv);
    p.set_poly_modulus_degree(2048);
    return 0;
}
CPP

set -x
g++ "$TMP/seal_smoke.cpp" -o "$TMP/seal_smoke" ${CGO_CXXFLAGS:-} ${CGO_LDFLAGS:-}
gpp_exit=$?
set +x
echo "g++ exit: $gpp_exit"
if [ "$gpp_exit" -eq 0 ]; then
    echo "--- ldd on the produced binary ---"
    ldd "$TMP/seal_smoke" 2>&1
    echo "--- can we run it? ---"
    "$TMP/seal_smoke" && echo "RUNTIME: ok"
fi

echo
echo "--- repeat the test the harness actually runs --------"
# The bundle's failing step is `make eval-tiptoe-go`, which `cargo run`s
# eval-harness's `tiptoe_go_runner` binary, which `go build`s the
# vendored ahenzinger/tiptoe Go reference. underhood/rlwe is what
# pulls in <seal/seal.h>. Recreate the link step directly here without
# needing the harness to drive it.
GO_REPO=tmp/tiptoe-go
if [ -d "$GO_REPO" ]; then
    echo "Found vendored Go ref at $GO_REPO; trying go build of paired-runner…"
    cd "$GO_REPO"
    go build -o /tmp/paired-runner ./cmd/paired-runner/ 2>&1
    echo "go build exit: $?"
else
    echo "(no $GO_REPO checkout in working dir — skip the full go build test)"
fi

echo
echo "================ smoke_seal_link done ================"
echo "Pass criterion: g++ exit = 0  AND  ldd resolves libseal*  AND"
echo "                go build exit = 0 (if Go ref was present)."
exit $gpp_exit
