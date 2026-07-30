# Source before building/running the GPU eval. Points the Rust GPU build
# (cudarc + cuvs-sys + nvcc-compiled kernels) at the RAPIDS conda env, since
# this host has no system /usr/local/cuda. Mirrors deploy/Dockerfile.
export RAPIDS=/home/rpereira/micromamba/envs/rapids
export PATH=$RAPIDS/bin:$PATH
export CUDA_HOME=$RAPIDS
export CMAKE_PREFIX_PATH=$RAPIDS
export LIBCLANG_PATH=$RAPIDS/lib
export BINDGEN_EXTRA_CLANG_ARGS="-I$RAPIDS/targets/x86_64-linux/include -I$RAPIDS/include"
export RUSTFLAGS="-L $RAPIDS/lib -L $RAPIDS/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH=$RAPIDS/lib:$RAPIDS/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
