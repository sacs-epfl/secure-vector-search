// Optimised BN server-side full GEMV over F_p (p = 2^61 − 1).
//
// Computes the same response vector as
// `scorer_bntm::crypto::compute_products`:
//
//   r[i] = sum_{j=0..N}  m_enc[i*N + j] * v_enc[j]   (mod P)
//
// for i ∈ [0, m). Output is a length-m vector.
//
// Constants are SEC128:
//   N = 1024       (BN ring dimension; matches BnTmParams::Sec128.n())
//   P = 2^61 - 1   (Mersenne prime)
//
// Kernel design
// -------------
// **Warp-per-row layout.** One thread block = one output row. 256
// threads in the block cooperatively compute the N=1024-element dot
// product. The block then writes a single u64 to r[blockIdx.x].
//
//   gridDim:  (m, 1, 1)     — one block per output row
//   blockDim: (256, 1, 1)   — 256 threads per block
//   shared mem: 8192 bytes  — caches v_enc[0..N] for the block
//
// This shape (vs. the original one-thread-per-row) gets two perf wins
// over the original kernel committed at 1cc39a9:
//
//   1. **Coalesced m_enc reads.** Adjacent threads in a warp read
//      adjacent elements of the same row (stride 8 bytes), so a warp
//      issues a single 256-byte memory transaction instead of 32
//      separate 8-byte loads. Effective HBM2 bandwidth on V100
//      jumps from ~30 GB/s to ~900 GB/s peak.
//
//   2. **Per-thread inner loop is fully unrollable.** Each thread
//      handles only N/256 = 4 products. With #pragma unroll the
//      compiler emits straight-line code, no loop overhead, no
//      periodic-reduce branch.
//
// Numeric range / overflow safety
// -------------------------------
// Each per-thread partial accumulates 4 u128 products without
// intermediate reduction. Each product is < P² < 2^122; four
// products sum to < 2^124. Fits in u128 with margin. After the
// per-thread mersenne_reduce the partial is < P < 2^61.
//
// The cross-warp reduction sums up to 256 u64 values < P. Doing this
// naively in u64 would overflow (P × 256 > 2^69). We reduce in two
// stages:
//
//   - Warp-shuffle reduce 32 lanes per warp: each accumulation step
//     adds two values < P, sum < 2P < 2^62. One conditional subtract
//     after each butterfly keeps it in [0, P).
//
//   - Cross-warp reduce 8 warp results via shared memory: sequential
//     adds in the first warp, with `add_mod` after each step (same
//     pattern, sum < 2P fits in u64).
//
// Mersenne arithmetic
// -------------------
// Same fp_reduce_u128 + add_mod / sub_mod helpers as the original
// kernel; reused verbatim from the 1cc39a9 commit. `sub_mod` is new
// here because the warp-shuffle reduction needs add (no sub) but I
// keep both for symmetry with the CPU reference.

#include <cuda_runtime.h>
#include <stdint.h>

#define BNTM_N  1024
#define BNTM_N1 512
#define BNTM_P ((uint64_t)((1ULL << 61) - 1ULL))

// Pre-reduce hi mod P so (hi * 8) doesn't overflow u64. See the
// original kernel's doc-comment (1cc39a9) for the bug this fix
// addresses; the function is verbatim from there.
__device__ __forceinline__ uint64_t fp_reduce_u128(uint64_t lo, uint64_t hi) {
    uint64_t hi_red = (hi & BNTM_P) + (hi >> 61);
    if (hi_red >= BNTM_P) hi_red -= BNTM_P;

    uint64_t hi_eff = hi_red << 3;

    uint64_t sum_lo = lo + hi_eff;
    uint64_t carry  = (sum_lo < lo) ? 1ULL : 0ULL;

    uint64_t s1_lo = sum_lo & BNTM_P;
    uint64_t s1_hi = (carry << 3) | (sum_lo >> 61);
    uint64_t v     = s1_lo + s1_hi;

    if (v >= BNTM_P) v -= BNTM_P;
    return v;
}

// Add two values in [0, P) and return the result in [0, P). Both
// inputs < P < 2^61 so a + b < 2^62 fits in u64.
__device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s >= BNTM_P) s -= BNTM_P;
    return s;
}

// Each block handles one output row.
extern "C" __global__ void bntm_compute_products(
    const uint64_t* __restrict__ m_enc,   // m × N row-major
    const uint64_t* __restrict__ v_enc,   // N
    uint64_t*       __restrict__ r,       // m
    uint32_t                     m
) {
    uint32_t i   = blockIdx.x;     // output row index
    uint32_t tid = threadIdx.x;    // 0..255

    if (i >= m) return;

    // Stage 1: cooperatively load v_enc[0..N] into shared memory.
    // Each thread loads N/blockDim.x = 4 elements with stride
    // blockDim.x = 256 (coalesced).
    __shared__ uint64_t v_shared[BNTM_N];
    #pragma unroll
    for (uint32_t k = 0; k < BNTM_N; k += 256) {
        v_shared[tid + k] = v_enc[tid + k];
    }
    __syncthreads();

    // Stage 2: per-thread partial dot product. Thread `tid` handles
    // elements at indices {tid, tid+256, tid+512, tid+768} of the
    // row — adjacent threads in a warp read adjacent elements
    // (coalesced stride-8B). Inner loop unrolled (4 iters).
    uint64_t row_base = (uint64_t)i * BNTM_N;

    uint64_t acc_lo = 0;
    uint64_t acc_hi = 0;

    #pragma unroll
    for (uint32_t k = 0; k < BNTM_N; k += 256) {
        uint64_t a = m_enc[row_base + tid + k];
        uint64_t b = v_shared[tid + k];
        uint64_t prod_lo = a * b;
        uint64_t prod_hi = __umul64hi(a, b);

        // 128-bit add: acc += prod.
        uint64_t new_lo = acc_lo + prod_lo;
        uint64_t carry  = (new_lo < acc_lo) ? 1ULL : 0ULL;
        acc_lo = new_lo;
        acc_hi = acc_hi + prod_hi + carry;
    }

    uint64_t partial = fp_reduce_u128(acc_lo, acc_hi);

    // Stage 3: warp-level reduction. Each warp collapses its 32
    // partial sums into one value via butterfly shuffles.
    #pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
        uint64_t other = __shfl_xor_sync(0xFFFFFFFFu, partial, offset);
        partial = add_mod(partial, other);
    }

    // Stage 4: cross-warp reduction. The first lane of each warp
    // writes its partial to shared memory; the first warp then sums
    // the 8 (= blockDim.x / 32) values sequentially.
    __shared__ uint64_t warp_partials[8];
    if ((tid & 31) == 0) {
        warp_partials[tid >> 5] = partial;
    }
    __syncthreads();

    if (tid < 8) {
        // Load only the slot owned by this thread to avoid races on
        // the shared-mem read; we only need 8 values total.
        uint64_t v = warp_partials[tid];
        // Butterfly across the 8 lanes of this single warp's bottom
        // half (mask covers tids 0..7).
        #pragma unroll
        for (uint32_t offset = 4; offset > 0; offset >>= 1) {
            uint64_t other = __shfl_xor_sync(0x000000FFu, v, offset);
            v = add_mod(v, other);
        }
        if (tid == 0) {
            r[i] = v;
        }
    }
}

// Dense GEMV over F_p at the n_1 = 512 inner dimension.
//
// Computes the same response vector as a CPU `dense_matvec` over an
// `m × n_1` matrix and an `n_1` query:
//
//   r[i] = sum_{j=0..N1}  mat[i*N1 + j] * v[j]   (mod P)
//
// Used twice per decode: once for `AL · G` and once for
// `H · (L⊤ · v_enc)`. The two output buffers are combined by the
// small `add_mod` kernel below.
//
// Shape is identical to `bntm_compute_products` above — warp-per-row,
// 256 threads, two-stage warp + cross-warp reduction — only the inner
// loop length and the shared-memory cache differ:
//
//   N1 = 512   ⇒ inner loop runs N1/blockDim = 2 iters per thread
//                (vs. 4 for N = 1024)
//   v_shared   ⇒ 512 × 8 = 4096 bytes (vs. 8192 bytes)
//
// Numeric range / overflow safety. Each per-thread partial accumulates
// 2 u128 products without intermediate reduction; each product < P² <
// 2^122, so the sum < 2^123 fits in u128 with margin. The warp + cross-
// warp reduction is unchanged from the N=1024 kernel and inherits its
// `add_mod` invariants. The `add_mod` / `fp_reduce_u128` helpers above
// are reused verbatim.
extern "C" __global__ void bntm_decode_dense(
    const uint64_t* __restrict__ mat,     // m × N1 row-major
    const uint64_t* __restrict__ v,       // N1
    uint64_t*       __restrict__ r,       // m
    uint32_t                     m
) {
    uint32_t i   = blockIdx.x;     // output row index
    uint32_t tid = threadIdx.x;    // 0..255

    if (i >= m) return;

    // Stage 1: cooperatively load v[0..N1] into shared memory. Two
    // strided loads per thread (N1 / 256 = 2).
    __shared__ uint64_t v_shared[BNTM_N1];
    #pragma unroll
    for (uint32_t k = 0; k < BNTM_N1; k += 256) {
        v_shared[tid + k] = v[tid + k];
    }
    __syncthreads();

    // Stage 2: per-thread partial dot product over the 2 elements
    // assigned to this thread (indices {tid, tid+256}). Coalesced
    // stride-8B reads on `mat`; inner loop fully unrolled.
    uint64_t row_base = (uint64_t)i * BNTM_N1;

    uint64_t acc_lo = 0;
    uint64_t acc_hi = 0;

    #pragma unroll
    for (uint32_t k = 0; k < BNTM_N1; k += 256) {
        uint64_t a = mat[row_base + tid + k];
        uint64_t b = v_shared[tid + k];
        uint64_t prod_lo = a * b;
        uint64_t prod_hi = __umul64hi(a, b);

        // 128-bit add: acc += prod.
        uint64_t new_lo = acc_lo + prod_lo;
        uint64_t carry  = (new_lo < acc_lo) ? 1ULL : 0ULL;
        acc_lo = new_lo;
        acc_hi = acc_hi + prod_hi + carry;
    }

    uint64_t partial = fp_reduce_u128(acc_lo, acc_hi);

    // Stage 3: warp-shuffle reduction. Identical to the N=1024 kernel.
    #pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
        uint64_t other = __shfl_xor_sync(0xFFFFFFFFu, partial, offset);
        partial = add_mod(partial, other);
    }

    // Stage 4: cross-warp reduction over the 8 warps in the block.
    __shared__ uint64_t warp_partials[8];
    if ((tid & 31) == 0) {
        warp_partials[tid >> 5] = partial;
    }
    __syncthreads();

    if (tid < 8) {
        uint64_t v_red = warp_partials[tid];
        #pragma unroll
        for (uint32_t offset = 4; offset > 0; offset >>= 1) {
            uint64_t other = __shfl_xor_sync(0x000000FFu, v_red, offset);
            v_red = add_mod(v_red, other);
        }
        if (tid == 0) {
            r[i] = v_red;
        }
    }
}

// Elementwise add over F_p, length-m.
//
// Combines the two `bntm_decode_dense` outputs (AL·G and
// H·(L⊤·v_enc)) into a single dense partial sum on device, avoiding
// an extra D2H/H2D pair. Both inputs are already in [0, P) so the
// reduction is a single add + conditional subtract per element.
//
//   grid: ceil(m / 256), block: 256.
extern "C" __global__ void bntm_add_mod_vec(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t*       __restrict__ out,
    uint32_t                     m
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    uint64_t s = a[i] + b[i];
    if (s >= BNTM_P) s -= BNTM_P;
    out[i] = s;
}
