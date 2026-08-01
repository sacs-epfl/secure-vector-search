// EMVP server-side block-GEMV over F_p (p = 2^61 − 1).
//
// Computes the same partial-product matrix as
// `scorer_emvp::crypto::compute_products`:
//
//   partials[si * m + i] = sum_{l=0..b}  m_hat[i*N + si*b + l] * q_hat[si*b + l]   (mod P)
//
// for si ∈ [0, S) and i ∈ [0, m). Output is block-major (S × m).
//
// Constants are SEC128:
//   N = 1292       (= S × B)
//   S = 76         (block count)
//   B = 17         (block size)
//   P = 2^61 - 1   (Mersenne prime)
//
// Kernel design
// -------------
//   - One thread per (i, si) output cell, blocked into 16×16 thread blocks.
//   - Grid: (ceil(m/16), ceil(S/16)) — a tiny grid for our typical
//     m=316, S=76: 20 × 5 = 100 thread blocks. Plenty of warps; SM
//     occupancy is bound by per-thread register usage rather than block
//     count at this size.
//   - Each thread sequentially walks its B=17 inner loop, accumulating
//     in F_p. No shared memory needed for the inner loop — q_hat is
//     small (76*17*8 = 10.3 KB total, fits in L1) and m_hat[i, ·] is
//     row-private.
//
// Mersenne reduction
// ------------------
// 64-bit × 64-bit → 128-bit on the device:
//   __umul64hi for the high 64 bits, regular `*` for the low 64.
// Then fold via the Mersenne identity 2^61 ≡ 1 (mod P):
//   x = (x mod 2^61) + (x >> 61)        (mod P)
// Two folds suffice because the input lives in [0, P)² < 2^122
// and the first fold lands in < 2^62 + 2^61 < 2^63, the second in
// < 2^61 + 4 ≈ P + 4 — one final conditional subtract handles the
// last carry.
//
// Accumulator stays in [0, P): reduce after every add. P < 2^61 so
// `acc + x` fits in u64 without overflow.

#include <cuda_runtime.h>
#include <stdint.h>

// SEC128 (must match scorer_emvp::params::SEC128). The Rust side asserts
// these at runtime against `SEC128` so a drift between the two trips a
// loud error rather than producing silent garbage.
#define EMVP_N 1292
#define EMVP_S 76
#define EMVP_B 17

// p = 2^61 - 1. Cast to uint64_t at the use site; declare as a literal
// so the compiler keeps it in an immediate.
#define EMVP_P ((uint64_t)((1ULL << 61) - 1ULL))

// Mersenne reduction of a full 128-bit product. The two-fold pattern
// matches scorer-bntm/src/crypto.rs:38-50 (the canonical reference);
// we use __umul64hi here because device intrinsics produce the hi/lo
// split in two ops.
__device__ __forceinline__ uint64_t fp_mul(uint64_t a, uint64_t b) {
    uint64_t hi = __umul64hi(a, b);
    uint64_t lo = a * b;

    // x = (lo, hi) as a 128-bit value.
    // Stage 1: x mod 2^61 = lo & P  (since P = 2^61 - 1 is the bottom
    //          61 bits all-ones), and x >> 61 = (hi << 3) | (lo >> 61).
    uint64_t mod61 = lo & EMVP_P;
    uint64_t high  = (hi << 3) | (lo >> 61);
    uint64_t s1    = mod61 + high;             // < 2^62 + 2^61

    // Stage 2: fold s1 again. s1 < 2^63, so its top bits past 61 fit
    // in 2 bits.
    uint64_t s2_lo = s1 & EMVP_P;
    uint64_t s2_hi = s1 >> 61;
    uint64_t v     = s2_lo + s2_hi;            // < P + 4

    if (v >= EMVP_P) v -= EMVP_P;
    return v;
}

__device__ __forceinline__ uint64_t fp_add(uint64_t a, uint64_t b) {
    uint64_t v = a + b;
    if (v >= EMVP_P) v -= EMVP_P;
    return v;
}

// Each thread computes one output cell partials[si * m + i].
//   gridDim:  (ceil(m/16), ceil(S/16))
//   blockDim: (16, 16)
extern "C" __global__ void emvp_compute_products(
    const uint64_t* __restrict__ m_hat,   // m × N row-major
    const uint64_t* __restrict__ q_hat,   // N
    uint64_t*       __restrict__ partials, // S × m block-major
    uint32_t                     m
) {
    uint32_t i  = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t si = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || si >= EMVP_S) return;

    uint32_t row_base = i * EMVP_N + si * EMVP_B;
    uint32_t q_base   = si * EMVP_B;

    uint64_t acc = 0;
    #pragma unroll
    for (int l = 0; l < EMVP_B; ++l) {
        uint64_t prod = fp_mul(m_hat[row_base + l], q_hat[q_base + l]);
        acc = fp_add(acc, prod);
    }
    partials[(uint64_t)si * (uint64_t)m + (uint64_t)i] = acc;
}
