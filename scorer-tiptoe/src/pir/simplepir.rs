//! SimplePIR LHE matrix-vector composition.
//!
//! In-house port of SimplePIR's LHE flavour. The protocol computes
//! `Enc(M_db · q)` under encryption given an encrypted query
//! `q ∈ Z_p^c`:
//!
//! - **Setup (corpus-fixed).** Server holds a database matrix
//!   `M_db ∈ Z_p^{r × c}` (row-major). A public matrix `A ∈ Z_q^{c × n}`
//!   is derived from a 32-byte PRG seed (`expand_a`). The server
//!   precomputes the hint `H = M_db · A ∈ Z_q^{r × n}`. The seed and
//!   the hint together form a [`PirSetup`].
//!
//! - **Per-query, client side.** Client holds a fresh ternary
//!   [`SecretKey`] `s ∈ {0,1,2}^n` (per Tiptoe's per-query single-use
//!   token semantics). The encrypted query is
//!   `ct = A·s + e + Δ·q ∈ Z_q^c` (computed by [`encrypt_lhe`], which
//!   streams `A` row-by-row from the seed via parallel
//!   `ChaCha20Rng::set_word_pos` seeks — no `c·n` matrix is ever
//!   materialised).
//!
//! - **Server answer.** `ans = M_db · ct ∈ Z_q^r`. By linearity,
//!   `ans = (M_db·A)·s + M_db·e + Δ·(M_db·q) = H·s + small_err
//!         + Δ·(M_db·q)` (mod `q`).
//!
//! - **Client recover.** Compute `H·s` from the stored hint and the
//!   client's secret, subtract from `ans`, round each component to
//!   nearest multiple of `Δ` and reduce mod `P`. The result is
//!   `M_db·q mod P`.

use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use super::lwe::{Params, SecretKey, compute_a_s, decrypt_vec, sample_gauss};

/// 32-byte PRG seed that determines the public matrix `A`. Server and
/// client must agree on it (server publishes the seed alongside the
/// hint; both derive `A` deterministically).
pub type PirSeed = [u8; 32];

/// Server-side preprocessed state for a fixed database `M_db`.
pub struct PirSetup {
    /// PRG seed for the public matrix `A ∈ Z_q^{c × n}`.
    pub seed: PirSeed,
    /// Hint `H = M_db · A`, row-major `r × n`.
    pub hint: Vec<u64>,
    /// Database height (rows).
    pub r: usize,
    /// Database width (columns); equals query length `c`.
    pub c: usize,
    /// Lattice secret dimension `n`. Mirrors `Params::n`.
    pub n: usize,
}

/// Derive the public `c × n` matrix `A` from a PRG seed. Each entry is
/// a uniformly random `u64`, sampled deterministically from
/// `ChaCha20(seed)` in row-major order.
pub fn expand_a(seed: &PirSeed, c: usize, n: usize) -> Vec<u64> {
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let mut a = vec![0u64; c * n];
    for slot in a.iter_mut() {
        *slot = rng.random();
    }
    a
}

/// Server-side: build the [`PirSetup`] from a row-major database
/// `M_db ∈ Z_p^{r × c}` and a fresh PRG seed.
///
/// `M_db` entries are conceptually `Z_p` values (`< params.p()`) but
/// passed as `u64` since arithmetic happens mod `q = 2^64`.
pub fn preprocess(m_db: &[u64], r: usize, c: usize, seed: PirSeed, params: &Params) -> PirSetup {
    preprocess_with_progress(m_db, r, c, seed, params, |_| {})
}

/// Same as [`preprocess`] but invokes `on_row(i)` after row `i` of the
/// hint `H = M_db · A` has been written. Used by the eval-harness to
/// drive a progress bar through the `O(r·c·n)` step. With the parallel,
/// loop-interchanged `mat_mul_u64` implementation below, row `i`
/// callbacks may arrive out of order across rayon workers; consumers
/// must treat `on_row` as a "rows done" counter, not an in-order
/// signal (the existing `IndicatifProgress::on_encrypt` impl is just
/// a counter, so this is a non-issue in practice).
pub fn preprocess_with_progress<F: Fn(usize) + Sync + Send>(
    m_db: &[u64],
    r: usize,
    c: usize,
    seed: PirSeed,
    params: &Params,
    on_row: F,
) -> PirSetup {
    assert_eq!(
        m_db.len(),
        r * c,
        "M_db must be row-major r × c: got len={}, expected {}",
        m_db.len(),
        r * c
    );
    let n = params.n;
    let a = expand_a(&seed, c, n);
    let hint = mat_mul_u64(m_db, r, c, &a, n, on_row);
    PirSetup {
        seed,
        hint,
        r,
        c,
        n,
    }
}

/// Client-side: encrypt a query vector `q ∈ Z_p^c` under the given
/// SimplePIR setup. Output is `ct = A·s + e + Δ·q ∈ Z_q^c`.
///
/// Streams `A` row-by-row from `setup.seed` rather than
/// materialising the `c × n` matrix. The byte order of the PRG
/// output matches [`expand_a`]'s row-major fill exactly, so the
/// ciphertext is bit-identical to the materialised-A path.
///
/// Parallelism: rows are partitioned into chunks of
/// `c / rayon::current_num_threads()` rows each; each worker
/// clones `ChaCha20Rng::from_seed(setup.seed)`, seeks to its
/// starting position via `set_word_pos` (each row consumes
/// `2·n` u32 words because each `rng.random::<u64>()` draws two
/// u32 words from ChaCha20's output stream), generates only its
/// slice of `A`, and accumulates partial `A·s` entries. Memory
/// footprint per worker is the row buffer `n × 8` bytes
/// (~16 KB at n = 2048), independent of `c`.
///
/// The noise (`sample_gauss`) and message-addition loop is kept
/// serial so the caller's `rng` state advances in the same
/// `i = 0..c` order as a sequential implementation. This is what
/// the bit-equality gate
/// (`tests::encrypt_lhe_matches_materialised_a`) checks.
pub fn encrypt_lhe<R: Rng + ?Sized>(
    setup: &PirSetup,
    secret: &SecretKey,
    query: &[u64],
    params: &Params,
    rng: &mut R,
) -> Vec<u64> {
    assert_eq!(
        query.len(),
        setup.c,
        "query length must equal database width c"
    );
    assert_eq!(secret.len(), setup.n, "secret dim must equal lattice dim n");

    let c = setup.c;
    let n = setup.n;
    let s_slice = secret.as_slice();
    let seed = setup.seed;

    let mut ct = vec![0u64; c];
    let threads = rayon::current_num_threads().max(1);
    let chunk_rows = c.div_ceil(threads).max(1);
    ct.par_chunks_mut(chunk_rows)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * chunk_rows;
            let mut chunk_rng = ChaCha20Rng::from_seed(seed);
            let word_pos = (start as u128) * (n as u128) * 2;
            chunk_rng.set_word_pos(word_pos);
            let mut row_buf = vec![0u64; n];
            for out_i in out_chunk.iter_mut() {
                for slot in row_buf.iter_mut() {
                    *slot = chunk_rng.random();
                }
                let mut acc: u64 = 0;
                for (&a_ij, &s_j) in row_buf.iter().zip(s_slice.iter()) {
                    acc = acc.wrapping_add(a_ij.wrapping_mul(s_j as u64));
                }
                *out_i = acc;
            }
        });

    let delta = params.delta();
    let p = params.p();
    for i in 0..c {
        let e = sample_gauss(params.sigma, rng) as u64;
        let m_red = query[i] % p;
        ct[i] = ct[i]
            .wrapping_add(e)
            .wrapping_add(delta.wrapping_mul(m_red));
    }
    ct
}

/// Server-side: compute `ans = M_db · ct ∈ Z_q^r`. By LHE linearity
/// this is `Enc(M_db · q)` modulo a small noise term that the client
/// rounds out in [`client_recover`].
pub fn server_answer(m_db: &[u64], r: usize, c: usize, ct: &[u64]) -> Vec<u64> {
    assert_eq!(m_db.len(), r * c);
    assert_eq!(ct.len(), c);
    let mut ans = vec![0u64; r];
    for (i, ans_i) in ans.iter_mut().enumerate() {
        let row = &m_db[i * c..(i + 1) * c];
        let mut acc: u64 = 0;
        for (&m_val, &c_val) in row.iter().zip(ct.iter()) {
            acc = acc.wrapping_add(m_val.wrapping_mul(c_val));
        }
        *ans_i = acc;
    }
    ans
}

/// Client-side: recover `M_db · q mod P` from the server's answer.
///
/// Computes `H·s` from the stored hint and the client's secret,
/// subtracts from `ans`, and rounds each component.
pub fn client_recover(
    ans: &[u64],
    setup: &PirSetup,
    secret: &SecretKey,
    params: &Params,
) -> Vec<u64> {
    assert_eq!(ans.len(), setup.r);
    assert_eq!(secret.len(), setup.n);
    let h_s = compute_a_s(&setup.hint, secret, setup.r);
    decrypt_vec(ans, &h_s, params)
}

/// Naive `r × c` × `c × k` → `r × k` matrix multiplication over
/// `Z_q = Z_{2^64}` (wrapping). Internal helper. `on_row(i)` fires
/// after row `i` of the result has been written, so the caller can
/// drive a progress bar through the slow path.
/// Compute `y = m · x` where `m` is `r × c` and `x` is `c × k`, all
/// row-major `u64` arrays under `wrapping_mul` / `wrapping_add` (mod
/// `2^64`).
///
/// Two performance properties matter for `preprocess_with_progress`'s
/// `r·c·n` hot path:
///
/// 1. **Outer `i` loop parallelised with rayon.** Each worker owns a
///    disjoint chunk of rows of `y`, so there is no write contention.
/// 2. **Loop order `i → j → kk` so the inner `kk` loop walks both
///    `y_row` and `x_row` contiguously.** At `k = 2048` (Tiptoe-text
///    LWE n), each row of `y` is 16 KB — comfortably L1-resident for
///    the duration of the `j` loop. `x_row` streams in 16 KB at a
///    time and is L2-prefetcher-friendly. The original `i → kk → j`
///    order strided into `x[j*k + kk]` with stride `k·8 = 16 KB`,
///    forcing an L1 miss every inner iteration — the bottleneck that
///    made the 100k cold build documented as "multi-hour" before
///    this rewrite. The saxpy-shaped inner loop autovectorises on
///    AVX2 (`vpmuludq` + `vpaddq`) and AVX-512 (`vpmullq` + `vpaddq`)
///    under `-C target-cpu=native`.
fn mat_mul_u64<F: Fn(usize) + Sync + Send>(
    m: &[u64],
    r: usize,
    c: usize,
    x: &[u64],
    k: usize,
    on_row: F,
) -> Vec<u64> {
    assert_eq!(m.len(), r * c);
    assert_eq!(x.len(), c * k);
    let mut y = vec![0u64; r * k];
    y.par_chunks_mut(k).enumerate().for_each(|(i, y_row)| {
        let m_row = &m[i * c..(i + 1) * c];
        for j in 0..c {
            let m_val = m_row[j];
            let x_row = &x[j * k..(j + 1) * k];
            for kk in 0..k {
                y_row[kk] = y_row[kk].wrapping_add(m_val.wrapping_mul(x_row[kk]));
            }
        }
        on_row(i);
    });
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pir::lwe::gen_secret;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn rng() -> ChaCha20Rng {
        ChaCha20Rng::seed_from_u64(42)
    }

    #[test]
    fn expand_a_is_deterministic_in_seed() {
        let seed = [3u8; 32];
        let a1 = expand_a(&seed, 8, 16);
        let a2 = expand_a(&seed, 8, 16);
        assert_eq!(a1, a2);
        let a3 = expand_a(&[4u8; 32], 8, 16);
        assert_ne!(a1, a3);
    }

    #[test]
    fn mat_mul_correctness_small() {
        // Hand-computed sanity:
        //   M = [[1, 2, 3], [4, 5, 6]]   (2 × 3)
        //   X = [[1, 0], [0, 1], [1, 1]] (3 × 2)
        //   M·X = [[1·1+2·0+3·1, 1·0+2·1+3·1],
        //          [4·1+5·0+6·1, 4·0+5·1+6·1]]
        //       = [[4, 5], [10, 11]]
        let m = vec![1u64, 2, 3, 4, 5, 6];
        let x = vec![1u64, 0, 0, 1, 1, 1];
        let rows_seen = std::sync::Mutex::new(Vec::new());
        let y = mat_mul_u64(&m, 2, 3, &x, 2, |i| rows_seen.lock().unwrap().push(i));
        assert_eq!(y, vec![4, 5, 10, 11]);
        // rayon may emit row callbacks out of order across workers; sort
        // before asserting the membership-equal check.
        let mut seen = rows_seen.into_inner().unwrap();
        seen.sort();
        assert_eq!(seen, vec![0, 1]);
    }

    #[test]
    fn unit_query_extracts_column() {
        // The vanilla-PIR special case: q = e_j (one-hot at column j).
        // Then M_db · q is column j of M_db.
        let params = Params::tiptoe_text();
        let mut r = rng();

        let r_dim = 4;
        let c_dim = 8;
        let m_db: Vec<u64> = (0..r_dim * c_dim)
            .map(|i| (i as u64 * 31 + 5) % params.p())
            .collect();
        let seed = [7u8; 32];
        let setup = preprocess(&m_db, r_dim, c_dim, seed, &params);
        assert_eq!(setup.hint.len(), r_dim * params.n);
        assert_eq!(setup.c, c_dim);
        assert_eq!(setup.r, r_dim);

        for col in 0..c_dim {
            let mut query = vec![0u64; c_dim];
            query[col] = 1;

            let secret = gen_secret(&params, &mut r);
            let ct = encrypt_lhe(&setup, &secret, &query, &params, &mut r);
            let ans = server_answer(&m_db, r_dim, c_dim, &ct);
            let recovered = client_recover(&ans, &setup, &secret, &params);

            let expected: Vec<u64> = (0..r_dim).map(|i| m_db[i * c_dim + col]).collect();
            assert_eq!(recovered, expected, "col={col}");
        }
    }

    #[test]
    fn general_lhe_query_decrypts_to_matrix_vector_mod_p() {
        // The LHE flavour: q is an arbitrary Z_p vector, not one-hot.
        let params = Params::tiptoe_text();
        let mut r = rng();

        let r_dim = 4;
        let c_dim = 6;
        let m_db: Vec<u64> = (0..r_dim * c_dim)
            .map(|i| (i as u64 * 17 + 3) % params.p())
            .collect();
        let query: Vec<u64> = (1..=c_dim as u64).collect();

        let seed = [11u8; 32];
        let setup = preprocess(&m_db, r_dim, c_dim, seed, &params);

        let secret = gen_secret(&params, &mut r);
        let ct = encrypt_lhe(&setup, &secret, &query, &params, &mut r);
        let ans = server_answer(&m_db, r_dim, c_dim, &ct);
        let recovered = client_recover(&ans, &setup, &secret, &params);

        let p = params.p();
        let expected: Vec<u64> = (0..r_dim)
            .map(|i| {
                let mut acc: u64 = 0;
                for (j, &q_j) in query.iter().enumerate() {
                    acc = acc.wrapping_add(m_db[i * c_dim + j].wrapping_mul(q_j));
                }
                acc % p
            })
            .collect();
        assert_eq!(recovered, expected);
    }

    #[test]
    fn zero_query_returns_zero_vector() {
        let params = Params::tiptoe_text();
        let mut r = rng();

        let r_dim = 3;
        let c_dim = 5;
        let m_db: Vec<u64> = (0..r_dim * c_dim).map(|i| i as u64 % params.p()).collect();
        let seed = [13u8; 32];
        let setup = preprocess(&m_db, r_dim, c_dim, seed, &params);

        let query = vec![0u64; c_dim];
        let secret = gen_secret(&params, &mut r);
        let ct = encrypt_lhe(&setup, &secret, &query, &params, &mut r);
        let ans = server_answer(&m_db, r_dim, c_dim, &ct);
        let recovered = client_recover(&ans, &setup, &secret, &params);
        assert_eq!(recovered, vec![0u64; r_dim]);
    }

    /// Bit-equality gate: the streaming
    /// `encrypt_lhe` must produce the same ciphertext as the
    /// pre-fix materialise-A path
    /// (`expand_a` + `lwe::encrypt_vec`) when given the same
    /// `(setup.seed, secret, query, noise_rng_state)`.
    ///
    /// Why this test exists: the perf rewrite for `encrypt_lhe`
    /// reshapes the computation (parallel row-chunked PRG +
    /// matvec) but is required to be byte-for-byte equivalent to
    /// the sequential implementation so that all downstream
    /// recall figures and the `analysis/tiptoe_diff.py`
    /// validation gate vs the Go reference remain valid without
    /// re-baselining. The ciphertext is the natural seam: it's
    /// deterministic given the inputs above (the only stochastic
    /// element is `sample_gauss`, which uses the caller's `rng`
    /// in identical i = 0..c order on both paths).
    ///
    /// Tests at `c = 1024` so each rayon chunk processes ≥ 1
    /// row (10-core M4 → chunk size 103 rows) while keeping the
    /// PRG word-position arithmetic non-trivial.
    #[test]
    fn encrypt_lhe_matches_materialised_a() {
        use crate::pir::lwe::encrypt_vec;

        let params = Params::tiptoe_text();
        let setup_seed = [0xABu8; 32];
        let secret_seed = 12_345u64;
        let query_seed = 67_890u64;
        let noise_seed = 11_111u64;

        let r_dim = 4;
        let c_dim = 1024;
        let n = params.n;

        // PirSetup is constructed directly — the hint isn't used
        // by encrypt_lhe so we skip the expensive preprocess.
        let setup = PirSetup {
            seed: setup_seed,
            hint: Vec::new(),
            r: r_dim,
            c: c_dim,
            n,
        };

        let mut secret_rng = ChaCha20Rng::seed_from_u64(secret_seed);
        let secret = gen_secret(&params, &mut secret_rng);

        let mut query_rng = ChaCha20Rng::seed_from_u64(query_seed);
        let query: Vec<u64> = (0..c_dim)
            .map(|_| query_rng.random_range(0u64..params.p()))
            .collect();

        // Path A: pre-fix (materialise A + encrypt_vec). Uses its
        // own noise rng seeded from the same value as Path B.
        let mut noise_rng_a = ChaCha20Rng::seed_from_u64(noise_seed);
        let a_mat = expand_a(&setup_seed, c_dim, n);
        let ct_pre_fix = encrypt_vec(&secret, &query, &a_mat, &params, &mut noise_rng_a);

        // Path B: streaming encrypt_lhe.
        let mut noise_rng_b = ChaCha20Rng::seed_from_u64(noise_seed);
        let ct_streaming = encrypt_lhe(&setup, &secret, &query, &params, &mut noise_rng_b);

        assert_eq!(
            ct_pre_fix, ct_streaming,
            "streaming encrypt_lhe must be byte-for-byte equal to materialised-A path"
        );
    }

    #[test]
    fn fresh_secrets_per_query_decrypt_correctly() {
        // Tiptoe's single-use token semantics: a fresh secret per
        // query. The protocol must handle this — the hint is corpus-
        // fixed but H·s changes per query.
        let params = Params::tiptoe_text();
        let mut r = rng();

        let r_dim = 4;
        let c_dim = 6;
        let m_db: Vec<u64> = (0..r_dim * c_dim)
            .map(|i| (i as u64 * 23 + 1) % params.p())
            .collect();
        let seed = [17u8; 32];
        let setup = preprocess(&m_db, r_dim, c_dim, seed, &params);

        let query = vec![2u64, 0, 5, 1, 3, 7];
        let p = params.p();
        let expected: Vec<u64> = (0..r_dim)
            .map(|i| {
                let mut acc: u64 = 0;
                for (j, &q_j) in query.iter().enumerate() {
                    acc = acc.wrapping_add(m_db[i * c_dim + j].wrapping_mul(q_j));
                }
                acc % p
            })
            .collect();

        for trial in 0..3 {
            let secret = gen_secret(&params, &mut r);
            let ct = encrypt_lhe(&setup, &secret, &query, &params, &mut r);
            let ans = server_answer(&m_db, r_dim, c_dim, &ct);
            let recovered = client_recover(&ans, &setup, &secret, &params);
            assert_eq!(recovered, expected, "trial={trial}");
        }
    }
}
