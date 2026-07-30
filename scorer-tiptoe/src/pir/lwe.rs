//! Regev-LWE primitive used by SimplePIR and Tiptoe's LHE ranking step.
//!
//! In-house port of the Go reference's LWE layer.
//!
//! Conventions:
//!
//! - Ciphertext modulus is implicit `q = 2^64`; arithmetic is `u64`
//!   wrapping. Tiptoe's ranking layer uses the 64-bit regime
//!   exclusively (`logq = 64`); the 32-bit regime is for the URL
//!   service and is out of scope.
//! - Plaintext modulus `P` must be a power of 2 — required by the LHE
//!   flavour ("LHE requires p | q"). We encode it as `log2_p: u8` so
//!   this is true by construction.
//! - LWE secrets are **ternary in `{0, 1, 2}`** (not `{-1, 0, 1}`).
//!   Stored as `Vec<u8>`, cast to `u64` for arithmetic.
//! - Error distribution: discrete Gaussian centred at 0, std σ from
//!   `Params`. The Go reference samples via a precomputed CDF table
//!   (~6500 entries); we use `rand_distr::Normal` rounded to the
//!   nearest integer, which produces noise of the same magnitude. The
//!   recall-vs-Go-ref validation gate is the empirical check on whether
//!   the exact sampler shape matters at our parameters.
//!
//! Module surface:
//!
//! - [`Params`] — LWE parameter set; `Params::tiptoe_text()` returns the
//!   Tiptoe-text-search regime mirrored from the Go reference.
//! - [`SecretKey`] — ternary secret vector.
//! - [`gen_secret`] — fresh random secret.
//! - [`sample_gauss`] — single discrete-Gaussian draw.
//! - [`encrypt_vec`] — encrypt a Z_p^m message vector under a public
//!   `m × n` matrix `A` (caller-supplied).
//! - [`compute_a_s`] — compute `A · s` (the "hint" when the database is
//!   identity; used by the LWE-only roundtrip test). At the SimplePIR
//!   layer this becomes `H · s = (M_db · A) · s` and is precomputed
//!   server-side once per database.
//! - [`decrypt_vec`] — given a ciphertext and `A · s`, recover the
//!   plaintext vector.

use rand::{Rng, RngExt};
use rand_distr::{Distribution, Normal};

/// LWE parameters mirrored from the Go reference (logq = 64 regime).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Lattice secret dimension. `n = 2048` for Tiptoe text-search.
    pub n: usize,
    /// `log2(plaintext modulus)`. `P = 2^log2_p`. Must be `≤ 63`.
    /// Tiptoe text: `log2_p = 17` (P = 2^17).
    pub log2_p: u8,
    /// LWE error standard deviation. `σ = 81920.0` for logq = 64.
    pub sigma: f64,
}

impl Params {
    /// Tiptoe text-search ranking parameters.
    pub const fn tiptoe_text() -> Self {
        Self {
            n: 2048,
            log2_p: 17,
            sigma: 81920.0,
        }
    }

    /// Plaintext modulus `P = 2^log2_p`.
    #[inline]
    pub const fn p(&self) -> u64 {
        1u64 << self.log2_p
    }

    /// Plaintext multiplier `Δ = q / P = 2^(64 - log2_p)`. Computable in
    /// `u64` here because `64 - log2_p < 64` (the Go reference needs
    /// big-integer arithmetic because `1 << 64` overflows `uint64`).
    #[inline]
    pub const fn delta(&self) -> u64 {
        1u64 << (64 - self.log2_p)
    }

    /// Round a noisy ciphertext-modulus value to the nearest multiple
    /// of `Δ` and reduce mod `P`:
    ///
    /// ```go
    /// v := (x + p.Delta/2) / p.Delta
    /// return v % p.P
    /// ```
    ///
    /// The wrapping addition is intentional: when the underlying
    /// "real" value of `x` is slightly negative (e.g. `Δ·m + e` with
    /// `e < 0`, represented as `(Δ·m + e) mod 2^64`), `x + Δ/2`
    /// overflows `u64` exactly when needed. Because `q / Δ = P`
    /// (`P` divides `q`), the overflow is absorbed by the final
    /// `mod P`.
    #[inline]
    pub const fn round(&self, x: u64) -> u64 {
        let delta = self.delta();
        let v = x.wrapping_add(delta / 2) / delta;
        v % self.p()
    }
}

/// LWE secret key — ternary in {0, 1, 2}, length `Params::n`.
///
/// Stored as `u8` to make the value range visible at the type level
/// and to halve memory vs `u64`. Cast to `u64` (zero-extending) for
/// arithmetic.
#[derive(Clone)]
pub struct SecretKey(Vec<u8>);

impl std::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Redact: never print secret bytes.
        write!(f, "SecretKey(<redacted, n={}>)", self.0.len())
    }
}

impl SecretKey {
    /// Length (must equal `Params::n`).
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Read access for SimplePIR matrix-multiply paths in this crate.
    #[allow(dead_code)] // consumed by `pir/simplepir.rs`.
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

/// Sample a fresh LWE secret. Each component is uniformly drawn from
/// `{0, 1, 2}`.
pub fn gen_secret<R: Rng + ?Sized>(params: &Params, rng: &mut R) -> SecretKey {
    let mut data = vec![0u8; params.n];
    for slot in data.iter_mut() {
        *slot = rng.random_range(0u8..3);
    }
    SecretKey(data)
}

/// Single discrete-Gaussian draw centred at 0 with std `σ`.
///
/// **Approximation, not a port:** the Go reference samples via a
/// precomputed CDF table (~6500 entries). We use `rand_distr::Normal`
/// rounded to the nearest integer, which gives noise of the same
/// magnitude. The recall-vs-Go-ref validation gate is the empirical
/// check on whether the distribution shape (vs magnitude) matters at
/// our parameters.
pub fn sample_gauss<R: Rng + ?Sized>(sigma: f64, rng: &mut R) -> i64 {
    let normal = Normal::new(0.0, sigma).expect("σ > 0 by construction");
    normal.sample(rng).round() as i64
}

/// Compute `A · s` where `A` is row-major `m × n` over `Z_q = Z_{2^64}`
/// and `s` is the ternary secret. Output: column vector of length `m`.
///
/// At the SimplePIR layer this generalises to `H · s` where
/// `H = M_db · A` (the precomputed hint). Here we expose the bare
/// `A · s` form because the LWE-only roundtrip test treats the
/// "database" as identity.
pub fn compute_a_s(a: &[u64], s: &SecretKey, m: usize) -> Vec<u64> {
    let n = s.0.len();
    assert_eq!(
        a.len(),
        m * n,
        "A must be row-major m × n: got len={}, expected {}",
        a.len(),
        m * n
    );
    let mut out = vec![0u64; m];
    for (i, out_i) in out.iter_mut().enumerate() {
        let row = &a[i * n..(i + 1) * n];
        let mut acc: u64 = 0;
        for (&a_ij, &s_j) in row.iter().zip(s.0.iter()) {
            // s[j] ∈ {0, 1, 2}; cast to u64 (zero-extend).
            acc = acc.wrapping_add(a_ij.wrapping_mul(s_j as u64));
        }
        *out_i = acc;
    }
    out
}

/// Encrypt a plaintext vector `msg ∈ Z_p^m` under public matrix
/// `A ∈ Z_q^{m × n}` (row-major) and secret `s`. Output ciphertext is
/// `ct = A·s + e + Δ·msg ∈ Z_q^m`.
pub fn encrypt_vec<R: Rng + ?Sized>(
    s: &SecretKey,
    msg: &[u64],
    a: &[u64],
    params: &Params,
    rng: &mut R,
) -> Vec<u64> {
    let m = msg.len();
    assert_eq!(a.len(), m * s.0.len(), "A dimension mismatch");
    let mut ct = compute_a_s(a, s, m);
    let delta = params.delta();
    let p = params.p();
    for i in 0..m {
        let e = sample_gauss(params.sigma, rng) as u64; // u64 wrap = signed mod 2^64
        let m_red = msg[i] % p;
        ct[i] = ct[i]
            .wrapping_add(e)
            .wrapping_add(delta.wrapping_mul(m_red));
    }
    ct
}

/// Decrypt a ciphertext given a precomputed `A · s` (the "hint
/// product"). Output: plaintext vector in `Z_p^m`. Subtract the
/// intermediate, then round each component.
pub fn decrypt_vec(ct: &[u64], a_s: &[u64], params: &Params) -> Vec<u64> {
    assert_eq!(ct.len(), a_s.len(), "ct and A·s must have same length");
    ct.iter()
        .zip(a_s.iter())
        .map(|(&c, &h)| params.round(c.wrapping_sub(h)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn rng() -> ChaCha20Rng {
        ChaCha20Rng::seed_from_u64(42)
    }

    /// Sample a uniform `m × n` matrix over `Z_q = Z_{2^64}`.
    fn random_a<R: Rng + ?Sized>(rng: &mut R, m: usize, n: usize) -> Vec<u64> {
        let mut a = vec![0u64; m * n];
        for slot in a.iter_mut() {
            *slot = rng.random();
        }
        a
    }

    #[test]
    fn params_text_constants() {
        let p = Params::tiptoe_text();
        assert_eq!(p.n, 2048);
        assert_eq!(p.log2_p, 17);
        assert_eq!(p.sigma, 81920.0);
        assert_eq!(p.p(), 1 << 17);
        assert_eq!(p.delta(), 1u64 << 47);
        // q / Δ = 2^64 / 2^47 = 2^17 = P (the LHE divisibility constraint).
    }

    #[test]
    fn round_zero() {
        let p = Params::tiptoe_text();
        assert_eq!(p.round(0), 0);
    }

    #[test]
    fn round_exact_multiples_of_delta() {
        let p = Params::tiptoe_text();
        let delta = p.delta();
        for m in [0u64, 1, 2, 100, p.p() - 1] {
            assert_eq!(p.round(delta.wrapping_mul(m)), m, "exact m={m}");
        }
    }

    #[test]
    fn round_handles_small_positive_noise() {
        let p = Params::tiptoe_text();
        let delta = p.delta();
        // Δ·m + e for small positive e; should still round to m.
        for e in [0u64, 1, 100, 1_000_000, delta / 2 - 1] {
            assert_eq!(p.round(delta.wrapping_mul(7).wrapping_add(e)), 7, "e={e}");
        }
    }

    #[test]
    fn round_handles_small_negative_noise() {
        let p = Params::tiptoe_text();
        let delta = p.delta();
        // Δ·m + e for small NEGATIVE e (represented as wrap-around);
        // should still round to m.
        for e_pos in [1u64, 100, 1_000_000, delta / 2 - 1] {
            let neg = (0u64).wrapping_sub(e_pos);
            assert_eq!(
                p.round(delta.wrapping_mul(7).wrapping_add(neg)),
                7,
                "e=-{e_pos}"
            );
        }
    }

    #[test]
    fn gen_secret_is_ternary() {
        let mut rng = rng();
        let p = Params::tiptoe_text();
        let s = gen_secret(&p, &mut rng);
        assert_eq!(s.len(), 2048);
        for &v in s.as_slice() {
            assert!(v <= 2, "secret entry {v} out of range");
        }
    }

    #[test]
    fn gen_secret_distribution_is_balanced() {
        // Sanity check that all three values appear at our N=2048.
        let mut rng = rng();
        let p = Params::tiptoe_text();
        let s = gen_secret(&p, &mut rng);
        let mut counts = [0usize; 3];
        for &v in s.as_slice() {
            counts[v as usize] += 1;
        }
        for (i, c) in counts.iter().enumerate() {
            // At N=2048 with uniform [0,3), each bucket ~ 683 ± O(√N).
            // Loose bound: each must be at least N/4.
            assert!(*c > 2048 / 4, "value {i} severely under-represented: {c}");
        }
    }

    #[test]
    fn debug_redacts_secret() {
        let mut rng = rng();
        let p = Params::tiptoe_text();
        let s = gen_secret(&p, &mut rng);
        let dbg = format!("{s:?}");
        assert!(dbg.contains("redacted"), "Debug must redact: got {dbg:?}");
        // No actual byte values leak: the only digit-bearing token
        // permitted is the length annotation.
        assert!(dbg.contains("n=2048"));
    }

    #[test]
    fn gauss_sampler_magnitude() {
        // Sanity: mean should be ~0 and std should be ~σ. Loose bounds.
        let sigma = 81920.0;
        let mut rng = rng();
        let n = 10_000;
        let samples: Vec<i64> = (0..n).map(|_| sample_gauss(sigma, &mut rng)).collect();
        let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let var: f64 = samples
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let std = var.sqrt();
        assert!(mean.abs() < sigma, "|mean|={mean} too large vs σ={sigma}");
        assert!(
            (std - sigma).abs() < sigma * 0.1,
            "std={std} not within 10% of σ={sigma}"
        );
    }

    #[test]
    fn encrypt_decrypt_roundtrip_small() {
        // Use Tiptoe-text params (real N) with a small message vector m=4.
        let params = Params::tiptoe_text();
        let mut rng = rng();
        let s = gen_secret(&params, &mut rng);
        let m = 4usize;
        let a = random_a(&mut rng, m, params.n);
        let msg = vec![0u64, 1, params.p() - 1, params.p() / 2];

        let ct = encrypt_vec(&s, &msg, &a, &params, &mut rng);
        let a_s = compute_a_s(&a, &s, m);
        let recovered = decrypt_vec(&ct, &a_s, &params);

        assert_eq!(recovered, msg);
    }

    #[test]
    fn homomorphic_addition_under_encryption() {
        // Enc(v1) + Enc(v2) (componentwise wrapping u64 add) decrypts
        // to (v1 + v2) mod P. This is the essence of the LHE flavour.
        let params = Params::tiptoe_text();
        let mut rng = rng();
        let s = gen_secret(&params, &mut rng);
        let m = 4usize;
        let a = random_a(&mut rng, m, params.n);
        let v1 = vec![3u64, 100, params.p() / 4, 7];
        let v2 = vec![5u64, 200, params.p() / 4, 11];

        let ct1 = encrypt_vec(&s, &v1, &a, &params, &mut rng);
        let ct2 = encrypt_vec(&s, &v2, &a, &params, &mut rng);

        // Adding ciphertexts requires also adding the A·s contributions
        // on the recovery side: ct1 + ct2 = 2·A·s + (e1+e2) + Δ·(v1+v2).
        let ct_sum: Vec<u64> = ct1
            .iter()
            .zip(ct2.iter())
            .map(|(&a, &b)| a.wrapping_add(b))
            .collect();
        let a_s = compute_a_s(&a, &s, m);
        let two_a_s: Vec<u64> = a_s.iter().map(|&v| v.wrapping_add(v)).collect();
        let recovered = decrypt_vec(&ct_sum, &two_a_s, &params);

        let p = params.p();
        let expected: Vec<u64> = v1.iter().zip(&v2).map(|(a, b)| (a + b) % p).collect();
        assert_eq!(recovered, expected);
    }

    #[test]
    fn ciphertext_independent_of_message_size_within_p() {
        // Boundary: msg values near P-1 still decrypt correctly.
        let params = Params::tiptoe_text();
        let mut rng = rng();
        let s = gen_secret(&params, &mut rng);
        let m = 3usize;
        let a = random_a(&mut rng, m, params.n);
        let msg = vec![params.p() - 1, params.p() - 2, params.p() - 3];

        let ct = encrypt_vec(&s, &msg, &a, &params, &mut rng);
        let a_s = compute_a_s(&a, &s, m);
        let recovered = decrypt_vec(&ct, &a_s, &params);

        assert_eq!(recovered, msg);
    }
}
