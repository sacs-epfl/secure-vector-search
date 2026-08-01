//! Hardness parameters and ring constants for the Braverman–Newman
//! trapdoored-matrix scorer.
//!
//! The concrete parameter tuple is a chosen fallback: no published
//! reference implementation exists to mirror. Retune triggers are
//! documented alongside each parameter.

/// Mersenne prime p = 2^61 − 1. Same ring as EMVP — keeps prime-field
/// arithmetic shared.
pub const P: u64 = (1u64 << 61) - 1;

/// Bytes per field element (u64).
pub const FIELD_BYTES: usize = 8;

/// f32 → F_p quantisation scale: round(v · Q). Q = 2^{20}, inherited
/// from EMVP and confirmed by sweep. An analytical bound would be
/// unreliable here (inner-product shapes differ across protocols), so
/// the empirical sweep is the authoritative check.
pub const Q: u64 = 1u64 << 20;

/// One BN parameter set.
///
/// **Convention.** Our `l_subspace` is shape `n × n_1`. In the standard
/// `LH + S` formulation `L` is `m × n_1`, so our `L` is its transpose
/// `L⊤`. The arithmetic in this file and in `crypto.rs` is in the
/// `HL⊤ + S` form throughout.
#[derive(Debug, Clone, Copy)]
pub enum BnTmParams {
    /// Standard-LPN parameter set, λ = 128, δ = 0.125 (`n_1` = 128),
    /// ε = 0.7 (`μ` = 0.125).
    ///
    /// δ retuned from the original δ = 0.5 fallback. A fixed
    /// δ = 0.5 defeats the trapdoor's own asymptotic point: paper §5
    /// (arXiv:2502.13060v3) states genuine client speedup needs
    /// δ = o(1), not a constant fraction of `n`.
    ///
    /// ε (and therefore `μ`) retuned separately (2026-07-28, see
    /// [`epsilon`](Self::epsilon)): the original ε = 0.5 gave μ = 1/32,
    /// which left the `δ·μ > 1/n` §7 floor cleared by only a small
    /// constant factor at our concrete `n = 1024` — nowhere near
    /// enough margin for the subsampling / Gaussian-elimination attack
    /// the paper names to actually cost anywhere close to 128 bits.
    /// ε = 0.7 quadruples that margin (4× → 16× at this δ) at
    /// negligible decode cost, since `μ` only weights the already-cheap
    /// sparse `S`/`T` terms. Equivalent to `Sec128Delta(3)`.
    Sec128,
    /// Same λ = 128 / `n` = 1024 / ε = 0.7 base as [`Self::Sec128`],
    /// with an explicit δ = 1 / 2^k for `k` in `1..=4`
    /// (δ ∈ {0.5, 0.25, 0.125, 0.0625}) — a sweep axis for exploring
    /// the decode-latency / security-margin tradeoff.
    ///
    /// `k = 4` (δ = 0.0625) is the smallest value this crate accepts,
    /// and it is **not** "the conservative choice" just because it
    /// clears the `δ·μ > 1/n` floor — at k=4 it's the *weakest* point
    /// in this sweep (smallest secret dimension `n_1`, least margin
    /// above the floor: 8× at ε = 0.7, was 2× before the ε retune).
    /// `k ≥ 5` would sit at or below the floor where the paper (§7)
    /// says subsampling / Gaussian-elimination attacks become
    /// efficient, so `n1()` panics rather than silently shipping an
    /// insecure parameter choice — but clearing that floor is a
    /// necessary condition, not a sufficient one, for any particular
    /// bit-security target at our concrete (small) `n`.
    ///
    /// The zero-noise recovery (§7.2) is δ-independent, so recall is
    /// exact at every `k` in range — only decode latency and setup
    /// bytes move. This crate does **not** compute the paper's formal
    /// `ν(δ,ε,λ)` bound (that requires porting or approximating the
    /// LPN-hardness estimator of [29] in the paper); `k` is chosen
    /// conservatively above the known hard floor, not proven optimal.
    Sec128Delta(u32),
}

impl BnTmParams {
    /// `δ = 1/2^k` exponent for this parameter set.
    fn delta_pow2(&self) -> u32 {
        match self {
            Self::Sec128 => 3,
            Self::Sec128Delta(k) => *k,
        }
    }

    /// Ring dimension. Relations: `n_1 = δn`, `μ = n^{ε-1}`.
    ///
    /// **Retune trigger:** move *up* if recall@10 on the 100k corpus
    /// drops more than 5 pp below plaintext-IVF at full probe (the
    /// protocol has zero LPN noise, so a recall gap implies the
    /// quantisation budget at this `n` is too tight). Move *down* only
    /// if the setup-bytes ceiling forces it.
    pub const fn n(&self) -> usize {
        1024
    }

    /// Subspace dimension `n_1 = δ · n = n / 2^k`.
    ///
    /// Panics if `delta_pow2() > 4` (this crate's reviewed, chosen
    /// ceiling — see [`Self::Sec128Delta`]) or if the paper's own
    /// `δ·μ > 1/n` security floor (§7) isn't cleared, checked
    /// *dynamically* against the live `delta()` / `mu()` values rather
    /// than a hardcoded margin comment — a stale comment is exactly
    /// how the ε = 0.5 / μ = 1/32 margin went unnoticed for as long as
    /// it did.
    pub fn n1(&self) -> usize {
        let k = self.delta_pow2();
        assert!(
            (1..=4).contains(&k),
            "BnTmParams: k = {k} is outside this crate's reviewed δ security \
             range (1..=4) — widen deliberately (re-deriving the δ·μ > 1/n \
             floor at the new k), not silently"
        );
        let floor = 1.0 / (self.n() as f64 * self.mu());
        assert!(
            self.delta() > floor,
            "BnTmParams: δ = 1/2^{k} = {} is at or below the δ·μ > 1/n security \
             floor (paper §7) at n = {}, μ = {} (floor = {floor})",
            self.delta(),
            self.n(),
            self.mu()
        );
        self.n() >> k
    }

    /// Subspace fraction `δ = 1 / 2^k`.
    ///
    /// **Retune trigger:** prefer the smallest `k` (in `1..=4`) that
    /// shows a measured decode-latency improvement in the eval-harness
    /// sweep. Move *up* (larger δ, smaller `k`) only if a
    /// cryptanalytic result against standard LPN at our `(n, μ)`
    /// surfaces, or if the per-query online response at `nprobe = 32`
    /// exceeds 1 MiB (EMVP+IVF is *projected* at ~6 MiB, so 1 MiB is
    /// BN's headroom).
    pub fn delta(&self) -> f64 {
        1.0 / (1u64 << self.delta_pow2()) as f64
    }

    /// Hardness exponent `ε`. Raised from the original 0.5 to 0.7
    /// (2026-07-28): at ε = 0.5, μ = 1/32 was low enough that the
    /// subsampling / Gaussian-elimination attack the paper names in §7
    /// (efficient once `δ·μ ≤ 1/n`) had far less margin than the
    /// `δ·μ > 1/n` check alone suggested — clearing that asymptotic
    /// floor by a small constant factor at our concrete `n = 1024`
    /// does not imply anything close to 128-bit concrete hardness.
    /// This is exactly the cryptanalytic-margin retune trigger this
    /// comment already named. ε = 0.7 gives μ = 1/8 = 0.125 — chosen
    /// so `μ` sits at parity with (not above) the default `Sec128`'s
    /// `δ = 0.125`, keeping the sparse `S`/`T` terms no more expensive
    /// than the dominant dense `O(m·n_1)` decode terms the scorer
    /// relies on, while quadrupling the `δ·μ` margin above the paper's §7
    /// floor at every δ in the `Sec128Delta` sweep (Sec128: 4× → 16×;
    /// k=4 / δ=0.0625, the sweep's weakest point: 2× → 8×).
    ///
    /// **Caveat, stated plainly:** these margin multiples are relative
    /// to the paper's asymptotic threshold, not a computed bit-security
    /// number — we still have no LPN-hardness estimator (`ν(δ,ε,λ)`'s
    /// `ι` term) to say what
    /// concrete security this buys. Raise further if a sharper
    /// cryptanalytic estimate says 0.125 still isn't enough at n=1024.
    pub const fn epsilon(&self) -> f64 {
        0.7
    }

    /// Noise rate / sparsity `μ = n^{ε - 1}`. Computed at runtime
    /// rather than hard-coded so a future const-eval-friendly
    /// rebinding of `(n, ε)` stays consistent. For Sec128 this
    /// returns 1024^{-0.3} = 1/8 = 0.125.
    ///
    /// **Retune trigger fired 2026-07-28:** a security-margin issue
    /// *did* surface against Conjecture 2 at our old `(n, ε=0.5)` —
    /// the `δ·μ > 1/n` floor was cleared by only a small constant
    /// factor, not enough margin for the paper's own named §7 attack
    /// to cost anywhere near 128 bits at our concrete `n`. Moved *up*
    /// via ε (0.5 → 0.7) accordingly; see [`epsilon`](Self::epsilon).
    ///
    /// **Retune trigger (still live):** move *down* (sparser) if the
    /// verification overhead exceeds 15% of `latency-us`, since
    /// Protocol 2's per-trial cost scales with the support size of `S`
    /// and `T` — but not below the point where `δ·μ > 1/n` stops
    /// holding with real margin; see `epsilon()`'s doc for the current
    /// reasoning.
    pub fn mu(&self) -> f64 {
        (self.n() as f64).powf(self.epsilon() - 1.0)
    }

    /// Matrix-multiplication exponent surrogate for the cost-bound
    /// formulae. Practical implementations run naive `O(n^3)` matmul
    /// on commodity hardware, so we set `ω = 3` for our setup-cost
    /// projections; the asymptotic `ω ≈ 2.37` is an upper-bound
    /// theoretical statement, not what the eval harness measures.
    pub const fn omega(&self) -> f64 {
        3.0
    }

    /// λ' for a 2^{-128} soundness target via Protocol 2. Per-trial
    /// soundness is ≈ |R|⁻¹, so λ' = ⌈128 / log₂|R|⌉. For our
    /// R = F_p with p = 2^61 − 1: λ' = ⌈128 / 61⌉ = 3, giving
    /// 2^{-183} actual soundness. A 32-bit integer ring (|R| = 2^32)
    /// would instead need λ' = 4. Derived from FIELD_BYTES so a
    /// future ring swap stays correct.
    pub const fn verification_trials(&self) -> usize {
        // Conservative lower bound on log2(|R|) given FIELD_BYTES = 8
        // for our Mersenne prime p = 2^61 − 1 (log2|R| ≈ 61). Holds
        // for any prime field whose bit-length is within 4 of
        // 8·FIELD_BYTES.
        let log2_ring_lower = 8 * FIELD_BYTES - 4;
        128_usize.div_ceil(log2_ring_lower)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sec128_relations_consistent() {
        let p = BnTmParams::Sec128;
        // n_1 = δ · n
        assert_eq!(p.n1(), (p.delta() * p.n() as f64) as usize);
        // μ = n^{ε - 1}: for n=1024, ε=0.7 this is 1/8 = 0.125
        assert!((p.mu() - 0.125).abs() < 1e-9);
    }

    #[test]
    fn lambda_prime_matches_target() {
        // 2^{-128} target over F_p with p = 2^61 − 1: λ' = 3.
        // A |R| = 2^32 ring would need λ' = 4; our ring is ~512×
        // larger so 3 trials suffice.
        let lp = BnTmParams::Sec128.verification_trials();
        assert_eq!(lp, 3);
        // Sanity: λ' · log2|R_lower| ≥ 128.
        let log2_ring_lower = 8 * FIELD_BYTES - 4;
        assert!(lp * log2_ring_lower >= 128);
    }

    #[test]
    fn ring_constants_static() {
        assert_eq!(P, (1u64 << 61) - 1);
        assert_eq!(FIELD_BYTES, 8);
        assert_eq!(Q, 1u64 << 20);
    }

    #[test]
    fn sec128_equivalent_to_delta_pow2_3() {
        let a = BnTmParams::Sec128;
        let b = BnTmParams::Sec128Delta(3);
        assert_eq!(a.n1(), b.n1());
        assert_eq!(a.delta(), b.delta());
        assert_eq!(a.n1(), 128);
        assert_eq!(a.delta(), 0.125);
    }

    #[test]
    fn delta_sweep_values_and_floor_margin() {
        // δ ∈ {0.5, 0.25, 0.125, 0.0625} for k ∈ 1..=4, all clearing the
        // δ·μ > 1/n floor (μ = 0.125 at our n = 1024, ε = 0.7 — raised
        // 2026-07-28 from the original ε = 0.5 / μ = 0.03125, which left
        // far less margin above this same floor).
        let floor = 1.0 / (BnTmParams::Sec128.n() as f64 * BnTmParams::Sec128.mu());
        assert!((floor - 0.0078125).abs() < 1e-9);
        let expect = [
            (1u32, 512usize, 0.5),
            (2, 256, 0.25),
            (3, 128, 0.125),
            (4, 64, 0.0625),
        ];
        for (k, n1, delta) in expect {
            let p = BnTmParams::Sec128Delta(k);
            assert_eq!(p.n1(), n1, "k={k}");
            assert!((p.delta() - delta).abs() < 1e-12, "k={k}");
            assert!(
                p.delta() > floor,
                "k={k}: δ={} must clear the security floor {floor}",
                p.delta()
            );
        }
    }

    #[test]
    #[should_panic(expected = "reviewed δ security range")]
    fn delta_pow2_five_outside_reviewed_range_panics() {
        // k=5 → δ=1/32=0.03125. With the retuned μ=0.125 this actually
        // clears the bare δ·μ > 1/n floor (floor=0.0078125), but this
        // crate only reviews/ships k in 1..=4 — n1() must still refuse
        // rather than silently widen the accepted range.
        let _ = BnTmParams::Sec128Delta(5).n1();
    }
}
