//! `f32` ↔ `Z_p` quantisation for Tiptoe.
//!
//! Fixed-point encoding pipeline mirroring the Go reference's clamp and
//! quantisation, reconciled with the deployed "signed 4-bit" wording.
//!
//! Conventions:
//!
//! - `quantisation_bits` is the **magnitude** width. Tiptoe deploys 5
//!   slot-bits = 4 magnitude bits + 1 sign bit, so our default of 4
//!   matches the deployed range `[-15, 15]`.
//! - The plaintext modulus `P` is supplied as `log2_p` (it must be a
//!   power of two — the LHE flavour requires `P | q`, see
//!   `pir::lwe::Params`).
//! - Signed values are represented in `Z_p` via the standard
//!   "negative-mod-p" convention: `x ↦ x mod P` with `x` in
//!   `[-(2^q − 1), 2^q − 1]`.

use thiserror::Error;

/// Encoding-time precondition violations. Quantisation itself is
/// infallible once parameters are valid (`f32` components are clamped
/// to the slot range, matching the Go reference).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EncodingError {
    /// `quantisation_bits` is too large for the chosen plaintext modulus
    /// `P = 2^log2_p`: the signed slot range `[-(2^q − 1), 2^q − 1]`
    /// must fit strictly under `P/2` so signed values map injectively
    /// into `Z_p`.
    #[error(
        "quantisation_bits = {q} too large for plaintext modulus log2_p = {log2_p}: \
         signed slot range exceeds P/2"
    )]
    QuantisationOverflow { q: u8, log2_p: u8 },

    /// A non-finite `f32` (NaN/±∞) reached the encoder. Embeddings are
    /// expected unit-normalised; non-finite components are upstream bugs.
    #[error("non-finite component at index {index}")]
    NonFinite { index: usize },
}

/// Maximum signed magnitude `2^q − 1` for `q = quantisation_bits`.
#[inline]
pub const fn max_magnitude(quantisation_bits: u8) -> i32 {
    (1i32 << quantisation_bits) - 1
}

/// Validate that signed `quantisation_bits`-magnitude values map
/// injectively into `Z_p` (i.e., `2^q − 1 < P/2 = 2^(log2_p − 1)`).
pub fn validate_params(quantisation_bits: u8, log2_p: u8) -> Result<(), EncodingError> {
    if log2_p == 0 || quantisation_bits >= log2_p {
        return Err(EncodingError::QuantisationOverflow {
            q: quantisation_bits,
            log2_p,
        });
    }
    Ok(())
}

/// Quantise a single `f32` component to a `Z_p` element.
///
/// Steps: scale by `mag = 2^q − 1`, round to nearest integer, clamp to
/// `[-mag, mag]` (matching the Go reference's `embeddings.Clamp`), then
/// reduce modulo `P`. Caller must validate `quantisation_bits` against
/// `log2_p` via [`validate_params`] beforehand.
#[inline]
pub fn quantise_component(x: f32, quantisation_bits: u8, p: u64) -> u64 {
    let mag = max_magnitude(quantisation_bits) as i64;
    let scaled = (x * mag as f32).round() as i64;
    let clamped = scaled.clamp(-mag, mag);
    (clamped.rem_euclid(p as i64)) as u64
}

/// Encode a length-`d` `f32` vector into `Z_p^d`, writing into `out`.
/// `out.len()` must equal `v.len()`.
pub fn encode_vector(
    v: &[f32],
    quantisation_bits: u8,
    p: u64,
    out: &mut [u64],
) -> Result<(), EncodingError> {
    debug_assert_eq!(v.len(), out.len(), "encode_vector: length mismatch");
    for (i, &x) in v.iter().enumerate() {
        if !x.is_finite() {
            return Err(EncodingError::NonFinite { index: i });
        }
        out[i] = quantise_component(x, quantisation_bits, p);
    }
    Ok(())
}

/// Inverse of [`quantise_component`] for a single slot. The inner
/// product of two encoded vectors lives in `Z_p`; this maps a single
/// `Z_p` element back to its signed `f32` representation, useful for
/// score-magnitude tests and for dequantising the recovered `M·q̃` row.
///
/// Maps `[0, P/2) → [0, mag]` and `[P/2, P) → [-mag, 0)`, then divides
/// by `mag` to undo the scaling.
#[inline]
pub fn dequantise_component(z: u64, quantisation_bits: u8, p: u64) -> f32 {
    let half = p / 2;
    let signed = if z >= half {
        (z as i64) - (p as i64)
    } else {
        z as i64
    };
    let mag = max_magnitude(quantisation_bits) as f32;
    signed as f32 / mag
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pir::lwe::Params;

    #[test]
    fn max_magnitude_matches_paper() {
        assert_eq!(max_magnitude(4), 15);
        assert_eq!(max_magnitude(5), 31);
        assert_eq!(max_magnitude(8), 255);
    }

    #[test]
    fn quantise_zero_is_zero() {
        let p = Params::tiptoe_text().p();
        assert_eq!(quantise_component(0.0, 4, p), 0);
        assert_eq!(quantise_component(0.0, 8, p), 0);
    }

    #[test]
    fn quantise_one_is_max_magnitude() {
        let p = Params::tiptoe_text().p();
        assert_eq!(quantise_component(1.0, 4, p), 15);
    }

    #[test]
    fn quantise_negative_one_is_p_minus_max_magnitude() {
        let p = Params::tiptoe_text().p();
        assert_eq!(quantise_component(-1.0, 4, p), p - 15);
    }

    #[test]
    fn quantise_clamps_out_of_range_input() {
        let p = Params::tiptoe_text().p();
        // f32 inputs outside [-1, 1] are clamped to the slot edges,
        // matching the Go reference's `embeddings.Clamp`.
        assert_eq!(quantise_component(2.5, 4, p), 15);
        assert_eq!(quantise_component(-2.5, 4, p), p - 15);
    }

    #[test]
    fn dequantise_inverts_quantise_within_step() {
        let p = Params::tiptoe_text().p();
        let q = 4;
        let mag = max_magnitude(q) as f32;
        for k in -15..=15 {
            let x = k as f32 / mag;
            let z = quantise_component(x, q, p);
            let back = dequantise_component(z, q, p);
            assert!(
                (back - x).abs() < 0.5 / mag,
                "k={k}: x={x}, z={z}, back={back}"
            );
        }
    }

    #[test]
    fn encode_vector_round_trip_inside_slot_range() {
        let p = Params::tiptoe_text().p();
        let q = 4;
        let mag = max_magnitude(q) as f32;
        let v: Vec<f32> = (-7..=7).map(|k| k as f32 / mag).collect();
        let mut out = vec![0u64; v.len()];
        encode_vector(&v, q, p, &mut out).unwrap();
        for (i, (&x, &z)) in v.iter().zip(out.iter()).enumerate() {
            let back = dequantise_component(z, q, p);
            assert!((back - x).abs() < 0.5 / mag, "i={i}");
        }
    }

    #[test]
    fn encode_vector_rejects_non_finite() {
        let p = Params::tiptoe_text().p();
        let v = vec![0.1, f32::NAN, 0.3];
        let mut out = vec![0u64; 3];
        let err = encode_vector(&v, 4, p, &mut out).unwrap_err();
        assert_eq!(err, EncodingError::NonFinite { index: 1 });

        let v = vec![0.1, 0.2, f32::INFINITY];
        let err = encode_vector(&v, 4, p, &mut out).unwrap_err();
        assert_eq!(err, EncodingError::NonFinite { index: 2 });
    }

    #[test]
    fn validate_params_accepts_paper_default() {
        // Paper default at the Tiptoe-text regime: q=4 magnitude bits,
        // P = 2^17. Range [-15, 15] sits well below P/2 = 2^16.
        validate_params(4, 17).unwrap();
        validate_params(5, 17).unwrap();
        validate_params(8, 17).unwrap();
        validate_params(15, 17).unwrap();
    }

    #[test]
    fn validate_params_rejects_overflow() {
        // q ≥ log2_p means signed slots can't fit injectively in Z_p.
        assert!(matches!(
            validate_params(17, 17),
            Err(EncodingError::QuantisationOverflow { .. })
        ));
        assert!(matches!(
            validate_params(20, 17),
            Err(EncodingError::QuantisationOverflow { .. })
        ));
    }

    #[test]
    fn signed_dot_product_matches_plaintext_within_slot_budget() {
        // Sanity check: with quantisation_bits=4 and small d, the inner
        // product of two encoded vectors agrees mod P with the integer
        // inner product of their signed slot representations.
        let p = Params::tiptoe_text().p();
        let q = 4;
        let mag = max_magnitude(q) as f32;
        let a: Vec<f32> = vec![0.5, -0.3, 0.7, -1.0];
        let b: Vec<f32> = vec![1.0, -0.5, 0.2, 0.4];

        let mut a_enc = vec![0u64; a.len()];
        let mut b_enc = vec![0u64; b.len()];
        encode_vector(&a, q, p, &mut a_enc).unwrap();
        encode_vector(&b, q, p, &mut b_enc).unwrap();

        // Inner product computed in Z_p (server's view).
        let dot_zp: u64 = a_enc.iter().zip(b_enc.iter()).fold(0u64, |acc, (&x, &y)| {
            acc.wrapping_add(x.wrapping_mul(y)) % p
        });

        // Reference: the same inner product computed on signed slot
        // values, then reduced into Z_p.
        let dot_signed: i64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let xs = (x * mag).round() as i64;
                let ys = (y * mag).round() as i64;
                xs * ys
            })
            .sum();
        let dot_signed_zp = dot_signed.rem_euclid(p as i64) as u64;

        assert_eq!(dot_zp, dot_signed_zp);
    }
}
