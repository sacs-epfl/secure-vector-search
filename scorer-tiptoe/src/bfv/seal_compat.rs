//! Parameter-parity gate against the Tiptoe Go reference's SEAL 4.1.1
//! BFV defaults.
//!
//! We adopt `fhe` / `fhe-traits` for the BFV layer contingent on this
//! check passing. The expected values from the Go reference are:
//!
//! - `poly_modulus_degree = 2048`
//! - `plain_modulus = 65537` (= 2¹⁶ + 1, prime)
//! - `coeff_modulus = CoeffModulus::Create(2048, 65537, {38})` — single
//!   38-bit prime modulus chain
//!
//! `fhe`'s `BfvParametersBuilder::set_moduli_sizes(&[38])` requests a
//! 38-bit NTT-compatible prime; the *specific* prime fhe picks need
//! not equal SEAL's prime (different libraries reasonably pick
//! different NTT-compatible primes for the same bit-width). Parity is
//! claimed at the **bit-width and structural level** — any mismatch
//! changes the noise budget and the BFV reconstruction's correctness
//! margin, so we fail-closed.

use fhe::bfv::BfvParameters;

/// Expected SEAL 4.1.1 BFV parameter values from the Go reference.
pub const EXPECTED_POLY_DEGREE: usize = 2048;
pub const EXPECTED_PLAINTEXT_MODULUS: u64 = 65537;
pub const EXPECTED_COEFF_MODULUS_BITS: &[usize] = &[38];

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SealCompatError {
    #[error("polynomial degree mismatch: fhe gives {actual}, SEAL 4.1.1 reference is {expected}")]
    PolyDegree { actual: usize, expected: usize },

    #[error("plaintext modulus mismatch: fhe gives {actual}, SEAL 4.1.1 reference is {expected}")]
    PlaintextModulus { actual: u64, expected: u64 },

    #[error(
        "coeff modulus chain length mismatch: fhe gives {actual} primes, SEAL 4.1.1 reference is {expected}"
    )]
    CoeffModulusLength { actual: usize, expected: usize },

    #[error(
        "coeff modulus bit-widths mismatch: fhe gives {actual:?}, SEAL 4.1.1 reference is {expected:?}"
    )]
    CoeffModulusBits {
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
}

/// Verify that `params` matches the SEAL 4.1.1 reference parameters
/// for Tiptoe's BFV layer.
///
/// Returns `Ok(())` on parity, otherwise the first kind of mismatch
/// detected. Caller is expected to fail-closed (panic / propagate)
/// rather than continue silently — any mismatch invalidates the BFV
/// noise-budget calculation that the Go reference's deployment relies
/// on.
pub fn verify(params: &BfvParameters) -> Result<(), SealCompatError> {
    if params.degree() != EXPECTED_POLY_DEGREE {
        return Err(SealCompatError::PolyDegree {
            actual: params.degree(),
            expected: EXPECTED_POLY_DEGREE,
        });
    }
    if params.plaintext() != EXPECTED_PLAINTEXT_MODULUS {
        return Err(SealCompatError::PlaintextModulus {
            actual: params.plaintext(),
            expected: EXPECTED_PLAINTEXT_MODULUS,
        });
    }
    let sizes = params.moduli_sizes();
    if sizes.len() != EXPECTED_COEFF_MODULUS_BITS.len() {
        return Err(SealCompatError::CoeffModulusLength {
            actual: sizes.len(),
            expected: EXPECTED_COEFF_MODULUS_BITS.len(),
        });
    }
    if sizes != EXPECTED_COEFF_MODULUS_BITS {
        return Err(SealCompatError::CoeffModulusBits {
            actual: sizes.to_vec(),
            expected: EXPECTED_COEFF_MODULUS_BITS.to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fhe::bfv::BfvParametersBuilder;

    fn good_params() -> BfvParameters {
        BfvParametersBuilder::new()
            .set_degree(EXPECTED_POLY_DEGREE)
            .set_plaintext_modulus(EXPECTED_PLAINTEXT_MODULUS)
            .set_moduli_sizes(EXPECTED_COEFF_MODULUS_BITS)
            .build()
            .expect("canonical SEAL-parity params must build under fhe")
    }

    #[test]
    fn canonical_params_pass() {
        let p = good_params();
        verify(&p).expect("canonical params must verify");
    }

    #[test]
    fn wrong_poly_degree_fails() {
        let p = BfvParametersBuilder::new()
            .set_degree(4096)
            .set_plaintext_modulus(EXPECTED_PLAINTEXT_MODULUS)
            .set_moduli_sizes(EXPECTED_COEFF_MODULUS_BITS)
            .build()
            .expect("4096-degree should still build under fhe");
        match verify(&p) {
            Err(SealCompatError::PolyDegree { actual, expected }) => {
                assert_eq!(actual, 4096);
                assert_eq!(expected, EXPECTED_POLY_DEGREE);
            }
            other => panic!("expected PolyDegree error, got {other:?}"),
        }
    }

    #[test]
    fn wrong_plaintext_modulus_fails() {
        let p = BfvParametersBuilder::new()
            .set_degree(EXPECTED_POLY_DEGREE)
            .set_plaintext_modulus(1024)
            .set_moduli_sizes(EXPECTED_COEFF_MODULUS_BITS)
            .build()
            .expect("plaintext=1024 should still build under fhe");
        match verify(&p) {
            Err(SealCompatError::PlaintextModulus { actual, expected }) => {
                assert_eq!(actual, 1024);
                assert_eq!(expected, EXPECTED_PLAINTEXT_MODULUS);
            }
            other => panic!("expected PlaintextModulus error, got {other:?}"),
        }
    }

    #[test]
    fn wrong_coeff_modulus_length_fails() {
        // Two 38-bit moduli instead of one — a multi-level BFV setup.
        let p = BfvParametersBuilder::new()
            .set_degree(EXPECTED_POLY_DEGREE)
            .set_plaintext_modulus(EXPECTED_PLAINTEXT_MODULUS)
            .set_moduli_sizes(&[38, 38])
            .build()
            .expect("two-modulus chain should still build under fhe");
        match verify(&p) {
            Err(SealCompatError::CoeffModulusLength { actual, expected }) => {
                assert_eq!(actual, 2);
                assert_eq!(expected, 1);
            }
            other => panic!("expected CoeffModulusLength error, got {other:?}"),
        }
    }

    #[test]
    fn wrong_coeff_modulus_bits_fails() {
        let p = BfvParametersBuilder::new()
            .set_degree(EXPECTED_POLY_DEGREE)
            .set_plaintext_modulus(EXPECTED_PLAINTEXT_MODULUS)
            .set_moduli_sizes(&[40])
            .build()
            .expect("40-bit modulus should still build under fhe");
        match verify(&p) {
            Err(SealCompatError::CoeffModulusBits { actual, expected }) => {
                assert_eq!(actual, vec![40]);
                assert_eq!(expected, vec![38]);
            }
            other => panic!("expected CoeffModulusBits error, got {other:?}"),
        }
    }
}
