pub mod sweep;
pub mod tempering;

const TWO_POW_64: f32 = 18_446_744_073_709_551_616.0;

/// Threshold on a uniform `u64` draw that is accepted with probability `p`.
///
/// `draw < threshold(p)` holds with probability `p` to within 2^-64, so rare moves keep
/// their tiny rates instead of the 2^-24 floor of a 24-bit uniform.
#[inline]
pub(crate) fn threshold(p: f32) -> u64 {
    // Saturates to u64::MAX for p >= 1 and to 0 for p <= 0.
    (p * TWO_POW_64) as u64
}
