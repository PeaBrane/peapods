pub mod sweep;
pub mod tempering;

use rand::RngCore;
use rand_xoshiro::Xoshiro256StarStar;

const TWO_POW_62: f32 = 4_611_686_018_427_387_904.0;

/// Threshold on [`uniform_draw`] that is accepted with probability `p`.
///
/// `uniform_draw(rng) < threshold(p)` holds with probability `p` to within 2^-62, so
/// rare moves keep their tiny rates instead of the 2^-24 floor of a 24-bit uniform.
/// The i64 scale keeps the float conversion to one signed instruction on x86-64.
#[inline]
pub(crate) fn threshold(p: f32) -> i64 {
    // Saturates: p >= 1 exceeds every draw and p <= 0 (or NaN) maps to 0.
    (p * TWO_POW_62) as i64
}

/// Uniform integer in [0, 2^62) from one RNG step.
#[inline]
pub(crate) fn uniform_draw(rng: &mut Xoshiro256StarStar) -> i64 {
    (rng.next_u64() >> 2) as i64
}

/// Uniform in [0, 1] with f32 relative precision at every magnitude, so ln(u) has no
/// 2^-24 floor (u = 0 occurs with probability 2^-62).
#[inline]
pub(crate) fn uniform_f32(rng: &mut Xoshiro256StarStar) -> f32 {
    uniform_draw(rng) as f32 / TWO_POW_62
}
