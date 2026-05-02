//! x86_64 SSE2 backend for siglip2's hot paths.
//!
//! Selected by [`crate::simd`]'s dispatcher when AVX2+FMA are not
//! available but SSE2 is. SSE2 is universally available on x86_64
//! (it's part of the base ISA), so this is the always-on fallback for
//! pre-Haswell x86_64 hardware. Each kernel carries
//! `#[target_feature(enable = "sse2")]`.
//!
//! Numerical contract:
//! - `dot_768`: agrees with `scalar::dot_768` to within `1e-4` (no FMA
//!   here — uses separate mul+add, so a different rounding path than
//!   the AVX2 backend).
//! - `normalize_patchify_row`: ≤1 ULP vs scalar.

use core::arch::x86_64::*;

/// 768-element f32 dot product using SSE2 128-bit registers
/// (4 lanes per vector, no FMA). 768 = 4 × 192 → 192 mul+add pairs
/// across 4 parallel accumulators (48 outer iterations × 4 chains).
///
/// # Safety
///
/// 1. SSE2 must be present (always true on x86_64 — this is part of
///    the base ISA — but the dispatcher still verifies).
/// 2. `a.len() >= 768` and `b.len() >= 768`.
#[inline]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn dot_768(a: &[f32], b: &[f32]) -> f32 {
  debug_assert!(a.len() >= 768);
  debug_assert!(b.len() >= 768);

  let mut acc0 = _mm_setzero_ps();
  let mut acc1 = _mm_setzero_ps();
  let mut acc2 = _mm_setzero_ps();
  let mut acc3 = _mm_setzero_ps();

  let pa = a.as_ptr();
  let pb = b.as_ptr();

  // 768 / 16 = 48 outer iterations, each processes 4 × 4-lane chunks.
  let mut i = 0usize;
  while i < 768 {
    // SAFETY: i + 16 ≤ 768 each iteration; pa/pb point to slices of
    // length ≥ 768 by precondition.
    unsafe {
      let a0 = _mm_loadu_ps(pa.add(i));
      let a1 = _mm_loadu_ps(pa.add(i + 4));
      let a2 = _mm_loadu_ps(pa.add(i + 8));
      let a3 = _mm_loadu_ps(pa.add(i + 12));
      let b0 = _mm_loadu_ps(pb.add(i));
      let b1 = _mm_loadu_ps(pb.add(i + 4));
      let b2 = _mm_loadu_ps(pb.add(i + 8));
      let b3 = _mm_loadu_ps(pb.add(i + 12));
      acc0 = _mm_add_ps(acc0, _mm_mul_ps(a0, b0));
      acc1 = _mm_add_ps(acc1, _mm_mul_ps(a1, b1));
      acc2 = _mm_add_ps(acc2, _mm_mul_ps(a2, b2));
      acc3 = _mm_add_ps(acc3, _mm_mul_ps(a3, b3));
    }
    i += 16;
  }

  // No `unsafe` block needed: `_mm_add_ps`, `_mm_shuffle_ps`,
  // `_mm_add_ss`, and `_mm_cvtss_f32` are all `safe fn` in std::arch
  // only loads/stores like `_mm_loadu_ps` are `unsafe fn`. Wrapping
  // safe ops in `unsafe { }` triggers `unused_unsafe`, which becomes
  // a hard error under the workflow's `RUSTFLAGS=-Dwarnings` policy
  // on the x86 CI matrix.
  let s01 = _mm_add_ps(acc0, acc1);
  let s23 = _mm_add_ps(acc2, acc3);
  let s = _mm_add_ps(s01, s23);
  // Horizontal sum of 4 lanes via two shuffles + adds.
  let shuf = _mm_shuffle_ps(s, s, 0b_01_00_11_10);
  let sum2 = _mm_add_ps(s, shuf);
  let shuf2 = _mm_shuffle_ps(sum2, sum2, 0b_00_00_00_01);
  let total = _mm_add_ss(sum2, shuf2);
  _mm_cvtss_f32(total)
}

/// Normalize-and-patchify one 16-pixel RGB row using SSE2 (no
/// `_mm_cvtepu8_epi32` from SSE4.1 — uses the SSE2 unpack chain
/// `_mm_unpacklo_epi8` → `_mm_unpacklo_epi16` → `_mm_cvtepi32_ps`,
/// then `_mm_mul_ps` + `_mm_add_ps` since SSE2 has no FMA).
///
/// 48 input bytes processed as three 16-byte loads. Each 16-byte
/// chunk widens to four __m128 of f32 (4 floats each = 16 floats per
/// chunk).
///
/// # Safety
///
/// 1. SSE2 must be present (dispatcher-verified).
/// 2. `src.len() >= 48` and `dst.len() >= 48`.
#[inline]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn normalize_patchify_row(src: &[u8], dst: &mut [f32]) {
  debug_assert!(src.len() >= 48);
  debug_assert!(dst.len() >= 48);

  let scale = _mm_set1_ps(1.0f32 / 127.5);
  let bias = _mm_set1_ps(-1.0f32);
  let zero = _mm_setzero_si128();

  let ps = src.as_ptr();
  let pd = dst.as_mut_ptr();

  // Process 16 bytes per outer iteration → 4 × 4-lane stores.
  let mut chunk = 0usize;
  while chunk < 48 {
    // SAFETY: chunk + 16 ≤ 48 each iteration; lengths ≥ 48 by
    // precondition.
    unsafe {
      let bytes = _mm_loadu_si128(ps.add(chunk) as *const __m128i);
      // u8 → u16 (low and high halves).
      let lo16 = _mm_unpacklo_epi8(bytes, zero);
      let hi16 = _mm_unpackhi_epi8(bytes, zero);
      // u16 → u32 (4 chunks of 4).
      let q0 = _mm_unpacklo_epi16(lo16, zero);
      let q1 = _mm_unpackhi_epi16(lo16, zero);
      let q2 = _mm_unpacklo_epi16(hi16, zero);
      let q3 = _mm_unpackhi_epi16(hi16, zero);
      // i32 → f32 (cvtepi32_ps interprets as signed, but our values
      // fit in [0, 255] so the high bit is always zero).
      let f0 = _mm_cvtepi32_ps(q0);
      let f1 = _mm_cvtepi32_ps(q1);
      let f2 = _mm_cvtepi32_ps(q2);
      let f3 = _mm_cvtepi32_ps(q3);
      // No FMA in SSE2: scale and add separately.
      let r0 = _mm_add_ps(_mm_mul_ps(f0, scale), bias);
      let r1 = _mm_add_ps(_mm_mul_ps(f1, scale), bias);
      let r2 = _mm_add_ps(_mm_mul_ps(f2, scale), bias);
      let r3 = _mm_add_ps(_mm_mul_ps(f3, scale), bias);
      _mm_storeu_ps(pd.add(chunk), r0);
      _mm_storeu_ps(pd.add(chunk + 4), r1);
      _mm_storeu_ps(pd.add(chunk + 8), r2);
      _mm_storeu_ps(pd.add(chunk + 12), r3);
    }
    chunk += 16;
  }
}

/// Multiply 768 floats by `factor` in place. Used by
/// `Embedding::from_model_output` for the L2-normalization divide
/// (`v[i] /= norm` rewritten as `v[i] *= 1/norm`).
///
/// # Safety
///
/// 1. SSE2 must be present (dispatcher-verified).
/// 2. `v.len() >= 768`.
#[inline]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn scale_768_inplace(v: &mut [f32], factor: f32) {
  debug_assert!(v.len() >= 768);
  let f = _mm_set1_ps(factor);
  let p = v.as_mut_ptr();
  let mut i = 0usize;
  while i < 768 {
    // SAFETY: i + 16 ≤ 768 each iteration; v.len() ≥ 768 by precondition.
    unsafe {
      let v0 = _mm_loadu_ps(p.add(i));
      let v1 = _mm_loadu_ps(p.add(i + 4));
      let v2 = _mm_loadu_ps(p.add(i + 8));
      let v3 = _mm_loadu_ps(p.add(i + 12));
      _mm_storeu_ps(p.add(i), _mm_mul_ps(v0, f));
      _mm_storeu_ps(p.add(i + 4), _mm_mul_ps(v1, f));
      _mm_storeu_ps(p.add(i + 8), _mm_mul_ps(v2, f));
      _mm_storeu_ps(p.add(i + 12), _mm_mul_ps(v3, f));
    }
    i += 16;
  }
}

// `cfg(not(miri))`: see neon.rs for rationale. These tests call SSE2
// intrinsics directly; miri can't evaluate `llvm.x86.sse2.*`.
#[cfg(all(test, not(miri)))]
mod tests {
  use super::*;
  use crate::simd::scalar;

  fn pair(seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(768);
    let mut b = Vec::with_capacity(768);
    for i in 0..768u32 {
      let xa = (i.wrapping_mul(2654435761).wrapping_add(seed)) as f32;
      let xb = (i.wrapping_mul(40503).wrapping_add(seed.wrapping_mul(13))) as f32;
      a.push((xa * 1e-9).sin());
      b.push((xb * 1e-9).cos());
    }
    (a, b)
  }

  #[test]
  fn sse2_dot_768_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("sse2") {
      return;
    }
    for seed in 0..8 {
      let (a, b) = pair(seed);
      // SAFETY: SSE2 verified above; lengths are 768.
      let got = unsafe { dot_768(&a, &b) };
      let want = scalar::dot_768(&a, &b);
      let diff = (got - want).abs();
      assert!(
        diff < 1e-4,
        "seed={seed}: SSE2 {got} scalar {want} diff {diff}"
      );
    }
  }

  #[test]
  fn sse2_normalize_patchify_row_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("sse2") {
      return;
    }
    for seed in 0u8..16 {
      let src: Vec<u8> = (0..48)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(seed))
        .collect();
      let mut got = vec![0.0f32; 48];
      let mut want = vec![0.0f32; 48];
      // SAFETY: SSE2 verified above; lengths are 48.
      unsafe { normalize_patchify_row(&src, &mut got) };
      scalar::normalize_patchify_row(&src, &mut want);
      for i in 0..48 {
        let diff = (got[i] - want[i]).abs();
        assert!(
          diff < 1e-6,
          "seed={seed} i={i}: SSE2 {} scalar {} diff {diff}",
          got[i],
          want[i]
        );
      }
    }
  }

  #[test]
  fn sse2_scale_768_inplace_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("sse2") {
      return;
    }
    for seed in 0..4 {
      let (a, _) = pair(seed);
      let mut got = a.clone();
      let mut want = a.clone();
      // SAFETY: SSE2 verified above; lengths are 768.
      unsafe { scale_768_inplace(&mut got, 0.5) };
      scalar::scale_768_inplace(&mut want, 0.5);
      for i in 0..768 {
        let diff = (got[i] - want[i]).abs();
        assert!(
          diff < 1e-6,
          "seed={seed} i={i}: SSE2 {} scalar {} diff {diff}",
          got[i],
          want[i]
        );
      }
    }
  }
}
