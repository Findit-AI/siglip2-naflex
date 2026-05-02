//! x86_64 AVX2 + FMA backend for siglip2's two hot paths.
//!
//! Selected by [`crate::simd`]'s dispatcher after AVX2 + FMA detection
//! returns true. Each kernel carries
//! `#[target_feature(enable = "avx2,fma")]`. Numerical contract
//! mirrors the NEON backend's — agreement to within `1e-4` for
//! `dot_768`; ≤1 ULP for `normalize_patchify_row`.

use core::arch::x86_64::*;

/// 768-element f32 dot product using AVX2 256-bit registers + FMA.
/// 768 = 8 × 96 → 96 × `_mm256_fmadd_ps` across 4 independent
/// accumulators (24 iterations of 4 chains = 96 FMAs).
///
/// # Safety
///
/// 1. AVX2 + FMA must be present (dispatcher-verified).
/// 2. `a.len() >= 768` and `b.len() >= 768`.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_768(a: &[f32], b: &[f32]) -> f32 {
  debug_assert!(a.len() >= 768);
  debug_assert!(b.len() >= 768);

  let mut acc0 = _mm256_setzero_ps();
  let mut acc1 = _mm256_setzero_ps();
  let mut acc2 = _mm256_setzero_ps();
  let mut acc3 = _mm256_setzero_ps();

  let pa = a.as_ptr();
  let pb = b.as_ptr();

  // 768 / 32 = 24 outer iterations, each loads 4 × 8-lane vectors per
  // operand and FMAs into 4 parallel accumulators.
  let mut i = 0usize;
  while i < 768 {
    // SAFETY: i + 32 ≤ 768 each iteration; pa/pb point to slices of
    // length ≥ 768 by precondition. Pointer offsets stay in bounds;
    // AVX2 + FMA loads/FMAs are sound under
    // `#[target_feature(enable = "avx2,fma")]`.
    unsafe {
      let a0 = _mm256_loadu_ps(pa.add(i));
      let a1 = _mm256_loadu_ps(pa.add(i + 8));
      let a2 = _mm256_loadu_ps(pa.add(i + 16));
      let a3 = _mm256_loadu_ps(pa.add(i + 24));
      let b0 = _mm256_loadu_ps(pb.add(i));
      let b1 = _mm256_loadu_ps(pb.add(i + 8));
      let b2 = _mm256_loadu_ps(pb.add(i + 16));
      let b3 = _mm256_loadu_ps(pb.add(i + 24));
      acc0 = _mm256_fmadd_ps(a0, b0, acc0);
      acc1 = _mm256_fmadd_ps(a1, b1, acc1);
      acc2 = _mm256_fmadd_ps(a2, b2, acc2);
      acc3 = _mm256_fmadd_ps(a3, b3, acc3);
    }
    i += 32;
  }

  // Reduce: 4 vectors → 1 vector → scalar.
  let s01 = _mm256_add_ps(acc0, acc1);
  let s23 = _mm256_add_ps(acc2, acc3);
  let s = _mm256_add_ps(s01, s23);
  // Horizontal sum of 8 lanes: split high/low 128-bit halves, add,
  // then sum the 4 lanes of the resulting __m128.
  let lo = _mm256_castps256_ps128(s);
  let hi = _mm256_extractf128_ps(s, 1);
  let sum128 = _mm_add_ps(lo, hi);
  // sum128 = [a, b, c, d]; want a + b + c + d.
  let shuf = _mm_movehdup_ps(sum128); // [b, b, d, d]
  let sums = _mm_add_ps(sum128, shuf); // [a+b, _, c+d, _]
  let shuf2 = _mm_movehl_ps(sums, sums); // [c+d, ...]
  let total = _mm_add_ss(sums, shuf2);
  _mm_cvtss_f32(total)
}

/// Normalize-and-patchify one 16-pixel RGB row using AVX2 + FMA.
/// 48 input bytes split into 3 × 16-byte chunks (each loaded as the
/// low 16 bytes of an `__m128i` zero-extended to `__m256i`), each
/// expanded to 16 f32 across two `__m256` registers via
/// `_mm256_cvtepi32_ps` after `_mm256_cvtepu8_epi32`. FMA computes
/// `f * scale + bias` for `scale = 1/127.5, bias = -1.0`.
///
/// # Safety
///
/// 1. AVX2 + FMA must be present (dispatcher-verified).
/// 2. `src.len() >= 48` and `dst.len() >= 48`.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn normalize_patchify_row(src: &[u8], dst: &mut [f32]) {
  debug_assert!(src.len() >= 48);
  debug_assert!(dst.len() >= 48);

  let scale = _mm256_set1_ps(1.0f32 / 127.5);
  let bias = _mm256_set1_ps(-1.0f32);

  let ps = src.as_ptr();
  let pd = dst.as_mut_ptr();

  // Process 8 bytes (one __m256 of f32) at a time. 48 bytes / 8 = 6
  // iterations. We use a u64 load + zero-extend rather than a 16-byte
  // load + lo/hi split, for code clarity.
  let mut i = 0usize;
  while i < 48 {
    // SAFETY: i + 8 ≤ 48 each iteration; src/dst lengths ≥ 48 by
    // precondition. Pointer offsets stay in bounds; AVX2/FMA
    // loads/converts/FMAs/stores are sound under
    // `#[target_feature(enable = "avx2,fma")]`.
    unsafe {
      // Load 8 bytes as a __m128i (low 8 lanes used).
      let bytes64 = (ps.add(i) as *const u64).read_unaligned();
      let v8 = _mm_cvtsi64_si128(bytes64 as i64);
      // Zero-extend 8 × u8 → 8 × i32 (in __m256i).
      let v32 = _mm256_cvtepu8_epi32(v8);
      // i32 → f32.
      let vf = _mm256_cvtepi32_ps(v32);
      // FMA: f * scale + bias.
      let out = _mm256_fmadd_ps(vf, scale, bias);
      _mm256_storeu_ps(pd.add(i), out);
    }
    i += 8;
  }
}

/// Multiply 768 floats by `factor` in place using AVX2.
///
/// # Safety
///
/// 1. AVX2 + FMA must be present (dispatcher-verified). Multiply
///    alone needs only AVX, but we share the target_feature with the
///    other AVX2 kernels for consistency.
/// 2. `v.len() >= 768`.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn scale_768_inplace(v: &mut [f32], factor: f32) {
  debug_assert!(v.len() >= 768);
  let f = _mm256_set1_ps(factor);
  let p = v.as_mut_ptr();
  let mut i = 0usize;
  while i < 768 {
    // SAFETY: i + 32 ≤ 768 each iteration; v.len() ≥ 768 by precondition.
    unsafe {
      let v0 = _mm256_loadu_ps(p.add(i));
      let v1 = _mm256_loadu_ps(p.add(i + 8));
      let v2 = _mm256_loadu_ps(p.add(i + 16));
      let v3 = _mm256_loadu_ps(p.add(i + 24));
      _mm256_storeu_ps(p.add(i), _mm256_mul_ps(v0, f));
      _mm256_storeu_ps(p.add(i + 8), _mm256_mul_ps(v1, f));
      _mm256_storeu_ps(p.add(i + 16), _mm256_mul_ps(v2, f));
      _mm256_storeu_ps(p.add(i + 24), _mm256_mul_ps(v3, f));
    }
    i += 32;
  }
}

// `cfg(not(miri))`: see neon.rs for rationale. These tests call AVX2
// intrinsics directly; miri can't evaluate `llvm.x86.avx2.*`.
#[cfg(all(test, not(miri)))]
mod tests {
  use super::*;
  use crate::simd::scalar;

  fn deterministic_pair(seed: u32) -> (Vec<f32>, Vec<f32>) {
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
  fn avx2_dot_768_matches_scalar_within_tolerance() {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
      return;
    }
    for seed in 0..8 {
      let (a, b) = deterministic_pair(seed);
      // SAFETY: AVX2 + FMA verified above; lengths are 768.
      let got = unsafe { dot_768(&a, &b) };
      let want = scalar::dot_768(&a, &b);
      let diff = (got - want).abs();
      assert!(
        diff < 1e-4,
        "seed={seed}: AVX2 dot {got}, scalar {want}, diff {diff}"
      );
    }
  }

  #[test]
  fn avx2_normalize_patchify_row_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
      return;
    }
    for seed in 0u8..16 {
      let src: Vec<u8> = (0..48)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(seed))
        .collect();
      let mut got = vec![0.0f32; 48];
      let mut want = vec![0.0f32; 48];
      // SAFETY: AVX2 + FMA verified above; lengths are 48.
      unsafe { normalize_patchify_row(&src, &mut got) };
      scalar::normalize_patchify_row(&src, &mut want);
      for i in 0..48 {
        let diff = (got[i] - want[i]).abs();
        assert!(
          diff < 1e-6,
          "seed={seed} i={i}: AVX2 {} scalar {} diff {diff}",
          got[i],
          want[i]
        );
      }
    }
  }

  #[test]
  fn avx2_scale_768_inplace_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
      return;
    }
    for seed in 0..4 {
      let (a, _) = deterministic_pair(seed);
      let mut got = a.clone();
      let mut want = a.clone();
      // SAFETY: AVX2 + FMA verified; lengths are 768.
      unsafe { scale_768_inplace(&mut got, 0.5) };
      scalar::scale_768_inplace(&mut want, 0.5);
      for i in 0..768 {
        let diff = (got[i] - want[i]).abs();
        assert!(
          diff < 1e-6,
          "seed={seed} i={i}: AVX2 {} scalar {} diff {diff}",
          got[i],
          want[i]
        );
      }
    }
  }
}
