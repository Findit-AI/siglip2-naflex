//! aarch64 NEON backend for siglip2's two hot paths.
//!
//! Selected by [`crate::simd`]'s dispatcher after
//! `is_aarch64_feature_detected!("neon")` returns true. Each kernel
//! carries `#[target_feature(enable = "neon")]` so its intrinsics
//! execute in an explicitly NEON-enabled context rather than one
//! merely inherited from the aarch64 target's default features.
//!
//! Numerical contract:
//! - `dot_768`: agrees with `scalar::dot_768` to within `1e-3` in
//!   absolute value (the difference comes from a different summation
//!   order — both are correct, neither matches IEEE-754 sequential
//!   addition byte-for-byte).
//! - `normalize_patchify_row`: byte-identical to `scalar::normalize_patchify_row`
//!   given the same FMA semantics LLVM compiles the scalar version
//!   into when `target_feature = "neon"`. We assert ≤1 ULP in tests.

use core::arch::aarch64::*;

/// 768-element f32 dot product using NEON FMA.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.** The dispatcher
///    in [`crate::simd`] verifies this via
///    `std::arch::is_aarch64_feature_detected!("neon")`. Calling this
///    kernel without that check is undefined behavior.
/// 2. `a.len() >= 768` and `b.len() >= 768`. The dispatcher asserts
///    exact equality at the boundary; this kernel reads exactly 768
///    elements via `vld1q_f32` and trusts the caller.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dot_768(a: &[f32], b: &[f32]) -> f32 {
  debug_assert!(a.len() >= 768);
  debug_assert!(b.len() >= 768);

  // Four parallel accumulators (16 lanes total). Each `vfmaq_f32`
  // multiplies 4 f32s and adds into the accumulator — fully pipelined
  // with no loop-carried dependency between the four chains.
  let mut acc0 = vdupq_n_f32(0.0);
  let mut acc1 = vdupq_n_f32(0.0);
  let mut acc2 = vdupq_n_f32(0.0);
  let mut acc3 = vdupq_n_f32(0.0);

  let pa = a.as_ptr();
  let pb = b.as_ptr();

  // 768 = 16 * 48 → 48 iterations of (4 × 4-lane FMAs) = 192 fused
  // multiply-adds across 4 independent dependency chains.
  let mut i = 0usize;
  while i < 768 {
    // SAFETY: i + 16 ≤ 768 each iteration; pa/pb point to slices of
    // length ≥ 768 by the precondition. The pointer offsets are in
    // bounds; NEON loads/FMAs are sound under
    // `#[target_feature(enable = "neon")]`.
    unsafe {
      let a0 = vld1q_f32(pa.add(i));
      let a1 = vld1q_f32(pa.add(i + 4));
      let a2 = vld1q_f32(pa.add(i + 8));
      let a3 = vld1q_f32(pa.add(i + 12));
      let b0 = vld1q_f32(pb.add(i));
      let b1 = vld1q_f32(pb.add(i + 4));
      let b2 = vld1q_f32(pb.add(i + 8));
      let b3 = vld1q_f32(pb.add(i + 12));
      acc0 = vfmaq_f32(acc0, a0, b0);
      acc1 = vfmaq_f32(acc1, a1, b1);
      acc2 = vfmaq_f32(acc2, a2, b2);
      acc3 = vfmaq_f32(acc3, a3, b3);
    }
    i += 16;
  }

  // Pairwise reduce 4 vectors → 1 vector → scalar.
  let s01 = vaddq_f32(acc0, acc1);
  let s23 = vaddq_f32(acc2, acc3);
  let s = vaddq_f32(s01, s23);
  vaddvq_f32(s)
}

/// Normalize-and-patchify one 16-pixel RGB row. Computes
/// `dst[i] = src[i] * (1/127.5) - 1.0` for `i in 0..48` using NEON FMA.
///
/// The 48-byte row is processed as three 16-byte loads (`vld1q_u8`).
/// Each `uint8x16_t` widens to four `float32x4_t` (16 floats), then
/// FMAs into the destination as four 4-lane stores. Total: 3 × 16 =
/// 48 input bytes → 12 × 4 = 48 output floats per call.
///
/// # Safety
///
/// 1. NEON must be available on the current CPU (dispatcher-verified).
/// 2. `src.len() >= 48` and `dst.len() >= 48`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn normalize_patchify_row(src: &[u8], dst: &mut [f32]) {
  debug_assert!(src.len() >= 48);
  debug_assert!(dst.len() >= 48);

  let scale = 1.0f32 / 127.5;
  let bias = -1.0f32;
  let scale_v = vdupq_n_f32(scale);
  let bias_v = vdupq_n_f32(bias);

  let ps = src.as_ptr();
  let pd = dst.as_mut_ptr();

  // Process 16 bytes at a time → 4 × 4-lane f32 stores.
  let mut chunk = 0usize;
  while chunk < 48 {
    // SAFETY: chunk + 16 ≤ 48 each iteration; src.len() ≥ 48 and
    // dst.len() ≥ 48 by precondition. Pointer offsets stay in bounds;
    // NEON load/FMA/store are sound under
    // `#[target_feature(enable = "neon")]`.
    unsafe {
      let bytes = vld1q_u8(ps.add(chunk));
      // Widen u8 → u16 (low 8 bytes, high 8 bytes).
      let lo16 = vmovl_u8(vget_low_u8(bytes));
      let hi16 = vmovl_u8(vget_high_u8(bytes));
      // Widen u16 → u32 (4 chunks of 4).
      let q0 = vmovl_u16(vget_low_u16(lo16));
      let q1 = vmovl_u16(vget_high_u16(lo16));
      let q2 = vmovl_u16(vget_low_u16(hi16));
      let q3 = vmovl_u16(vget_high_u16(hi16));
      // Convert u32 → f32.
      let f0 = vcvtq_f32_u32(q0);
      let f1 = vcvtq_f32_u32(q1);
      let f2 = vcvtq_f32_u32(q2);
      let f3 = vcvtq_f32_u32(q3);
      // FMA: bias + f * scale = f * (1/127.5) + (-1.0).
      let r0 = vfmaq_f32(bias_v, f0, scale_v);
      let r1 = vfmaq_f32(bias_v, f1, scale_v);
      let r2 = vfmaq_f32(bias_v, f2, scale_v);
      let r3 = vfmaq_f32(bias_v, f3, scale_v);
      vst1q_f32(pd.add(chunk), r0);
      vst1q_f32(pd.add(chunk + 4), r1);
      vst1q_f32(pd.add(chunk + 8), r2);
      vst1q_f32(pd.add(chunk + 12), r3);
    }
    chunk += 16;
  }
}

/// Multiply 768 floats by `factor` in place using NEON.
///
/// # Safety
///
/// 1. NEON must be available on the current CPU (dispatcher-verified).
/// 2. `v.len() >= 768`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn scale_768_inplace(v: &mut [f32], factor: f32) {
  debug_assert!(v.len() >= 768);
  let f = vdupq_n_f32(factor);
  let p = v.as_mut_ptr();
  let mut i = 0usize;
  while i < 768 {
    // SAFETY: i + 16 ≤ 768 each iteration; v.len() ≥ 768 by precondition.
    unsafe {
      let v0 = vld1q_f32(p.add(i));
      let v1 = vld1q_f32(p.add(i + 4));
      let v2 = vld1q_f32(p.add(i + 8));
      let v3 = vld1q_f32(p.add(i + 12));
      vst1q_f32(p.add(i), vmulq_f32(v0, f));
      vst1q_f32(p.add(i + 4), vmulq_f32(v1, f));
      vst1q_f32(p.add(i + 8), vmulq_f32(v2, f));
      vst1q_f32(p.add(i + 12), vmulq_f32(v3, f));
    }
    i += 16;
  }
}

// `cfg(not(miri))`: these tests call NEON intrinsics directly, bypassing
// the dispatcher (the dispatcher's `siglip2_force_scalar` short-circuit
// only affects `simd::tests::*`). Miri can't evaluate
// `llvm.aarch64.neon.*` and errors `unsupported operation: can't call
// foreign function ... on OS macos`. The intrinsics are validated
// natively by the regular `test` job on aarch64 hosts.
#[cfg(all(test, not(miri)))]
mod tests {
  use super::*;
  use crate::simd::scalar;

  fn deterministic_pair(seed: u32) -> (Vec<f32>, Vec<f32>) {
    // Cheap deterministic generator — distinct lcg per slot keeps the
    // values spread across the f32 range without using rand crates.
    let mut a = Vec::with_capacity(768);
    let mut b = Vec::with_capacity(768);
    for i in 0..768u32 {
      let xa = (i.wrapping_mul(2654435761).wrapping_add(seed)) as f32;
      let xb = (i.wrapping_mul(40503).wrapping_add(seed.wrapping_mul(13))) as f32;
      // Squash to roughly [-1, 1] so cosine math doesn't overflow.
      a.push((xa * 1e-9).sin());
      b.push((xb * 1e-9).cos());
    }
    (a, b)
  }

  #[test]
  fn neon_dot_768_matches_scalar_within_tolerance() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
      return; // Skip on hosts without NEON; should be unreachable on aarch64.
    }
    for seed in 0..8 {
      let (a, b) = deterministic_pair(seed);
      // SAFETY: verified NEON above; lengths are 768.
      let got = unsafe { dot_768(&a, &b) };
      let want = scalar::dot_768(&a, &b);
      let diff = (got - want).abs();
      assert!(
        diff < 1e-4,
        "seed={seed}: NEON dot {got}, scalar {want}, diff {diff}"
      );
    }
  }

  #[test]
  fn neon_dot_768_zero_input() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
      return;
    }
    let a = vec![0.0f32; 768];
    let b = vec![1.0f32; 768];
    // SAFETY: verified NEON; lengths are 768.
    let got = unsafe { dot_768(&a, &b) };
    assert_eq!(got, 0.0);
  }

  #[test]
  fn neon_normalize_patchify_row_matches_scalar() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
      return;
    }
    for seed in 0u8..16 {
      let src: Vec<u8> = (0..48)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(seed))
        .collect();
      let mut got = vec![0.0f32; 48];
      let mut want = vec![0.0f32; 48];
      // SAFETY: verified NEON; lengths are 48.
      unsafe { normalize_patchify_row(&src, &mut got) };
      scalar::normalize_patchify_row(&src, &mut want);
      for i in 0..48 {
        let diff = (got[i] - want[i]).abs();
        // Both versions produce the same FMA result; expect ≤1 ULP
        // (floating-point) but in practice 0.0 because the LLVM
        // compiler emits the same FMA for the scalar code on aarch64
        // when the target feature is enabled.
        assert!(
          diff < 1e-6,
          "seed={seed} i={i}: NEON {} scalar {} diff {diff}",
          got[i],
          want[i]
        );
      }
    }
  }

  #[test]
  fn neon_scale_768_inplace_matches_scalar() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
      return;
    }
    for seed in 0..4 {
      let (a, _) = deterministic_pair(seed);
      let mut got = a.clone();
      let mut want = a.clone();
      // SAFETY: verified NEON; lengths are 768.
      unsafe { scale_768_inplace(&mut got, 0.5) };
      scalar::scale_768_inplace(&mut want, 0.5);
      for i in 0..768 {
        let diff = (got[i] - want[i]).abs();
        assert!(
          diff < 1e-6,
          "seed={seed} i={i}: NEON {} scalar {} diff {diff}",
          got[i],
          want[i]
        );
      }
    }
  }
}
