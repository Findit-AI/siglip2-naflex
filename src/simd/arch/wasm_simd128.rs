//! wasm32 simd128 backend for siglip2's hot paths.
//!
//! Selected by [`crate::simd`]'s dispatcher when running on wasm32
//! with `target_feature = "simd128"`. WASM has no runtime CPU feature
//! detection — simd128 availability is fixed at module produce time
//! so the dispatcher's `simd128_available()` is a `const fn`.
//!
//! Numerical contract:
//! - `dot_768`: agrees with `scalar::dot_768` to within `1e-4`. Like
//!   the SSE2 backend, simd128 has no FMA in the baseline (the
//!   relaxed-simd extension adds `f32x4_relaxed_madd`, but that's a
//!   separate target_feature we don't require here). Uses separate
//!   mul + add.
//! - `normalize_patchify_row`: ≤1 ULP vs scalar.

use core::arch::wasm32::*;

/// 768-element f32 dot product using simd128.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time check via
///    `cfg!(target_feature = "simd128")` at the dispatcher).
/// 2. `a.len() >= 768` and `b.len() >= 768`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn dot_768(a: &[f32], b: &[f32]) -> f32 {
  debug_assert!(a.len() >= 768);
  debug_assert!(b.len() >= 768);

  let mut acc0 = f32x4_splat(0.0);
  let mut acc1 = f32x4_splat(0.0);
  let mut acc2 = f32x4_splat(0.0);
  let mut acc3 = f32x4_splat(0.0);

  let pa = a.as_ptr();
  let pb = b.as_ptr();

  // 768 / 16 = 48 outer iterations × 4 chains × 4-lane FMAs (mul+add).
  let mut i = 0usize;
  while i < 768 {
    // SAFETY: i + 16 ≤ 768; lengths ≥ 768.
    unsafe {
      let a0 = v128_load(pa.add(i) as *const v128);
      let a1 = v128_load(pa.add(i + 4) as *const v128);
      let a2 = v128_load(pa.add(i + 8) as *const v128);
      let a3 = v128_load(pa.add(i + 12) as *const v128);
      let b0 = v128_load(pb.add(i) as *const v128);
      let b1 = v128_load(pb.add(i + 4) as *const v128);
      let b2 = v128_load(pb.add(i + 8) as *const v128);
      let b3 = v128_load(pb.add(i + 12) as *const v128);
      acc0 = f32x4_add(acc0, f32x4_mul(a0, b0));
      acc1 = f32x4_add(acc1, f32x4_mul(a1, b1));
      acc2 = f32x4_add(acc2, f32x4_mul(a2, b2));
      acc3 = f32x4_add(acc3, f32x4_mul(a3, b3));
    }
    i += 16;
  }

  // Reduce to scalar.
  let s = f32x4_add(f32x4_add(acc0, acc1), f32x4_add(acc2, acc3));
  // Extract lanes and sum. (No vaddvq equivalent in baseline simd128.)
  f32x4_extract_lane::<0>(s)
    + f32x4_extract_lane::<1>(s)
    + f32x4_extract_lane::<2>(s)
    + f32x4_extract_lane::<3>(s)
}

/// Normalize-and-patchify one 16-pixel RGB row using simd128.
///
/// 48 input bytes processed as three 16-byte loads. Each chunk widens
/// to four `v128` of f32 via the simd128 widening conversions.
///
/// # Safety
///
/// 1. simd128 must be available.
/// 2. `src.len() >= 48` and `dst.len() >= 48`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn normalize_patchify_row(src: &[u8], dst: &mut [f32]) {
  debug_assert!(src.len() >= 48);
  debug_assert!(dst.len() >= 48);

  let scale = f32x4_splat(1.0f32 / 127.5);
  let bias = f32x4_splat(-1.0f32);

  let ps = src.as_ptr();
  let pd = dst.as_mut_ptr();

  let mut chunk = 0usize;
  while chunk < 48 {
    // SAFETY: chunk + 16 ≤ 48; lengths ≥ 48.
    unsafe {
      let bytes = v128_load(ps.add(chunk) as *const v128);
      // u8x16 → u16x8 (low and high halves).
      let lo16 = u16x8_extend_low_u8x16(bytes);
      let hi16 = u16x8_extend_high_u8x16(bytes);
      // u16x8 → u32x4 (4 chunks of 4).
      let q0 = u32x4_extend_low_u16x8(lo16);
      let q1 = u32x4_extend_high_u16x8(lo16);
      let q2 = u32x4_extend_low_u16x8(hi16);
      let q3 = u32x4_extend_high_u16x8(hi16);
      // u32x4 → f32x4.
      let f0 = f32x4_convert_u32x4(q0);
      let f1 = f32x4_convert_u32x4(q1);
      let f2 = f32x4_convert_u32x4(q2);
      let f3 = f32x4_convert_u32x4(q3);
      // No baseline FMA: scale + bias separately.
      let r0 = f32x4_add(f32x4_mul(f0, scale), bias);
      let r1 = f32x4_add(f32x4_mul(f1, scale), bias);
      let r2 = f32x4_add(f32x4_mul(f2, scale), bias);
      let r3 = f32x4_add(f32x4_mul(f3, scale), bias);
      v128_store(pd.add(chunk) as *mut v128, r0);
      v128_store(pd.add(chunk + 4) as *mut v128, r1);
      v128_store(pd.add(chunk + 8) as *mut v128, r2);
      v128_store(pd.add(chunk + 12) as *mut v128, r3);
    }
    chunk += 16;
  }
}

/// Multiply 768 floats by `factor` in place using simd128.
///
/// # Safety
///
/// 1. simd128 must be available.
/// 2. `v.len() >= 768`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn scale_768_inplace(v: &mut [f32], factor: f32) {
  debug_assert!(v.len() >= 768);
  let f = f32x4_splat(factor);
  let p = v.as_mut_ptr();
  let mut i = 0usize;
  while i < 768 {
    // SAFETY: i + 16 ≤ 768; v.len() ≥ 768.
    unsafe {
      let v0 = v128_load(p.add(i) as *const v128);
      let v1 = v128_load(p.add(i + 4) as *const v128);
      let v2 = v128_load(p.add(i + 8) as *const v128);
      let v3 = v128_load(p.add(i + 12) as *const v128);
      v128_store(p.add(i) as *mut v128, f32x4_mul(v0, f));
      v128_store(p.add(i + 4) as *mut v128, f32x4_mul(v1, f));
      v128_store(p.add(i + 8) as *mut v128, f32x4_mul(v2, f));
      v128_store(p.add(i + 12) as *mut v128, f32x4_mul(v3, f));
    }
    i += 16;
  }
}

// `cfg(not(miri))`: symmetry with the other arch test gates. Wasm
// targets aren't in the miri matrix today, but if they were, miri
// can't evaluate `v128_load`/`v128_store` either.
#[cfg(all(test, not(miri)))]
mod tests {
  //! Scalar-parity tests for the wasm simd128 kernels. Compiled and
  //! executed under `wasm32-wasip1` with `RUSTFLAGS=-C target-feature=+simd128`,
  //! driven via wasmtime as the cargo target runner — see the
  //! `wasm-simd128-runtime` CI job.
  //!
  //! Without runtime parity coverage a lane-order, reduction, or
  //! widening-conversion mistake in the unsafe simd128 backend would
  //! ship undetected.
  use super::*;
  use crate::simd::scalar;

  fn deterministic_pair(seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(768);
    let mut b = Vec::with_capacity(768);
    for i in 0..768u32 {
      let xa = (i.wrapping_mul(2_654_435_761).wrapping_add(seed)) as f32;
      let xb = (i.wrapping_mul(40_503).wrapping_add(seed.wrapping_mul(13))) as f32;
      a.push((xa * 1e-9).sin());
      b.push((xb * 1e-9).cos());
    }
    (a, b)
  }

  #[test]
  fn wasm_simd128_dot_768_matches_scalar() {
    for seed in 0..8 {
      let (a, b) = deterministic_pair(seed);
      // SAFETY: simd128 is statically required by `#[target_feature]`
      // on the kernel; this test only compiles for wasm32 builds, and
      // CI invokes it with `+simd128` enabled.
      let got = unsafe { dot_768(&a, &b) };
      let want = scalar::dot_768(&a, &b);
      let diff = (got - want).abs();
      assert!(
        diff < 1e-4,
        "seed={seed}: simd128 {got}, scalar {want}, diff {diff}"
      );
    }
  }

  #[test]
  fn wasm_simd128_dot_768_zero_input() {
    let a = vec![0.0f32; 768];
    let b = vec![1.0f32; 768];
    // SAFETY: as above.
    let got = unsafe { dot_768(&a, &b) };
    assert_eq!(got, 0.0);
  }

  #[test]
  fn wasm_simd128_normalize_patchify_row_matches_scalar() {
    for seed in 0u8..16 {
      let src: Vec<u8> = (0..48)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(seed))
        .collect();
      let mut got = vec![0.0f32; 48];
      let mut want = vec![0.0f32; 48];
      // SAFETY: lengths are 48; simd128 statically required.
      unsafe { normalize_patchify_row(&src, &mut got) };
      scalar::normalize_patchify_row(&src, &mut want);
      for i in 0..48 {
        let diff = (got[i] - want[i]).abs();
        assert!(
          diff < 1e-6,
          "seed={seed} i={i}: simd128 {} scalar {} diff {diff}",
          got[i],
          want[i]
        );
      }
    }
  }

  #[test]
  fn wasm_simd128_scale_768_inplace_matches_scalar() {
    for seed in 0..4 {
      let (a, _) = deterministic_pair(seed);
      let mut got = a.clone();
      let mut want = a.clone();
      // SAFETY: lengths are 768; simd128 statically required.
      unsafe { scale_768_inplace(&mut got, 0.5) };
      scalar::scale_768_inplace(&mut want, 0.5);
      for i in 0..768 {
        let diff = (got[i] - want[i]).abs();
        assert!(
          diff < 1e-6,
          "seed={seed} i={i}: simd128 {} scalar {} diff {diff}",
          got[i],
          want[i]
        );
      }
    }
  }
}
