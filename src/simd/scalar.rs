//! Scalar reference implementations of the siglip2 hot paths.
//!
//! Always compiled. SIMD backends in [`super::arch`] dispatch to these
//! as their tail fallback, and the dispatcher in [`super`] picks the
//! best available backend per-call.
//!
//! These implementations are written for clarity and to define the
//! numerical contract every SIMD backend must match within the
//! documented tolerance.

/// Scalar dot product of two 768-element f32 slices, computed with
/// four parallel accumulators to break the loop-carried dependency
/// chain that prevents auto-vectorization. With `-O3` LLVM picks this
/// up and emits SIMD on most targets without intrinsics; the
/// per-arch backends in [`super::arch`] provide a tighter ceiling for
/// targets where it matters.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn dot_768(a: &[f32], b: &[f32]) -> f32 {
  debug_assert_eq!(a.len(), 768);
  debug_assert_eq!(b.len(), 768);
  // Four-way parallel accumulator: each lane sums every fourth pair.
  // This breaks the serial dependency on a single accumulator and is
  // the simplest pattern that lets LLVM autovectorize cleanly.
  let mut s0 = 0.0f32;
  let mut s1 = 0.0f32;
  let mut s2 = 0.0f32;
  let mut s3 = 0.0f32;
  let mut i = 0;
  while i < 768 {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
    i += 4;
  }
  (s0 + s1) + (s2 + s3)
}

/// Scalar normalize-and-patchify for one 16-pixel RGB row. Computes
/// `dst[i] = src[i] * (1.0 / 127.5) - 1.0` exactly — the FMA-friendly
/// rearrangement of `(src[i] / 255 - 0.5) / 0.5` that SigLIP2's image
/// processor uses on the upstream PyTorch side.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn normalize_patchify_row(src: &[u8], dst: &mut [f32]) {
  debug_assert_eq!(src.len(), 48);
  debug_assert_eq!(dst.len(), 48);
  let scale = 1.0f32 / 127.5;
  for i in 0..48 {
    dst[i] = (src[i] as f32) * scale - 1.0;
  }
}

/// Multiply 768 floats by `factor` in place. Used by
/// `Embedding::from_model_output` to do the L2-normalization divide
/// (`v[i] /= norm` rewritten as `v[i] *= 1/norm`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn scale_768_inplace(v: &mut [f32], factor: f32) {
  debug_assert_eq!(v.len(), 768);
  for x in v.iter_mut() {
    *x *= factor;
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn dot_768_zero_inputs() {
    let a = vec![0.0f32; 768];
    let b = vec![1.0f32; 768];
    assert_eq!(dot_768(&a, &b), 0.0);
  }

  #[test]
  fn dot_768_unit_e0() {
    let mut a = vec![0.0f32; 768];
    let mut b = vec![0.0f32; 768];
    a[0] = 1.0;
    b[0] = 1.0;
    assert_eq!(dot_768(&a, &b), 1.0);
  }

  #[test]
  fn dot_768_constant() {
    let a = vec![0.5f32; 768];
    let b = vec![2.0f32; 768];
    let got = dot_768(&a, &b);
    let want = 0.5f32 * 2.0 * 768.0;
    assert!((got - want).abs() < 1e-3, "got {got} want {want}");
  }

  #[test]
  fn normalize_patchify_row_endpoints() {
    let mut src = vec![0u8; 48];
    src[0] = 0;
    src[1] = 255;
    src[2] = 128;
    src[3] = 127;
    let mut dst = vec![0.0f32; 48];
    normalize_patchify_row(&src, &mut dst);
    // 0 -> -1.0
    assert!((dst[0] - (-1.0)).abs() < 1e-6, "got {}", dst[0]);
    // 255 -> 255/127.5 - 1 = 1.0
    assert!((dst[1] - 1.0).abs() < 1e-6, "got {}", dst[1]);
    // 128 -> 128/127.5 - 1 ≈ 0.00392
    assert!((dst[2] - 0.0039215_f32).abs() < 1e-5, "got {}", dst[2]);
    // 127 -> 127/127.5 - 1 ≈ -0.00392
    assert!((dst[3] - (-0.0039215_f32)).abs() < 1e-5, "got {}", dst[3]);
  }
}
