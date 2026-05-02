//! x86_64 AVX-512 backend — `dot_768` only.
//!
//! Selected by [`crate::simd`]'s dispatcher when AVX-512F is present.
//! AVX-512F (foundation) is the most widely deployed AVX-512 subset
//! and includes 512-bit f32 FMA, which is all `dot_768` needs.
//!
//! **Why only `dot_768`?**
//! - `normalize_patchify_row` works on 48 bytes per call — narrower
//!   than a single AVX-512 register (64 bytes), so widening doesn't
//!   help. The AVX2 path already processes the row in 3 lanes.
//! - `scale_768_inplace` autovectorizes cleanly under `-O3`, so the
//!   benchmark shows zero gain from AVX2/NEON over scalar there. AVX-512
//!   wouldn't change that.
//! - For `dot_768`, the 16-lane FMA halves the iteration count vs AVX2
//!   (48 vs 96 FMAs), worth the extra backend file on Sapphire
//!   Rapids / Zen 4 / EPYC Genoa.
//!
//! **Throttling note (deployment guidance).** Older AVX-512 hardware
//! (Skylake-X, Cascade Lake, Ice Lake — 2017–2020) drops all-core
//! frequency 5–25% under sustained AVX-512 load, depending on SKU.
//! For mixed workloads on those chips, the dispatcher's AVX-512 path
//! can run *slower* than AVX2 because the throttle affects adjacent
//! scalar code too. Zen 4 (2022), Sapphire Rapids (2023), and
//! EPYC Genoa (2022) fixed this — modern cloud-instance CPUs are
//! safe.
//!
//! There's no way to detect throttling at runtime — Intel never
//! published a definitive SKU list, and the threshold depends on
//! stepping. If you're deploying on older AVX-512 hardware and a
//! benchmark shows AVX-512 hurts in your workload, build with
//! `RUSTFLAGS='--cfg siglip2_disable_avx512' cargo build --release`
//! to force the dispatcher to skip AVX-512 and fall back to AVX2 +
//! FMA. The cfg flag is documented alongside `siglip2_force_scalar`
//! and `siglip2_disable_avx2` in `crate::simd`'s module docstring;
//! the CI matrix exercises it as the `avx2-max` tier.
//!
//! Numerical contract: `dot_768` agrees with `scalar::dot_768` to
//! within `1e-4` (different summation order).

use core::arch::x86_64::*;

/// 768-element f32 dot product using AVX-512F (16-lane FMA).
///
/// 768 / 16 = 48 iterations, organized as 12 outer iterations × 4
/// parallel accumulators (4 chains, 12 FMAs each = 48 FMAs total).
///
/// # Safety
///
/// 1. AVX-512F must be present on the current CPU
///    (dispatcher-verified via `is_x86_feature_detected!("avx512f")`).
/// 2. `a.len() >= 768` and `b.len() >= 768`.
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn dot_768(a: &[f32], b: &[f32]) -> f32 {
  debug_assert!(a.len() >= 768);
  debug_assert!(b.len() >= 768);

  let mut acc0 = _mm512_setzero_ps();
  let mut acc1 = _mm512_setzero_ps();
  let mut acc2 = _mm512_setzero_ps();
  let mut acc3 = _mm512_setzero_ps();

  let pa = a.as_ptr();
  let pb = b.as_ptr();

  // 768 / 64 = 12 outer iterations, each loads 4 × 16-lane vectors
  // per operand and FMAs into 4 parallel accumulators.
  let mut i = 0usize;
  while i < 768 {
    // SAFETY: i + 64 ≤ 768 each iteration; pa/pb point to slices of
    // length ≥ 768 by precondition.
    unsafe {
      let a0 = _mm512_loadu_ps(pa.add(i));
      let a1 = _mm512_loadu_ps(pa.add(i + 16));
      let a2 = _mm512_loadu_ps(pa.add(i + 32));
      let a3 = _mm512_loadu_ps(pa.add(i + 48));
      let b0 = _mm512_loadu_ps(pb.add(i));
      let b1 = _mm512_loadu_ps(pb.add(i + 16));
      let b2 = _mm512_loadu_ps(pb.add(i + 32));
      let b3 = _mm512_loadu_ps(pb.add(i + 48));
      acc0 = _mm512_fmadd_ps(a0, b0, acc0);
      acc1 = _mm512_fmadd_ps(a1, b1, acc1);
      acc2 = _mm512_fmadd_ps(a2, b2, acc2);
      acc3 = _mm512_fmadd_ps(a3, b3, acc3);
    }
    i += 64;
  }

  // No `unsafe` block needed: `_mm512_add_ps` and `_mm512_reduce_add_ps`
  // are `safe fn` in std::arch — only loads/stores (`_mm512_loadu_ps`)
  // are `unsafe fn`. The function carries `#[target_feature(enable =
  // "avx512f")]` so the AVX-512 ISA is in scope here regardless.
  // Wrapping safe ops in `unsafe { }` triggers `unused_unsafe`, which
  // becomes a hard error under the workflow's `RUSTFLAGS=-Dwarnings`
  // policy on the x86 CI matrix (avx512-sde + cross).
  let s01 = _mm512_add_ps(acc0, acc1);
  let s23 = _mm512_add_ps(acc2, acc3);
  let s = _mm512_add_ps(s01, s23);
  // `_mm512_reduce_add_ps` is the canonical 16-lane horizontal sum;
  // LLVM lowers it to a balanced shuffle+add tree.
  _mm512_reduce_add_ps(s)
}

// `cfg(not(miri))`: see neon.rs for rationale. These tests call
// AVX-512 intrinsics directly; miri can't evaluate `llvm.x86.avx512.*`.
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
  fn avx512_dot_768_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
      return;
    }
    for seed in 0..8 {
      let (a, b) = pair(seed);
      // SAFETY: AVX-512F verified above; lengths are 768.
      let got = unsafe { dot_768(&a, &b) };
      let want = scalar::dot_768(&a, &b);
      let diff = (got - want).abs();
      assert!(
        diff < 1e-4,
        "seed={seed}: AVX-512 {got} scalar {want} diff {diff}"
      );
    }
  }

  #[test]
  fn avx512_dot_768_zero_input() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
      return;
    }
    let a = vec![0.0f32; 768];
    let b = vec![1.0f32; 768];
    // SAFETY: AVX-512F verified above; lengths are 768.
    let got = unsafe { dot_768(&a, &b) };
    assert_eq!(got, 0.0);
  }
}
