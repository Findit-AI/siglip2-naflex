//! Crate-internal SIMD primitives for siglip2's hot paths.
//!
//! Pattern mirrors the `colconv` crate: a `scalar` reference module is
//! always compiled, per-arch SIMD kernels live under `arch::*`, and a
//! per-call dispatcher selects the best available backend at runtime.
//!
//! Backends:
//! - `scalar` — always compiled, reference implementation.
//! - `arch::neon` — aarch64 NEON (FMA-capable).
//! - `arch::x86_avx512` — x86_64 AVX-512F. 512-bit vectors. **`dot_768`
//!   only** — see the module docstring for why the other primitives
//!   skip this tier.
//! - `arch::x86_avx2` — x86_64 AVX2 + FMA. 256-bit vectors.
//! - `arch::x86_sse2` — x86_64 SSE2 fallback (no FMA, 128-bit vectors).
//! - `arch::wasm_simd128` — wasm32 simd128 (no FMA in baseline).
//!
//! Dispatch model: feature detection runs at call time on aarch64 /
//! x86_64 (`is_aarch64_feature_detected!` / `is_x86_feature_detected!`
//! — std caches the result in an atomic, so per-call overhead is a
//! relaxed load + branch). On wasm32, simd128 availability is fixed
//! at module produce time, so detection is a `const fn`. Each SIMD
//! kernel itself carries `#[target_feature(enable = "...")]` so its
//! intrinsics execute in an explicitly feature-enabled context, not
//! one inherited from the target's default features.
//!
//! Numerical contract: SIMD backends are not byte-identical to scalar
//! (different summation order changes f32 rounding) but agree to
//! within `1e-4` for `dot_768` and ≤1 ULP for the other primitives.
//! Tests in each backend module enforce this.
//!
//! The `siglip2_force_scalar` cfg, when set via
//! `RUSTFLAGS='--cfg siglip2_force_scalar'`, short-circuits every
//! `*_available()` helper to `false` so the dispatcher always falls
//! through to the scalar reference path. Useful for benchmarking the
//! baseline and for coverage of the scalar code on machines that
//! would otherwise always pick a SIMD path.
//!
//! Runtime kill switch (x86_64 only): `SIGLIP2_DISABLE_AVX512=1` in
//! the process environment forces the AVX-512 dispatch off at runtime,
//! falling back to AVX2. This is a deploy-time escape hatch for
//! mixed-CPU fleets where some hosts have older AVX-512 implementations
//! that downclock under sustained load (Skylake-X, Cascade Lake) and
//! others don't (Sapphire Rapids, Zen 4, Granite Rapids). The env var
//! is read once at first dispatch and cached via `OnceLock`, so the
//! per-call overhead is a relaxed atomic load + branch — same shape
//! as the cfg short-circuit. Not symmetric with AVX2 because AVX-512
//! is the only backend with a documented downclock concern.

pub(crate) mod arch;
pub(crate) mod scalar;

// ---- runtime CPU feature detection ----------------------------------

/// NEON availability on aarch64. NEON is a baseline aarch64 feature
/// (every aarch64 CPU has it), so this returns `true` unconditionally
/// on aarch64 unless `siglip2_force_scalar` is set.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn neon_available() -> bool {
  if cfg!(siglip2_force_scalar) {
    return false;
  }
  std::arch::is_aarch64_feature_detected!("neon")
}

/// AVX-512F availability on x86_64. AVX-512F is the foundation
/// subset and includes 512-bit f32 FMA, which is all our `dot_768`
/// kernel needs. Only used to pick the AVX-512 path for `dot_768`
/// the other primitives don't benefit (see `arch::x86_avx512`).
///
/// Three escape hatches, in order:
/// 1. `siglip2_force_scalar` cfg — compile-time, all SIMD off.
/// 2. `siglip2_disable_avx512` cfg — compile-time, AVX-512 → AVX2.
///    Used by the CI benchmark / coverage matrices to exercise the
///    AVX2 dispatcher branch on AVX-512-capable runners.
/// 3. `SIGLIP2_DISABLE_AVX512=1` env var — runtime, AVX-512 → AVX2.
///    Cached via `OnceLock` so the env-var read happens once at first
///    dispatch. Provides a deploy-time switch for mixed-CPU fleets
///    where downclock-prone AVX-512 hosts coexist with newer ones.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn avx512_available() -> bool {
  if cfg!(siglip2_force_scalar) || cfg!(siglip2_disable_avx512) {
    return false;
  }
  static DISABLED_BY_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
  let disabled = *DISABLED_BY_ENV
    .get_or_init(|| parse_disable_env(std::env::var("SIGLIP2_DISABLE_AVX512").ok().as_deref()));
  if disabled {
    return false;
  }
  std::arch::is_x86_feature_detected!("avx512f")
}

/// Parser pulled out so the truthiness rule is unit-testable without
/// needing to round-trip `OnceLock` (which caches its first read for
/// the lifetime of the process). Any value other than the literal
/// `"0"` and the empty string is treated as "disable AVX-512"
/// `=1`, `=true`, `=yes` all flip the switch. `unset` and `=0` and
/// `=` (empty) leave AVX-512 enabled. Mirrors the `RUST_BACKTRACE` /
/// `RUST_LOG` convention (any non-empty/non-zero value enables),
/// inverted because this env var *disables* rather than enables.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn parse_disable_env(value: Option<&str>) -> bool {
  !matches!(value, None | Some("") | Some("0"))
}

/// AVX2 + FMA availability on x86_64. Both must be present for the
/// AVX2 kernel — FMA is a separate cpuid bit that ships alongside AVX2
/// on Haswell+ but is technically independent.
///
/// The `siglip2_disable_avx2` cfg, when set, short-circuits this to
/// `false` so the dispatcher falls back to SSE2 even on AVX2 hardware.
/// Used by the CI matrix to exercise the SSE2 branch on runners that
/// would otherwise always pick AVX2.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn avx2_available() -> bool {
  if cfg!(siglip2_force_scalar) || cfg!(siglip2_disable_avx2) {
    return false;
  }
  std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

/// SSE2 availability on x86_64. SSE2 is part of the x86_64 base ISA,
/// so this is effectively `true` on every x86_64 CPU. The runtime
/// check is kept for symmetry with AVX2 detection and so that
/// `siglip2_force_scalar` can short-circuit it.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn sse2_available() -> bool {
  if cfg!(siglip2_force_scalar) {
    return false;
  }
  std::arch::is_x86_feature_detected!("sse2")
}

/// simd128 availability on wasm32. WASM has no runtime CPU detection
/// (SIMD support is fixed at module produce time), so this is always
/// a compile-time check.
#[cfg(target_arch = "wasm32")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn simd128_available() -> bool {
  !cfg!(siglip2_force_scalar) && cfg!(target_feature = "simd128")
}

// ---- public dispatchers ---------------------------------------------

/// Dot product of two 768-element f32 slices. Used by `Embedding::cosine`.
///
/// Selects the best available SIMD backend at runtime; falls through
/// to the scalar reference on architectures without a dedicated kernel.
///
/// # Panics
///
/// `a.len() != 768` or `b.len() != 768`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn dot_768(a: &[f32], b: &[f32]) -> f32 {
  assert_eq!(a.len(), 768, "dot_768: a.len() must be 768");
  assert_eq!(b.len(), 768, "dot_768: b.len() must be 768");

  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      // SAFETY: neon_available verified NEON is present; lengths are 768.
      return unsafe { arch::neon::dot_768(a, b) };
    }
  }

  #[cfg(target_arch = "x86_64")]
  {
    // AVX-512 first: 16-lane FMA halves the iteration count vs AVX2
    // on hardware that has it (Sapphire Rapids, Zen 4, EPYC Genoa).
    if avx512_available() {
      // SAFETY: avx512_available verified AVX-512F; lengths are 768.
      return unsafe { arch::x86_avx512::dot_768(a, b) };
    }
    if avx2_available() {
      // SAFETY: avx2_available verified AVX2 + FMA; lengths are 768.
      return unsafe { arch::x86_avx2::dot_768(a, b) };
    }
    if sse2_available() {
      // SAFETY: sse2_available verified SSE2; lengths are 768.
      return unsafe { arch::x86_sse2::dot_768(a, b) };
    }
  }

  #[cfg(target_arch = "wasm32")]
  {
    if simd128_available() {
      // SAFETY: simd128 compile-time verified; lengths are 768.
      return unsafe { arch::wasm_simd128::dot_768(a, b) };
    }
  }

  scalar::dot_768(a, b)
}

/// Normalize-and-patchify one 16-pixel RGB row. Computes
/// `dst[i] = src[i] * (1/127.5) - 1.0` for `i in 0..48`, exactly the
/// SigLIP convention `(x/255 - 0.5) / 0.5`. Used by
/// `naflex::preprocess_into` for every row of every patch.
///
/// `src` is 48 contiguous bytes (16 RGB pixels = 48 channels);
/// `dst` is 48 contiguous f32. Both must be exactly that length.
///
/// # Panics
///
/// `src.len() != 48` or `dst.len() != 48`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn normalize_patchify_row(src: &[u8], dst: &mut [f32]) {
  assert_eq!(
    src.len(),
    48,
    "normalize_patchify_row: src.len() must be 48"
  );
  assert_eq!(
    dst.len(),
    48,
    "normalize_patchify_row: dst.len() must be 48"
  );

  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      // SAFETY: neon_available verified; lengths are 48.
      unsafe {
        arch::neon::normalize_patchify_row(src, dst);
      }
      return;
    }
  }

  #[cfg(target_arch = "x86_64")]
  {
    if avx2_available() {
      // SAFETY: avx2_available verified; lengths are 48.
      unsafe {
        arch::x86_avx2::normalize_patchify_row(src, dst);
      }
      return;
    }
    if sse2_available() {
      // SAFETY: sse2_available verified; lengths are 48.
      unsafe {
        arch::x86_sse2::normalize_patchify_row(src, dst);
      }
      return;
    }
  }

  #[cfg(target_arch = "wasm32")]
  {
    if simd128_available() {
      // SAFETY: simd128 compile-time verified; lengths are 48.
      unsafe {
        arch::wasm_simd128::normalize_patchify_row(src, dst);
      }
      return;
    }
  }

  scalar::normalize_patchify_row(src, dst);
}

/// Multiply 768 floats by `factor` in place. Used by
/// `Embedding::from_model_output` and `Embedding::try_from` for the
/// L2-normalization divide (`v[i] /= norm` rewritten as
/// `v[i] *= 1/norm`, which is mathematically equivalent and a single
/// SIMD multiplication per lane).
///
/// # Panics
///
/// `v.len() != 768`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn scale_768_inplace(v: &mut [f32], factor: f32) {
  assert_eq!(v.len(), 768, "scale_768_inplace: v.len() must be 768");

  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      // SAFETY: neon_available verified; length is 768.
      unsafe {
        arch::neon::scale_768_inplace(v, factor);
      }
      return;
    }
  }

  #[cfg(target_arch = "x86_64")]
  {
    if avx2_available() {
      // SAFETY: avx2_available verified; length is 768.
      unsafe {
        arch::x86_avx2::scale_768_inplace(v, factor);
      }
      return;
    }
    if sse2_available() {
      // SAFETY: sse2_available verified; length is 768.
      unsafe {
        arch::x86_sse2::scale_768_inplace(v, factor);
      }
      return;
    }
  }

  #[cfg(target_arch = "wasm32")]
  {
    if simd128_available() {
      // SAFETY: simd128 compile-time verified; length is 768.
      unsafe {
        arch::wasm_simd128::scale_768_inplace(v, factor);
      }
      return;
    }
  }

  scalar::scale_768_inplace(v, factor);
}

#[cfg(test)]
mod tests {
  use super::*;

  fn deterministic_pair() -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..768).map(|i| ((i as f32) * 0.001).sin()).collect();
    let b: Vec<f32> = (0..768).map(|i| ((i as f32) * 0.002).cos()).collect();
    (a, b)
  }

  #[test]
  fn dot_768_dispatcher_matches_scalar() {
    let (a, b) = deterministic_pair();
    let got = dot_768(&a, &b);
    let want = scalar::dot_768(&a, &b);
    assert!(
      (got - want).abs() < 1e-3,
      "dispatcher and scalar should agree (got {got}, want {want})"
    );
  }

  #[test]
  fn normalize_patchify_row_dispatcher_matches_scalar() {
    let src: Vec<u8> = (0..48).map(|i| (i as u8).wrapping_mul(17)).collect();
    let mut got = vec![0.0f32; 48];
    let mut want = vec![0.0f32; 48];
    normalize_patchify_row(&src, &mut got);
    scalar::normalize_patchify_row(&src, &mut want);
    for i in 0..48 {
      let diff = (got[i] - want[i]).abs();
      assert!(
        diff < 1e-6,
        "i={i}: got {} want {} diff {diff}",
        got[i],
        want[i]
      );
    }
  }

  #[test]
  fn scale_768_inplace_dispatcher_matches_scalar() {
    let (a, _) = deterministic_pair();
    let mut got = a.clone();
    let mut want = a.clone();
    scale_768_inplace(&mut got, 0.5);
    scalar::scale_768_inplace(&mut want, 0.5);
    for i in 0..768 {
      let diff = (got[i] - want[i]).abs();
      assert!(
        diff < 1e-6,
        "i={i}: got {} want {} diff {diff}",
        got[i],
        want[i]
      );
    }
  }

  #[test]
  #[should_panic(expected = "dot_768: a.len() must be 768")]
  fn dot_768_panics_on_wrong_length() {
    let a = vec![0.0f32; 100];
    let b = vec![0.0f32; 768];
    dot_768(&a, &b);
  }

  #[test]
  #[should_panic(expected = "normalize_patchify_row: src.len() must be 48")]
  fn normalize_patchify_panics_on_wrong_length() {
    let src = vec![0u8; 47];
    let mut dst = vec![0.0f32; 48];
    normalize_patchify_row(&src, &mut dst);
  }

  #[test]
  #[should_panic(expected = "scale_768_inplace: v.len() must be 768")]
  fn scale_768_inplace_panics_on_wrong_length() {
    let mut v = vec![0.0f32; 100];
    scale_768_inplace(&mut v, 0.5);
  }

  /// Pin the truthiness rule for the runtime AVX-512 kill switch.
  /// Direct end-to-end testing isn't possible here because the
  /// dispatcher caches the first read for the lifetime of the process
  /// (via `OnceLock`), so cargo's parallel test runner would race
  /// any env-var manipulation.
  #[cfg(target_arch = "x86_64")]
  #[test]
  fn parse_disable_env_truthiness() {
    // Unset → keep AVX-512 enabled.
    assert!(!parse_disable_env(None));
    // Empty / `0` → keep AVX-512 enabled.
    assert!(!parse_disable_env(Some("")));
    assert!(!parse_disable_env(Some("0")));
    // Anything else → disable AVX-512.
    assert!(parse_disable_env(Some("1")));
    assert!(parse_disable_env(Some("true")));
    assert!(parse_disable_env(Some("yes")));
    assert!(
      parse_disable_env(Some("00")),
      "trailing chars beyond `0` count as truthy"
    );
    assert!(
      parse_disable_env(Some(" ")),
      "whitespace counts as truthy (no trim)"
    );
  }
}
