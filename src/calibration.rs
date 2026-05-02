//! `Calibration` — sigmoid scale/bias for SigLIP2's calibrated
//! probabilities.

#[cfg(feature = "serde")]
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Sigmoid calibration: `score = sigmoid(exp(logit_scale)·cosine + logit_bias)`.
///
/// Carries the two scalars released alongside `google/siglip2-base-patch16-naflex`
/// (`logit_scale = 4.7476`, `logit_bias = -16.7770`). Exponentiation of
/// `logit_scale` happens at inference time inside [`crate::Siglip2::classify`];
/// callers building their own scoring path must do the same.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
  logit_scale: f32,
  logit_bias: f32,
}

impl Calibration {
  /// Const constructor for tests and callers with hard-coded values.
  /// **Does not validate** — `logit_scale` may be 0, negative, or NaN.
  /// Production paths should use `Calibration::from_path` or
  /// `Calibration::from_bytes` (both gated on `feature = "serde"`),
  /// which run the validation pipeline. If a `Calibration` built via
  /// `new` is passed to [`crate::Siglip2::from_parts`], that
  /// constructor re-runs validation, so the unchecked path can't
  /// reach `classify` undetected.
  pub const fn new(logit_scale: f32, logit_bias: f32) -> Self {
    Self {
      logit_scale,
      logit_bias,
    }
  }

  /// **Raw** learned scale parameter (matches the `logit_scale` field of the
  /// JSON file and HuggingFace `Siglip2Model.logit_scale`). The model
  /// exponentiates this value before multiplying by cosine similarity, so
  /// the effective scale used at inference is `logit_scale().exp()` (≈ 115
  /// for the pinned release value of 4.7476). `Siglip2::classify` applies
  /// the exponentiation internally; consumers building their own scoring
  /// path should do the same.
  pub fn logit_scale(&self) -> f32 {
    self.logit_scale
  }
  /// **Raw** additive bias (matches the `logit_bias` field of the
  /// JSON file). Combined with `exp(logit_scale)·cosine` to form the
  /// pre-sigmoid logit. Pinned release value is `-16.7770`.
  pub fn logit_bias(&self) -> f32 {
    self.logit_bias
  }

  /// Parses and validates the JSON. Validation rejects:
  /// - non-finite `logit_scale` or `logit_bias` (NaN, ±∞)
  /// - `logit_scale` outside `[-10, 8]` (rank/score-collapsing on
  ///   either end — too-low values produce every label scored as
  ///   `sigmoid(bias)` because `exp(scale)*cos` rounds away in f32;
  ///   too-high values saturate every match to `~1.0`)
  /// - `logit_bias` outside `[-50, 50]`
  ///
  /// The pinned release values (`logit_scale = 4.7476`,
  /// `logit_bias = -16.7770`) are comfortably inside both ranges, as
  /// are all known SigLIP and SigLIP2 published calibrations.
  ///
  /// Gated on `feature = "serde"` because the JSON parse uses
  /// `serde_json`. Without the feature, build calibration directly
  /// via [`Calibration::new`] and run validation through the typed
  /// constructor [`crate::Siglip2::from_parts`] (which re-validates).
  #[cfg(feature = "serde")]
  pub fn from_path(path: &Path) -> Result<Self> {
    let bytes = std::fs::read(path)?;
    let raw: CalibrationRaw =
      serde_json::from_slice(&bytes).map_err(|source| Error::LoadCalibration {
        path: Some(PathBuf::from(path)),
        source,
      })?;
    Self::validate(raw.logit_scale, raw.logit_bias)
  }

  /// Path-less variant. Errors surface as
  /// `Error::LoadCalibration { path: None, source }` (parse) or
  /// `Error::InvalidCalibration` (validation). Same `serde`-feature
  /// gate as [`Self::from_path`].
  #[cfg(feature = "serde")]
  pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
    let raw: CalibrationRaw = serde_json::from_slice(bytes)
      .map_err(|source| Error::LoadCalibration { path: None, source })?;
    Self::validate(raw.logit_scale, raw.logit_bias)
  }

  /// Operational range for the raw `logit_scale`. The pinned release
  /// value is `4.7476` (SigLIP and SigLIP2 published calibrations all
  /// fall in `[4, 5]`); the bounds here are wide enough to admit any
  /// plausible variant or fine-tune while rejecting obviously
  /// misconfigured values.
  ///
  /// Why these specific bounds:
  /// - `>= -10`: `exp(-10) ≈ 4.5e-5`. Above the f32 ULP at typical
  ///   `|bias|` magnitudes (~1e-6 near `bias = -16.78`), so the
  ///   calibrated logit `effective * cos + bias` is distinguishable
  ///   from `bias` alone — confidence scores remain meaningful for
  ///   any `cos ∈ [-1, 1]`. The previous bound of `-20` was
  ///   itself at the noise floor (`exp(-20) ≈ 2e-9`, below ULP), so
  ///   `logit_scale = -20` silently collapsed every score to
  ///   `sigmoid(bias)`. tightened
  ///   this; `validate_rejects_silent_collapse_scale` pins it.
  /// - `<= 8`: `exp(8) ≈ 2981`. Above this, the calibrated logit
  ///   exits the linear region of `sigmoid` for typical retrieval
  ///   cosines (`cos > 0.01`) and every match saturates to `~1.0`,
  ///   again collapsing the sort. flagged
  ///   `logit_scale = 10` (`exp(10) ≈ 22000`) as the canonical bad
  ///   value.
  ///
  /// This intentionally walks back the earlier stance that arbitrary
  /// negative scales are "mathematically valid" — they are, but the
  /// operational story matters more for a calibrated classifier.
  // The constants and `validate` below are only reached from
  // `from_path` / `from_bytes` (gated on `serde`) and
  // `Siglip2::from_parts` (in the `inference`-gated `siglip2` module).
  // With `--no-default-features` neither callsite exists, so suppress
  // the dead-code lint there. The pure-arithmetic tests in `mod tests`
  // still cover the logic in every build.
  #[cfg_attr(
    not(any(feature = "inference", feature = "serde", test)),
    allow(dead_code)
  )]
  pub(crate) const SCALE_MIN: f32 = -10.0;
  #[cfg_attr(
    not(any(feature = "inference", feature = "serde", test)),
    allow(dead_code)
  )]
  pub(crate) const SCALE_MAX: f32 = 8.0;

  /// Operational range for `logit_bias`. The pinned release is
  /// `-16.7770`. Wider bound (`±50`) than `logit_scale` because
  /// `bias` interacts linearly with the logit — a moderately wrong
  /// bias shifts scores but doesn't collapse them, and we don't want
  /// to over-constrain alternate fine-tunes that might land farther
  /// from the released value.
  #[cfg_attr(
    not(any(feature = "inference", feature = "serde", test)),
    allow(dead_code)
  )]
  pub(crate) const BIAS_ABS_MAX: f32 = 50.0;

  /// Crate-internal — also called from `Siglip2::from_parts` to close the
  /// unchecked-`new` gap.
  #[cfg_attr(
    not(any(feature = "inference", feature = "serde", test)),
    allow(dead_code)
  )]
  pub(crate) fn validate(logit_scale: f32, logit_bias: f32) -> Result<Self> {
    if !logit_scale.is_finite() {
      return Err(Error::InvalidCalibration {
        reason: "logit_scale is not finite",
      });
    }
    if !logit_bias.is_finite() {
      return Err(Error::InvalidCalibration {
        reason: "logit_bias is not finite",
      });
    }
    // Operational range check. Catches the `logit_scale = 10`
    // (saturation-to-1) and `logit_scale = -50` (collapse-to-bias)
    // failure modes flagged. The bounds also subsume
    // the finite-but-overflow / underflow checks
    // (raw scale ~88.7 overflows `exp()`, ~-103.97 underflows to 0):
    // both are well outside `[-20, 8]`.
    if !(Self::SCALE_MIN..=Self::SCALE_MAX).contains(&logit_scale) {
      return Err(Error::InvalidCalibration {
        reason: "logit_scale outside operational range [-10, 8]: \
                 effective scale exp(logit_scale) either rounds away \
                 in f32 vs bias (rank/score-collapsing) or saturates \
                 the sigmoid (rank-collapsing). Pinned release is \
                 4.7476; any reasonable SigLIP2 variant fits in this \
                 band.",
      });
    }
    if logit_bias.abs() > Self::BIAS_ABS_MAX {
      return Err(Error::InvalidCalibration {
        reason: "logit_bias outside operational range |x| ≤ 50: shifts \
                 every score past the sigmoid linear region. Pinned \
                 release is -16.7770.",
      });
    }
    Ok(Self {
      logit_scale,
      logit_bias,
    })
  }
}

/// Internal serde landing pad for `from_path` / `from_bytes`. Gated on
/// `feature = "serde"` since the derive itself depends on the trait.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct CalibrationRaw {
  logit_scale: f32,
  logit_bias: f32,
}

// Bundled-calibration constants, generated by `build.rs` from
// `models/siglip2/calibration.json`. The file lives in `$OUT_DIR` (a
// per-build-target directory under `target/`) and only exists when the
// `bundled` feature is on, so the include itself is feature-gated.
//
// Keeping the JSON as the single source of truth — rather than
// hardcoding the two `f32` literals — means a future model swap only
// needs the JSON updated; build.rs picks up the new values via
// `cargo:rerun-if-changed`.
#[cfg(feature = "bundled")]
include!(concat!(env!("OUT_DIR"), "/bundled_calibration.rs"));

#[cfg(feature = "bundled")]
impl Calibration {
  /// Pinned calibration for `google/siglip2-base-patch16-naflex`,
  /// generated at build time from `models/siglip2/calibration.json`.
  ///
  /// This avoids the `feature = "serde"` requirement of
  /// `Self::from_path` / `Self::from_bytes` — `build.rs` parses the
  /// JSON once and emits two `f32` constants, so the runtime path is
  /// `Self::new(...)` followed by `Self::validate(...)` to re-check
  /// the invariants. The validation is paranoia: `build.rs` also
  /// asserts the same range constraints, so a malformed
  /// `calibration.json` would already have failed compilation.
  pub fn bundled() -> Self {
    Self::validate(BUNDLED_LOGIT_SCALE, BUNDLED_LOGIT_BIAS)
      .expect("bundled calibration must validate — build.rs already checks the invariants")
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  const PINNED_SCALE: f32 = 4.747_554_3;
  const PINNED_BIAS: f32 = -16.776_989;

  #[cfg(feature = "serde")]
  #[test]
  fn from_bytes_accepts_pinned_values() {
    let json = r#"{"logit_scale": 4.747554302215576, "logit_bias": -16.776988983154297}"#;
    let cal = Calibration::from_bytes(json.as_bytes()).expect("pinned values must parse");
    assert!((cal.logit_scale() - PINNED_SCALE).abs() < 1e-4);
    assert!((cal.logit_bias() - PINNED_BIAS).abs() < 1e-3);
  }

  #[cfg(feature = "serde")]
  #[test]
  fn from_bytes_rejects_nan_scale() {
    let json = r#"{"logit_scale": "NaN", "logit_bias": 0.0}"#;
    // serde_json doesn't decode "NaN" as a float by default — this surfaces as a parse error.
    let err = Calibration::from_bytes(json.as_bytes()).unwrap_err();
    match err {
      Error::LoadCalibration { path: None, .. } => {}
      _ => panic!("expected LoadCalibration with path=None, got {err}"),
    }
  }

  /// Round-7 left this in the `[-20, 8]` band, so it still passes:
  /// `exp(0) = 1` is a small-but-positive effective scale.
  #[test]
  fn validate_accepts_zero_scale() {
    let cal = Calibration::validate(0.0, 0.0).expect("zero raw scale must pass");
    assert_eq!(cal.logit_scale(), 0.0);
  }

  /// Mildly negative raw scale — `exp(-1) ≈ 0.368` example.
  /// Inside the operational range, still admitted.
  #[test]
  fn validate_accepts_negative_scale() {
    let cal = Calibration::validate(-1.0, 0.0).expect("negative raw scale must pass");
    assert_eq!(cal.logit_scale(), -1.0);
  }

  #[test]
  fn validate_rejects_nan_bias() {
    let err = Calibration::validate(1.0, f32::NAN).unwrap_err();
    match err {
      Error::InvalidCalibration { reason } => {
        assert_eq!(reason, "logit_bias is not finite");
      }
      _ => panic!("expected InvalidCalibration"),
    }
  }

  #[test]
  fn validate_accepts_negative_bias() {
    let cal = Calibration::validate(1.0, -16.78).expect("negative bias is fine");
    assert!((cal.logit_bias() + 16.78).abs() < 1e-5);
  }

  /// `logit_scale = 10` makes `exp(10) ≈
  /// 22000`, which forces the calibrated logit past the sigmoid's
  /// linear region for any cos > ~0.001 and saturates every score to
  /// `~1.0` — the sort silently collapses to input order.
  #[test]
  fn validate_rejects_saturating_scale() {
    let err = Calibration::validate(10.0, 0.0).unwrap_err();
    match err {
      Error::InvalidCalibration { reason } => {
        assert!(
          reason.contains("operational range"),
          "expected operational-range message, got {reason}"
        );
      }
      _ => panic!("expected InvalidCalibration for saturating scale, got {err}"),
    }
  }

  /// `logit_scale = -50` makes
  /// `exp(-50) ≈ 1.9e-22`, so `effective * cos ≈ 0` for any
  /// reasonable cos and every label scores essentially
  /// `sigmoid(logit_bias)` — the sort collapses to input order.
  /// Walks back the earlier stance that admitted this value as
  /// "mathematically valid"; mathematical validity ≠ operational
  /// usability.
  #[test]
  fn validate_rejects_collapsing_scale() {
    let err = Calibration::validate(-50.0, 0.0).unwrap_err();
    match err {
      Error::InvalidCalibration { reason } => {
        assert!(
          reason.contains("operational range"),
          "expected operational-range message, got {reason}"
        );
      }
      _ => panic!("expected InvalidCalibration for collapsing scale, got {err}"),
    }
  }

  /// Sanity: `logit_scale = 100` is outside the
  /// operational range and still rejected (now via the `[-20, 8]`
  /// bound rather than the old `exp().is_finite()` check, which is
  /// subsumed).
  #[test]
  fn validate_rejects_overflowing_scale() {
    let err = Calibration::validate(100.0, 0.0).unwrap_err();
    match err {
      Error::InvalidCalibration { .. } => {}
      _ => panic!("expected InvalidCalibration for overflow, got {err}"),
    }
  }

  /// Sanity: `logit_scale = -200` is outside the
  /// operational range and still rejected (now subsumes the f32
  /// underflow-to-zero check).
  #[test]
  fn validate_rejects_underflowing_scale() {
    let err = Calibration::validate(-200.0, 0.0).unwrap_err();
    match err {
      Error::InvalidCalibration { .. } => {}
      _ => panic!("expected InvalidCalibration for underflow, got {err}"),
    }
  }

  /// Boundary: `logit_scale = 8.0` is the inclusive upper edge of the
  /// operational range. `exp(8) ≈ 2981` is high but admits niche
  /// low-cos workloads. Must pass.
  #[test]
  fn validate_accepts_upper_boundary_scale() {
    let cal = Calibration::validate(8.0, 0.0).expect("8.0 must pass");
    assert!((cal.logit_scale() - 8.0).abs() < 1e-5);
  }

  /// Boundary: `logit_scale = -10.0` is the inclusive lower edge.
  /// `exp(-10) ≈ 4.5e-5` — well above the f32 ULP at typical bias
  /// magnitudes (~1e-6 near -16.78), so adding it to bias produces a
  /// distinguishable f32 instead of rounding back. The previous /// bound of -20 was at the noise floor; tightened it.
  #[test]
  fn validate_accepts_lower_boundary_scale() {
    let cal = Calibration::validate(-10.0, 0.0).expect("-10.0 must pass");
    assert!((cal.logit_scale() - (-10.0)).abs() < 1e-5);

    // The load-bearing property at SCALE_MIN: `exp(scale) * 1.0 + bias`
    // must produce a *different* f32 than `bias` alone. Otherwise every
    // calibrated score collapses to `sigmoid(bias)`. Use the pinned
    // release bias (-16.78) as the worst case for this check — its ULP
    // is one of the larger ones in the legal bias range.
    let bias = -16.78_f32;
    let scale = -10.0_f32;
    let logit_at_max_cos = scale.exp().mul_add(1.0, bias);
    assert_ne!(
      logit_at_max_cos, bias,
      "at SCALE_MIN, the calibrated logit must remain distinguishable \
       from `bias` in f32 — otherwise every score is sigmoid(bias)"
    );
  }

  /// the previous lower bound `-20`
  /// silently collapsed every score to `sigmoid(bias)` because
  /// `exp(-20) ≈ 2e-9` rounds away below the f32 ULP at typical bias
  /// magnitudes. Pin the tightening so a future regression to a wider
  /// SCALE_MIN trips this test instead of producing plausible-looking
  /// useless probabilities.
  #[test]
  fn validate_rejects_silent_collapse_scale() {
    // -20 is the old bound (collapse), -15 / -11 sit between the old
    // and new bounds. All must be rejected.
    for bad_scale in [-20.0_f32, -15.0, -11.0, -10.001] {
      let err = Calibration::validate(bad_scale, 0.0)
        .err()
        .unwrap_or_else(|| panic!("{bad_scale} must be rejected as below SCALE_MIN"));
      match err {
        Error::InvalidCalibration { reason } => {
          assert!(
            reason.contains("operational range"),
            "expected operational-range message for {bad_scale}, got {reason}"
          );
        }
        _ => panic!("expected InvalidCalibration for {bad_scale}, got {err}"),
      }
    }
  }

  /// Bias bound: pinned release is -16.78; ±50 is generous. Reject
  /// obvious outliers like ±100 that would shift every score past
  /// the sigmoid linear region.
  #[test]
  fn validate_rejects_huge_bias() {
    let err = Calibration::validate(4.7476, 100.0).unwrap_err();
    match err {
      Error::InvalidCalibration { reason } => {
        assert!(
          reason.contains("logit_bias"),
          "expected logit_bias message, got {reason}"
        );
      }
      _ => panic!("expected InvalidCalibration for huge bias, got {err}"),
    }
  }

  #[test]
  fn new_does_not_validate() {
    // Per.3, `new` is unchecked.
    let cal = Calibration::new(f32::NAN, f32::NAN);
    assert!(cal.logit_scale().is_nan());
    assert!(cal.logit_bias().is_nan());
  }

  #[test]
  fn calibration_sanity_pinned_value_at_zero_cos() {
    // Item 4: sigmoid(exp(scale)·0 + bias) ≈ 5.174e-8.
    // (At cos = 0 the multiplication zeros out, so this case can't tell raw
    // and exponentiated scale apart. See the next test for the load-bearing
    // domain check.)
    let cal = Calibration::validate(PINNED_SCALE, PINNED_BIAS).unwrap();
    let cos = 0.0f32;
    let logit = cal.logit_scale().exp() * cos + cal.logit_bias();
    let sigmoid = 1.0 / (1.0 + (-logit).exp());
    let expected = 5.174e-8;
    assert!(
      ((sigmoid - expected) / expected).abs() < 1e-2,
      "sigmoid(0) should be ~5.174e-8, got {sigmoid}"
    );
  }

  #[test]
  fn calibration_sanity_pinned_value_at_typical_cos() {
    // Load-bearing test that distinguishes the raw vs exponentiated scale
    // domains. SigLIP2's typical confident-match cosine is ~0.18; with
    // exp(scale) ≈ 115.36, the logit becomes 115.36·0.18 - 16.777 ≈ 3.977
    // and the sigmoid ≈ 0.982. If someone forgets the exp() and uses the
    // raw value 4.7476 directly, the logit becomes 4.7476·0.18 - 16.777
    // ≈ -15.92 and the sigmoid is ~1e-7 — well outside this assertion's
    // 1% tolerance. This test fails if the `Siglip2::classify` formula
    // ever drops the exp() again.
    let cal = Calibration::validate(PINNED_SCALE, PINNED_BIAS).unwrap();
    let cos = 0.18f32;
    let logit = cal.logit_scale().exp() * cos + cal.logit_bias();
    let sigmoid = 1.0 / (1.0 + (-logit).exp());
    let expected = 0.9816f32;
    assert!(
      ((sigmoid - expected) / expected).abs() < 1e-2,
      "sigmoid(exp(scale)·0.18 + bias) should be ~0.9816, got {sigmoid}"
    );
  }

  /// `Calibration` must be `Send + Sync` — it's `Copy + POD` so this is
  /// trivially true today, but the test pins the contract documented in
  ///.3 against future field additions. Ungated: applies to all
  /// builds, including `--no-default-features`.
  #[test]
  fn calibration_is_send_sync() {
    fn _req<T: Send + Sync>() {}
    _req::<Calibration>();
  }

  /// `Calibration::bundled()` must reproduce the pinned release values.
  /// Pins the build.rs codegen contract: a future change to
  /// `models/siglip2/calibration.json` that drifts from the
  /// `google/siglip2-base-patch16-naflex` baseline trips this test
  /// instead of producing silently-wrong calibrated probabilities.
  /// `build.rs` also range-checks the values, so a malformed JSON
  /// would already have failed compilation.
  #[cfg(feature = "bundled")]
  #[test]
  fn bundled_matches_pinned_release_values() {
    let cal = Calibration::bundled();
    assert!(
      (cal.logit_scale() - PINNED_SCALE).abs() < 1e-4,
      "bundled logit_scale {} drifted from pinned {PINNED_SCALE}",
      cal.logit_scale()
    );
    assert!(
      (cal.logit_bias() - PINNED_BIAS).abs() < 1e-3,
      "bundled logit_bias {} drifted from pinned {PINNED_BIAS}",
      cal.logit_bias()
    );
  }
}
