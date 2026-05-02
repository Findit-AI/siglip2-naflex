//! `Embedding`, `LabeledScore[Owned]`.

use std::sync::Arc;

use crate::error::{Error, Result};

/// L2-normalized embedding. Length is `BASE_NAFLEX_DIM` (768) in 0.1.0.
///
/// `Embedding` deliberately does **not** implement `Serialize` or `Deserialize`.
/// An auto-derived `Deserialize` would bypass the dim and L2-norm invariants
/// that `TryFrom<Vec<f32>>` exists to enforce. Round-trip via the inner
/// representation:
///
/// ```ignore
/// // Serialize via the inner slice (`&[f32]: Serialize`):
/// let json = serde_json::to_string(embedding.as_slice())?;
///
/// // Deserialize via the validated path:
/// let v: Vec<f32> = serde_json::from_str(&json)?;
/// let embedding  = Embedding::try_from(v)?;  // validates dim + L2-norm
/// ```
#[derive(Clone, Debug)]
pub struct Embedding(Arc<[f32]>);

impl Embedding {
  /// 0.1.0 supports only the base/patch16/naflex variant.
  pub const BASE_NAFLEX_DIM: usize = 768;

  /// L2-norm tolerance for the unit-norm invariant.
  pub const NORM_EPSILON: f32 = 5e-4;

  /// Number of `f32` lanes. Always [`Self::BASE_NAFLEX_DIM`] (768)
  /// for any `Embedding` produced by this crate's public constructors.
  pub fn dim(&self) -> usize {
    self.0.len()
  }

  /// Borrowed view of the underlying `f32` data — the standard input
  /// for downstream similarity / vector-store code that wants a `&[f32]`.
  pub fn as_slice(&self) -> &[f32] {
    &self.0
  }

  /// Returns the inner `Arc<[f32]>`. O(1) — atomic refcount only,
  /// no data copy.
  pub fn into_inner(self) -> Arc<[f32]> {
    self.0
  }

  /// Convenience: copy into a fresh `Vec<f32>`. Equivalent to `as_slice().to_vec()`.
  pub fn into_vec(self) -> Vec<f32> {
    self.as_slice().to_vec()
  }

  /// Dot product. Both operands must be unit-norm; valid because every
  /// `Embedding` in this crate is L2-normalized at construction.
  ///
  /// Panics if `self.dim() != other.dim()`. In 0.1.0 only the 768-dim
  /// base/naflex variant exists, so this is trivially satisfied.
  ///
  /// Internally dispatches through the crate-private SIMD layer
  /// picks NEON on aarch64, AVX2+FMA on x86_64, or a four-accumulator
  /// scalar fallback on every other target. The runtime feature check
  /// is a cached atomic load + branch.
  pub fn cosine(&self, other: &Embedding) -> f32 {
    assert_eq!(
      self.dim(),
      other.dim(),
      "Embedding::cosine: dim mismatch (variants must match)"
    );
    crate::simd::dot_768(self.as_slice(), other.as_slice())
  }

  /// Crate-internal: build an `Embedding` from raw model output. The
  /// SigLIP2 NaFlex ONNX exports emit unnormalized `pooler_output` and
  /// **explicitly delegate L2-normalization to the consumer** (per the
  /// `Findit-AI/indexer` release body — the per-tower I/O block notes
  /// "L2-normalize at consumer side" for both vision and text
  /// `pooler_output`). So this path normalizes *unconditionally*: any
  /// finite, non-zero, dim-correct vector is rescaled to unit norm.
  /// Rejection only happens for dim mismatch, all-zero output (degenerate
  /// model state), or non-finite components.
  ///
  /// The `TryFrom<Vec<f32>>` path keeps the strict near-unit-norm check
  /// — that's for *caller-supplied* embeddings (e.g., deserialized from
  /// a vector store) which should already be unit-norm; silent renorm
  /// there would mask data corruption.
  #[cfg(feature = "inference")]
  pub(crate) fn from_model_output(data: &[f32]) -> Result<Self> {
    if data.len() != Self::BASE_NAFLEX_DIM {
      return Err(Error::EmbeddingDim {
        expected: Self::BASE_NAFLEX_DIM,
        got: data.len(),
      });
    }
    // `dot(data, data)` gives ||v||² in one SIMD pass directly over
    // the borrowed slice. NaN/Inf in `data` propagates through the
    // dot, so the `norm.is_finite()` check below catches non-finite
    // components without a separate scan.
    let norm_sq = crate::simd::dot_768(data, data);
    let norm = norm_sq.sqrt();
    if !norm.is_finite() || norm == 0.0 {
      return Err(Error::NotNormalized {
        norm,
        epsilon: Self::NORM_EPSILON,
      });
    }
    let factor = 1.0 / norm;
    // One allocation. `Arc::<[T]>::from_iter` has a `TrustedLen`
    // specialization in std that allocates the Arc directly with
    // the right layout and writes elements in place — no Vec
    // intermediate. Slice `Iter<f32>` and the wrapping `Map` both
    // implement `TrustedLen`, so `.collect::<Arc<[f32]>>()` lands
    // on the fast path. drops the
    // per-embedding `data[..].to_vec()` alloc that previously sat
    // in `run_image_session` / `run_text_session`.
    let arc: Arc<[f32]> = data.iter().map(|&x| x * factor).collect();
    Ok(Self(arc))
  }
}

impl TryFrom<Vec<f32>> for Embedding {
  type Error = Error;

  /// Validates dim (`Error::EmbeddingDim`) and L2-norm
  /// (`Error::NotNormalized`, tolerance `NORM_EPSILON`). This path is for
  /// **caller-supplied** embeddings — typically deserialized from a
  /// vector store — that should already be unit-norm; we reject (rather
  /// than silently renormalize) so corruption can't slip through.
  ///
  /// Vectors whose `||v||₂` is within `NORM_EPSILON` of 1.0 are
  /// snapped to exactly 1.0 (in-place renorm preserves the cosine
  /// invariant under tiny f32 drift).
  ///
  /// Use the encoder methods (`embed_pixels`, `embed`, etc.) rather than
  /// this constructor for fresh model outputs — the encoders go through
  /// `from_model_output` which handles raw, unnormalized pooler outputs.
  fn try_from(mut v: Vec<f32>) -> Result<Self> {
    if v.len() != Self::BASE_NAFLEX_DIM {
      return Err(Error::EmbeddingDim {
        expected: Self::BASE_NAFLEX_DIM,
        got: v.len(),
      });
    }
    let norm_sq = crate::simd::dot_768(&v, &v);
    let norm = norm_sq.sqrt();
    if !norm.is_finite() || (norm - 1.0).abs() > Self::NORM_EPSILON {
      return Err(Error::NotNormalized {
        norm,
        epsilon: Self::NORM_EPSILON,
      });
    }
    crate::simd::scale_768_inplace(&mut v, 1.0 / norm);
    Ok(Self(v.into()))
  }
}

/// Borrowed label + score returned by `Siglip2::classify`. `score` is
/// `sigmoid(scale·cos + bias) ∈ [0, 1]`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct LabeledScore<'a> {
  label: &'a str,
  score: f32,
}

impl<'a> LabeledScore<'a> {
  /// Crate-internal constructor used by `Siglip2::classify`. Gated on
  /// the `inference` feature because that's the only call site; without
  /// inference, `LabeledScore` is reachable as a deserialization target
  /// (via `Deserialize`) but never constructed by us.
  #[cfg(feature = "inference")]
  pub(crate) fn new(label: &'a str, score: f32) -> Self {
    Self { label, score }
  }

  /// The label string borrowed from the slice originally passed to
  /// `Siglip2::classify`.
  pub fn label(&self) -> &'a str {
    self.label
  }
  /// Calibrated probability in `[0, 1]` (from `sigmoid(scale·cos + bias)`).
  /// Saturates to exactly 1.0 on confident matches; sort by cosine for
  /// reliable rank order.
  pub fn score(&self) -> f32 {
    self.score
  }
  /// Convert to an owned form whose label outlives the slice originally
  /// passed to `classify`.
  pub fn to_owned(&self) -> LabeledScoreOwned {
    LabeledScoreOwned::new(self.label, self.score)
  }
}

/// Owned form of `LabeledScore`. Use when the label needs to outlive the
/// `&[&str]` originally passed to `classify`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LabeledScoreOwned {
  label: smol_str::SmolStr,
  score: f32,
}

impl LabeledScoreOwned {
  /// Public constructor for tests, mocks, and callers whose source isn't
  /// `LabeledScore::to_owned()` or `serde::Deserialize`. `score` is
  /// **unchecked** — production scores from `Siglip2::classify` are in
  /// `[0, 1]`, but this constructor accepts any `f32`.
  pub fn new(label: impl Into<smol_str::SmolStr>, score: f32) -> Self {
    Self {
      label: label.into(),
      score,
    }
  }

  /// The owned label as a `&str`.
  pub fn label(&self) -> &str {
    self.label.as_str()
  }
  /// Calibrated probability in `[0, 1]` (when produced by
  /// `Siglip2::classify`). Caller-constructed `LabeledScoreOwned` is
  /// unchecked.
  pub fn score(&self) -> f32 {
    self.score
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn unit_vec(dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[0] = 1.0;
    v
  }

  #[test]
  fn try_from_accepts_unit_norm_768() {
    let v = unit_vec(768);
    let e = Embedding::try_from(v).expect("unit-norm 768-dim should succeed");
    assert_eq!(e.dim(), 768);
    assert!((e.cosine(&e) - 1.0).abs() < 1e-5);
  }

  #[test]
  fn try_from_rejects_wrong_dim() {
    let v = vec![0.0; 100];
    let err = Embedding::try_from(v).unwrap_err();
    match err {
      Error::EmbeddingDim { expected, got } => {
        assert_eq!(expected, 768);
        assert_eq!(got, 100);
      }
      _ => panic!("expected EmbeddingDim, got {err}"),
    }
  }

  #[test]
  fn try_from_rejects_non_unit_norm() {
    let v = vec![0.5f32; 768]; // norm ≈ sqrt(192) ≈ 13.86
    let err = Embedding::try_from(v).unwrap_err();
    match err {
      Error::NotNormalized { .. } => {}
      _ => panic!("expected NotNormalized, got {err}"),
    }
  }

  /// SigLIP2 NaFlex ONNX exports emit
  /// unnormalized pooler_output and explicitly delegate L2-norm to the
  /// consumer. `from_model_output` (the encoder path) must therefore
  /// accept and normalize arbitrary-norm vectors. Without this, every
  /// real model run would fail with `NotNormalized`.
  #[cfg(feature = "inference")]
  #[test]
  fn from_model_output_normalizes_arbitrary_norm() {
    // norm = sqrt(768) ≈ 27.7 — well outside the strict ε.
    let v = vec![1.0f32; 768];
    let e = Embedding::from_model_output(&v).expect("arbitrary-norm output must be normalized");
    let cos = e.cosine(&e);
    assert!(
      (cos - 1.0).abs() < 1e-5,
      "post-norm cosine should be 1.0; got {cos}"
    );
    // First component should be 1/sqrt(768) ≈ 0.03608.
    assert!(
      (e.as_slice()[0] - (1.0 / (768.0_f32).sqrt())).abs() < 1e-6,
      "expected normalized component"
    );
  }

  #[cfg(feature = "inference")]
  #[test]
  fn from_model_output_rejects_zero_norm() {
    let v = vec![0.0f32; 768];
    let err = Embedding::from_model_output(&v).unwrap_err();
    match err {
      Error::NotNormalized { norm, .. } => assert_eq!(norm, 0.0),
      _ => panic!("expected NotNormalized for zero output, got {err}"),
    }
  }

  #[cfg(feature = "inference")]
  #[test]
  fn from_model_output_rejects_nan_component() {
    let mut v = vec![0.5f32; 768];
    v[100] = f32::NAN;
    let err = Embedding::from_model_output(&v).unwrap_err();
    match err {
      Error::NotNormalized { norm, .. } => assert!(norm.is_nan()),
      _ => panic!("expected NotNormalized for NaN, got {err}"),
    }
  }

  #[test]
  fn try_from_still_rejects_far_from_unit_norm() {
    // The `TryFrom` path is strict — for caller-supplied embeddings
    // (e.g. deserialized from a vector store), we don't want to silently
    // mask corruption. Vectors way off unit-norm must be rejected, not
    // normalized.
    let v = vec![0.5f32; 768]; // norm ≈ sqrt(192) ≈ 13.86
    let err = Embedding::try_from(v).unwrap_err();
    match err {
      Error::NotNormalized { .. } => {}
      _ => panic!("TryFrom must keep the strict norm check, got {err}"),
    }
  }

  #[test]
  fn try_from_renormalizes_within_tolerance() {
    // norm = sqrt(1 + (NORM_EPSILON/2)^2) ≈ 1 + tiny — within tolerance
    let mut v = unit_vec(768);
    v[1] = Embedding::NORM_EPSILON / 2.0;
    let e = Embedding::try_from(v).expect("near-unit norm should be accepted");
    let dot = e.cosine(&e);
    assert!(
      (dot - 1.0).abs() < 1e-5,
      "renormalized cosine should be 1.0; got {dot}"
    );
  }

  #[test]
  #[should_panic(expected = "Embedding::cosine: dim mismatch")]
  fn cosine_panics_on_dim_mismatch() {
    // Manually construct two embeddings of different sizes via the
    // tuple-struct constructor — only possible inside the same module.
    let a = Embedding(vec![1.0f32, 0.0].into());
    let b = Embedding(vec![1.0f32, 0.0, 0.0].into());
    let _ = a.cosine(&b);
  }

  #[test]
  fn into_vec_round_trips() {
    let v = unit_vec(768);
    let e = Embedding::try_from(v.clone()).unwrap();
    let back = e.into_vec();
    assert_eq!(back.len(), 768);
    assert!((back[0] - 1.0).abs() < 1e-6);
  }

  /// `Embedding` must be `Send + Sync` — `Arc<[f32]>` already is, but
  /// pinning the bound here means a future inner-rep change can't
  /// silently regress the contract documented in.
  /// Ungated: applies to all builds, including `--no-default-features`.
  #[test]
  fn embedding_is_send_sync() {
    fn _req<T: Send + Sync>() {}
    _req::<Embedding>();
    _req::<LabeledScore<'_>>();
    _req::<LabeledScoreOwned>();
  }

  #[test]
  fn labeled_score_owned_new_constructs() {
    let s = LabeledScoreOwned::new("dog", 0.42);
    assert_eq!(s.label(), "dog");
    assert!((s.score() - 0.42).abs() < 1e-6);
  }
}
