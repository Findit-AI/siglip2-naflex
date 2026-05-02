//! 6 for the full surface and rationale (defaults match the
//! existing `findit-siglip2-vision` service's settings).
//!
//! `GraphOptimizationLevel` and `Options::optimization_level` are
//! re-exported / present only with `feature = "inference"` — they
//! reach into `ort` types that don't exist on wasm builds.

#[cfg(feature = "inference")]
pub use ort::session::builder::GraphOptimizationLevel;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// `optimization_level` adapts `GraphOptimizationLevel` (an `ort` type
// without serde derives) into / out of a serde-friendly mirror enum.
// Both halves of the conjunction are required: `inference` for the
// `GraphOptimizationLevel` type itself, `serde` for the trait
// machinery. Mirrors the egemma pattern.
#[cfg(all(feature = "inference", feature = "serde"))]
mod optimization_level {
  use super::GraphOptimizationLevel;
  use serde::*;

  #[derive(
    Debug, Default, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize,
  )]
  #[serde(rename_all = "snake_case")]
  enum OptimizationLevel {
    Disable,
    #[default]
    Level1,
    Level2,
    Level3,
    All,
  }

  impl From<GraphOptimizationLevel> for OptimizationLevel {
    #[inline]
    fn from(value: GraphOptimizationLevel) -> Self {
      match value {
        GraphOptimizationLevel::Disable => Self::Disable,
        GraphOptimizationLevel::Level1 => Self::Level1,
        GraphOptimizationLevel::Level2 => Self::Level2,
        GraphOptimizationLevel::Level3 => Self::Level3,
        GraphOptimizationLevel::All => Self::All,
      }
    }
  }

  impl From<OptimizationLevel> for GraphOptimizationLevel {
    #[inline]
    fn from(value: OptimizationLevel) -> Self {
      match value {
        OptimizationLevel::Disable => Self::Disable,
        OptimizationLevel::Level1 => Self::Level1,
        OptimizationLevel::Level2 => Self::Level2,
        OptimizationLevel::Level3 => Self::Level3,
        OptimizationLevel::All => Self::All,
      }
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn serialize<S>(level: &GraphOptimizationLevel, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    OptimizationLevel::from(*level).serialize(serializer)
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn deserialize<'de, D>(deserializer: D) -> Result<GraphOptimizationLevel, D::Error>
  where
    D: Deserializer<'de>,
  {
    OptimizationLevel::deserialize(deserializer).map(Into::into)
  }

  // Must stay in lock-step with `Options::new()` so that deserializing a
  // config that omits `optimization_level` yields the same baseline level
  // a normal `Options::default()` would.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn default() -> GraphOptimizationLevel {
    GraphOptimizationLevel::Level1
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_max_num_patches() -> u32 {
  256
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_batch_size() -> usize {
  8
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_max_batch_size() -> usize {
  1024
}

/// Knobs that control how requests are grouped into ORT inference
/// runs. Carried on [`Options`] and queried internally by both
/// [`crate::ImageEncoder`] and [`crate::TextEncoder`].
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BatchOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_max_num_patches"))]
  max_num_patches: u32,
  #[cfg_attr(feature = "serde", serde(default = "default_batch_size"))]
  batch_size: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_max_batch_size"))]
  max_batch_size: usize,
}

impl BatchOptions {
  /// `BatchOptions` with the crate defaults: `max_num_patches = 256`,
  /// `batch_size = 8`, `max_batch_size = 1024`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      max_num_patches: default_max_num_patches(),
      batch_size: default_batch_size(),
      max_batch_size: default_max_batch_size(),
    }
  }

  /// Cap on patches per image. 0.1.0 hard-codes this at 256 — the
  /// only value the bundled ONNX export accepts.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_num_patches(&self) -> u32 {
    self.max_num_patches
  }

  /// Target ORT micro-batch size. Inputs above this length are split
  /// into chunks of this size before being handed to `Session::run`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn batch_size(&self) -> usize {
    self.batch_size
  }

  /// Hard cap on a single chunk's length — also bounds scratch
  /// allocations sized off `BatchOptions`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_batch_size(&self) -> usize {
    self.max_batch_size
  }

  /// Builder-style override for [`Self::max_num_patches`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_num_patches(mut self, n: u32) -> Self {
    self.set_max_num_patches(n);
    self
  }

  /// Builder-style override for [`Self::batch_size`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_batch_size(mut self, n: usize) -> Self {
    self.set_batch_size(n);
    self
  }

  /// Builder-style override for [`Self::max_batch_size`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_batch_size(mut self, n: usize) -> Self {
    self.set_max_batch_size(n);
    self
  }

  /// Setter form of [`Self::with_max_num_patches`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_num_patches(&mut self, n: u32) -> &mut Self {
    self.max_num_patches = n;
    self
  }

  /// Setter form of [`Self::with_batch_size`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_batch_size(&mut self, n: usize) -> &mut Self {
    self.batch_size = n;
    self
  }

  /// Setter form of [`Self::with_max_batch_size`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_batch_size(&mut self, n: usize) -> &mut Self {
    self.max_batch_size = n;
    self
  }

  /// Reject `batch_size == 0` (the silent `.max(1)` coercion footgun) and
  /// `batch_size > max_batch_size` (a config error that wastes scratch
  /// memory and never produces a chunk that large in practice — the
  /// runtime reject-too-large guard already caps batch length at
  /// `max_batch_size`).
  ///
  /// Image preprocessing applies an additional `max_num_patches` check
  /// at `Preprocessor::new` (the export currently bakes in 256). This
  /// helper handles only the cross-modality batch-shape invariants so
  /// `TextEncoder` and `ImageEncoder` can't drift on what counts as a
  /// valid `BatchOptions`.
  pub(crate) fn validate(&self) -> Result<(), crate::Error> {
    if self.batch_size == 0 || self.batch_size > self.max_batch_size {
      return Err(crate::Error::InvalidBatchSize {
        batch_size: self.batch_size,
        max_batch_size: self.max_batch_size,
      });
    }
    Ok(())
  }
}

impl Default for BatchOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_intra_threads() -> usize {
  1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_inter_threads() -> usize {
  1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_parallel_execution() -> bool {
  false
}

/// Threading knobs forwarded to the ORT session.
/// Mirrors `ort::session::builder::SessionBuilder::with_intra_threads`,
/// `with_inter_threads`, and `with_parallel_execution`.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ThreadOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_intra_threads"))]
  intra_threads: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_inter_threads"))]
  inter_threads: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_parallel_execution"))]
  parallel_execution: bool,
}

impl ThreadOptions {
  /// `ThreadOptions` with the crate defaults: single intra-/inter-op
  /// thread and `parallel_execution = false`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      intra_threads: default_intra_threads(),
      inter_threads: default_inter_threads(),
      parallel_execution: default_parallel_execution(),
    }
  }

  /// Number of threads ORT uses inside a single op.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn intra_threads(&self) -> usize {
    self.intra_threads
  }

  /// Number of threads ORT uses to run independent ops concurrently.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn inter_threads(&self) -> usize {
    self.inter_threads
  }

  /// Whether ORT may execute independent ops in parallel
  /// (`SessionBuilder::with_parallel_execution`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn parallel_execution(&self) -> bool {
    self.parallel_execution
  }

  /// Builder-style override for [`Self::intra_threads`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_intra_threads(mut self, n: usize) -> Self {
    self.set_intra_threads(n);
    self
  }

  /// Builder-style override for [`Self::inter_threads`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_inter_threads(mut self, n: usize) -> Self {
    self.set_inter_threads(n);
    self
  }

  /// Builder-style override for [`Self::parallel_execution`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_parallel_execution(mut self, p: bool) -> Self {
    self.set_parallel_execution(p);
    self
  }

  /// Setter form of [`Self::with_intra_threads`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_intra_threads(&mut self, n: usize) -> &mut Self {
    self.intra_threads = n;
    self
  }

  /// Setter form of [`Self::with_inter_threads`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_inter_threads(&mut self, n: usize) -> &mut Self {
    self.inter_threads = n;
    self
  }

  /// Setter form of [`Self::with_parallel_execution`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_parallel_execution(&mut self, p: bool) -> &mut Self {
    self.parallel_execution = p;
    self
  }
}

impl Default for ThreadOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

/// Top-level configuration carried by every constructor on
/// [`crate::TextEncoder`], [`crate::ImageEncoder`], and
/// [`crate::Siglip2`]. Aggregates [`BatchOptions`], [`ThreadOptions`],
/// and (under the `inference` feature) [`GraphOptimizationLevel`].
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  #[cfg(feature = "inference")]
  #[cfg_attr(
    feature = "serde",
    serde(with = "optimization_level", default = "optimization_level::default")
  )]
  optimization_level: GraphOptimizationLevel,
  #[cfg_attr(feature = "serde", serde(default))]
  batch: BatchOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  threads: ThreadOptions,
}

impl Options {
  /// `Options` with the crate defaults — `Level1` graph optimization,
  /// default [`BatchOptions`], default [`ThreadOptions`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      #[cfg(feature = "inference")]
      optimization_level: GraphOptimizationLevel::Level1,
      batch: BatchOptions::new(),
      threads: ThreadOptions::new(),
    }
  }

  /// Graph-optimization level forwarded to ORT's `SessionBuilder`.
  /// Defaults to `Level1` because higher levels can subtly alter
  /// numerics — surface them as opt-in.
  #[cfg(feature = "inference")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn optimization_level(&self) -> GraphOptimizationLevel {
    self.optimization_level
  }

  /// Borrow the [`BatchOptions`] sub-config.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn batch(&self) -> BatchOptions {
    self.batch
  }

  /// Borrow the [`ThreadOptions`] sub-config.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn threads(&self) -> ThreadOptions {
    self.threads
  }

  /// Builder-style override for [`Self::optimization_level`].
  #[cfg(feature = "inference")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_optimization_level(mut self, l: GraphOptimizationLevel) -> Self {
    self.optimization_level = l;
    self
  }

  /// Builder-style override for the [`BatchOptions`] sub-config.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_batch(mut self, b: BatchOptions) -> Self {
    self.batch = b;
    self
  }

  /// Builder-style override for the [`ThreadOptions`] sub-config.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_threads(mut self, t: ThreadOptions) -> Self {
    self.threads = t;
    self
  }

  /// Setter form of [`Self::with_optimization_level`].
  #[cfg(feature = "inference")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_optimization_level(&mut self, l: GraphOptimizationLevel) -> &mut Self {
    self.optimization_level = l;
    self
  }

  /// Setter form of [`Self::with_batch`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_batch(&mut self, b: BatchOptions) -> &mut Self {
    self.batch = b;
    self
  }

  /// Setter form of [`Self::with_threads`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_threads(&mut self, t: ThreadOptions) -> &mut Self {
    self.threads = t;
    self
  }
}

impl Default for Options {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[cfg(feature = "inference")]
  #[test]
  fn defaults_match_spec() {
    let o = Options::default();
    assert_eq!(o.optimization_level(), GraphOptimizationLevel::Level1);
    assert_eq!(o.batch().max_num_patches(), 256);
    assert_eq!(o.batch().batch_size(), 8);
    assert_eq!(o.batch().max_batch_size(), 1024);
    assert_eq!(o.threads().intra_threads(), 1);
    assert_eq!(o.threads().inter_threads(), 1);
    assert!(!o.threads().parallel_execution());
  }

  #[cfg(feature = "inference")]
  #[test]
  fn builder_chains_compose() {
    let o = Options::default()
      .with_optimization_level(GraphOptimizationLevel::Level3)
      .with_batch(BatchOptions::default().with_batch_size(32))
      .with_threads(ThreadOptions::default().with_intra_threads(4));

    assert_eq!(o.optimization_level(), GraphOptimizationLevel::Level3);
    assert_eq!(o.batch().batch_size(), 32);
    assert_eq!(o.threads().intra_threads(), 4);
  }

  #[cfg(feature = "inference")]
  #[test]
  fn setters_chain_in_place() {
    let mut o = Options::default();
    o.set_optimization_level(GraphOptimizationLevel::Level2)
      .set_batch(BatchOptions::default().with_batch_size(16));

    assert_eq!(o.optimization_level(), GraphOptimizationLevel::Level2);
    assert_eq!(o.batch().batch_size(), 16);
  }

  #[test]
  fn options_is_copy() {
    fn _require_copy<T: Copy>() {}
    _require_copy::<Options>();
    _require_copy::<BatchOptions>();
    _require_copy::<ThreadOptions>();
  }

  // The validation rules below are also exercised through
  // `Preprocessor::new` (`preproc::tests::rejects_{zero,oversized}_batch_size`)
  // and through `TextEncoder::from_ort_session_with_options` at integration
  // time (`tests/integration.rs::text_encoder_*_batch_*`). These unit tests
  // pin the helper itself so a future refactor that bypasses one caller
  // still gets covered by the other.

  #[test]
  fn validate_rejects_zero_batch_size() {
    let bad = BatchOptions::default().with_batch_size(0);
    match bad.validate() {
      Err(crate::Error::InvalidBatchSize {
        batch_size: 0,
        max_batch_size: 1024,
      }) => {}
      other => panic!("expected InvalidBatchSize {{ 0, 1024 }}, got {other:?}"),
    }
  }

  #[test]
  fn validate_rejects_batch_size_above_max() {
    let bad = BatchOptions::default()
      .with_batch_size(2048)
      .with_max_batch_size(1024);
    match bad.validate() {
      Err(crate::Error::InvalidBatchSize {
        batch_size: 2048,
        max_batch_size: 1024,
      }) => {}
      other => panic!("expected InvalidBatchSize {{ 2048, 1024 }}, got {other:?}"),
    }
  }

  #[test]
  fn validate_accepts_batch_size_equal_to_max() {
    BatchOptions::default()
      .with_batch_size(1024)
      .with_max_batch_size(1024)
      .validate()
      .expect("batch_size == max_batch_size is the boundary case and must validate");
  }

  #[test]
  fn validate_accepts_default() {
    BatchOptions::default()
      .validate()
      .expect("default BatchOptions must validate (8 / 1024)");
  }

  // Regression: the serde default for `optimization_level` once returned
  // `Disable` while `Options::default()` returned `Level1`, so a config
  // file that omitted the field silently produced sessions with a
  // different ORT optimization level than normal code.
  #[cfg(all(feature = "inference", feature = "serde"))]
  #[test]
  fn deserializing_empty_object_equals_default() {
    let from_empty: Options = serde_json::from_str("{}").expect("empty options");
    let dflt = Options::default();
    assert_eq!(from_empty.optimization_level(), dflt.optimization_level());
    assert_eq!(
      from_empty.batch().max_num_patches(),
      dflt.batch().max_num_patches()
    );
    assert_eq!(from_empty.batch().batch_size(), dflt.batch().batch_size());
    assert_eq!(
      from_empty.batch().max_batch_size(),
      dflt.batch().max_batch_size()
    );
    assert_eq!(
      from_empty.threads().intra_threads(),
      dflt.threads().intra_threads()
    );
    assert_eq!(
      from_empty.threads().inter_threads(),
      dflt.threads().inter_threads()
    );
    assert_eq!(
      from_empty.threads().parallel_execution(),
      dflt.threads().parallel_execution()
    );
  }
}
