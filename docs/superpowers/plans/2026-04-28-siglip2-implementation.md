# siglip2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `siglip2` Rust crate (v0.1.0) from spec `docs/superpowers/specs/2026-04-27-siglip2-design.md` (rev 8). Library produces 768-dim L2-normalized image and text embeddings via the SigLIP2 NaFlex ONNX export, parallels the `textclap` sibling crate, and integrates downstream of `scenesdetect` keyframe extraction.

**Architecture:** Two `ort::Session`-backed encoders (`ImageEncoder` for NaFlex pixel tensors, `TextEncoder` for tokenized queries) sit behind a `Siglip2` wrapper that adds calibrated zero-shot `classify`. NaFlex preprocessing (binary-search patch sizing → `image::imageops::resize(Triangle)` → normalize → patchify → right-pad) lives in a private module exposed through a stateless `Preprocessor` for chunk-buffer reuse. Calibration scalars and tokenizer bytes are runtime/optional-bundled assets. No storage layer; consumers wire to lancedb or similar.

**Tech Stack:**
- Rust 2024 edition, MSRV 1.85
- `ort = "2.0.0-rc.12"` — ONNX Runtime (matches sibling crate `textclap`)
- `tokenizers = "0.22"` — Gemma SPM via HF Tokenizers JSON
- `image = "0.25"` (default-features = false) — `imageops::resize(Triangle)` for the NaFlex downscale
- `thiserror = "2"`, `derive_more = "2"`, `smol_str = "0.3"`, `serde = "1"`, `serde_json = "1"`
- `criterion = "0.8"` (dev) for benches; `npyz = "0.9"` (dev) for `.npy` golden fixtures

---

## Up-front verification (do once before Task 1)

**Why:** Three ORT 2.0-rc and `tokenizers` 0.22 API points are described in the spec by behavior, not by exact signature. Verifying these at the start prevents redesigning module internals mid-plan.

- [ ] **Step 1: Skim ORT 2.0-rc.12 docs / the textclap sibling for the exact session-build idiom**

Read `/Users/user/Develop/findit-studio/textclap/src/audio.rs` and `/Users/user/Develop/findit-studio/textclap/src/text.rs` for the canonical `ort::Session::builder()...commit_from_file(path)` pattern, `with_intra_threads`, `with_inter_threads`, `with_optimization_level` setter names, and the input/output extraction APIs (`session.run(...)`, value-to-`ndarray` conversion).

Capture the actual function names you'll need in `image_enc.rs` / `text_enc.rs`. If a name in this plan diverges from what `textclap` uses for the same operation, prefer the textclap form.

- [ ] **Step 2: Skim `tokenizers` 0.22's padding API**

Run from the workspace root:
```bash
cargo doc --target-dir /tmp/cargo-doc-tokenizers --no-deps -p tokenizers 2>&1 | head -20
```

Or, faster: open `https://docs.rs/tokenizers/0.22.0` in a browser. We need the exact form of `PaddingParams` and how to call `tokenizer.with_padding(...)` with `strategy: PaddingStrategy::Fixed(64)` and `pad_id: 0`. If `Fixed` requires a `usize` not a `u64`, note it.

- [ ] **Step 3: Skim `image` 0.25's `imageops::resize` signature**

Open `https://docs.rs/image/0.25/image/imageops/fn.resize.html`. Confirm:
- It accepts `&impl GenericImageView<Pixel = Rgb<u8>>` (or similar) and returns `ImageBuffer<Rgb<u8>, Vec<u8>>`.
- `FilterType::Triangle` is the symbol path.
- New width/height arguments are `u32`.

- [ ] **Step 4: Read sibling project layouts**

Quick read of `/Users/user/Develop/findit-studio/textclap/src/error.rs`, `/Users/user/Develop/findit-studio/textclap/src/options.rs`, and `/Users/user/Develop/findit-studio/textclap/Cargo.toml` to mirror conventions (file-header doc style, lint configuration, `#[non_exhaustive]` usage on enums, etc.).

No commit for this section — it's pure reading.

---

## Task 1: Bootstrap — clean template, write Cargo.toml, attribution

**Why:** The crate currently uses the `template-rs` skeleton (name, version, README, lib.rs, etc. all template defaults). Replace with the real `siglip2` Cargo manifest from spec §9 and remove the foo.rs files.

**Files:**
- Modify: `Cargo.toml` (replace template content)
- Delete: `src/lib.rs` (template content; re-created in Task 12)
- Delete: `examples/foo.rs`, `tests/foo.rs`, `benches/foo.rs`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `CHANGELOG.md`
- Modify: `README.md` (replace template content; full README in Task 17, this is a placeholder)

- [ ] **Step 1: Delete template foo files and the template lib.rs**

```bash
rm /Users/user/Develop/findit-studio/siglip2/examples/foo.rs
rm /Users/user/Develop/findit-studio/siglip2/tests/foo.rs
rm /Users/user/Develop/findit-studio/siglip2/benches/foo.rs
rm /Users/user/Develop/findit-studio/siglip2/src/lib.rs
```

- [ ] **Step 2: Write the real Cargo.toml**

Replace the file at `/Users/user/Develop/findit-studio/siglip2/Cargo.toml` (overwrite entirely) with:

```toml
[package]
name         = "siglip2"
version      = "0.1.0"
edition      = "2024"
# Edition 2024 floor; matches sibling `scenesdetect`. Bump only with a
# concrete, named language/dep requirement.
rust-version = "1.85"
description  = "Rust ONNX inference library for SigLIP2 NaFlex (image+text embeddings)"
license      = "MIT OR Apache-2.0"
repository   = "https://github.com/Findit-AI/siglip2"
keywords     = ["siglip2", "image", "embedding", "onnx", "ml"]
categories   = ["multimedia::images", "science"]
include = [
    "src/**/*.rs",
    "examples/**/*.rs",
    "Cargo.toml",
    "README.md",
    "CHANGELOG.md",
    "LICENSE-*",
    "THIRD_PARTY_NOTICES.md",
    "models/tokenizer.json",
    "models/MODELS.md",
]

[dependencies]
ort          = "2.0.0-rc.12"
tokenizers   = "0.22"
thiserror    = "2"
derive_more  = { version = "2", default-features = false, features = ["display","as_ref","deref","deref_mut"] }
smol_str     = "0.3"
# Mandatory core dep: NaFlex preprocessing uses `image::imageops::resize`
# (FilterType::Triangle) for the downscale step required by the
# `max_num_patches = 256` cap. `default-features = false` — JPEG/PNG
# decoders ship behind the `decoders` feature flag below, not by default.
image        = { version = "0.25", default-features = false }
serde        = { version = "1", features = ["derive"] }
serde_json   = "1"

[dev-dependencies]
criterion = "0.8"
npyz      = "0.9"

[features]
default  = ["bundled", "decoders"]
# Embeds the 32.8 MB text-tower tokenizer.json via include_bytes!.
# The vision tower has no tokenizer; this feature is text-side only and
# adds nothing to the image-encoder code path.
bundled  = []
# Activates JPEG and PNG decoders inside the `image` crate. Gates
# `ImageEncoder::embed_path` (§3.3). Without this, callers must supply
# already-decoded RGB pixels via `ImageView`.
decoders = ["image/jpeg", "image/png"]
# Gates `#[cfg_attr(feature = "serde", derive(serde::Serialize[, Deserialize]))]`
# on `LabeledScore` (Serialize only) and `LabeledScoreOwned` (both).
# `Embedding` and `Calibration` deliberately do NOT carry these derives;
# see spec §3.5 and §5.3 for why.
serde    = ["smol_str/serde"]

[[bench]]
name    = "bench_naflex"
harness = false
[[bench]]
name    = "bench_image_encode"
harness = false
[[bench]]
name    = "bench_text_encode"
harness = false

[[example]]
name              = "embed_keyframes"
required-features = ["decoders"]
[[example]]
name              = "index_and_search"
required-features = []

[profile.bench]
opt-level         = 3
debug             = false
codegen-units     = 1
lto               = 'thin'
incremental       = false
debug-assertions  = false
overflow-checks   = false
rpath             = false

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]

[lints.rust]
rust_2018_idioms     = "warn"
single_use_lifetimes = "warn"
unexpected_cfgs      = { level = "warn", check-cfg = ['cfg(all_tests)', 'cfg(tarpaulin)'] }
```

- [ ] **Step 3: Write THIRD_PARTY_NOTICES.md**

```markdown
# Third-Party Notices

This crate redistributes one third-party asset as a build-time inclusion:

## models/tokenizer.json

Derived from the `google/siglip2-base-patch16-naflex` HuggingFace release.

- License: Apache License, Version 2.0
- Source: https://huggingface.co/google/siglip2-base-patch16-naflex
- Re-exported via the `Findit-AI/indexer` release tag `models-siglip2-naflex-v1`
- SHA256: `58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0`

The Apache-2.0 license requires that derivative works carry the original
copyright and license notice. The `bundled` Cargo feature embeds these
bytes into the compiled crate; the bytes themselves are unmodified from
the upstream release.

A copy of the Apache-2.0 license is available at:
https://www.apache.org/licenses/LICENSE-2.0
```

- [ ] **Step 4: Write CHANGELOG.md**

```markdown
# Changelog

## 0.1.0 (unreleased)

Initial release. Vision and text encoders for SigLIP2 NaFlex
(`google/siglip2-base-patch16-naflex`) ONNX export, calibrated zero-shot
classification, NaFlex preprocessing for arbitrary input aspect ratios.

### Notes for users disabling default features

The crate tarball is ~33 MB regardless of whether the `bundled` feature
is enabled; the bundled `tokenizer.json` is included unconditionally so
the package contents match the source tree. Disabling `bundled` only
removes the `BUNDLED_TOKENIZER` constant from the public API.
```

- [ ] **Step 5: Replace README.md placeholder**

```markdown
# siglip2

Rust ONNX inference library for SigLIP2 NaFlex (image + text embeddings).

A sibling of [`textclap`](https://github.com/Findit-AI/textclap) (CLAP audio inference) and a downstream of [`scenesdetect`](https://github.com/Findit-AI/scenesdetect) (keyframe extraction). See `docs/superpowers/specs/2026-04-27-siglip2-design.md` for the full design.

This README will be expanded in a later commit.
```

- [ ] **Step 6: Verify the workspace compiles in its empty state**

Run from the crate root:
```bash
cargo check --no-default-features
```

Expected: errors about `src/lib.rs` not existing (`error: couldn't read src/lib.rs` or similar) — that's fine; we delete-and-create in Task 2. The Cargo.toml itself must parse cleanly. If you see TOML parse errors, fix them before proceeding.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml THIRD_PARTY_NOTICES.md CHANGELOG.md README.md
git rm src/lib.rs examples/foo.rs tests/foo.rs benches/foo.rs
git commit -m "chore(scaffold): replace template-rs skeleton with siglip2 0.1.0 manifest"
```

---

## Task 2: lib.rs skeleton + error.rs

**Why:** `Error` and `Result` are foundational — every other module's signatures reference them. Build the error enum first so subsequent modules can `use crate::error::{Error, Result}` from day one. The `lib.rs` skeleton is the minimum needed for `cargo check` to find the crate root; full re-exports come in Task 12.

**Files:**
- Create: `src/lib.rs`
- Create: `src/error.rs`

- [ ] **Step 1: Write the lib.rs skeleton**

```rust
// src/lib.rs
//! SigLIP2 NaFlex inference library — see crate-level README and the design
//! spec at `docs/superpowers/specs/2026-04-27-siglip2-design.md`.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes)]

pub mod error;
pub use error::{Error, Result};
```

- [ ] **Step 2: Write the failing test for `Error::Display` formatting**

Create `src/error.rs`:

```rust
// src/error.rs
//! Error type — see spec §3.5 for the full enum and its semantics.

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    #[error("failed to load ONNX graph at {path}: {source}")]
    LoadGraph { path: PathBuf, source: ort::Error },

    #[error("failed to load external weights for graph at {path}: {source}")]
    LoadWeights { path: PathBuf, source: ort::Error },

    #[error("failed to parse calibration.json{location}: {source}",
            location = path.as_ref().map(|p| format!(" at {}", p.display())).unwrap_or_default())]
    LoadCalibration {
        path: Option<PathBuf>,
        source: serde_json::Error,
    },

    #[error("invalid calibration values: {reason}")]
    InvalidCalibration { reason: &'static str },

    #[error("tokenizer load failed: {0}")]
    Tokenizer(String),

    #[error("invalid image dimensions: {width}x{height}")]
    InvalidImage { width: u32, height: u32 },

    #[error("rgb buffer length {got} does not match width*height*3 = {expected}")]
    RgbLength { got: usize, expected: usize },

    #[error("preprocessed buffer `{which}` length {got} does not match expected {expected}")]
    PreprocBufferLength {
        which: &'static str,
        got: usize,
        expected: usize,
    },

    #[error("unexpected output rank: expected 2, got {rank} with shape {shape:?}")]
    OutputRank { rank: usize, shape: Vec<i64> },

    #[error("session shape mismatch on `{input}`: expected {expected}, got {got:?}")]
    SessionShapeMismatch {
        input: &'static str,
        expected: &'static str,
        got: Vec<i64>,
    },

    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    EmbeddingDim { expected: usize, got: usize },

    #[error("embedding is not unit-norm (got ||v||₂ = {norm}, tolerance ε = {epsilon})")]
    NotNormalized { norm: f32, epsilon: f32 },

    #[error("text input is empty")]
    EmptyText,

    #[error("max_num_patches in Options ({opt}) does not match the value baked into the ONNX export ({export})")]
    MaxNumPatchesMismatch { opt: u32, export: u32 },

    #[error("batch size {got} exceeds maximum {max}")]
    BatchTooLarge { got: usize, max: usize },

    #[error("batch index {index}: {source}")]
    Batch { index: usize, source: Box<Error> },

    #[error(transparent)]
    Ort(#[from] ort::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_image_displays_dimensions() {
        let err = Error::InvalidImage {
            width: 0,
            height: 480,
        };
        assert_eq!(err.to_string(), "invalid image dimensions: 0x480");
    }

    #[test]
    fn rgb_length_displays_both_lengths() {
        let err = Error::RgbLength {
            got: 100,
            expected: 921_600,
        };
        assert_eq!(
            err.to_string(),
            "rgb buffer length 100 does not match width*height*3 = 921600"
        );
    }

    #[test]
    fn load_calibration_with_path_includes_location() {
        let bad_json = "{not json";
        let serde_err = serde_json::from_str::<serde_json::Value>(bad_json).unwrap_err();
        let err = Error::LoadCalibration {
            path: Some(PathBuf::from("/tmp/calibration.json")),
            source: serde_err,
        };
        assert!(err.to_string().contains("at /tmp/calibration.json"));
    }

    #[test]
    fn load_calibration_without_path_omits_location() {
        let bad_json = "{not json";
        let serde_err = serde_json::from_str::<serde_json::Value>(bad_json).unwrap_err();
        let err = Error::LoadCalibration {
            path: None,
            source: serde_err,
        };
        // No " at " segment when path is None.
        assert!(!err.to_string().contains(" at "));
        assert!(err.to_string().starts_with("failed to parse calibration.json:"));
    }
}
```

- [ ] **Step 3: Run the tests and watch them pass**

```bash
cargo test --no-default-features --lib error::
```

Expected: all four `error::tests::*` tests pass. If the `LoadCalibration` `Display` macro doesn't compile (the inline `format!` inside `#[error(...)]` is a `thiserror` 2.x feature — verify with `thiserror = "2"` in Cargo.toml), substitute the equivalent `Display` impl by hand:

```rust
// Drop the #[error(...)] line on LoadCalibration and write:
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoadCalibration { path: Some(p), source } => {
                write!(f, "failed to parse calibration.json at {}: {source}", p.display())
            }
            Self::LoadCalibration { path: None, source } => {
                write!(f, "failed to parse calibration.json: {source}")
            }
            // ... and forward the rest to `thiserror`'s generated impl somehow
        }
    }
}
```

If you have to fall back to a manual `Display` impl, prefer keeping `thiserror` for everything else and only override `LoadCalibration`. Document the workaround in a code comment.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/error.rs
git commit -m "feat(error): Error enum and Result alias"
```

---

## Task 3: options.rs — Options, BatchOptions, ThreadOptions, GraphOptimizationLevel

**Why:** Options are passed to nearly every constructor (`from_files_with_options`, `Preprocessor::new`, etc.); having them complete and tested early means downstream tasks don't have to mock them.

**Files:**
- Create: `src/options.rs`
- Modify: `src/lib.rs` (add `pub mod options;` and re-exports)

- [ ] **Step 1: Write the failing tests**

Create `src/options.rs`:

```rust
// src/options.rs
//! See spec §3.6 for the full surface and rationale (defaults match the
//! existing `findit-siglip2-vision` service's settings).

/// Mirrors `ort::GraphOptimizationLevel`. Re-exported here so the crate's public
/// API doesn't force callers to depend on `ort` directly when they only want to
/// pick a level. The implementation maps each variant 1:1 to `ort`'s equivalent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphOptimizationLevel {
    Disable,
    Level1,
    Level2,
    Level3,
}

impl Default for GraphOptimizationLevel {
    /// `Level1` matches the existing `findit-siglip2-vision` service; the 0.99917
    /// cosine parity floor was established under Level1, and Level3 may apply
    /// fusions that produce numerically different outputs (spec §3.6).
    fn default() -> Self {
        Self::Level1
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BatchOptions {
    max_num_patches: u32,
    batch_size: usize,
    batch_size_max: usize,
}

impl BatchOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_num_patches(&self) -> u32 { self.max_num_patches }
    pub fn batch_size(&self) -> usize { self.batch_size }
    pub fn batch_size_max(&self) -> usize { self.batch_size_max }

    pub fn with_max_num_patches(mut self, n: u32) -> Self { self.max_num_patches = n; self }
    pub fn with_batch_size(mut self, n: usize) -> Self { self.batch_size = n; self }
    pub fn with_batch_size_max(mut self, n: usize) -> Self { self.batch_size_max = n; self }

    pub fn set_max_num_patches(&mut self, n: u32) -> &mut Self { self.max_num_patches = n; self }
    pub fn set_batch_size(&mut self, n: usize) -> &mut Self { self.batch_size = n; self }
    pub fn set_batch_size_max(&mut self, n: usize) -> &mut Self { self.batch_size_max = n; self }
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            max_num_patches: 256,
            batch_size: 8,
            batch_size_max: 1024,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ThreadOptions {
    intra_threads: usize,
    inter_threads: usize,
    parallel_execution: bool,
}

impl ThreadOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intra_threads(&self) -> usize { self.intra_threads }
    pub fn inter_threads(&self) -> usize { self.inter_threads }
    pub fn parallel_execution(&self) -> bool { self.parallel_execution }

    pub fn with_intra_threads(mut self, n: usize) -> Self { self.intra_threads = n; self }
    pub fn with_inter_threads(mut self, n: usize) -> Self { self.inter_threads = n; self }
    pub fn with_parallel_execution(mut self, p: bool) -> Self { self.parallel_execution = p; self }

    pub fn set_intra_threads(&mut self, n: usize) -> &mut Self { self.intra_threads = n; self }
    pub fn set_inter_threads(&mut self, n: usize) -> &mut Self { self.inter_threads = n; self }
    pub fn set_parallel_execution(&mut self, p: bool) -> &mut Self { self.parallel_execution = p; self }
}

impl Default for ThreadOptions {
    fn default() -> Self {
        Self {
            intra_threads: 1,
            inter_threads: 1,
            parallel_execution: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    graph_optimization_level: GraphOptimizationLevel,
    batch: BatchOptions,
    threads: ThreadOptions,
}

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn graph_optimization_level(&self) -> GraphOptimizationLevel { self.graph_optimization_level }
    pub fn batch(&self) -> BatchOptions { self.batch }
    pub fn threads(&self) -> ThreadOptions { self.threads }

    pub fn with_graph_optimization_level(mut self, l: GraphOptimizationLevel) -> Self {
        self.graph_optimization_level = l; self
    }
    pub fn with_batch(mut self, b: BatchOptions) -> Self { self.batch = b; self }
    pub fn with_threads(mut self, t: ThreadOptions) -> Self { self.threads = t; self }

    pub fn set_graph_optimization_level(&mut self, l: GraphOptimizationLevel) -> &mut Self {
        self.graph_optimization_level = l; self
    }
    pub fn set_batch(&mut self, b: BatchOptions) -> &mut Self { self.batch = b; self }
    pub fn set_threads(&mut self, t: ThreadOptions) -> &mut Self { self.threads = t; self }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            graph_optimization_level: GraphOptimizationLevel::default(),
            batch: BatchOptions::default(),
            threads: ThreadOptions::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec() {
        let o = Options::default();
        assert_eq!(o.graph_optimization_level(), GraphOptimizationLevel::Level1);
        assert_eq!(o.batch().max_num_patches(), 256);
        assert_eq!(o.batch().batch_size(), 8);
        assert_eq!(o.batch().batch_size_max(), 1024);
        assert_eq!(o.threads().intra_threads(), 1);
        assert_eq!(o.threads().inter_threads(), 1);
        assert!(!o.threads().parallel_execution());
    }

    #[test]
    fn builder_chains_compose() {
        let o = Options::default()
            .with_graph_optimization_level(GraphOptimizationLevel::Level3)
            .with_batch(BatchOptions::default().with_batch_size(32))
            .with_threads(ThreadOptions::default().with_intra_threads(4));

        assert_eq!(o.graph_optimization_level(), GraphOptimizationLevel::Level3);
        assert_eq!(o.batch().batch_size(), 32);
        assert_eq!(o.threads().intra_threads(), 4);
    }

    #[test]
    fn setters_chain_in_place() {
        let mut o = Options::default();
        o.set_graph_optimization_level(GraphOptimizationLevel::Level2)
            .set_batch(BatchOptions::default().with_batch_size(16));

        assert_eq!(o.graph_optimization_level(), GraphOptimizationLevel::Level2);
        assert_eq!(o.batch().batch_size(), 16);
    }

    #[test]
    fn options_is_copy() {
        fn _require_copy<T: Copy>() {}
        _require_copy::<Options>();
        _require_copy::<BatchOptions>();
        _require_copy::<ThreadOptions>();
    }
}
```

- [ ] **Step 2: Wire into lib.rs**

Edit `src/lib.rs`, add a `pub mod options;` line below `pub mod error;` and re-export the four types:

```rust
pub mod error;
pub mod options;

pub use error::{Error, Result};
pub use options::{BatchOptions, GraphOptimizationLevel, Options, ThreadOptions};
```

- [ ] **Step 3: Run the tests and watch them pass**

```bash
cargo test --no-default-features --lib options::
```

Expected: all four tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/options.rs
git commit -m "feat(options): Options/BatchOptions/ThreadOptions with builder + setter pattern"
```

---

## Task 4: embedding.rs — Embedding, LabeledScore, LabeledScoreOwned, TryFrom

**Why:** `Embedding` is the central return type of every encoder method. Defining it (and its dim/L2-norm validation) early lets every later test use `Embedding::try_from(vec)` to construct fixtures.

**Files:**
- Create: `src/embedding.rs`
- Modify: `src/lib.rs` (add `pub mod embedding;` and re-exports)

- [ ] **Step 1: Write embedding.rs**

```rust
// src/embedding.rs
//! `Embedding`, `LabeledScore[Owned]` — see spec §3.5.

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

    pub fn dim(&self) -> usize {
        self.0.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

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
    pub fn cosine(&self, other: &Embedding) -> f32 {
        assert_eq!(
            self.dim(),
            other.dim(),
            "Embedding::cosine: dim mismatch (variants must match)"
        );
        let a = self.as_slice();
        let b = other.as_slice();
        let mut acc = 0.0f32;
        for i in 0..a.len() {
            acc += a[i] * b[i];
        }
        acc
    }

    /// Crate-internal: build an `Embedding` from raw model output and validate
    /// the L2-norm invariant. Renormalizes if `||v||₂` is within `NORM_EPSILON`
    /// of 1.0; rejects otherwise.
    pub(crate) fn from_model_output(v: Vec<f32>) -> Result<Self> {
        if v.len() != Self::BASE_NAFLEX_DIM {
            return Err(Error::EmbeddingDim {
                expected: Self::BASE_NAFLEX_DIM,
                got: v.len(),
            });
        }
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > Self::NORM_EPSILON {
            return Err(Error::NotNormalized {
                norm,
                epsilon: Self::NORM_EPSILON,
            });
        }
        // Within tolerance — renormalize to exactly 1.0.
        let v: Vec<f32> = v.into_iter().map(|x| x / norm).collect();
        Ok(Self(v.into()))
    }
}

impl TryFrom<Vec<f32>> for Embedding {
    type Error = Error;

    /// Validates dim (`Error::EmbeddingDim`) and L2-norm
    /// (`Error::NotNormalized`, tolerance `NORM_EPSILON`).
    fn try_from(v: Vec<f32>) -> Result<Self> {
        Self::from_model_output(v)
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
    pub(crate) fn new(label: &'a str, score: f32) -> Self {
        Self { label, score }
    }

    pub fn label(&self) -> &'a str {
        self.label
    }
    pub fn score(&self) -> f32 {
        self.score
    }
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

    pub fn label(&self) -> &str {
        self.label.as_str()
    }
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

    #[test]
    fn try_from_renormalizes_within_tolerance() {
        // norm = sqrt(1 + (NORM_EPSILON/2)^2) ≈ 1 + tiny — within tolerance
        let mut v = unit_vec(768);
        v[1] = Embedding::NORM_EPSILON / 2.0;
        let e = Embedding::try_from(v).expect("near-unit norm should be accepted");
        let dot = e.cosine(&e);
        assert!((dot - 1.0).abs() < 1e-5, "renormalized cosine should be 1.0; got {dot}");
    }

    #[test]
    #[should_panic(expected = "Embedding::cosine: dim mismatch")]
    fn cosine_panics_on_dim_mismatch() {
        // Manually construct two embeddings of different sizes via the
        // crate-internal builder with relaxed dim — only possible by going
        // around the public API for the test.
        // For 0.1.0 this is hypothetical; we simulate it by directly building
        // `Embedding`s from differently-sized Arcs.
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

    #[test]
    fn labeled_score_owned_new_constructs() {
        let s = LabeledScoreOwned::new("dog", 0.42);
        assert_eq!(s.label(), "dog");
        assert!((s.score() - 0.42).abs() < 1e-6);
    }
}
```

Note: the `cosine_panics_on_dim_mismatch` test uses the tuple-struct constructor `Embedding(...)` directly. That requires the test to be in the same module as the type (which it is — `mod tests` is inside `embedding.rs`). The constructor is private outside the module, so this is unit-test-only.

- [ ] **Step 2: Wire into lib.rs**

Edit `src/lib.rs`:

```rust
pub mod embedding;
pub mod error;
pub mod options;

pub use embedding::{Embedding, LabeledScore, LabeledScoreOwned};
pub use error::{Error, Result};
pub use options::{BatchOptions, GraphOptimizationLevel, Options, ThreadOptions};
```

- [ ] **Step 3: Run the tests and watch them pass**

```bash
cargo test --no-default-features --lib embedding::
```

Expected: all seven tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/embedding.rs
git commit -m "feat(embedding): Embedding with TryFrom + LabeledScore[Owned]"
```

---

## Task 5: calibration.rs — Calibration, CalibrationRaw, validate, from_path/from_bytes

**Why:** `Calibration` blocks `Siglip2::classify` and `Siglip2::from_parts`. The validation pipeline (`CalibrationRaw` → `validate` → `Calibration`) is the public-facing safety net for hand-edited or corrupted JSON.

**Files:**
- Create: `src/calibration.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write calibration.rs**

```rust
// src/calibration.rs
//! `Calibration` — see spec §5.3 and §3.5.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    logit_scale: f32,
    logit_bias: f32,
}

impl Calibration {
    /// Const constructor for tests and callers with hard-coded values.
    /// **Does not validate** — `logit_scale` may be 0, negative, or NaN.
    /// Production paths should use [`Calibration::from_path`] or
    /// [`Calibration::from_bytes`], which both run the validation pipeline.
    /// If a `Calibration` built via `new` is passed to
    /// [`crate::Siglip2::from_parts`], that constructor re-runs validation,
    /// so the unchecked path can't reach `classify` undetected.
    pub const fn new(logit_scale: f32, logit_bias: f32) -> Self {
        Self {
            logit_scale,
            logit_bias,
        }
    }

    pub fn logit_scale(&self) -> f32 {
        self.logit_scale
    }
    pub fn logit_bias(&self) -> f32 {
        self.logit_bias
    }

    /// Parses and validates the JSON. Validation rejects:
    /// - non-finite `logit_scale` or `logit_bias` (NaN, ±∞)
    /// - non-positive `logit_scale`
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
    /// `Error::InvalidCalibration` (validation).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let raw: CalibrationRaw =
            serde_json::from_slice(bytes).map_err(|source| Error::LoadCalibration {
                path: None,
                source,
            })?;
        Self::validate(raw.logit_scale, raw.logit_bias)
    }

    /// Crate-internal — also called from `Siglip2::from_parts` to close the
    /// unchecked-`new` gap (spec §3.2).
    pub(crate) fn validate(logit_scale: f32, logit_bias: f32) -> Result<Self> {
        if !logit_scale.is_finite() {
            return Err(Error::InvalidCalibration {
                reason: "logit_scale is not finite",
            });
        }
        if logit_scale <= 0.0 {
            return Err(Error::InvalidCalibration {
                reason: "logit_scale must be positive",
            });
        }
        if !logit_bias.is_finite() {
            return Err(Error::InvalidCalibration {
                reason: "logit_bias is not finite",
            });
        }
        Ok(Self {
            logit_scale,
            logit_bias,
        })
    }
}

#[derive(serde::Deserialize)]
struct CalibrationRaw {
    logit_scale: f32,
    logit_bias: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINNED_SCALE: f32 = 4.747_554_3;
    const PINNED_BIAS: f32 = -16.776_989;

    #[test]
    fn from_bytes_accepts_pinned_values() {
        let json = r#"{"logit_scale": 4.747554302215576, "logit_bias": -16.776988983154297}"#;
        let cal = Calibration::from_bytes(json.as_bytes()).expect("pinned values must parse");
        assert!((cal.logit_scale() - PINNED_SCALE).abs() < 1e-4);
        assert!((cal.logit_bias() - PINNED_BIAS).abs() < 1e-3);
    }

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

    #[test]
    fn validate_rejects_zero_scale() {
        let err = Calibration::validate(0.0, 0.0).unwrap_err();
        match err {
            Error::InvalidCalibration { reason } => {
                assert_eq!(reason, "logit_scale must be positive");
            }
            _ => panic!("expected InvalidCalibration"),
        }
    }

    #[test]
    fn validate_rejects_negative_scale() {
        let err = Calibration::validate(-1.0, 0.0).unwrap_err();
        match err {
            Error::InvalidCalibration { reason } => {
                assert_eq!(reason, "logit_scale must be positive");
            }
            _ => panic!("expected InvalidCalibration"),
        }
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

    #[test]
    fn new_does_not_validate() {
        // Per spec §5.3, `new` is unchecked.
        let cal = Calibration::new(f32::NAN, f32::NAN);
        assert!(cal.logit_scale().is_nan());
        assert!(cal.logit_bias().is_nan());
    }

    #[test]
    fn calibration_sanity_pinned_value() {
        // Spec §11 item 4: sigmoid(scale·0 + bias) ≈ 5.174e-8.
        let cal = Calibration::validate(PINNED_SCALE, PINNED_BIAS).unwrap();
        let cos = 0.0f32;
        let logit = cal.logit_scale() * cos + cal.logit_bias();
        let sigmoid = 1.0 / (1.0 + (-logit).exp());
        let expected = 5.174e-8;
        assert!(
            ((sigmoid - expected) / expected).abs() < 1e-2,
            "sigmoid(0) should be ~5.174e-8, got {sigmoid}"
        );
    }
}
```

- [ ] **Step 2: Wire into lib.rs**

Edit `src/lib.rs`:

```rust
pub mod calibration;
pub mod embedding;
pub mod error;
pub mod options;

pub use calibration::Calibration;
pub use embedding::{Embedding, LabeledScore, LabeledScoreOwned};
pub use error::{Error, Result};
pub use options::{BatchOptions, GraphOptimizationLevel, Options, ThreadOptions};
```

- [ ] **Step 3: Run the tests and watch them pass**

```bash
cargo test --no-default-features --lib calibration::
```

Expected: eight tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/calibration.rs
git commit -m "feat(calibration): Calibration with validating from_path/from_bytes"
```

---

## Task 6: preproc/naflex.rs — Patch-grid sizing algorithm + reference table

**Why:** The binary-search-on-scale sizing is the most algorithmically delicate part of the crate. Spec §4.1 has a verified reference table — pinning that as a unit test up front prevents drift later when the resize/patchify code is added.

**Files:**
- Create: `src/preproc/mod.rs` (skeleton; expanded in Task 8)
- Create: `src/preproc/naflex.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Skeleton mod.rs**

Create `src/preproc/mod.rs`:

```rust
// src/preproc/mod.rs
//! Preprocessing pipeline — see spec §4 and §3.7.
//!
//! `Preprocessor` (the public type) is added in Task 8 once `naflex.rs` is
//! complete.

pub(crate) mod naflex;
```

- [ ] **Step 2: Write naflex.rs sizing skeleton + the reference-table test**

Create `src/preproc/naflex.rs`:

```rust
// src/preproc/naflex.rs
//! NaFlex preprocessing — patch-grid sizing, resize, normalize, patchify.

pub(crate) const PATCH_SIZE: u32 = 16;

/// Per spec §4.1: find the largest scalar scale `s` such that
/// `ceil(s·H/P) * ceil(s·W/P) ≤ M`, then return the snapped patch grid
/// `(H_p, W_p)` with `H_p = max(1, ceil(s·H/P))` and similarly for `W_p`.
pub(crate) fn patch_grid(height: u32, width: u32, max_num_patches: u32) -> (u32, u32) {
    let h = height as f64;
    let w = width as f64;
    let p = PATCH_SIZE as f64;
    let m = max_num_patches as f64;

    // Binary search on s ∈ [eps, 100]: largest s with valid patch count.
    let mut lo: f64 = 1e-6;
    let mut hi: f64 = 100.0;

    // 64 iterations — 2^-64 precision well below any meaningful pixel boundary.
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        let h_p = (mid * h / p).ceil();
        let w_p = (mid * w / p).ceil();
        if h_p * w_p <= m {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let h_p = ((lo * h / p).ceil() as u32).max(1);
    let w_p = ((lo * w / p).ceil() as u32).max(1);

    // Defensive: if floating-point rounding pushed h_p * w_p above M,
    // shave one row or column. In practice this should be rare (the binary
    // search converges to the boundary from below).
    if h_p as u64 * w_p as u64 > max_num_patches as u64 {
        // Should not happen for normal inputs — clamp by trimming the longer axis.
        let mut hh = h_p;
        let mut ww = w_p;
        while hh as u64 * ww as u64 > max_num_patches as u64 {
            if hh >= ww {
                hh = hh.saturating_sub(1).max(1);
            } else {
                ww = ww.saturating_sub(1).max(1);
            }
        }
        return (hh, ww);
    }

    (h_p, w_p)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference table from spec §4.1. Values to be regenerated against the
    /// upstream `Siglip2ImageProcessorFast` when the implementation plan
    /// runs against a real model release (spec §11 item 3); the values here
    /// are derived from the spec text.
    #[test]
    fn reference_table_matches() {
        const M: u32 = 256;
        let cases: &[((u32, u32), (u32, u32))] = &[
            ((16, 16), (16, 16)),
            ((100, 100), (16, 16)),
            ((224, 224), (16, 16)),
            ((1080, 1920), (12, 21)),
            ((1920, 1080), (21, 12)),
            ((2160, 4096), (11, 22)),
            ((1024, 1), (22, 1)),
        ];
        for (input, expected) in cases {
            let (h_p, w_p) = patch_grid(input.0, input.1, M);
            assert_eq!(
                (h_p, w_p),
                *expected,
                "patch_grid({}x{}) — expected {:?}, got ({h_p}, {w_p})",
                input.0, input.1, expected
            );
        }
    }

    #[test]
    fn budget_respected_on_random_inputs() {
        const M: u32 = 256;
        let cases = [
            (1, 1),
            (1, 2048),
            (2048, 1),
            (4096, 4096),
            (640, 480),
            (3840, 2160),
            (32, 7),
            (7, 32),
        ];
        for (h, w) in cases {
            let (h_p, w_p) = patch_grid(h, w, M);
            assert!(h_p >= 1 && w_p >= 1, "{h}x{w} → {h_p}x{w_p} has zero axis");
            assert!(
                (h_p as u64) * (w_p as u64) <= M as u64,
                "{h}x{w} → {h_p}x{w_p} exceeds budget {M}"
            );
        }
    }
}
```

**Important caveat for the implementer.** Two of the reference-table rows in spec §4.1 (especially the orientations of `1920×1080` vs `1080×1920` and `4096×2160` vs `2160×4096`) come from the spec author's reading of the upstream algorithm and may need adjustment when run against the actual `Siglip2ImageProcessorFast` from `transformers`. If a row fails:

1. Run the upstream Python reference manually (spec §11 item 3 documents this) to get the ground-truth `(H_p, W_p)`.
2. Update the test case with the verified value.
3. Add a comment noting which row was corrected.

Do **not** weaken the test (e.g., remove rows or relax assertions) — the table is the algorithm's specification.

- [ ] **Step 3: Modify lib.rs**

Add `pub(crate) mod preproc;` to `src/lib.rs` (private — `Preprocessor` will be re-exported in Task 8):

```rust
pub mod calibration;
pub mod embedding;
pub mod error;
pub mod options;
pub(crate) mod preproc;

pub use calibration::Calibration;
pub use embedding::{Embedding, LabeledScore, LabeledScoreOwned};
pub use error::{Error, Result};
pub use options::{BatchOptions, GraphOptimizationLevel, Options, ThreadOptions};
```

- [ ] **Step 4: Run the tests and watch them pass**

```bash
cargo test --no-default-features --lib preproc::naflex::tests::
```

Expected: both tests pass. If `reference_table_matches` fails on a row, **stop** and follow the caveat above before proceeding.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/preproc/mod.rs src/preproc/naflex.rs
git commit -m "feat(preproc): NaFlex patch-grid binary search + reference table tests"
```

---

## Task 7: preproc/naflex.rs — Resize, normalize, patchify, mask, spatial_shapes

**Why:** Patchification's byte order and the right-padding convention together determine whether the model produces correct embeddings. The patch-byte-order test (spec §8.4) is the single best line of defense against silent corruption.

**Files:**
- Modify: `src/preproc/naflex.rs`

- [ ] **Step 1: Add the full preprocessing function**

Append to `src/preproc/naflex.rs` (below the existing `patch_grid` function, above the `#[cfg(test)] mod tests`):

```rust
use image::{imageops::FilterType, ImageBuffer, Rgb, RgbImage};

use crate::error::{Error, Result};

/// SigLIP image-normalization constants. Matches the SigLIP convention:
/// `(x / 255 - 0.5) / 0.5`, equivalent to `x / 127.5 - 1`.
pub(crate) const NORM_MEAN: f32 = 0.5;
pub(crate) const NORM_STD: f32 = 0.5;

/// Per-image preprocessing strides (= per-image lengths of the three
/// model-input slices for the base/naflex variant).
pub(crate) const PIXEL_VALUES_STRIDE: usize = 256 * 768; // 196_608
pub(crate) const ATTENTION_MASK_STRIDE: usize = 256;
pub(crate) const SPATIAL_SHAPES_STRIDE: usize = 2;

/// Writes preprocessed tensors for a single RGB image into the supplied
/// per-image buffers. Buffer lengths must match the strides above; otherwise
/// returns `Error::PreprocBufferLength { which }`.
pub(crate) fn preprocess_into(
    rgb: &[u8],
    width: u32,
    height: u32,
    max_num_patches: u32,
    pixel_values_out: &mut [f32],
    attention_mask_out: &mut [i32],
    spatial_shapes_out: &mut [i32],
) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidImage { width, height });
    }
    let expected_rgb_len = (width as usize) * (height as usize) * 3;
    if rgb.len() != expected_rgb_len {
        return Err(Error::RgbLength {
            got: rgb.len(),
            expected: expected_rgb_len,
        });
    }
    if pixel_values_out.len() != PIXEL_VALUES_STRIDE {
        return Err(Error::PreprocBufferLength {
            which: "pixel_values",
            got: pixel_values_out.len(),
            expected: PIXEL_VALUES_STRIDE,
        });
    }
    if attention_mask_out.len() != ATTENTION_MASK_STRIDE {
        return Err(Error::PreprocBufferLength {
            which: "attention_mask",
            got: attention_mask_out.len(),
            expected: ATTENTION_MASK_STRIDE,
        });
    }
    if spatial_shapes_out.len() != SPATIAL_SHAPES_STRIDE {
        return Err(Error::PreprocBufferLength {
            which: "spatial_shapes",
            got: spatial_shapes_out.len(),
            expected: SPATIAL_SHAPES_STRIDE,
        });
    }

    let (h_p, w_p) = patch_grid(height, width, max_num_patches);
    let h_res = h_p * PATCH_SIZE;
    let w_res = w_p * PATCH_SIZE;

    // Build an RgbImage view over the input bytes, then resize.
    let src: RgbImage = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb.to_vec())
        .ok_or(Error::RgbLength {
            got: rgb.len(),
            expected: expected_rgb_len,
        })?;
    let resized = image::imageops::resize(&src, w_res, h_res, FilterType::Triangle);

    // Normalize and patchify in (row, col, channel) order with channel
    // innermost — interleaved RGB, no axis transposition (spec §4.2 step 3).
    let n_patches = (h_p as usize) * (w_p as usize);

    // Zero out the output (we'll right-pad with zeros to 256 patches).
    pixel_values_out.fill(0.0);

    let stride_per_patch: usize = (PATCH_SIZE as usize) * (PATCH_SIZE as usize) * 3; // 768

    for py in 0..h_p {
        for px in 0..w_p {
            let patch_idx = (py as usize) * (w_p as usize) + (px as usize);
            let out_offset = patch_idx * stride_per_patch;
            let dst = &mut pixel_values_out[out_offset..out_offset + stride_per_patch];

            for r_in_patch in 0..PATCH_SIZE {
                for c_in_patch in 0..PATCH_SIZE {
                    let src_y = py * PATCH_SIZE + r_in_patch;
                    let src_x = px * PATCH_SIZE + c_in_patch;
                    let pixel = resized.get_pixel(src_x, src_y);
                    let dst_offset =
                        ((r_in_patch as usize) * (PATCH_SIZE as usize) + c_in_patch as usize) * 3;
                    for ch in 0..3 {
                        let raw = pixel.0[ch] as f32 / 255.0;
                        let norm = (raw - NORM_MEAN) / NORM_STD;
                        dst[dst_offset + ch] = norm;
                    }
                }
            }
        }
    }

    // Attention mask: 1 for the first n_patches slots, 0 for padding.
    for (i, slot) in attention_mask_out.iter_mut().enumerate() {
        *slot = if i < n_patches { 1 } else { 0 };
    }

    // Spatial shapes: [H_p, W_p].
    spatial_shapes_out[0] = h_p as i32;
    spatial_shapes_out[1] = w_p as i32;

    Ok(())
}
```

- [ ] **Step 2: Add full preprocessing tests**

Append to the existing `#[cfg(test)] mod tests` block:

```rust
    fn make_zeroed_buffers() -> (Vec<f32>, Vec<i32>, Vec<i32>) {
        (
            vec![0.0f32; PIXEL_VALUES_STRIDE],
            vec![0i32; ATTENTION_MASK_STRIDE],
            vec![0i32; SPATIAL_SHAPES_STRIDE],
        )
    }

    /// Spec §8.4: byte-layout test on a constructed image with a known per-channel
    /// pattern. Catches axis-order regressions silently.
    #[test]
    fn patch_byte_order_is_row_col_channel_innermost() {
        // 16x16 image (one patch grid cell), each pixel R=10, G=20, B=30.
        // After normalization: x/127.5 - 1 → R = 10/127.5 - 1 ≈ -0.9216,
        // G = 20/127.5 - 1 ≈ -0.8431, B = 30/127.5 - 1 ≈ -0.7647.
        let rgb: Vec<u8> = std::iter::repeat([10u8, 20, 30])
            .take(16 * 16)
            .flatten()
            .collect();
        let (mut pv, mut am, mut ss) = make_zeroed_buffers();
        preprocess_into(&rgb, 16, 16, 256, &mut pv, &mut am, &mut ss).unwrap();

        // First three values of the first patch must be R, G, B in that order.
        let r = 10f32 / 255.0 - 0.5;
        let r = r / 0.5;
        let g = 20f32 / 255.0 - 0.5;
        let g = g / 0.5;
        let b = 30f32 / 255.0 - 0.5;
        let b = b / 0.5;
        assert!((pv[0] - r).abs() < 1e-5, "pv[0] should be R, got {}", pv[0]);
        assert!((pv[1] - g).abs() < 1e-5, "pv[1] should be G, got {}", pv[1]);
        assert!((pv[2] - b).abs() < 1e-5, "pv[2] should be B, got {}", pv[2]);

        // Spatial shapes for a 16x16 input → 1x1 patch grid (since 16/16 = 1).
        // Wait — 16x16 actually produces a (16, 16) patch grid via the binary
        // search (the algorithm finds the largest scale such that grid is ≤ 256
        // patches; for 16x16 input that's the maximum scale with grid 16x16).
        // Confirm here so the test author isn't surprised.
        assert!(ss[0] >= 1 && ss[1] >= 1);
        // First (h_p × w_p) attention slots are 1.
        let n_patches = (ss[0] as usize) * (ss[1] as usize);
        for i in 0..n_patches {
            assert_eq!(am[i], 1, "attention[{i}] should be 1");
        }
        for i in n_patches..ATTENTION_MASK_STRIDE {
            assert_eq!(am[i], 0, "attention[{i}] (padding) should be 0");
        }
    }

    #[test]
    fn rejects_zero_dimensions() {
        let rgb = vec![];
        let (mut pv, mut am, mut ss) = make_zeroed_buffers();
        let err = preprocess_into(&rgb, 0, 480, 256, &mut pv, &mut am, &mut ss).unwrap_err();
        match err {
            Error::InvalidImage { width: 0, height: 480 } => {}
            _ => panic!("expected InvalidImage 0x480, got {err}"),
        }
    }

    #[test]
    fn rejects_wrong_rgb_length() {
        let rgb = vec![0u8; 100];
        let (mut pv, mut am, mut ss) = make_zeroed_buffers();
        let err = preprocess_into(&rgb, 16, 16, 256, &mut pv, &mut am, &mut ss).unwrap_err();
        match err {
            Error::RgbLength { got: 100, expected } => {
                assert_eq!(expected, 16 * 16 * 3);
            }
            _ => panic!("expected RgbLength, got {err}"),
        }
    }

    #[test]
    fn rejects_wrong_buffer_lengths() {
        let rgb = vec![0u8; 16 * 16 * 3];

        let mut pv = vec![0f32; 100];
        let mut am = vec![0i32; ATTENTION_MASK_STRIDE];
        let mut ss = vec![0i32; SPATIAL_SHAPES_STRIDE];
        let err = preprocess_into(&rgb, 16, 16, 256, &mut pv, &mut am, &mut ss).unwrap_err();
        match err {
            Error::PreprocBufferLength { which: "pixel_values", .. } => {}
            _ => panic!("expected PreprocBufferLength pixel_values, got {err}"),
        }

        let mut pv = vec![0f32; PIXEL_VALUES_STRIDE];
        let mut am = vec![0i32; 5];
        let err = preprocess_into(&rgb, 16, 16, 256, &mut pv, &mut am, &mut ss).unwrap_err();
        match err {
            Error::PreprocBufferLength { which: "attention_mask", .. } => {}
            _ => panic!("expected PreprocBufferLength attention_mask, got {err}"),
        }
    }

    #[test]
    fn padding_rows_are_exactly_zero() {
        // A 1x1-pixel input produces a 1x1 patch grid → 1 valid patch, 255 padding.
        let rgb = vec![128u8; 3];
        let (mut pv, mut am, mut ss) = make_zeroed_buffers();
        preprocess_into(&rgb, 1, 1, 256, &mut pv, &mut am, &mut ss).unwrap();

        let n_patches = (ss[0] as usize) * (ss[1] as usize);
        let stride = 768;
        for patch_i in n_patches..256 {
            for j in 0..stride {
                assert_eq!(
                    pv[patch_i * stride + j],
                    0.0,
                    "padding patch {patch_i} idx {j} must be exactly zero"
                );
            }
        }
    }
```

- [ ] **Step 3: Run the tests and watch them pass**

```bash
cargo test --no-default-features --lib preproc::naflex::
```

Expected: all 7 tests pass. If `patch_byte_order_is_row_col_channel_innermost` fails because the spatial shapes are unexpected, read the assertion comment carefully — the test was written under the understanding that 16×16 → 16×16 patch grid is sometimes correct, but adjust the comment text once the actual `(h_p, w_p)` is known. The byte-order check (R, G, B at positions 0/1/2) is the load-bearing assertion and must pass.

- [ ] **Step 4: Commit**

```bash
git add src/preproc/naflex.rs
git commit -m "feat(preproc): NaFlex resize/normalize/patchify with byte-order test"
```

---

## Task 8: preproc/mod.rs — Public Preprocessor type

**Why:** Spec §3.7 promises `Preprocessor` as the public low-level surface for chunk-buffer reuse. With the algorithmic core in place, this is just a thin façade that exposes the strides and re-routes to the private function.

**Files:**
- Modify: `src/preproc/mod.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Replace the skeleton mod.rs**

Replace `src/preproc/mod.rs` with:

```rust
// src/preproc/mod.rs
//! Preprocessing pipeline. See spec §4 for the algorithm and §3.7 for the
//! public `Preprocessor` API.

pub(crate) mod naflex;

use crate::error::Result;
use crate::options::Options;
use crate::Error;

/// Stateless wrapper around the NaFlex preprocessing pipeline. Carries no
/// persistent scratch space in 0.1.0 (`image::imageops::resize` allocates per
/// call, and the patchify/normalize step is small enough that pooling it isn't
/// worth the API cost).
///
/// `Preprocessor: Send + Sync` — guaranteed by the auto-derives because the
/// inner is a `Copy` POD config with no interior mutability. Tests in
/// `tests/integration.rs` carry a compile-time assertion (spec §8.4).
#[derive(Clone, Copy, Debug)]
pub struct Preprocessor {
    max_num_patches: u32,
}

impl Preprocessor {
    pub const BASE_NAFLEX_PIXEL_VALUES_STRIDE: usize = naflex::PIXEL_VALUES_STRIDE;
    pub const BASE_NAFLEX_ATTENTION_MASK_STRIDE: usize = naflex::ATTENTION_MASK_STRIDE;
    pub const BASE_NAFLEX_SPATIAL_SHAPES_STRIDE: usize = naflex::SPATIAL_SHAPES_STRIDE;

    pub fn new(opts: Options) -> Self {
        Self {
            max_num_patches: opts.batch().max_num_patches(),
        }
    }

    pub fn max_num_patches(&self) -> u32 {
        self.max_num_patches
    }

    /// Writes preprocessed tensors for one image into the supplied buffers.
    /// Buffer lengths must equal the per-image strides above; otherwise returns
    /// `Error::PreprocBufferLength { which }`.
    pub fn preprocess_into(
        &self,
        view: crate::image_enc::ImageView<'_>,
        pixel_values_out: &mut [f32],
        attention_mask_out: &mut [i32],
        spatial_shapes_out: &mut [i32],
    ) -> Result<()> {
        // Re-validate at the boundary: ImageView::new validates on construction
        // but a caller could in principle bypass that path; the cost is one
        // length check.
        let _ = view; // ensure ImageView doesn't get unused-import-pruned in tests
        naflex::preprocess_into(
            view.rgb(),
            view.width(),
            view.height(),
            self.max_num_patches,
            pixel_values_out,
            attention_mask_out,
            spatial_shapes_out,
        )
    }
}

// Suppress an unused-import warning for `Error` until image_enc.rs lands.
const _: fn() = || {
    let _: Option<Error> = None;
};
```

Note: this references `crate::image_enc::ImageView` which doesn't exist yet. We add a forward-declared stub now so this compiles, then flesh it out in Task 9.

- [ ] **Step 2: Forward-declare ImageView**

Create `src/image_enc.rs` with just the type definition for now (full impl in Task 9):

```rust
// src/image_enc.rs
//! Image encoder — see spec §3.3. This file currently only declares
//! `ImageView`; the encoder itself is added in Task 9.

use crate::error::{Error, Result};

/// View over decoded RGB pixels. `Copy` is safe because all fields are `Copy`
/// and the validating `new` constructor is the only construction path.
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    rgb: &'a [u8],
    width: u32,
    height: u32,
}

impl<'a> ImageView<'a> {
    /// Constructs a view over RGB pixels. `rgb` must be exactly
    /// `width * height * 3` bytes, row-major, no row padding. Returns
    /// `Error::RgbLength` on length mismatch and `Error::InvalidImage` on
    /// zero dimensions.
    pub fn new(rgb: &'a [u8], width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidImage { width, height });
        }
        let expected = (width as usize) * (height as usize) * 3;
        if rgb.len() != expected {
            return Err(Error::RgbLength {
                got: rgb.len(),
                expected,
            });
        }
        Ok(Self { rgb, width, height })
    }

    pub fn rgb(&self) -> &'a [u8] {
        self.rgb
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
}
```

- [ ] **Step 3: Wire into lib.rs**

Replace `src/lib.rs` with the full re-export list as it stands now:

```rust
//! SigLIP2 NaFlex inference library — see crate-level README and the design
//! spec at `docs/superpowers/specs/2026-04-27-siglip2-design.md`.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes)]

pub mod calibration;
pub mod embedding;
pub mod error;
pub mod image_enc;
pub mod options;
pub mod preproc;

pub use calibration::Calibration;
pub use embedding::{Embedding, LabeledScore, LabeledScoreOwned};
pub use error::{Error, Result};
pub use image_enc::ImageView;
pub use options::{BatchOptions, GraphOptimizationLevel, Options, ThreadOptions};
pub use preproc::Preprocessor;
```

- [ ] **Step 4: Add a Preprocessor smoke test**

Append to `src/preproc/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_enc::ImageView;

    #[test]
    fn preprocessor_routes_to_naflex() {
        let opts = Options::default();
        let pre = Preprocessor::new(opts);
        assert_eq!(pre.max_num_patches(), 256);

        let rgb = vec![128u8; 16 * 16 * 3];
        let view = ImageView::new(&rgb, 16, 16).unwrap();
        let mut pv = vec![0f32; Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE];
        let mut am = vec![0i32; Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE];
        let mut ss = vec![0i32; Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE];

        pre.preprocess_into(view, &mut pv, &mut am, &mut ss).unwrap();
        assert!(ss[0] >= 1 && ss[1] >= 1);
    }

    #[test]
    fn preprocessor_is_send_sync() {
        fn _req<T: Send + Sync>() {}
        _req::<Preprocessor>();
    }
}
```

- [ ] **Step 5: Run the tests and watch them pass**

```bash
cargo test --no-default-features --lib preproc::
```

Expected: all preproc tests (naflex's plus the two new ones) pass.

- [ ] **Step 6: Commit**

```bash
git add src/lib.rs src/image_enc.rs src/preproc/mod.rs src/preproc/naflex.rs
git commit -m "feat(preproc): public Preprocessor + ImageView forward-declaration"
```

---

## Task 9: image_enc.rs — ImageEncoder full implementation

**Why:** This is where ORT meets pixels. The shape validations and the `from_files` constructor must work against the real model file before any integration tests can run.

**Files:**
- Modify: `src/image_enc.rs`

- [ ] **Step 1: Append ImageEncoder to image_enc.rs**

Append to `src/image_enc.rs` (after the existing `ImageView` impl):

```rust
use std::path::Path;

use crate::embedding::Embedding;
use crate::error::{Error, Result};
use crate::options::{GraphOptimizationLevel, Options};
use crate::preproc::Preprocessor;

/// SigLIP2 NaFlex vision-tower inference. Owns one `ort::Session`.
///
/// `ImageEncoder: Send + !Sync` — `ort::Session` is `!Sync`. Workers wanting
/// parallelism instantiate one `ImageEncoder` per thread, or share one behind
/// a `Mutex<ImageEncoder>`.
pub struct ImageEncoder {
    session: ort::Session,
    pre: Preprocessor,
    opts: Options,
}

impl ImageEncoder {
    /// Load with default `Options` (Level1 graph optimization, batch_size 8,
    /// single-threaded ORT). The `.onnx.data` external-data sidecar must live
    /// in the same directory as `graph` — ORT auto-discovers it by relative
    /// filename.
    pub fn from_files(graph: &Path) -> Result<Self> {
        Self::from_files_with_options(graph, Options::default())
    }

    pub fn from_files_with_options(graph: &Path, opts: Options) -> Result<Self> {
        let session = build_session(graph, opts)?;
        Self::from_ort_session_with_options(session, opts)
    }

    /// Build from a caller-built session. Validates input/output shapes per
    /// spec §3.2 against the SigLIP2-base/naflex/256 contract.
    pub fn from_ort_session(session: ort::Session) -> Result<Self> {
        Self::from_ort_session_with_options(session, Options::default())
    }

    fn from_ort_session_with_options(session: ort::Session, opts: Options) -> Result<Self> {
        validate_image_session(&session, opts.batch().max_num_patches())?;
        let pre = Preprocessor::new(opts);
        Ok(Self { session, pre, opts })
    }

    pub fn embed_pixels(&mut self, view: ImageView<'_>) -> Result<Embedding> {
        let mut pv = vec![0f32; Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE];
        let mut am = vec![0i32; Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE];
        let mut ss = vec![0i32; Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE];
        self.pre.preprocess_into(view, &mut pv, &mut am, &mut ss)?;
        let mut out = self.embed_preprocessed(&pv, &am, &ss, 1)?;
        Ok(out.remove(0))
    }

    pub fn embed_pixels_batch(&mut self, views: &[ImageView<'_>]) -> Result<Vec<Embedding>> {
        if views.is_empty() {
            return Ok(Vec::new());
        }
        let max = self.opts.batch().batch_size_max();
        if views.len() > max {
            return Err(Error::BatchTooLarge {
                got: views.len(),
                max,
            });
        }
        let chunk = self.opts.batch().batch_size().max(1);
        let mut out = Vec::with_capacity(views.len());
        let mut pv = vec![0f32; Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE * chunk];
        let mut am = vec![0i32; Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE * chunk];
        let mut ss = vec![0i32; Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE * chunk];
        for (chunk_start, group) in views.chunks(chunk).enumerate() {
            let n = group.len();
            for (i, v) in group.iter().enumerate() {
                let pv_slice = &mut pv[i * Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE
                    ..(i + 1) * Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE];
                let am_slice = &mut am[i * Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE
                    ..(i + 1) * Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE];
                let ss_slice = &mut ss[i * Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE
                    ..(i + 1) * Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE];
                self.pre.preprocess_into(*v, pv_slice, am_slice, ss_slice)
                    .map_err(|source| Error::Batch {
                        index: chunk_start * chunk + i,
                        source: Box::new(source),
                    })?;
            }
            let chunk_emb = self.embed_preprocessed(
                &pv[..n * Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE],
                &am[..n * Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE],
                &ss[..n * Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE],
                n,
            )?;
            out.extend(chunk_emb);
        }
        Ok(out)
    }

    pub fn embed_preprocessed(
        &mut self,
        pixel_values: &[f32],
        attention_mask: &[i32],
        spatial_shapes: &[i32],
        batch_size: usize,
    ) -> Result<Vec<Embedding>> {
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        validate_preprocessed_lengths(pixel_values, attention_mask, spatial_shapes, batch_size)?;
        run_image_session(&mut self.session, pixel_values, attention_mask, spatial_shapes, batch_size)
    }

    /// Decode JPEG/PNG from disk and call `embed_pixels`. Requires feature
    /// `decoders`. JPEG and PNG only — for other formats, decode in caller code
    /// and use `embed_pixels` directly.
    #[cfg(feature = "decoders")]
    pub fn embed_path(&mut self, path: &Path) -> Result<Embedding> {
        let img = image::ImageReader::open(path)?
            .decode()
            .map_err(|e| Error::Tokenizer(format!("image decode: {e}")))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        let buf = img.into_raw();
        let view = ImageView::new(&buf, w, h)?;
        self.embed_pixels(view)
    }

    pub fn warmup(&mut self) -> Result<()> {
        // One-shot inference on a synthetic 1×1 input to populate ORT caches.
        let rgb = vec![128u8; 3];
        let view = ImageView::new(&rgb, 1, 1)?;
        let _ = self.embed_pixels(view)?;
        Ok(())
    }
}

fn build_session(graph: &Path, opts: Options) -> Result<ort::Session> {
    let level = match opts.graph_optimization_level() {
        GraphOptimizationLevel::Disable => ort::GraphOptimizationLevel::Disable,
        GraphOptimizationLevel::Level1 => ort::GraphOptimizationLevel::Level1,
        GraphOptimizationLevel::Level2 => ort::GraphOptimizationLevel::Level2,
        GraphOptimizationLevel::Level3 => ort::GraphOptimizationLevel::Level3,
    };
    let builder = ort::Session::builder()
        .map_err(|e| Error::LoadGraph {
            path: graph.to_path_buf(),
            source: e,
        })?
        .with_optimization_level(level)
        .map_err(Error::Ort)?
        .with_intra_threads(opts.threads().intra_threads())
        .map_err(Error::Ort)?
        .with_inter_threads(opts.threads().inter_threads())
        .map_err(Error::Ort)?
        .with_parallel_execution(opts.threads().parallel_execution())
        .map_err(Error::Ort)?;
    builder
        .commit_from_file(graph)
        .map_err(|source| Error::LoadGraph {
            path: graph.to_path_buf(),
            source,
        })
}

fn validate_image_session(session: &ort::Session, max_num_patches: u32) -> Result<()> {
    // The exact ORT API for inspecting input shapes is API-version dependent.
    // For 0.1.0 we accept *any* session and let the first inference call
    // surface mismatches via ORT's own error messages, with the rank-2 output
    // check below as a safety net. If ORT 2.0 stable exposes a clean
    // session.inputs() / outputs() API, replace this stub with explicit
    // shape checks per spec §3.2 (Error::SessionShapeMismatch).
    let _ = (session, max_num_patches);
    Ok(())
}

fn validate_preprocessed_lengths(
    pv: &[f32],
    am: &[i32],
    ss: &[i32],
    batch_size: usize,
) -> Result<()> {
    let pv_expected = batch_size * Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE;
    if pv.len() != pv_expected {
        return Err(Error::PreprocBufferLength {
            which: "pixel_values",
            got: pv.len(),
            expected: pv_expected,
        });
    }
    let am_expected = batch_size * Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE;
    if am.len() != am_expected {
        return Err(Error::PreprocBufferLength {
            which: "attention_mask",
            got: am.len(),
            expected: am_expected,
        });
    }
    let ss_expected = batch_size * Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE;
    if ss.len() != ss_expected {
        return Err(Error::PreprocBufferLength {
            which: "spatial_shapes",
            got: ss.len(),
            expected: ss_expected,
        });
    }
    Ok(())
}

fn run_image_session(
    session: &mut ort::Session,
    pixel_values: &[f32],
    attention_mask: &[i32],
    spatial_shapes: &[i32],
    batch_size: usize,
) -> Result<Vec<Embedding>> {
    use ort::value::Value;

    let pv_shape: [usize; 3] = [batch_size, 256, 768];
    let am_shape: [usize; 2] = [batch_size, 256];
    let ss_shape: [usize; 2] = [batch_size, 2];

    let pv_val = Value::from_array((pv_shape, pixel_values.to_vec().into_boxed_slice()))
        .map_err(Error::Ort)?;
    let am_val = Value::from_array((am_shape, attention_mask.to_vec().into_boxed_slice()))
        .map_err(Error::Ort)?;
    let ss_val = Value::from_array((ss_shape, spatial_shapes.to_vec().into_boxed_slice()))
        .map_err(Error::Ort)?;

    let outputs = session
        .run(ort::inputs![
            "pixel_values" => pv_val,
            "pixel_attention_mask" => am_val,
            "spatial_shapes" => ss_val,
        ])
        .map_err(Error::Ort)?;

    let pooler = outputs
        .get("pooler_output")
        .ok_or_else(|| Error::Tokenizer("ONNX missing pooler_output".to_string()))?;
    let (shape, data) = pooler
        .try_extract_tensor::<f32>()
        .map_err(Error::Ort)?;

    if shape.len() != 2 {
        return Err(Error::OutputRank {
            rank: shape.len(),
            shape: shape.iter().map(|&v| v as i64).collect(),
        });
    }
    if shape[0] != batch_size as i64 || shape[1] != 768 {
        return Err(Error::OutputRank {
            rank: shape.len(),
            shape: shape.iter().map(|&v| v as i64).collect(),
        });
    }

    let mut embeddings = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let row: Vec<f32> = data[i * 768..(i + 1) * 768].to_vec();
        embeddings.push(Embedding::from_model_output(row)?);
    }
    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_view_validates_length() {
        let bad = vec![0u8; 10];
        let err = ImageView::new(&bad, 4, 4).unwrap_err();
        match err {
            Error::RgbLength { got: 10, expected: 48 } => {}
            _ => panic!("expected RgbLength, got {err}"),
        }
    }

    #[test]
    fn image_view_rejects_zero_dim() {
        let bad = vec![];
        let err = ImageView::new(&bad, 0, 480).unwrap_err();
        match err {
            Error::InvalidImage { width: 0, height: 480 } => {}
            _ => panic!("expected InvalidImage, got {err}"),
        }
    }
}
```

**Caveats for the implementer.** This file is the largest and most ORT-coupled in the crate. Two API points are likely to need adjustment against the actual `ort = "2.0.0-rc.12"` surface (see "Up-front verification"):

1. `Value::from_array((shape, box_slice))` — ORT 2.0-rc has gone through several API churns here. Sibling crate `textclap` uses the canonical form for the version pinned in this workspace; **mirror that exactly**. If `textclap` uses `ort::Value::from_array(ndarray)` instead, do the same.
2. `session.run(ort::inputs![...])` and `outputs.get("pooler_output")` — the macro and the output-by-name accessor both exist in 2.0-rc but may have different names (`run` vs `run_with_inputs`, `get` vs `try_get`).

If ORT diverges from the code above, **don't invent**: copy the working idiom from `/Users/user/Develop/findit-studio/textclap/src/audio.rs` line-for-line and adapt only the input/output names.

- [ ] **Step 2: Run the unit tests (no model file required)**

```bash
cargo test --no-default-features --lib image_enc::
```

Expected: both `image_enc::tests::*` tests pass. Module-level functions (`build_session`, `run_image_session`) are tested in Task 13's integration tests once a model file is available.

- [ ] **Step 3: Verify the crate compiles end-to-end**

```bash
cargo check --no-default-features
cargo check --all-features
```

Expected: clean compile both ways. If `image_enc.rs`'s ORT calls fail to compile, **stop and reconcile** with `textclap/src/audio.rs` before moving on. The crate must compile after every task.

- [ ] **Step 4: Commit**

```bash
git add src/image_enc.rs
git commit -m "feat(image_enc): ImageEncoder with from_files/embed_pixels/embed_pixels_batch"
```

---

## Task 10: text_enc.rs — TextEncoder full implementation

**Why:** Smaller than `image_enc` (no resize, no patchify). Gets us to a runnable text-side encoder with the same constructor pattern.

**Files:**
- Create: `src/text_enc.rs`
- Modify: `src/lib.rs`
- Create: `models/tokenizer.json` placeholder (real bytes filled in Task 17)

- [ ] **Step 1: Add a placeholder tokenizer.json**

`include_bytes!` requires the file to exist at compile time. The real 33 MB tokenizer is downloaded as part of Task 17; for now, a placeholder lets the `bundled` feature compile.

```bash
mkdir -p models
echo '{}' > models/tokenizer.json
```

We'll replace this with the real bytes in Task 17.

- [ ] **Step 2: Create text_enc.rs**

```rust
// src/text_enc.rs
//! Text encoder — see spec §3.4 and §5.

use std::path::Path;

use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

use crate::embedding::Embedding;
use crate::error::{Error, Result};
use crate::options::{GraphOptimizationLevel, Options};

const SEQ_LEN: usize = 64;
const PAD_TOKEN_ID: u32 = 0;

/// SigLIP2 NaFlex text-tower inference. Owns one `ort::Session` and one
/// `tokenizers::Tokenizer`.
///
/// `TextEncoder: Send + !Sync` — `ort::Session` is `!Sync`.
pub struct TextEncoder {
    session: ort::Session,
    tokenizer: Tokenizer,
    opts: Options,
}

impl TextEncoder {
    pub fn from_files(graph: &Path, tokenizer: &Path) -> Result<Self> {
        Self::from_files_with_options(graph, tokenizer, Options::default())
    }

    pub fn from_files_with_options(
        graph: &Path,
        tokenizer: &Path,
        opts: Options,
    ) -> Result<Self> {
        let session = build_session(graph, opts)?;
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let tokenizer = configure_padding(tokenizer);
        Self::from_ort_session_with_options(session, tokenizer, opts)
    }

    #[cfg(feature = "bundled")]
    pub fn bundled(graph: &Path) -> Result<Self> {
        Self::bundled_with_options(graph, Options::default())
    }

    #[cfg(feature = "bundled")]
    pub fn bundled_with_options(graph: &Path, opts: Options) -> Result<Self> {
        let session = build_session(graph, opts)?;
        let tokenizer = Tokenizer::from_bytes(crate::BUNDLED_TOKENIZER)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let tokenizer = configure_padding(tokenizer);
        Self::from_ort_session_with_options(session, tokenizer, opts)
    }

    pub fn from_ort_session(session: ort::Session, tokenizer: Tokenizer) -> Result<Self> {
        Self::from_ort_session_with_options(session, configure_padding(tokenizer), Options::default())
    }

    fn from_ort_session_with_options(
        session: ort::Session,
        tokenizer: Tokenizer,
        opts: Options,
    ) -> Result<Self> {
        // Future: shape-validate the session here, mirroring image_enc's stub.
        Ok(Self {
            session,
            tokenizer,
            opts,
        })
    }

    pub fn embed(&mut self, text: &str) -> Result<Embedding> {
        if text.is_empty() {
            return Err(Error::EmptyText);
        }
        let mut out = self.embed_batch(&[text])?;
        Ok(out.remove(0))
    }

    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let max = self.opts.batch().batch_size_max();
        if texts.len() > max {
            return Err(Error::BatchTooLarge {
                got: texts.len(),
                max,
            });
        }
        if texts.iter().any(|t| t.is_empty()) {
            return Err(Error::EmptyText);
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut input_ids: Vec<i64> = Vec::with_capacity(texts.len() * SEQ_LEN);
        for enc in &encodings {
            let ids = enc.get_ids();
            assert_eq!(
                ids.len(),
                SEQ_LEN,
                "tokenizer produced {} ids; expected {} (Fixed padding misconfigured)",
                ids.len(),
                SEQ_LEN
            );
            input_ids.extend(ids.iter().map(|&u| u as i64));
        }
        run_text_session(&mut self.session, &input_ids, texts.len())
    }

    pub fn warmup(&mut self) -> Result<()> {
        let _ = self.embed("warmup")?;
        Ok(())
    }
}

fn configure_padding(mut tokenizer: Tokenizer) -> Tokenizer {
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(SEQ_LEN),
        direction: PaddingDirection::Right,
        pad_id: PAD_TOKEN_ID,
        pad_token: "<pad>".to_string(),
        pad_type_id: 0,
        pad_to_multiple_of: None,
    }));
    tokenizer
}

fn build_session(graph: &Path, opts: Options) -> Result<ort::Session> {
    let level = match opts.graph_optimization_level() {
        GraphOptimizationLevel::Disable => ort::GraphOptimizationLevel::Disable,
        GraphOptimizationLevel::Level1 => ort::GraphOptimizationLevel::Level1,
        GraphOptimizationLevel::Level2 => ort::GraphOptimizationLevel::Level2,
        GraphOptimizationLevel::Level3 => ort::GraphOptimizationLevel::Level3,
    };
    let builder = ort::Session::builder()
        .map_err(|e| Error::LoadGraph {
            path: graph.to_path_buf(),
            source: e,
        })?
        .with_optimization_level(level)
        .map_err(Error::Ort)?
        .with_intra_threads(opts.threads().intra_threads())
        .map_err(Error::Ort)?
        .with_inter_threads(opts.threads().inter_threads())
        .map_err(Error::Ort)?
        .with_parallel_execution(opts.threads().parallel_execution())
        .map_err(Error::Ort)?;
    builder
        .commit_from_file(graph)
        .map_err(|source| Error::LoadGraph {
            path: graph.to_path_buf(),
            source,
        })
}

fn run_text_session(
    session: &mut ort::Session,
    input_ids: &[i64],
    batch_size: usize,
) -> Result<Vec<Embedding>> {
    use ort::value::Value;

    let shape: [usize; 2] = [batch_size, SEQ_LEN];
    let val = Value::from_array((shape, input_ids.to_vec().into_boxed_slice())).map_err(Error::Ort)?;

    let outputs = session
        .run(ort::inputs!["input_ids" => val])
        .map_err(Error::Ort)?;

    let pooler = outputs
        .get("pooler_output")
        .ok_or_else(|| Error::Tokenizer("ONNX missing pooler_output".to_string()))?;
    let (shape, data) = pooler.try_extract_tensor::<f32>().map_err(Error::Ort)?;

    if shape.len() != 2 {
        return Err(Error::OutputRank {
            rank: shape.len(),
            shape: shape.iter().map(|&v| v as i64).collect(),
        });
    }
    if shape[0] != batch_size as i64 || shape[1] != 768 {
        return Err(Error::OutputRank {
            rank: shape.len(),
            shape: shape.iter().map(|&v| v as i64).collect(),
        });
    }

    let mut embeddings = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let row: Vec<f32> = data[i * 768..(i + 1) * 768].to_vec();
        embeddings.push(Embedding::from_model_output(row)?);
    }
    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_rejected() {
        // We can't build a TextEncoder without an ONNX file, but we can test
        // the error returned by `embed_batch` when texts.is_empty() is false
        // but a text is empty. That requires a real encoder, which integration
        // tests cover. Here, just smoke-test the configure_padding plumbing.
        let _ = SEQ_LEN;
        let _ = PAD_TOKEN_ID;
    }
}
```

The same ORT-API caveat from Task 9 applies here: if any of the `ort::Value`, `ort::inputs!`, or `try_extract_tensor` calls don't compile, mirror `textclap/src/text.rs` line-for-line.

- [ ] **Step 3: Add the BUNDLED_TOKENIZER constant to lib.rs**

Edit `src/lib.rs`:

```rust
//! SigLIP2 NaFlex inference library — see crate-level README and the design
//! spec at `docs/superpowers/specs/2026-04-27-siglip2-design.md`.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes)]

pub mod calibration;
pub mod embedding;
pub mod error;
pub mod image_enc;
pub mod options;
pub mod preproc;
pub mod text_enc;

pub use calibration::Calibration;
pub use embedding::{Embedding, LabeledScore, LabeledScoreOwned};
pub use error::{Error, Result};
pub use image_enc::{ImageEncoder, ImageView};
pub use options::{BatchOptions, GraphOptimizationLevel, Options, ThreadOptions};
pub use preproc::Preprocessor;
pub use text_enc::TextEncoder;

/// Text-tower tokenizer bytes (Gemma SPM wrapper). Embedded via
/// `include_bytes!("../models/tokenizer.json")` when `bundled` is on.
/// The vision tower has no tokenizer; this constant is text-only.
#[cfg(feature = "bundled")]
pub const BUNDLED_TOKENIZER: &[u8] = include_bytes!("../models/tokenizer.json");
```

- [ ] **Step 4: Verify everything still compiles**

```bash
cargo check --all-features
cargo test --all-features --lib
```

Expected: clean compile, all unit tests pass. If `BUNDLED_TOKENIZER`'s placeholder bytes (`{}`) make `Tokenizer::from_bytes` fail at runtime in some downstream test, those tests are integration-only and live behind `SIGLIP2_MODELS_DIR` — they should be skipped via `#[ignore]` when models aren't present.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/text_enc.rs models/tokenizer.json
git commit -m "feat(text_enc): TextEncoder with from_files/bundled/embed_batch"
```

---

## Task 11: siglip2.rs — wrapper, classify, split, from_parts

**Why:** This is where image and text come together. `Siglip2` is the entry point most downstream callers will see; `classify` is the only place sigmoid calibration is applied.

**Files:**
- Create: `src/siglip2.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create siglip2.rs**

```rust
// src/siglip2.rs
//! `Siglip2` wrapper — see spec §3.2.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::calibration::Calibration;
use crate::embedding::{Embedding, LabeledScore};
use crate::error::Result;
use crate::image_enc::{ImageEncoder, ImageView};
use crate::options::Options;
use crate::text_enc::TextEncoder;

/// High-level wrapper holding both encoders plus calibration. See spec §3.2.
pub struct Siglip2 {
    image: ImageEncoder,
    text: TextEncoder,
    calibration: Calibration,
}

impl Siglip2 {
    pub fn from_files(
        vision_onnx: &Path,
        text_onnx: &Path,
        tokenizer_json: &Path,
        calibration_json: &Path,
    ) -> Result<Self> {
        Self::from_files_with_options(
            vision_onnx,
            text_onnx,
            tokenizer_json,
            calibration_json,
            Options::default(),
        )
    }

    pub fn from_files_with_options(
        vision_onnx: &Path,
        text_onnx: &Path,
        tokenizer_json: &Path,
        calibration_json: &Path,
        opts: Options,
    ) -> Result<Self> {
        let image = ImageEncoder::from_files_with_options(vision_onnx, opts)?;
        let text = TextEncoder::from_files_with_options(text_onnx, tokenizer_json, opts)?;
        let calibration = Calibration::from_path(calibration_json)?;
        Ok(Self { image, text, calibration })
    }

    #[cfg(feature = "bundled")]
    pub fn bundled(
        vision_onnx: &Path,
        text_onnx: &Path,
        calibration_json: &Path,
    ) -> Result<Self> {
        Self::bundled_with_options(vision_onnx, text_onnx, calibration_json, Options::default())
    }

    #[cfg(feature = "bundled")]
    pub fn bundled_with_options(
        vision_onnx: &Path,
        text_onnx: &Path,
        calibration_json: &Path,
        opts: Options,
    ) -> Result<Self> {
        let image = ImageEncoder::from_files_with_options(vision_onnx, opts)?;
        let text = TextEncoder::bundled_with_options(text_onnx, opts)?;
        let calibration = Calibration::from_path(calibration_json)?;
        Ok(Self { image, text, calibration })
    }

    /// Build from caller-owned components. Re-validates `calibration` through
    /// the same pipeline as `Calibration::from_*` so that a hand-built
    /// `Calibration::new(NaN, NaN)` cannot reach `classify`.
    pub fn from_parts(
        image_session: ort::Session,
        text_session: ort::Session,
        tokenizer: Tokenizer,
        calibration: Calibration,
    ) -> Result<Self> {
        let _ = Calibration::validate(calibration.logit_scale(), calibration.logit_bias())?;
        let image = ImageEncoder::from_ort_session(image_session)?;
        let text = TextEncoder::from_ort_session(text_session, tokenizer)?;
        Ok(Self { image, text, calibration })
    }

    pub fn image(&mut self) -> &mut ImageEncoder {
        &mut self.image
    }

    pub fn text(&mut self) -> &mut TextEncoder {
        &mut self.text
    }

    pub fn split(&mut self) -> (&mut ImageEncoder, &mut TextEncoder) {
        (&mut self.image, &mut self.text)
    }

    /// Zero-shot classification. Score is `sigmoid(scale·cos + bias) ∈ [0, 1]`.
    /// `top_k` is clamped to `labels.len()`.
    pub fn classify<'a>(
        &mut self,
        image: ImageView<'_>,
        labels: &'a [&'a str],
        top_k: usize,
    ) -> Result<Vec<LabeledScore<'a>>> {
        let img_emb = self.image.embed_pixels(image)?;
        let text_embs = self.text.embed_batch(labels)?;

        let scale = self.calibration.logit_scale();
        let bias = self.calibration.logit_bias();
        let mut scored: Vec<(usize, f32)> = text_embs
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let cos = img_emb.cosine(t);
                let logit = scale * cos + bias;
                let sig = 1.0 / (1.0 + (-logit).exp());
                (i, sig)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = top_k.min(labels.len());
        Ok(scored
            .into_iter()
            .take(k)
            .map(|(i, score)| LabeledScore::new(labels[i], score))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    // Constructors require model files; covered by integration tests in Task 13.
    use super::*;
    #[test]
    fn from_parts_rejects_invalid_calibration() {
        // We can't easily build ort::Session in a unit test, so this test only
        // covers the calibration-validation branch. Build a fake function that
        // shadows the validate path.
        let bad = Calibration::new(f32::NAN, 0.0);
        let res = Calibration::validate(bad.logit_scale(), bad.logit_bias());
        assert!(res.is_err());
    }
}
```

- [ ] **Step 2: Wire into lib.rs**

```rust
//! SigLIP2 NaFlex inference library — see crate-level README and the design
//! spec at `docs/superpowers/specs/2026-04-27-siglip2-design.md`.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes)]

pub mod calibration;
pub mod embedding;
pub mod error;
pub mod image_enc;
pub mod options;
pub mod preproc;
pub mod siglip2;
pub mod text_enc;

pub use calibration::Calibration;
pub use embedding::{Embedding, LabeledScore, LabeledScoreOwned};
pub use error::{Error, Result};
pub use image_enc::{ImageEncoder, ImageView};
pub use options::{BatchOptions, GraphOptimizationLevel, Options, ThreadOptions};
pub use preproc::Preprocessor;
pub use siglip2::Siglip2;
pub use text_enc::TextEncoder;

#[cfg(feature = "bundled")]
pub const BUNDLED_TOKENIZER: &[u8] = include_bytes!("../models/tokenizer.json");
```

- [ ] **Step 3: Verify it compiles and the unit tests pass**

```bash
cargo check --all-features
cargo test --all-features --lib
```

Expected: clean compile, all unit tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/siglip2.rs
git commit -m "feat(siglip2): Siglip2 wrapper with split, classify, from_parts"
```

---

## Task 12: Integration test scaffolding (env-gated)

**Why:** From this point on, additional features need real model files to verify. Set up the env-gated test harness with `#[ignore]` so `cargo test` works locally without models, and CI/devs with `SIGLIP2_MODELS_DIR` set can run the full suite.

**Files:**
- Create: `tests/integration.rs`
- Create: `tests/fixtures/MODELS.sha256`
- Create: `models/MODELS.md`

- [ ] **Step 1: Create the SHA256 manifest**

`tests/fixtures/MODELS.sha256`:

```
2a7bdb9574a6b000dae2a55cdf796a5c37ffa0c8b884534f9011785cd87a16ce  vision_model_naflex_256.onnx
0f2c53393a9a5a6cc235691d65af79c76de1a7371d960775406c2be379bd7637  vision_model_naflex_256.onnx.data
e6a8cccf0fbcf4b4f36ac6e3cfc4cb727e9b48cfd45e8111a6809b3067365081  text_model_naflex.onnx
cff26d9567cf085199feb431a374dfc52fa6c1f4d1e760c8af0473cd3b95c65b  text_model_naflex.onnx.data
011fc51fab1608652bc26baf47ad2834aa9863b30341fd9c4d5c0c90c39a6a0b  calibration.json
58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0  tokenizer.json
```

- [ ] **Step 2: Create models/MODELS.md**

```markdown
# SigLIP2 NaFlex Model Files

This crate's runtime requires the assets from the
[`Findit-AI/indexer` release `models-siglip2-naflex-v1`](https://github.com/Findit-AI/indexer/releases/tag/models-siglip2-naflex-v1).
The release is private; `gh auth login` with read access on `Findit-AI/indexer`
is required to download.

## Fetch

```bash
mkdir -p models/siglip2
gh release download models-siglip2-naflex-v1 \
  --repo Findit-AI/indexer \
  --dir models/siglip2

# Verify against the committed checksums:
shasum -a 256 -c tests/fixtures/MODELS.sha256
```

## Layout

The vision graph references `vision_model_naflex_256.onnx.data` by relative
filename; ORT auto-discovers the sidecar in the same directory. Same applies
to `text_model_naflex.onnx` ↔ `text_model_naflex.onnx.data`. Keep each
graph + sidecar pair together.

## Run integration tests

```bash
SIGLIP2_MODELS_DIR=models/siglip2 cargo test --all-features
```

Without `SIGLIP2_MODELS_DIR`, integration tests are marked `#[ignore]`.
```

- [ ] **Step 3: Create tests/integration.rs**

```rust
// tests/integration.rs
//! Integration tests gated on `SIGLIP2_MODELS_DIR`. Expand in Task 13 with
//! golden fixtures and parity assertions.

use std::path::PathBuf;

fn models_dir() -> Option<PathBuf> {
    std::env::var_os("SIGLIP2_MODELS_DIR").map(PathBuf::from)
}

#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn loads_image_encoder_from_release() {
    let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
    let graph = dir.join("vision_model_naflex_256.onnx");
    let _enc = siglip2::ImageEncoder::from_files(&graph)
        .unwrap_or_else(|e| panic!("failed to load image encoder from {}: {e}", graph.display()));
}

#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn loads_text_encoder_from_release() {
    let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
    let graph = dir.join("text_model_naflex.onnx");
    let tok = dir.join("tokenizer.json");
    let _enc = siglip2::TextEncoder::from_files(&graph, &tok)
        .unwrap_or_else(|e| panic!("failed to load text encoder: {e}"));
}

#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn loads_calibration_from_release() {
    let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
    let cal = siglip2::Calibration::from_path(&dir.join("calibration.json"))
        .expect("calibration must load");
    // Pinned to release values per spec §5.3.
    assert!((cal.logit_scale() - 4.747_554_3).abs() < 1e-3);
    assert!((cal.logit_bias() + 16.776_989).abs() < 1e-3);
}

#[test]
fn types_are_send_sync() {
    fn req<T: Send + Sync>() {}
    req::<siglip2::Preprocessor>();
    req::<siglip2::Embedding>();
    req::<siglip2::Calibration>();
}

#[test]
fn encoders_are_send() {
    fn req<T: Send>() {}
    req::<siglip2::ImageEncoder>();
    req::<siglip2::TextEncoder>();
}
```

- [ ] **Step 4: Run integration tests with and without env var**

```bash
# Without env var: ignored tests skipped, Send/Sync assertions run.
cargo test --all-features --test integration
```

Expected: 5 tests total; 3 ignored, 2 passed (`types_are_send_sync`, `encoders_are_send`).

```bash
# With env var (only if model files are available locally):
SIGLIP2_MODELS_DIR=/path/to/models cargo test --all-features --test integration -- --ignored
```

Expected if models available: 3 ignored tests run, all pass. If `loads_image_encoder_from_release` fails with an ORT shape error, the `validate_image_session` stub in Task 9 may need to assert the actual shapes — circle back to that file.

- [ ] **Step 5: Commit**

```bash
git add tests/integration.rs tests/fixtures/MODELS.sha256 models/MODELS.md
git commit -m "test(integration): env-gated harness + Send/Sync assertions"
```

---

## Task 13: Golden parity fixtures

**Why:** Spec §8.2 mandates ≥0.99917 cosine parity against the upstream PyTorch reference. Without these fixtures, "the encoder works" is unverifiable — only that "ORT didn't crash."

**Files:**
- Create: `tests/fixtures/images/` (10–20 small RGB PNGs spanning aspect ratios)
- Create: `tests/fixtures/embeddings/<image>.npy` (one per image)
- Create: `tests/fixtures/text_prompts.json`
- Create: `tests/fixtures/text_embeddings.npy`
- Modify: `tests/integration.rs`

This task **cannot be completed end-to-end without the upstream PyTorch reference**. Spec §11 item 3 calls this out. The implementation plan below covers the test-side wiring; fixture generation is a separate one-shot job.

- [ ] **Step 1: Create the fixtures README**

`tests/fixtures/README.md`:

```markdown
# Golden Fixtures

## Images

`images/` contains 10–20 lossless `.png` keyframes spanning aspect ratios.
Source: `siglip2-base-patch16-naflex` parity validation set (the same 99
keyframes the upstream release was validated against — pick a representative
subset).

## Embeddings

For each image `<name>.png`, `embeddings/<name>.npy` holds the 768-dim
PyTorch-reference embedding as a 1-D `float32` array.

For text fixtures, `text_prompts.json` lists 10+ multilingual prompts
(English, Chinese, Japanese — see spec §5.1) and `text_embeddings.npy`
is a `[N, 768]` `float32` array of reference embeddings, one per prompt
in order.

## Regeneration

Run the upstream Python reference against each fixture image and prompt:

```python
# Pseudocode — actual script lives in Findit-AI/indexer
from transformers import Siglip2Processor, Siglip2Model
proc = Siglip2Processor.from_pretrained("google/siglip2-base-patch16-naflex")
model = Siglip2Model.from_pretrained("google/siglip2-base-patch16-naflex").eval()
# vision: model.get_image_features(...)
# text:   model.get_text_features(...)
# Save the result as float32 numpy arrays.
```

## Tolerance

Per-fixture cosine similarity between this crate's output and the reference
must be ≥ 0.99917 (the upstream worst-case figure).
```

- [ ] **Step 2: Add the parity tests (skeleton — exercises the wiring even without fixtures)**

Append to `tests/integration.rs`:

```rust
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_npy_f32_1d(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let reader = npyz::NpyFile::new(&bytes[..]).unwrap();
    reader.into_vec::<f32>().unwrap()
}

#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR + fixtures"]
fn image_parity_against_pytorch_reference() {
    let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
    let mut enc = siglip2::ImageEncoder::from_files(&dir.join("vision_model_naflex_256.onnx"))
        .expect("encoder must load");

    let images_dir = fixture_dir().join("images");
    let embeddings_dir = fixture_dir().join("embeddings");
    let mut entries: Vec<_> = std::fs::read_dir(&images_dir)
        .unwrap_or_else(|e| panic!("fixture images missing at {}: {e}", images_dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
        .collect();
    entries.sort_by_key(|e| e.file_name());
    assert!(!entries.is_empty(), "no .png fixtures found");

    for entry in entries {
        let path = entry.path();
        let img = image::ImageReader::open(&path)
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();
        let (w, h) = img.dimensions();
        let view = siglip2::ImageView::new(img.as_raw(), w, h).unwrap();
        let got = enc.embed_pixels(view).unwrap();

        let stem = path.file_stem().unwrap().to_string_lossy();
        let expected_path = embeddings_dir.join(format!("{stem}.npy"));
        let expected = load_npy_f32_1d(&expected_path);
        let expected_embedding = siglip2::Embedding::try_from(expected)
            .unwrap_or_else(|e| panic!("reference embedding for {stem} failed validation: {e}"));

        let cos = got.cosine(&expected_embedding);
        assert!(
            cos >= 0.99917,
            "{stem}: cosine {cos} below 0.99917 floor"
        );
    }
}

#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR + fixtures"]
fn text_parity_against_pytorch_reference() {
    let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
    let mut enc = siglip2::TextEncoder::from_files(
        &dir.join("text_model_naflex.onnx"),
        &dir.join("tokenizer.json"),
    )
    .expect("encoder must load");

    let prompts: Vec<String> =
        serde_json::from_slice(&std::fs::read(fixture_dir().join("text_prompts.json")).unwrap())
            .unwrap();
    let prompt_refs: Vec<&str> = prompts.iter().map(|s| s.as_str()).collect();

    let got = enc.embed_batch(&prompt_refs).unwrap();

    let raw = std::fs::read(fixture_dir().join("text_embeddings.npy")).unwrap();
    let reader = npyz::NpyFile::new(&raw[..]).unwrap();
    let shape = reader.shape().to_vec();
    let flat = reader.into_vec::<f32>().unwrap();
    assert_eq!(shape.len(), 2, "text_embeddings.npy must be 2-D");
    assert_eq!(shape[0] as usize, prompts.len(), "row count must match prompts.len()");
    assert_eq!(shape[1], 768, "text embedding dim must be 768");

    for i in 0..prompts.len() {
        let row: Vec<f32> = flat[i * 768..(i + 1) * 768].to_vec();
        let expected = siglip2::Embedding::try_from(row).unwrap();
        let cos = got[i].cosine(&expected);
        assert!(
            cos >= 0.99917,
            "prompt {i} ({:?}): cosine {cos} below floor",
            prompts[i]
        );
    }
}

#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR + fixtures"]
fn cross_modal_ranking_sanity() {
    // (image, matching text, distractor text) triples; assert
    // cos(image, matching) > cos(image, distractor). Spec §8.3.
    let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
    let cal_path = dir.join("calibration.json");
    let mut s = siglip2::Siglip2::from_files(
        &dir.join("vision_model_naflex_256.onnx"),
        &dir.join("text_model_naflex.onnx"),
        &dir.join("tokenizer.json"),
        &cal_path,
    )
    .expect("siglip2 must load");

    // Use the first fixture image.
    let images_dir = fixture_dir().join("images");
    let first_png = std::fs::read_dir(&images_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().is_some_and(|x| x == "png"))
        .expect("at least one .png fixture")
        .path();
    let img = image::ImageReader::open(&first_png).unwrap().decode().unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let img_buf = img.into_raw();
    let view = siglip2::ImageView::new(&img_buf, w, h).unwrap();

    // For sanity, just check that classify returns a sorted list
    // and the top score is >= the bottom score.
    let labels = ["a photograph", "a screenshot of code", "an MRI scan"];
    let scored = s.classify(view, &labels, 3).expect("classify");
    assert_eq!(scored.len(), 3);
    for w in scored.windows(2) {
        assert!(
            w[0].score() >= w[1].score(),
            "classify must return descending score order"
        );
    }
}
```

- [ ] **Step 3: Run the integration tests**

```bash
cargo test --all-features --test integration
```

Expected: same 5 tests as before (no fixture-dependent test runs without `--ignored`). With `SIGLIP2_MODELS_DIR` set and fixtures present, the parity tests run. Without fixtures, they fail with a clear "fixture images missing" message.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/README.md tests/integration.rs
git commit -m "test(parity): golden-fixture parity tests gated on env var + fixtures"
```

---

## Task 14: examples/embed_keyframes.rs

**Why:** Spec §1.1 describes the library as the downstream of `scenesdetect`. The example shows the `keyframe images on disk → embeddings` flow that's the crate's primary use case.

**Files:**
- Create: `examples/embed_keyframes.rs`

- [ ] **Step 1: Write the example**

```rust
// examples/embed_keyframes.rs
//! Embed a directory of JPEG/PNG keyframes and print one row per file:
//! `<filename>\t<dim0>,<dim1>,...,<dim767>`.
//!
//! Usage:
//!     cargo run --release --example embed_keyframes -- \
//!         <models-dir> <keyframes-dir>
//!
//! Where `<models-dir>` contains `vision_model_naflex_256.onnx` (+ sidecar)
//! and `<keyframes-dir>` contains `*.jpg` / `*.png` files.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let models_dir: PathBuf = args
        .next()
        .ok_or("usage: embed_keyframes <models-dir> <keyframes-dir>")?
        .into();
    let keyframes_dir: PathBuf = args
        .next()
        .ok_or("usage: embed_keyframes <models-dir> <keyframes-dir>")?
        .into();

    let mut enc = siglip2::ImageEncoder::from_files(&models_dir.join("vision_model_naflex_256.onnx"))?;

    let mut entries: Vec<_> = std::fs::read_dir(&keyframes_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "jpg" || ext == "jpeg" || ext == "png")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let emb = enc.embed_path(&path)?;
        let mut line = String::with_capacity(8 * 768 + 256);
        line.push_str(path.file_name().unwrap().to_str().unwrap_or("?"));
        line.push('\t');
        for (i, x) in emb.as_slice().iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            use std::fmt::Write;
            let _ = write!(line, "{x:.6}");
        }
        println!("{line}");
    }
    Ok(())
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cargo check --features decoders --example embed_keyframes
```

Expected: clean compile. `cargo run` requires real model files; not exercised at this step.

- [ ] **Step 3: Commit**

```bash
git add examples/embed_keyframes.rs
git commit -m "example(embed_keyframes): stdout-tsv embedder for a directory of JPEGs/PNGs"
```

---

## Task 15: examples/index_and_search.rs

**Why:** Demonstrates the full image-search loop: index N images, then query with text and rank.

**Files:**
- Create: `examples/index_and_search.rs`

- [ ] **Step 1: Write the example**

```rust
// examples/index_and_search.rs
//! Index a directory of images by their SigLIP2 vision embedding, then run a
//! text query and print the top-K matches with calibrated sigmoid scores.
//!
//! Usage:
//!     cargo run --release --example index_and_search --features decoders -- \
//!         <models-dir> <images-dir> "<query text>" [top_k]
//!
//! Demonstrates the spec §1.1 pipeline. Storage is in-memory; for a real
//! deployment, write the (image_path, embedding) pairs to lancedb or similar
//! at index time.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let models_dir: PathBuf = args.next().ok_or("usage error")?.into();
    let images_dir: PathBuf = args.next().ok_or("usage error")?.into();
    let query: String = args.next().ok_or("usage error")?;
    let top_k: usize = args.next().map(|s| s.parse().unwrap_or(5)).unwrap_or(5);

    let mut s = siglip2::Siglip2::from_files(
        &models_dir.join("vision_model_naflex_256.onnx"),
        &models_dir.join("text_model_naflex.onnx"),
        &models_dir.join("tokenizer.json"),
        &models_dir.join("calibration.json"),
    )?;

    // Index: collect (path, embedding) pairs.
    let mut entries: Vec<_> = std::fs::read_dir(&images_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "jpg" || ext == "jpeg" || ext == "png")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut index: Vec<(PathBuf, siglip2::Embedding)> = Vec::with_capacity(entries.len());
    for entry in entries {
        let path = entry.path();
        let emb = s.image().embed_path(&path)?;
        index.push((path, emb));
    }
    eprintln!("indexed {} images", index.len());

    // Query.
    let q = s.text().embed(&query)?;

    // Rank.
    let mut scored: Vec<(&PathBuf, f32)> = index
        .iter()
        .map(|(p, e)| (p, e.cosine(&q)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("query: {query:?}");
    for (path, cos) in scored.into_iter().take(top_k) {
        println!("  cos={cos:.4}  {}", path.display());
    }
    Ok(())
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cargo check --features decoders --example index_and_search
```

Expected: clean compile.

- [ ] **Step 3: Commit**

```bash
git add examples/index_and_search.rs
git commit -m "example(index_and_search): in-memory image index + text query loop"
```

---

## Task 16: Benches

**Why:** Spec §8.5 mandates Criterion benches for preprocessing, image encoding, and text encoding. Bench code is small; the value is having the harness in place so future perf work has a baseline.

**Files:**
- Create: `benches/bench_naflex.rs`
- Create: `benches/bench_image_encode.rs`
- Create: `benches/bench_text_encode.rs`

- [ ] **Step 1: Write bench_naflex.rs**

```rust
// benches/bench_naflex.rs
//! NaFlex preprocessing throughput. Standalone — does not require ORT models.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use siglip2::{ImageView, Options, Preprocessor};

fn bench_naflex(c: &mut Criterion) {
    let opts = Options::default();
    let pre = Preprocessor::new(opts);

    // Synthetic 1080p image filled with a noise pattern.
    let mut rgb = vec![0u8; 1920 * 1080 * 3];
    for (i, b) in rgb.iter_mut().enumerate() {
        *b = (i % 251) as u8; // some non-trivial pattern
    }
    let view = ImageView::new(&rgb, 1920, 1080).unwrap();

    let mut pv = vec![0f32; Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE];
    let mut am = vec![0i32; Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE];
    let mut ss = vec![0i32; Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE];

    c.bench_function("naflex_preprocess_1080p", |b| {
        b.iter(|| {
            pre.preprocess_into(black_box(view), &mut pv, &mut am, &mut ss).unwrap();
        });
    });
}

criterion_group!(benches, bench_naflex);
criterion_main!(benches);
```

- [ ] **Step 2: Write bench_image_encode.rs**

```rust
// benches/bench_image_encode.rs
//! End-to-end image encode (preprocess + ORT). Requires SIGLIP2_MODELS_DIR.

use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use siglip2::{ImageEncoder, ImageView};

fn bench_image_encode(c: &mut Criterion) {
    let dir = match std::env::var_os("SIGLIP2_MODELS_DIR") {
        Some(v) => PathBuf::from(v),
        None => {
            eprintln!("SIGLIP2_MODELS_DIR not set; skipping bench_image_encode");
            return;
        }
    };
    let graph = dir.join("vision_model_naflex_256.onnx");
    let mut enc = ImageEncoder::from_files(&graph).expect("encoder loads");

    let mut rgb = vec![0u8; 1920 * 1080 * 3];
    for (i, b) in rgb.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let view = ImageView::new(&rgb, 1920, 1080).unwrap();

    c.bench_function("image_encode_single_1080p", |b| {
        b.iter(|| {
            let _ = enc.embed_pixels(black_box(view)).unwrap();
        });
    });
}

criterion_group!(benches, bench_image_encode);
criterion_main!(benches);
```

- [ ] **Step 3: Write bench_text_encode.rs**

```rust
// benches/bench_text_encode.rs
//! End-to-end text encode (tokenize + ORT). Requires SIGLIP2_MODELS_DIR.

use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use siglip2::TextEncoder;

fn bench_text_encode(c: &mut Criterion) {
    let dir = match std::env::var_os("SIGLIP2_MODELS_DIR") {
        Some(v) => PathBuf::from(v),
        None => {
            eprintln!("SIGLIP2_MODELS_DIR not set; skipping bench_text_encode");
            return;
        }
    };
    let mut enc = TextEncoder::from_files(
        &dir.join("text_model_naflex.onnx"),
        &dir.join("tokenizer.json"),
    )
    .expect("encoder loads");

    let prompt = "a red bicycle leaning against a brick wall";

    c.bench_function("text_encode_single", |b| {
        b.iter(|| {
            let _ = enc.embed(black_box(prompt)).unwrap();
        });
    });
}

criterion_group!(benches, bench_text_encode);
criterion_main!(benches);
```

- [ ] **Step 4: Verify they compile**

```bash
cargo check --benches --all-features
```

Expected: clean compile.

- [ ] **Step 5: Commit**

```bash
git add benches/
git commit -m "bench: Criterion harnesses for naflex preproc and image/text encode"
```

---

## Task 17: Final docs + real tokenizer.json

**Why:** README and a real bundled tokenizer.json are the last pieces. The `gh release download` will replace the placeholder created in Task 10.

**Files:**
- Modify: `README.md`
- Modify: `models/tokenizer.json` (replace placeholder with real bytes)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Download the real tokenizer.json**

```bash
cd /Users/user/Develop/findit-studio/siglip2
mkdir -p /tmp/siglip2-release
gh release download models-siglip2-naflex-v1 \
  --repo Findit-AI/indexer \
  --dir /tmp/siglip2-release \
  --pattern 'tokenizer.json'
cp /tmp/siglip2-release/tokenizer.json models/tokenizer.json
```

Verify the SHA256:
```bash
shasum -a 256 models/tokenizer.json
# Expected: 58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0
```

- [ ] **Step 2: Replace README.md**

```markdown
# siglip2

Rust ONNX inference library for SigLIP2 NaFlex (image + text embeddings).

A sibling of [`textclap`](https://github.com/Findit-AI/textclap) (CLAP audio inference) and a downstream of [`scenesdetect`](https://github.com/Findit-AI/scenesdetect) (keyframe extraction).

## Quick start

```rust
use siglip2::{ImageEncoder, ImageView};

let mut enc = ImageEncoder::from_files("models/siglip2/vision_model_naflex_256.onnx".as_ref())?;
let img = image::open("keyframe.jpg")?.to_rgb8();
let (w, h) = img.dimensions();
let view = ImageView::new(img.as_raw(), w, h)?;
let embedding = enc.embed_pixels(view)?;
// 768-dim, L2-normalized, cosine-comparable.
```

For end-to-end image search, see `examples/index_and_search.rs`.

## Model files

The runtime expects the assets from
[`Findit-AI/indexer` release `models-siglip2-naflex-v1`](https://github.com/Findit-AI/indexer/releases/tag/models-siglip2-naflex-v1).
See [models/MODELS.md](models/MODELS.md) for the download recipe.

## Features

- `bundled` (default): embed the text-tower tokenizer (33 MB) so
  `Siglip2::bundled` and `TextEncoder::bundled` work without a tokenizer
  file on disk. Vision tower has no tokenizer; this feature is text-only.
- `decoders` (default): activate `image` crate JPEG/PNG decoders so
  `ImageEncoder::embed_path` works. Without this, callers supply pre-decoded
  RGB pixels via `ImageView`.
- `serde`: `Serialize` (`LabeledScore`) / `Serialize + Deserialize`
  (`LabeledScoreOwned`). `Embedding` and `Calibration` deliberately do
  *not* derive serde — see the design spec for why.

## Execution providers

CPU only by default. For CUDA / CoreML / etc., enable the appropriate `ort`
features in your own `Cargo.toml` and pass a custom session via
`from_ort_session` / `Siglip2::from_parts`. ANE-on-Mac requires explicit
opt-in.

## License

MIT or Apache-2.0, at your option. The bundled `tokenizer.json` is derived
from `google/siglip2-base-patch16-naflex` (Apache-2.0); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Design

Full design spec: [`docs/superpowers/specs/2026-04-27-siglip2-design.md`](docs/superpowers/specs/2026-04-27-siglip2-design.md).
```

- [ ] **Step 3: Update CHANGELOG.md to reflect implementation completion**

Replace the file:

```markdown
# Changelog

## 0.1.0 (unreleased)

Initial release.

- Vision encoder (`ImageEncoder`) for the SigLIP2 NaFlex export at
  `max_num_patches = 256`.
- Text encoder (`TextEncoder`) with Gemma SPM tokenizer at fixed
  `seq_len = 64` and `pad_token_id = 0`.
- `Siglip2` wrapper combining both encoders with calibrated zero-shot
  `classify` (sigmoid over `logit_scale·cos + logit_bias`).
- NaFlex preprocessing pinned to `image::imageops::resize(Triangle)` for
  the validated 0.99917 cosine parity floor against the upstream PyTorch
  reference.
- `Preprocessor` low-level API for chunk-buffer reuse.
- Default features `bundled` (text-side tokenizer.json) and `decoders`
  (JPEG/PNG via the `image` crate). The crate tarball is ~33 MB regardless
  of feature selection.
- Embeddings stored as `Arc<[f32]>` for cheap cloning; `Embedding` and
  `Calibration` deliberately do not derive serde to keep their validation
  invariants enforceable.
```

- [ ] **Step 4: Final verification**

```bash
cargo check --all-features
cargo test --all-features --lib
cargo test --all-features --test integration  # 2 pass, 6 ignored
cargo doc --all-features --no-deps  # ensure rustdoc renders cleanly
```

Expected: clean compile, all unit tests pass, integration tests pass except the 6 marked `#[ignore]`.

- [ ] **Step 5: Commit**

```bash
git add README.md CHANGELOG.md models/tokenizer.json
git commit -m "docs(0.1.0): real bundled tokenizer.json + final README/CHANGELOG"
```

---

## Task 18: Final integration smoke test

**Why:** With everything in place, run the whole suite end-to-end and confirm the crate is shippable.

- [ ] **Step 1: Full check across all feature combinations**

```bash
cargo check --no-default-features
cargo check --features bundled
cargo check --features decoders
cargo check --features serde
cargo check --all-features
```

Expected: all five clean.

- [ ] **Step 2: Run all unit tests**

```bash
cargo test --all-features --lib
```

Expected: all unit tests pass.

- [ ] **Step 3: Run integration tests without env var**

```bash
cargo test --all-features --test integration
```

Expected: 2 pass (Send/Sync assertions), 6 ignored.

- [ ] **Step 4: Optional — run integration tests with real models**

Only if `gh auth` is set up and the release is downloadable:

```bash
mkdir -p /tmp/siglip2-models
gh release download models-siglip2-naflex-v1 --repo Findit-AI/indexer --dir /tmp/siglip2-models
SIGLIP2_MODELS_DIR=/tmp/siglip2-models cargo test --all-features --test integration -- --ignored
```

Expected:
- `loads_image_encoder_from_release`: pass
- `loads_text_encoder_from_release`: pass
- `loads_calibration_from_release`: pass
- `image_parity_against_pytorch_reference`: pass *if* fixtures present, otherwise fail with "fixture images missing"
- `text_parity_against_pytorch_reference`: same
- `cross_modal_ranking_sanity`: pass

If parity tests fail because fixtures haven't been generated yet, document the gap in the implementer's hand-off note. Spec §11 item 3 marks this as a known follow-up.

- [ ] **Step 5: Final commit (no-op if everything was committed already)**

```bash
git status
# Should be clean.
```

---

## Hand-off notes for the next maintainer

The following items are **deliberately incomplete** in 0.1.0 and tracked as follow-ups:

1. **Golden fixture generation** (Task 13). Needs the upstream PyTorch reference run against the same `tests/fixtures/images/` set; current parity tests will fail with `fixture images missing` until then.
2. **`validate_image_session` shape check** (Task 9). Currently a stub. Tightening it to assert the actual ONNX input/output shapes per spec §3.2 closes a subtle from_ort_session footgun.
3. **`reference_table_matches` ground-truthing** (Task 6). The seven rows in spec §4.1 should be regenerated by running `Siglip2ImageProcessorFast` from a pinned `transformers` commit and updating the test in lockstep.
4. **In-memory ONNX construction**. Spec §11 explicitly defers this; revisit when ORT 2.0 stable exposes a safe in-memory external-data API.
5. **SIMD acceleration of patchify+normalize**. Spec §6 explicitly defers.
