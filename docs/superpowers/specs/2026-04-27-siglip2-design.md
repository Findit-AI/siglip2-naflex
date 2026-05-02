# siglip2 — SigLIP2 NaFlex Inference Library Design

**Status:** Approved (revision 17, post-Codex-adversarial-review-round-9)
**Date:** 2026-04-28
**Target version:** 0.1.0

> **Revision 17 changelog (vs revision 16).** Codex round-9 caught two
> issues: one a real arithmetic-overflow bug, one the still-unresolved
> fixture-generation hand-off.
>
> - **`validate_preprocessed_lengths` did unchecked `batch_size *
>   STRIDE`.** With `batch_size = 1 << 63`, every stride product wraps
>   to 0 in 64-bit usize (verified: `(1<<63) * 196_608 ≡ 0 mod 2⁶⁴`,
>   same for the other two strides). Empty slices then satisfied the
>   length check, and `run_image_session` would build an ORT tensor
>   with a nonsensical huge batch. Fixed: (a) `embed_preprocessed` now
>   enforces `batch_size <= max_batch_size` (mirroring the cap that
>   `embed_pixels_batch` already had — without this the lower-level
>   path was a back door); (b) `validate_preprocessed_lengths` uses
>   `checked_mul` and surfaces `Error::DimensionsOverflow` on overflow,
>   defense-in-depth even with the cap. Regression test
>   `validate_preprocessed_lengths_rejects_overflow` calls the helper
>   directly at `1 << 63` to exercise the arithmetic check.
>
> - **Parity fixtures still aren't in-tree** (carryover from
>   round-5/-6/-9). Codex's recommendation to "generate and commit
>   golden fixtures" requires a one-shot upstream PyTorch run that
>   isn't possible inside the crate's own CI. What is in this revision:
>   a committed `scripts/generate_parity_fixtures.py` — a single-file,
>   CPU-only Python that downloads `google/siglip2-base-patch16-naflex`,
>   encodes a caller-supplied images-dir and prompts-json, and writes
>   the `.npy`/`.json` fixtures into `tests/fixtures/`. The recipe is
>   now concrete (one command) rather than the prior README pseudocode;
>   the CI parity workflow continues to fail closed until the fixtures
>   land. `tests/fixtures/README.md` updated to point at the script.

> **Revision 16 changelog (vs revision 15).** Codex round-8 caught two
> validators that were too lenient — accepting inputs that would silently
> degrade or fail at inference rather than failing closed at construction.
>
> - **`Calibration::validate` accepted f32 underflow to zero.** Rev 15's
>   check was `effective.is_finite()` only. But `(-200_f32).exp() = 0.0`
>   in f32 — finite, so it passed. `Siglip2::classify` then computed
>   `0 * cos + bias = bias` for every label, silently collapsing
>   ranking to "input order" without erroring. Codex independently
>   verified the underflow boundary (`exp(-100)` ≈ `3.78e-44` survives;
>   `exp(-200)` collapses to exactly `0.0`). Fixed: validate the
>   **effective scale** is both finite AND strictly positive. New
>   regression test `validate_rejects_underflowing_scale` (with
>   `logit_scale = -200`); the existing
>   `validate_accepts_negative_scale` and a new
>   `validate_accepts_smallest_safe_negative_scale` (`logit_scale = -50`,
>   above the underflow boundary) keep round-7's "negative raw scale is
>   valid" decision intact for non-pathological values.
>
> - **`check_outlet` accepted concrete static batch dims where the
>   contract expects dynamic.** Rev 12 wrote: "wildcard on either side
>   accepts: -1 in expected = any". That's wrong for the batch axis —
>   our public `embed_*_batch` API sends arbitrary per-call chunk sizes,
>   so a graph exported with a static batch dim (e.g. `[1, 256, 768]`)
>   would pass construction-time validation and then fail at first
>   inference call with a chunk size != 1. Fixed: when expected
>   dim is `-1` (the contract says "this axis must be dynamic"), the
>   graph's actual dim must also be `-1` — concrete static dims are
>   rejected as `SessionShapeMismatch`. Concrete-expected dims (256,
>   768, 64, 2) still accept either an exact match OR a `-1` dynamic
>   axis. Two new tests:
>   `check_outlet_rejects_static_batch_dim` (constructs an `Outlet`
>   with `[1, 256, 768]`, asserts rejection) and
>   `check_outlet_accepts_dynamic_batch_dim` (sanity check that the
>   `[-1, 256, 768]` shape still passes).

> **Revision 15 changelog (vs revision 14).** Codex round-7 caught three
> contract-vs-implementation mismatches:
>
> - **Pooler outputs were rejected unless already pre-normalized.** The
>   release body explicitly says `L2-normalize at consumer side` for both
>   vision and text `pooler_output` (§1's I/O block). But
>   `Embedding::from_model_output` rejected any vector whose norm wasn't
>   within `5e-4` of 1.0. Real model outputs are arbitrary-norm, so every
>   first inference would have failed with `NotNormalized`. Fixed: split
>   the two construction paths cleanly.
>     - `from_model_output` (encoder path) now **always** normalizes
>       finite, non-zero, dim-correct vectors. Rejects only on dim
>       mismatch, all-zero (degenerate model state), or non-finite
>       components.
>     - `TryFrom<Vec<f32>>` (caller path) keeps the strict
>       near-unit-norm check — that path is for caller-supplied
>       embeddings (e.g. deserialized from a vector store) which should
>       already be unit-norm; silent renorm there would mask data
>       corruption.
>   New regression tests:
>   `from_model_output_normalizes_arbitrary_norm` (vector of all 1s
>   normalizes correctly), `from_model_output_rejects_zero_norm`,
>   `from_model_output_rejects_nan_component`,
>   `try_from_still_rejects_far_from_unit_norm`.
>
> - **Session validators allowed extra required inputs.** The releases
>   ship one-input text (`input_ids` only — bit-exact parity verified at
>   cosine 1.00000 against PyTorch) and three-input vision
>   (`pixel_values`, `pixel_attention_mask`, `spatial_shapes`). Our
>   validators checked each named input but didn't enforce input/output
>   counts, so a future re-export adding (e.g.) an `attention_mask` to
>   the text graph would surface as a confusing ORT runtime error
>   instead of a clean construction-time `SessionShapeMismatch`. Fixed:
>   both `validate_image_session` and `validate_text_session` now also
>   assert exact input/output counts (3/1 and 1/1 respectively).
>
> - **Calibration validator rejected mathematically valid raw scales.**
>   `Calibration::validate` rejected `logit_scale <= 0`, but the raw
>   scale is exponentiated at inference time — `exp(0) = 1`,
>   `exp(-1) ≈ 0.368` are perfectly valid effective scales producing
>   weak-but-well-defined sigmoid probabilities. The check confused raw
>   and effective scale. Fixed: dropped the `<= 0` rejection; only
>   `is_finite` and the `exp().is_finite()` overflow check remain.
>   Renamed tests `validate_rejects_zero_scale`,
>   `validate_rejects_negative_scale` to `validate_accepts_*` to match
>   the corrected semantics.

> **Revision 14 changelog (vs revision 13).** Codex round-6 caught two
> issues — both `[medium]`/`[high]`-severity:
>
> - **`(width as usize) * (height as usize) * 3` could silently
>   wrap to a small value on 64-bit release builds for legal `u32`
>   inputs near `u32::MAX`.** Independently verified:
>   `2_479_008_847 × 2_480_392_395 × 3 ≡ 4_079 (mod 2^64)`. A 4079-byte
>   rgb slice would then satisfy the length check and proceed with
>   completely wrong dimensions, leading to `ImageBuffer::from_raw`
>   panics or silent corruption. Fixed in both `ImageView::new` and
>   `naflex::preprocess_into` via `checked_mul` chained on the
>   width/height/3 multiplications, returning new
>   `Error::DimensionsOverflow { width, height }` variant on overflow.
>   Regression test `image_view_rejects_dimension_overflow` pins the
>   wrap-to-small case.
>
> - **Rev-13's parity workflow was not actually a parity gate.** The
>   workflow downloaded only runtime model artifacts and ran ALL
>   ignored tests, including the parity-against-PyTorch tests that
>   need fixtures we don't have in-tree. Result: a configured CI
>   would fail without showing why, an unconfigured CI would silently
>   skip. Codex correctly flagged this as misleading. Replaced with a
>   two-stage workflow:
>     1. `model-load-smoke` — runs only the load tests (image_encoder,
>        text_encoder, calibration). Verifies session shape contract,
>        does NOT prove parity.
>     2. `parity-against-pytorch` — runs the cosine-floor parity tests.
>        Checks for fixtures up-front; fails closed with a clear
>        `::error::` message when fixtures are missing. So a configured-
>        but-fixtureless CI is RED, not falsely green.
>   README and CHANGELOG carry a clearly labeled stage table; warning
>   text now says "Do not treat a green main-branch CI as parity-
>   verified unless the parity-against-pytorch job is actually running
>   and passing."

> **Revision 13 changelog (vs revision 12).** Codex round-5 found two
> issues — one a real algorithmic invariant violation, one a process
> gap — both `[high]`-severity:
>
> - **`patch_grid` could exceed the 256-patch budget on extreme `u32`
>   inputs.** The binary search assumes `scale_min = SCALE_EPS / 10 = 1e-6`
>   is feasible, but for `width > ~4·10⁹` at `height = 1` (legal `u32`
>   territory, even if unrealistic) the entire feasible range is below
>   that floor — `target_w` would already exceed the per-row budget at
>   `s = 1e-6`. The loop then never finds a feasible scale, `scale_min`
>   stays at the infeasible initial value, and the function returns a
>   grid that overflows the fixed `[256, 768]` `pixel_values` buffer
>   in `preprocess_into`. Independently verified with Python:
>   `patch_grid(1, 4_096_000_001, 256) → (1, 257)`. Fixed in
>   `preprocess_into` via a postcondition check that returns
>   `Error::ImageTooLarge { width, height, grid_patches, max_num_patches }`
>   before the patchify loop ever indexes the buffer. Two regression
>   tests: `patch_grid_overshoots_at_extreme_dims` documents the
>   underlying overshoot at `(1, 4_096_000_001)`;
>   `rejects_image_too_large_for_budget` exercises the postcondition
>   path synthetically (using `max_num_patches = 0` to avoid allocating
>   a 12 GB rgb buffer).
>
> - **Parity-against-PyTorch tests are `#[ignore]`-d, so the active
>   suite can ship while embedding parity is silently broken.** This
>   was a known hand-off gap, but Codex correctly noted it's not a
>   release gate. Addressed via:
>   1. New `.github/workflows/parity.yml` that downloads the runtime
>      model artifacts (gh release download) and runs the ignored
>      tests when the `FINDIT_INDEXER_TOKEN` repo secret is configured.
>      Without the secret the gate short-circuits, so forks and
>      unconfigured environments still pass green.
>   2. README and CHANGELOG carry an explicit warning that "green CI ≠
>      parity-verified unless the secret is configured."
>   3. Spec §11 hand-off note for the in-tree fixture generation
>      remains — that's still a separate one-shot upstream-PyTorch
>      task. The CI workflow exists to run the tests, not generate the
>      fixtures.

> **Revision 12 changelog (vs revision 11).** Codex round-4 caught one
> `[high]`-severity preprocessing-parity bug:
>
> - **Text inputs were tokenized without SigLIP2's required Lowercase
>   normalization.** Verified by reading both the bundled
>   `models/tokenizer.json` (whose `normalizer` is just
>   `Replace(" ", "▁")` — the SentencePiece-marker step) and the upstream
>   `transformers/models/siglip2/tokenization_siglip2.py:95-96` —
>   `Siglip2Tokenizer.__init__` does
>   `backend.normalizer = Sequence([Lowercase(), backend.normalizer])`
>   at runtime. That wrap is NOT serialized into the exported JSON, so a
>   Rust caller loading `tokenizer.json` directly via
>   `Tokenizer::from_file` / `Tokenizer::from_bytes` (without going
>   through Python's `Siglip2Tokenizer`) would silently encode
>   `"HELLO WORLD"` to a different token sequence than upstream's
>   `"hello world"` — different IDs → different embeddings → degraded
>   retrieval and wrong `classify` rankings on any mixed-case query or
>   label. Fixed in `configure_padding`: it now prepends `Lowercase` to
>   whatever normalizer the loaded JSON carries, exactly mirroring
>   upstream's runtime wrap. Regression test
>   `configure_padding_applies_lowercase` (gated on `bundled`) asserts
>   `"HELLO WORLD"` and `"hello world"` encode to identical IDs after
>   `configure_padding`. Without the fix the test fails because the
>   SentencePiece BPE only has lowercase merges, so uppercase routes
>   through `<unk>`.

> **Revision 11 changelog (vs revision 10).** Codex's third pass found
> two more boundary defects:
>
> - **`Calibration::validate` accepted raw scales that overflow `f32 exp()`.**
>   The pinned release value (`logit_scale = 4.7476`) is well within range,
>   but a corrupted file with `logit_scale ≈ 100` would pass validation
>   (finite, positive) and then `exp(100)` saturates to `f32::INFINITY`.
>   Downstream, `inf * cos = inf` saturates sigmoid scores to 1, but
>   `inf * 0.0 = NaN` for orthogonal cosines, breaking the `[0, 1]` score
>   contract and the `partial_cmp` sort order in `classify`. Fixed:
>   `validate` now also rejects scales whose `.exp()` is non-finite, with
>   a documented threshold of ~88.7 (the f32 overflow boundary). New
>   `validate_rejects_overflowing_scale` test pins a `100.0` rejection;
>   `validate_accepts_largest_safe_scale` confirms `80.0` (just under the
>   boundary) still passes.
> - **`configure_padding` only configured padding, not truncation.**
>   `PaddingStrategy::Fixed(64)` pads short inputs to 64 but does not
>   truncate long ones. The bundled `tokenizer.json` carries its own
>   truncation config so the bundled path silently worked, but a caller
>   passing a custom tokenizer to `from_ort_session` (the advertised
>   custom-EP path) would see over-64-token queries fail with
>   `Error::Batch { source: Tokenizer("…produced N ids; expected 64") }`
>   instead of being truncated to the static `[batch, 64]` `input_ids`
>   axis. Fixed: `configure_padding` now also calls `with_truncation`
>   with `max_length = SEQ_LEN`, `direction = Right`, `strategy =
>   LongestFirst`, `stride = 0` — infallible at stride 0 so the
>   `.expect()` is justified by the tokenizers crate's documented
>   precondition.

> **Revision 10 changelog (vs revision 9).** Codex's second adversarial
> pass found one `[high]` and two `[medium]` defects, all in the batch /
> session boundaries. All three adopted.
>
> - **Image-batch scratch was sized by `BatchOptions::batch_size`, not by
>   the actual `views.len()`.** With a valid `batch_size = 1024` setting,
>   even a single-image call would allocate ~770 MB of `pixel_values`
>   scratch — a configuration knob that could OOM on non-pathological
>   input. Fixed: scratch now allocated as `min(chunk, views.len())`,
>   plus a new `Preprocessor::new` validation that rejects
>   `batch_size = 0` or `batch_size > max_batch_size` at construction
>   (new `Error::InvalidBatchSize` variant). Two regression tests pin
>   the validation.
> - **`TextEncoder::embed_batch` ignored the configured micro-batch
>   size**, sending every text to one ORT run. Spec §3.4's "chunks by
>   `BatchOptions::batch_size`" was image-side only. Fixed to mirror the
>   image path: chunk by `batch_size`, run one ORT inference per chunk,
>   collect results in input order. Per-chunk failures use
>   `Error::Batch { index, source }`.
> - **`validate_image_session` / `validate_text_session` were no-op
>   stubs** — constructors accepted any ONNX graph as healthy, deferring
>   shape mismatches to first inference. Replaced both with real metadata
>   checks that inspect `session.inputs()` / `session.outputs()`,
>   verifying:
>   - input/output names exist;
>   - element type matches (`f32` for `pixel_values` / `pooler_output`,
>     `i32` for `pixel_attention_mask` / `spatial_shapes`, `i64` for
>     `input_ids`);
>   - rank matches;
>   - static dims match (`256` patches, `768` channels, `64` text seq_len,
>     `768` output dim).
>   The validation is shared via a `pub(crate) check_outlet` helper in
>   `image_enc.rs` so the two encoders' contracts stay in sync.
>
> All three fixes commit together as rev-10. The "block release until
> batch sizing is bounded and contracts are validated at construction"
> blocker from Codex's review is closed.

> **Revision 9 changelog (vs revision 8).** Codex's adversarial review of
> the implemented branch caught two `[high]`-severity bugs that all prior
> review rounds — and the implementation itself — missed because the
> sanity tests they would have failed used cos = 0, where both wrong and
> right formulas agree.
>
> - **Calibration sigmoid was applied in the wrong domain.** Rev 8 wrote
>   the score as `sigmoid(logit_scale · cos + logit_bias)`. The release
>   `calibration.json` stores the **raw** learned scale (matching
>   HuggingFace `Siglip2Model.logit_scale`), which the model
>   exponentiates at inference time. Without `.exp()`, every score caps
>   at `~6e-6` even for a perfect cosine match — calibrated classification
>   was effectively unusable. Fixed: `Siglip2::classify` now applies
>   `logit_scale().exp()`. The `Calibration::logit_scale()` rustdoc and
>   §5.3 prose call this out explicitly. A new sanity test at cos = 0.18
>   (typical confident match) asserts the sigmoid lands at ~0.9816, which
>   would fail loudly if the `.exp()` is dropped again.
> - **NaFlex `patch_grid` diverged from upstream at edge cases.** Rev 8's
>   "64-iteration binary search converging to ~2^-64 precision" was too
>   tight: at the boundary where `ceil()` flips, f64 noise tipped the
>   answer the wrong way for ~thousands of (h, w) pairs. Codex's
>   regression — `(3, 39)` should give `(4, 52)` per upstream, ours gave
>   `(4, 53)` — is one of 168 mismatches in the 200×200 grid alone.
>   Fixed: `patch_grid` is now a direct port of upstream's
>   `get_image_size_for_max_num_patches` with `eps = 1e-5` termination
>   and `ceil(s·X/P) · P` pixel-size snapping. The defensive post-clamp
>   loop is dropped because the loop invariant (`scale_min` always
>   feasible) trivially guarantees the budget. Reference table now
>   includes the `(3, 39) → (4, 52)` regression case.

> **Revision 8 changelog (vs revision 7).** Round-5 review found one P3
> validation gap and two cosmetic items; all adopted.
>
> - **Validation gap.** `Calibration::new` is unchecked (rev 3 design call,
>   for tests/hard-coded values), but `Siglip2::from_parts` consumed a
>   `Calibration` directly without re-validating — same shape as the
>   round-4 serde bypass. Closed by having `from_parts` call
>   `Calibration::validate` on the supplied calibration before building
>   the wrapper. `validate` is promoted from private to `pub(crate)` so
>   it's reachable from `siglip2.rs` (still not part of the public
>   surface) (§3.2, §5.3).
> - **O-7**: `ImageView` gains `Copy`. All fields (`&[u8]`, `u32`, `u32`)
>   are `Copy`; the validating `ImageView::new` constructor is the only
>   way to construct one, so `Copy` doesn't reopen the length-bypass
>   risk. The §3.7 example drops `view.clone()` (§3.3, §3.7).
> - **O-8**: Concrete round-trip example added to `Embedding`'s rustdoc
>   showing the `as_slice() → JSON → Vec<f32> → TryFrom` path (§3.5).
> - **O-9**: `Calibration::new` rustdoc cross-links to `from_path` /
>   `from_bytes` for production paths and notes that `from_parts`
>   re-validates (§5.3).

> **Revision 7 changelog (vs revision 6).** Round-4 review surfaced one
> real bug — auto-derived `Deserialize` on `Embedding` and `Calibration`
> bypassed the validation invariants those types' explicit constructors
> were designed to enforce. Project owner picked the cleanest fix: drop
> the serde derives on those two types entirely. Callers who need to
> serialize raw embeddings can do so via `Embedding::as_slice()` /
> `Embedding::into_vec()` (the inner `Vec<f32>` / `&[f32]` already
> implements `Serialize`).
>
> **Adopted:**
> - `Embedding` no longer has any `cfg_attr(feature = "serde", ...)`
>   derives. The L2-norm and dim invariants are now enforceable in full:
>   the only construction paths from outside the crate are
>   `TryFrom<Vec<f32>>` (validated) and the encoder methods themselves
>   (validated) (§3.5).
> - `Calibration` no longer has any `cfg_attr` serde derives. Internal
>   JSON parsing uses a private `CalibrationRaw` struct (always-on
>   `serde::Deserialize`); the public `Calibration` is constructible
>   only via `new`, `from_path`, or `from_bytes` — all three pass
>   through the same `validate` private method (§5.3).
> - `serde/rc` feature dropped from the hard `serde` dep — `Arc<[f32]>:
>   Deserialize` is no longer needed (§9).
> - `LabeledScore` / `LabeledScoreOwned` keep their `cfg_attr` serde
>   derives. Those types have no invariants to violate; `score: f32`
>   in `[0, 1]` is documented but not enforced, so direct deserialization
>   is safe (§3.5).
> - **O-1**: `embed_pixels_batch` failure semantics pinned: aborts on
>   the first failing input and returns `Error::Batch { index, source }`
>   carrying the offending index. Callers who want partial-success
>   semantics chunk caller-side. Documented (§3.3).
> - **O-2**: `Embedding::cosine` panic message gains explicit context
>   ("Embedding::cosine: dim mismatch (variants must match)"). (§3.5)
> - **O-3**: `LabeledScoreOwned::new` rustdoc states the `score` argument
>   is unchecked — provided primarily for tests/mocks (§3.5).
> - **O-4**: `ImageEncoder` / `TextEncoder` documented as `Send + !Sync`
>   (each owns an `ort::Session`, which is `!Sync`). Negative trait-
>   bound assertions are deferred (no clean stable-Rust idiom); the
>   constraint is rustdoc-only (§3.3, §3.4, §7.3).
> - **O-5**: One-line note that `tests/fixtures/MODELS.sha256` ships in
>   the GitHub repo only (it is in the `tests/` tree, not the crates.io
>   tarball). Bundled tokenizer bytes ship in the tarball directly so
>   tarball users trust them by reference (§8.1).
>
> **Declined:**
> - Reviewer's `O-6` (drop `Options::new()` since `default()` is
>   canonical): no action — both kept; reviewer themself rated this
>   "no action needed."

> **Revision 6 changelog (vs revision 5).** Round-3 review found zero
> blockers; this revision lands the six recommended P2 refinements plus
> the relevant P3 polish.
>
> **Adopted P2:**
> (P2-A) `Embedding::cosine` now `assert_eq!`s on dim mismatch with a
>     clear panic message — picked over `Result<f32, _>` to keep the
>     ergonomic `a.cosine(&b)` call site, since dim mismatch is a
>     programming error, not a runtime condition (§3.5).
> (P2-B) `Calibration::from_path` / `from_bytes` validate that
>     `logit_scale.is_finite() && logit_scale > 0.0 && logit_bias.is_finite()`;
>     a corrupted file produces `Error::InvalidCalibration { reason }`
>     rather than silently propagating into `classify` scores (§5.3, §3.5).
> (P2-C) `Siglip2::split(&mut self) -> (&mut ImageEncoder, &mut TextEncoder)`
>     added so `classify` (and any caller wanting to embed an image and
>     a label set in one go) can borrow both halves simultaneously (§3.2).
> (P2-D) `Preprocessor: Send + Sync` is now stated explicitly in the
>     rustdoc; tests carry a compile-time assertion (§3.7, §8.4).
> (P2-E) `examples/**/*.rs` added to the crate `include` list so
>     `cargo run --example embed_keyframes` works against the published
>     crate, matching silero's behavior (§9).
> (P2-F) `LabeledScoreOwned::new(label, score)` public constructor for
>     tests and mocks (§3.5).
>
> **Adopted P3:**
> - `Embedding::into_vec(self) -> Vec<f32>` convenience to avoid the
>   `try_unwrap → into_vec → unwrap_or_else(|a| a.to_vec())` dance (§3.5).
> - `ImageView::new` now returns `Result<Self>`, validating
>   `rgb.len() == width * height * 3` upfront. `const fn` is dropped
>   (validation precludes it). Caller-side cost is one `?` (§3.3).
> - `Options::default()` documented as canonical; `Options::new()` retained
>   as a stylistic alternative that calls through to `default()` (§3.6).
> - `ImageEncoder::from_part` / `TextEncoder::from_part` renamed to
>   `from_ort_session` for symmetry with sibling crates (silero, textclap).
>   `Siglip2::from_parts` (plural, four arguments) keeps its name (§3.2–§3.4).
> - `tests/fixtures/MODELS.sha256` lists the bundled `tokenizer.json`'s
>   hash too — single source of truth for runtime-asset checksums (§8.1).
> - `smol_str?/serde` → `smol_str/serde` (drop the `?`; `smol_str` is a
>   non-optional dep) (§9).
> - `top_k > labels.len()` documented to clamp rather than error (§3.2).
> - `embed_pixels_batch` internal chunking documented:
>   chunks by `BatchOptions::batch_size`, one ORT inference per chunk,
>   returned `Vec` preserves input order (§3.3).
> - `batch_size_max = 1024` arithmetic footnote: peak `pixel_values`
>   buffer ~770 MB, plus ~1.55 GB resident weights, totals ~2.3 GB
>   per worker at max batch (§3.6, §7.3).
>
> **Deferred to 0.2.0** (per reviewer's recommendation): splitting
> `bundled` into `bundled-text` to make room for hypothetical future
> image-side bundled assets — speculative until such an asset exists.

> **Revision 5 changelog (vs revision 4).** `Embedding`'s inner storage
> changed from `Box<[f32]>` to `Arc<[f32]>` at project-owner request.
> `Embedding::clone()` is now an atomic-refcount bump instead of a 3 KB
> heap memcpy, which matters because embeddings routinely move between
> batch buffers, search rankers, and vector-store writes. Same 16 B fat
> pointer on the stack; `Send + Sync` preserved; the post-construction
> immutability invariant means losing free `&mut [f32]` access via `Arc`
> isn't a constraint. The hard `serde` dep gains the `rc` feature so
> `Arc<[f32]>: Deserialize` is available (§3.5, §9, §10).

> **Revision 4 changelog (vs revision 3).** API ergonomics pass requested
> by the project owner:
> (A) Feature `serde-public` renamed to `serde` (the conventional name).
> (B) Feature `bundled-tokenizer` renamed to `bundled`. The constant is
>     still `BUNDLED_TOKENIZER` because the name identifies the bytes,
>     not just the feature gate; rustdoc on the constant continues to
>     state it's text-side only.
> (C) All public fields removed. `ImageView`, `LabeledScore`,
>     `LabeledScoreOwned`, `Options`, `BatchOptions`, `ThreadOptions`,
>     and `Calibration` now expose getters; the three `*Options` config
>     types also expose `with_*` builder methods (consume `self`) and
>     `set_*` in-place mutators (`&mut self`). `Calibration` is a value
>     object and gets neither — only `new`, `logit_scale()`, `logit_bias()`,
>     `from_path`, `from_bytes`. Error-enum struct variants keep their
>     named fields (those are accessed by pattern-matching, not getters).
> (D) Resize-library question resolved (project owner asked: "do we need
>     the image crate to resize?"). Clarification: the model accepts
>     arbitrary aspect ratios but enforces
>     `max_num_patches × patch_size² = 65,536 px` of total area, so a
>     downscale is mandatory on every real-world keyframe. Project owner
>     confirmed option A: keep `image` for resize but split the dep
>     structure cleanly — `image` is now a **non-optional** core dep
>     (`default-features = false`, no decoders), and the previous `image`
>     feature flag is renamed to **`decoders`** (gates the JPEG/PNG
>     decoders that `embed_path` needs). This preserves the validated
>     0.99917 cosine floor while making the dep boundary honest (§9, §3.3).

> **Revision 3 changelog (vs revision 2).** Round-2 review found four
> blockers introduced by the rev-2 rewrite plus several P2 polish items.
> All adopted except the suggestion to bundle `calibration.json`, which is
> declined to keep export-version coupling explicit. Material changes:
> (a) `serde` feature now actually gates `Serialize`/`Deserialize`
>     derives via `cfg_attr`, instead of merely toggling a transitive
>     `smol_str` feature with no observable effect (§9, §3.5);
>     `LabeledScoreOwned` is now explicitly defined (§3.5).
> (b) Default `GraphOptimizationLevel` lowered to `Level1` to match the
>     existing service's validated parity floor; CI runs the §8.2 fixtures
>     at both Level1 and Level3 (§3.6, §8.2).
> (c) `Error::LoadCalibration` carries `Option<PathBuf>` so `from_bytes`
>     failures don't have to invent a fake path (§3.5, §5.3).
> (d) `sigmoid(-16.776989) ≈ 5.17e-8` (was wrongly written 4.6e-8 in §11);
>     the parity assertion now uses a 1% relative tolerance.
> (e) MSRV lowered to 1.85 (the `edition = "2024"` floor, matches
>     `scenesdetect`); 1.95 in rev 2 was unverified inheritance from
>     `textclap` (§9).
> (f) `from_ort_sessions` renamed to `from_parts` (it takes a tokenizer
>     and a `Calibration`, not just sessions) and now validates each
>     session's input/output shapes at construction (§3.2, §3.3, §3.4).
> (g) `Embedding::DIM`, `Preprocessor::PIXEL_VALUES_STRIDE`, etc.
>     renamed with a `BASE_NAFLEX_` prefix to make their variant-
>     specificity explicit and to keep the door open for `siglip2-large`
>     (1024-dim) in 0.2.0 without naming collisions (§3.5, §3.7).
> (h) `Preprocessor::preprocess_into` is now `&self`; rev 2's "holds
>     reusable scratch space" claim was misleading because
>     `image::imageops::resize` allocates per call (§3.7).
> (i) `embed_pixels_batch(&[])` documented to return `Ok(vec![])`;
>     `BatchOptions::batch_size` clamped to a documented maximum of 1024
>     (`Error::BatchTooLarge`); `Error::PreprocBufferLength` carries a
>     `which: &'static str` to identify the offending buffer (§3.3, §3.5).
> (j) `THIRD_PARTY_NOTICES.md` added for Apache-2.0 attribution of the
>     bundled `tokenizer.json` (§9).
> (k) Tokenizer Rust↔Python parity note added; multilingual fixtures in
>     §8.2 are explicitly the parity net (§5.1).
> (l) Single source of truth for `MODELS.sha256` consolidated to
>     `tests/fixtures/MODELS.sha256` (§8.1, §9).
> (m) `Calibration::new(scale, bias)` const constructor added; `ImageView`
>     gains `Clone, Debug` (§5.3, §3.3).
> (n) Bundled-tokenizer scope clarified: the 33 MB `tokenizer.json` is the
>     **text encoder's** Gemma SPM wrapper. The vision encoder has no
>     tokenizer (it patchifies pixels). `BUNDLED_TOKENIZER` is therefore a
>     text-side asset; `Siglip2::bundled` uses it only via `TextEncoder`,
>     and `ImageEncoder` has no `bundled` constructor (§3.1, §3.3, §3.4,
>     §5.1, §9).
> (o) Calibration values pinned at full precision per the release body:
>     `logit_scale = 4.747554302215576`, `logit_bias = -16.776988983154297`
>     (rev 2 used 7-significant-digit truncations) (§5.3).
> (p) Text-side `pad_token_id` documented as `0` per the release body (§5.2).

> **Revision 2 changelog (vs revision 1).** Adversarial review surfaced
> nine substantive issues, all adopted. Material changes:
> (i) text tower also has an external-data sidecar (~1.08 GB) — every text
>     code path is now symmetric with the vision tower (§1, §3.4, §7.2, §7.4);
> (ii) `from_memory` constructors for the ONNX models are dropped from the 0.1.0
>     public API — `ort 2.0.0-rc.12` exposes no public way to bind external
>     initializer data from memory, and the temp-file workaround is unsafe
>     under concurrent loads (§7.1, §11);
> (iii) NaFlex sizing is pinned to the upstream binary-search-on-scale
>     algorithm with explicit pseudocode and a reference table (§4);
> (iv) in-patch byte ordering is pinned to (row, col, channel) channel-innermost
>     (§4 step 4);
> (v) text `input_ids` dtype corrected to `i64`; rank of vision/text outputs
>     hard-checked to be exactly 2 (§5, §3.5);
> (vi) sigmoid calibration (`calibration.json`) is loaded and applied inside
>     `classify` — was wrongly punted as int8-related in revision 1 (§5.3, §10);
> (vii) image resize pinned to `image::imageops::resize(FilterType::Triangle)`
>     to preserve the validated 0.99917 parity floor; `fast_image_resize`
>     is dropped (§4, §6, §9);
> (viii) low-level `Preprocessor` + `ImageEncoder::embed_preprocessed` API
>     added so callers can reuse one chunk-sized buffer and skip slots in
>     place (§3.7);
> (ix) constructor naming aligned with the silero crate's pattern;
>     ORT thread-control knobs added to `Options` (§3.6, §3.2–3.4); default
>     features now include `bundled` and `image` (§9).

## 1. Purpose

`siglip2` is a Rust inference library for **SigLIP2 NaFlex** (sigmoid-loss language-image
pre-training, "native flexible" aspect-ratio variant). It loads the vision-tower and text-tower ONNX
exports of `google/siglip2-base-patch16-naflex` and exposes them as a paired image/text encoder for
keyframe embedding and free-form text search.

The crate follows the API conventions of the sibling crates `textclap` (CLAP audio inference) and
`silero` (VAD) in the Findit-AI ecosystem.

The exact ONNX export targeted by 0.1.0 is the
[`Findit-AI/indexer` release `models-siglip2-naflex-v1`](https://github.com/Findit-AI/indexer/releases/tag/models-siglip2-naflex-v1):

| asset | role | size | runtime / archival |
|---|---|---|---|
| `vision_model_naflex_256.onnx`      | vision graph              | 1.0 MB   | runtime |
| `vision_model_naflex_256.onnx.data` | vision external weights   | 358 MB   | runtime |
| `text_model_naflex.onnx`            | text graph                | 0.9 MB   | runtime |
| `text_model_naflex.onnx.data`       | text external weights     | 1.08 GB  | runtime |
| `tokenizer.json`                    | **text-tower** tokenizer (Gemma SPM wrapper, HF Tokenizers JSON form) | 32.8 MB | runtime (text only) |
| `calibration.json`                  | sigmoid `logit_scale`, `logit_bias`    | 303 B   | runtime (for `classify`) |
| `tokenizer.model`                   | raw SentencePiece vocab   | 4.0 MB   | archival |
| `tokenizer_config.json`             | HF tokenizer config       | 40 KB    | archival |
| `special_tokens_map.json`           | special-tokens metadata   | 636 B    | archival |

Both ONNX files are external-data exports: the `.onnx` graph references its `.onnx.data`
sidecar by relative filename, and ORT auto-discovers the sidecar by looking for that name in the
same directory. Both files **must** live in the same directory for `from_files` to succeed.

The vision encoder consumes RGB pixels directly (NaFlex patchification, §4) — it has no tokenizer.
The text encoder is the only side of the crate that needs `tokenizer.json`, so the bundled
feature is a text-side asset and `ImageEncoder` has no `bundled` constructor (§3.3, §9).

The export is documented as **bit-exact** vs the upstream PyTorch reference across a 99-frame
validation set (median cosine 0.99997, min 0.99917). 0.1.0 inherits that floor as its
golden-fixture tolerance (§8).

### 1.1 Pipeline and the role of SigLIP2 within it

`siglip2` exposes two encoders that work as the **indexing** and **query** halves of an image-search
system. The model treats them as a contrastive pair — image embeddings and text embeddings live in
the same 768-dim space — but **they are used at different times in the pipeline**.

**Indexing path (write side, runs once per scene at keyframe-extraction time):**

```text
video → scenesdetect (cut detection, keyframe selection) → keyframe timestamps
  → caller-supplied frame extraction (e.g. ffmpeg) → decoded RGB image
  → ImageEncoder::embed_pixels(view) → 768-dim image embedding
  → caller writes { image_embedding, scene_id, ts, frame_path, ... } to a vector store (e.g. lancedb)
```

The image encoder runs **once per selected keyframe**, not once per video frame. Upstream
`scenesdetect` returns keyframe timestamps; the caller (or a sibling crate like `findit-keyframe`) is
responsible for turning those timestamps into decoded RGB pixel buffers. `siglip2` consumes pixels
and emits embeddings — it does not open video containers (see §2).

**Query path (read side, on demand when a user submits a text search):**

```text
user query text (e.g. "dog catching a frisbee on grass")
  → TextEncoder::embed(text) → 768-dim text embedding
  → caller runs cosine-similarity search against the image_embedding column
  → ranked keyframes → ranked scenes
```

The text encoder runs **once per search query**.

### 1.2 Use cases beyond live indexing + live query

- **Offline batch embedding (`embed_pixels_batch`).** Backfilling an index after first-time setup or
  re-indexing after a model update.
- **High-throughput preprocessor reuse (`Preprocessor` + `embed_preprocessed`).** A worker that
  decodes thousands of keyframes per chunk preallocates one chunk-sized `Vec<f32>` and writes
  preprocessed pixels into pre-sliced sub-ranges, calling the encoder once per chunk. See §3.7.
- **Ad-hoc / diagnostic classification (`Siglip2::classify`).** Zero-shot tagging of a single image
  against a fixed label set, with calibrated sigmoid scores.

### 1.3 What SigLIP2 NaFlex is good at — and what it isn't

**Domain-of-training.** SigLIP2 is trained on web-scale image-caption pairs (WebLI). It discriminates
object identity, scene content, coarse activities, broad style ("aerial photo", "pencil sketch"),
and cross-modal text-image alignment. It is suited to descriptive text queries like *"a red bicycle
leaning against a brick wall"*, *"two people shaking hands at a conference"*, *"close-up of a
sunflower"*.

**It is NOT a text-recognition (OCR) model.** Queries that depend on reading on-screen text or fine
typographic detail will perform poorly; index a separate OCR pipeline if those are needed.

**It is NOT a face-identification model.** Identity-level person search is out of scope.

**NaFlex strength.** Unlike the fixed `siglip2-base-patch16-224` variant, NaFlex preserves the
original aspect ratio of the input image, which materially improves retrieval on tall portraits,
wide cinematic frames, and other non-square keyframes that dominate real-world video corpora.

## 2. Non-goals

- **Image decoding** beyond the optional `embed_path` helper (`decoders` feature, default-on; see §9).
- **Storage / vector database integration.** Embeddings are emitted; storage and ANN search live in
  the caller. No `lancedb` dependency in this crate (§10).
- **CLI binary.** Library only.
- **In-memory ONNX construction (`from_memory` for vision/text).** The ORT 2.0 release-candidate
  exposes no public API for binding external initializer data from memory; the temp-file workaround
  is racy under concurrent loads. Reserved for a future minor when ORT stabilizes
  `add_external_initializers_from_array` in safe Rust (§7.1, §11). Tokenizer-side `from_memory` is
  unaffected.
- **Quantized / int8 export.** The released ONNX is f32; an int8 variant requires a separate export
  cycle and re-validation against the parity floor.
- **Re-exports at non-256 NaFlex patch budgets.** The exported graph is fixed at
  `max_num_patches = 256`; running it with any other budget is invalid (§3.6).
- **Async / runtime ownership.** Synchronous library.
- **Multi-variant SigLIP2 support.** 0.1.0 supports the base/patch16/naflex variant only. The
  embedding type's runtime-checked dimension makes growing to large/so400m a non-breaking
  follow-up.

## 3. Public API surface

### 3.1 Re-exports (`lib.rs`)

```rust
pub use embedding::{Embedding, LabeledScore, LabeledScoreOwned};
pub use image_enc::{ImageEncoder, ImageView};
pub use text_enc::TextEncoder;
pub use siglip2::Siglip2;
pub use preproc::Preprocessor;
pub use calibration::Calibration;
pub use error::{Error, Result};
pub use options::{Options, BatchOptions, ThreadOptions, GraphOptimizationLevel};

/// Text-tower tokenizer bytes (Gemma SPM wrapper). Embedded via
/// `include_bytes!("../models/tokenizer.json")` when `bundled` is on.
/// The vision tower has no tokenizer; this constant is text-only.
#[cfg(feature = "bundled")]
pub const BUNDLED_TOKENIZER: &[u8];
```

### 3.2 Top-level wrapper (silero-style constructor names)

```rust
impl Siglip2 {
    pub fn from_files(
        vision_onnx: &Path,
        text_onnx: &Path,
        tokenizer_json: &Path,
        calibration_json: &Path,
    ) -> Result<Self>;

    pub fn from_files_with_options(
        vision_onnx: &Path,
        text_onnx: &Path,
        tokenizer_json: &Path,
        calibration_json: &Path,
        opts: Options,
    ) -> Result<Self>;

    /// Uses BUNDLED_TOKENIZER. Requires feature `bundled`.
    /// `calibration_json` is still loaded from disk because its values are tied
    /// to the specific export, not to the crate.
    #[cfg(feature = "bundled")]
    pub fn bundled(
        vision_onnx: &Path,
        text_onnx: &Path,
        calibration_json: &Path,
    ) -> Result<Self>;

    #[cfg(feature = "bundled")]
    pub fn bundled_with_options(
        vision_onnx: &Path,
        text_onnx: &Path,
        calibration_json: &Path,
        opts: Options,
    ) -> Result<Self>;

    /// Build from caller-owned components (e.g. ORT sessions with custom
    /// execution providers). Validates each session's input/output shapes
    /// against the SigLIP2-base-NaFlex contract at construction:
    /// the image session must have `pixel_values: f32[_, 256, 768]`,
    /// `pixel_attention_mask: i32[_, 256]`, `spatial_shapes: i32[_, 2]`,
    /// `pooler_output: f32[_, 768]`; the text session must have
    /// `input_ids: i64[_, 64]` and a 2-D `f32` output. Mismatches surface
    /// as `Error::MaxNumPatchesMismatch`, `Error::OutputRank`, or
    /// `Error::SessionShapeMismatch`.
    ///
    /// Also **re-validates `calibration`** through the same `validate`
    /// path that `Calibration::from_path` / `from_bytes` use. This closes
    /// the gap where a caller built `Calibration` via `new` (which is
    /// deliberately unchecked, for tests/hard-coded values) and passed
    /// the result here — without this re-check, NaN or non-positive
    /// `logit_scale` would silently poison every `classify` score.
    /// Validation failures surface as `Error::InvalidCalibration`.
    pub fn from_parts(
        image_session: ort::Session,
        text_session:  ort::Session,
        tokenizer:     tokenizers::Tokenizer,
        calibration:   Calibration,
    ) -> Result<Self>;

    pub fn image(&mut self) -> &mut ImageEncoder;
    pub fn text(&mut self)  -> &mut TextEncoder;

    /// Borrow both encoders simultaneously. Use when a caller needs to embed
    /// an image and a text batch in one operation without serializing through
    /// the wrapper. `classify` uses this internally.
    pub fn split(&mut self) -> (&mut ImageEncoder, &mut TextEncoder);

    /// Zero-shot classification: ranks `labels` by calibrated sigmoid score against `image`.
    /// Score ∈ [0, 1] is `sigmoid(exp(logit_scale) · cos(image, text) + logit_bias)`.
    /// `logit_scale` in the JSON is the **raw** learned parameter; the model
    /// exponentiates at inference time (matches HuggingFace `Siglip2Model.forward`).
    /// For the pinned release values, `exp(4.7476) ≈ 115.36` is the effective
    /// scale, and a typical confident match (cos ≈ 0.18) yields ~0.98.
    /// `top_k` is clamped to `labels.len()`; passing `top_k > labels.len()` returns all
    /// labels ranked in descending score order rather than erroring.
    pub fn classify<'a>(
        &mut self,
        image: ImageView<'_>,
        labels: &'a [&'a str],
        top_k: usize,
    ) -> Result<Vec<LabeledScore<'a>>>;
}
```

`Calibration` is a small public struct holding `logit_scale: f32` and `logit_bias: f32` parsed from
`calibration.json`. `Calibration::from_path` loads and validates the file (§5.3).

### 3.3 Image encoder (NaFlex)

```rust
/// `Copy` is safe even with the validating `new` constructor because all fields
/// are `Copy` and the validation is purely on the input that's *already* in the
/// view — copying an existing `ImageView` re-uses bytes that have already passed
/// the length and dim checks. Cheap copies matter on the hot path (one per
/// keyframe in a video index).
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    rgb:    &'a [u8],
    width:  u32,
    height: u32,
}

impl<'a> ImageView<'a> {
    /// Constructs a view over RGB pixels. `rgb` must be exactly
    /// `width * height * 3` bytes, row-major, no row padding. Returns
    /// `Error::RgbLength` on mismatch and `Error::InvalidImage` on zero
    /// dimensions — failing here is preferable to deferring to first
    /// inference, where the error stack is harder to read.
    pub fn new(rgb: &'a [u8], width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidImage { width, height });
        }
        let expected = (width as usize) * (height as usize) * 3;
        if rgb.len() != expected {
            return Err(Error::RgbLength { got: rgb.len(), expected });
        }
        Ok(Self { rgb, width, height })
    }
    pub fn rgb(&self)    -> &'a [u8] { self.rgb }
    pub fn width(&self)  -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }
}

impl ImageEncoder {
    pub fn from_files(graph: &Path) -> Result<Self>;
    pub fn from_files_with_options(graph: &Path, opts: Options) -> Result<Self>;
    /// See `Siglip2::from_parts` for the session-shape contract; the same
    /// validation runs here.
    pub fn from_ort_session(session: ort::Session) -> Result<Self>;

    pub fn embed_pixels(&mut self, view: ImageView<'_>) -> Result<Embedding>;

    /// Returns `Ok(vec![])` for an empty input slice (no ORT call).
    /// Returns `Error::BatchTooLarge` when `views.len() > opts.batch.batch_size_max`.
    /// Internally chunks `views` into groups of size `BatchOptions::batch_size`
    /// and runs one ORT inference per chunk; the returned `Vec` preserves input
    /// order and has the same length as `views` on success.
    ///
    /// **Failure semantics.** Aborts on the first failing input and returns
    /// `Error::Batch { index, source }` carrying the offending zero-based index.
    /// Already-computed embeddings from earlier chunks are dropped (the partial-
    /// success mode would require a different return type, and callers needing
    /// it can chunk caller-side and call `embed_pixels` per item to track failures
    /// individually).
    pub fn embed_pixels_batch(&mut self, views: &[ImageView<'_>]) -> Result<Vec<Embedding>>;

    /// Runs ONNX on already-preprocessed tensors. See §3.7 for the buffer contract.
    /// `batch_size == 0` returns `Ok(vec![])` without invoking ORT.
    pub fn embed_preprocessed(
        &mut self,
        pixel_values:    &[f32],   // batch_size * BASE_NAFLEX_PIXEL_VALUES_STRIDE
        attention_mask:  &[i32],   // batch_size * BASE_NAFLEX_ATTENTION_MASK_STRIDE
        spatial_shapes:  &[i32],   // batch_size * BASE_NAFLEX_SPATIAL_SHAPES_STRIDE
        batch_size:      usize,
    ) -> Result<Vec<Embedding>>;

    /// Decode JPEG/PNG from disk and call `embed_pixels`. Requires feature `decoders`.
    /// Supported formats: JPEG and PNG only (the only decoders activated by `decoders`).
    /// For other formats, decode in caller code and use `embed_pixels` directly.
    #[cfg(feature = "decoders")]
    pub fn embed_path(&mut self, path: &Path) -> Result<Embedding>;

    pub fn warmup(&mut self) -> Result<()>;
}
```

`from_files` takes only the `.onnx` graph path; ORT discovers the `.onnx.data` sidecar in the same
directory automatically (§7.1). Documenting this on the rustdoc is sufficient; we do not validate
adjacency ourselves — ORT's error already names both files clearly.

`ImageEncoder` deliberately has no `bundled` constructor and no tokenizer parameter on any
constructor — the SigLIP2 vision tower consumes RGB pixels directly via NaFlex patchification (§4).
The bundled feature is a text-side asset (§3.4, §9).

**Threading contract.** `ImageEncoder: Send + !Sync` — it owns an `ort::Session`, which is
`!Sync`. Workers wanting parallelism instantiate one `ImageEncoder` per thread, or share one
behind a `Mutex<ImageEncoder>`. The same constraint applies to `TextEncoder` (§3.4). Negative
trait bounds aren't expressible cleanly in stable Rust, so this is a rustdoc-only guarantee;
the §8.4 trait-bound test asserts the positive `Send` bound.

`ImageView` is deliberately a small inline struct rather than a generic over `image::ImageBuffer` —
it keeps the core path free of the `image` crate and gives callers a stable shape regardless of how
their pixels were decoded.

### 3.4 Text encoder

`text_model_naflex.onnx` is also an external-data export (1.08 GB sidecar at
`text_model_naflex.onnx.data`). The text-tower constructors mirror the vision-tower's exactly,
plus the tokenizer parameter — the text tower is the only side of the crate that uses
`tokenizer.json` (the 32.8 MB Gemma SPM wrapper). When `bundled` is on,
`TextEncoder::bundled(graph)` and `TextEncoder::bundled_with_options(graph, opts)` use
`BUNDLED_TOKENIZER` from §3.1.

```rust
impl TextEncoder {
    pub fn from_files(graph: &Path, tokenizer: &Path) -> Result<Self>;
    pub fn from_files_with_options(graph: &Path, tokenizer: &Path, opts: Options) -> Result<Self>;

    /// Uses BUNDLED_TOKENIZER. Requires feature `bundled`.
    #[cfg(feature = "bundled")]
    pub fn bundled(graph: &Path) -> Result<Self>;

    #[cfg(feature = "bundled")]
    pub fn bundled_with_options(graph: &Path, opts: Options) -> Result<Self>;

    /// See `Siglip2::from_parts` for the session-shape contract.
    pub fn from_ort_session(session: ort::Session, tokenizer: tokenizers::Tokenizer)
        -> Result<Self>;

    pub fn embed(&mut self, text: &str) -> Result<Embedding>;
    /// `Ok(vec![])` on empty input; `Error::BatchTooLarge` when over the cap.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>>;
    pub fn warmup(&mut self) -> Result<()>;
}
```

There is no `from_memory` for text-side ONNX in 0.1.0 — same external-data limitation as vision.

### 3.5 Embedding & error types

```rust
// Deliberately no `Serialize`/`Deserialize` derives — see §3.5 prose below
// for why direct serde on Embedding would bypass the L2-norm and dim invariants.
#[derive(Clone, Debug)]
pub struct Embedding(Arc<[f32]>);
// length == BASE_NAFLEX_DIM == 768 in 0.1.0; L2-normalized invariant

impl Embedding {
    /// 0.1.0 supports only the base/patch16/naflex variant. The variant-specific
    /// prefix is deliberate so 0.2.0 can add (e.g.) `LARGE_DIM = 1024` without
    /// renaming this constant.
    pub const BASE_NAFLEX_DIM: usize = 768;

    pub fn dim(&self) -> usize { self.0.len() }
    pub fn as_slice(&self) -> &[f32];
    pub fn into_inner(self) -> Arc<[f32]>;

    /// Convenience: copy the embedding into a fresh `Vec<f32>`. Equivalent to
    /// `self.as_slice().to_vec()`; provided so callers don't have to chain
    /// `Arc::try_unwrap` for the common round-trip case (e.g. into a row of
    /// a parquet/lancedb writer).
    pub fn into_vec(self) -> Vec<f32>;

    /// Dot product. Both operands MUST be unit-norm; valid because all `Embedding`
    /// values returned by this crate are L2-normalized at construction.
    ///
    /// **Panics** via
    /// `assert_eq!(self.dim(), other.dim(), "Embedding::cosine: dim mismatch (variants must match)")`
    /// when the dims disagree. In 0.1.0 only the 768-dim base/naflex variant exists,
    /// so this is trivially satisfied; in a future multi-variant world, mixing
    /// variants in `cosine` is a programming error and `Result<f32, _>` would only
    /// push the recovery point further from the bug. `assert_eq!` (not
    /// `debug_assert_eq!`) so release builds catch it too.
    pub fn cosine(&self, other: &Embedding) -> f32;
}

// No `impl Serialize`, no `impl Deserialize` — deliberate.
//
// An auto-derived `Deserialize` would route directly to `Embedding(Arc::<[f32]>::deserialize(d)?)`,
// bypassing both the dim check (`Error::EmbeddingDim`) and the L2-norm check
// (`Error::NotNormalized`) that `TryFrom<Vec<f32>>` exists to enforce. A serialized
// vector with the wrong length, NaN values, or non-unit norm would deserialize
// silently and then poison `cosine` with a value outside `[-1, 1]`. The fix would
// be `#[serde(try_from = "Vec<f32>")]`, but we don't actually need serde on
// `Embedding` — vector stores (lancedb, faiss, arrow) use binary columnar formats,
// not JSON. Callers who genuinely need to serialize an embedding can go through
// `embedding.as_slice()` (which is `&[f32]: Serialize`) or `embedding.into_vec()`
// (which is `Vec<f32>: Serialize + Deserialize`); reconstruction goes through
// `Embedding::try_from(vec)`, which is the only validated path.
//
// Concrete round-trip example:
//
// ```rust
// // Serialize via the inner slice (`&[f32]: Serialize`):
// let json = serde_json::to_string(embedding.as_slice())?;
//
// // Deserialize via the validated path:
// let v: Vec<f32> = serde_json::from_str(&json)?;
// let embedding  = Embedding::try_from(v)?;  // validates dim + L2-norm
// ```

impl TryFrom<Vec<f32>> for Embedding {
    type Error = Error;
    /// Rejects vectors of wrong length (`Error::EmbeddingDim`) or non-unit norm
    /// (`Error::NotNormalized`, tolerance ε = 5e-4).
    fn try_from(v: Vec<f32>) -> Result<Self> { /* ... */ }
}

/// Borrowed label + score. Returned by `Siglip2::classify`. `score` is
/// `sigmoid(scale·cos + bias) ∈ [0, 1]`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct LabeledScore<'a> {
    label: &'a str,
    score: f32,
}

impl<'a> LabeledScore<'a> {
    pub fn label(&self) -> &'a str { self.label }
    pub fn score(&self) -> f32     { self.score }
    pub fn to_owned(&self) -> LabeledScoreOwned;
}

/// Owned analogue of `LabeledScore` for callers that need to outlive the
/// `&[&str]` they passed to `classify` (e.g. for serialization).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LabeledScoreOwned {
    label: smol_str::SmolStr,
    score: f32,
}

impl LabeledScoreOwned {
    /// Public constructor for tests, mocks, and callers building `LabeledScoreOwned`
    /// values from sources other than `LabeledScore::to_owned()` or `serde::Deserialize`.
    /// `score` is **unchecked** — production scores returned by `Siglip2::classify`
    /// are guaranteed to lie in `[0, 1]` (sigmoid output), but `new` accepts any
    /// `f32` (including NaN). Callers ingesting `LabeledScoreOwned` from untrusted
    /// sources should validate `score.is_finite() && (0.0..=1.0).contains(&score)`
    /// themselves.
    pub fn new(label: impl Into<smol_str::SmolStr>, score: f32) -> Self {
        Self { label: label.into(), score }
    }
    pub fn label(&self) -> &str { self.label.as_str() }
    pub fn score(&self) -> f32  { self.score }
}
```

`Arc<[f32]>` (rather than `[f32; 768]` or `Box<[f32]>`) gives constant-time `clone` (atomic
refcount bump instead of a 3 KB heap memcpy), which matters because embeddings routinely move
between batch buffers, vector-store writes, search rankers, and result types. The fat pointer is
the same 16 B on the stack as `Box<[f32]>`; `Send + Sync` is preserved. `Embedding` is
post-construction immutable in this API (the L2-norm invariant, §3.5), so the lack of free
`&mut [f32]` access via `Arc` is not a constraint — callers needing a mutable buffer go through
`embed_preprocessed` (§3.7) or `into_inner().into_iter().collect()`.

The variant-specific `BASE_NAFLEX_DIM` is enforced at every constructor in 0.1.0; future
variants (`siglip2-large` 1024-dim, `so400m` 1152-dim) slot in without a type-level breaking
change because the inner is a slice, not a fixed-size array.

```rust
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to load ONNX graph at {path}: {source}")]
    LoadGraph { path: PathBuf, source: ort::Error },

    #[error("failed to load external weights for graph at {path}: {source}")]
    LoadWeights { path: PathBuf, source: ort::Error },

    /// Path is `Some` for `Calibration::from_path`, `None` for `from_bytes`.
    #[error("failed to parse calibration.json{location}: {source}",
            location = path.as_ref().map(|p| format!(" at {}", p.display())).unwrap_or_default())]
    LoadCalibration { path: Option<PathBuf>, source: serde_json::Error },

    /// `calibration.json` parsed but contained semantically invalid values
    /// (NaN, non-finite, non-positive logit_scale, etc.). Catches corrupted
    /// or hand-edited files that would otherwise pollute every `classify` score.
    #[error("invalid calibration values: {reason}")]
    InvalidCalibration { reason: &'static str },

    #[error("tokenizer load failed: {0}")]
    Tokenizer(String),

    #[error("invalid image dimensions: {width}x{height}")]
    InvalidImage { width: u32, height: u32 },

    #[error("rgb buffer length {got} does not match width*height*3 = {expected}")]
    RgbLength { got: usize, expected: usize },

    /// `embed_preprocessed` / `preprocess_into` buffer length didn't match the
    /// expected `batch_size * stride`. `which` identifies the offending
    /// buffer (`"pixel_values"`, `"attention_mask"`, or `"spatial_shapes"`).
    #[error("preprocessed buffer `{which}` length {got} does not match expected {expected}")]
    PreprocBufferLength { which: &'static str, got: usize, expected: usize },

    /// Output tensor was not rank-2. Defends against accidental re-export that
    /// emits `last_hidden_state` ([B, T, 768]) instead of `pooler_output` ([B, 768]).
    #[error("unexpected output rank: expected 2, got {rank} with shape {shape:?}")]
    OutputRank { rank: usize, shape: Vec<i64> },

    #[error("session shape mismatch on `{input}`: expected {expected}, got {got:?}")]
    SessionShapeMismatch { input: &'static str, expected: &'static str, got: Vec<i64> },

    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    EmbeddingDim { expected: usize, got: usize },

    #[error("embedding is not unit-norm (got ||v||₂ = {norm}, tolerance ε = {epsilon})")]
    NotNormalized { norm: f32, epsilon: f32 },

    #[error("text input is empty")]
    EmptyText,

    #[error("max_num_patches in Options ({opt}) does not match the value baked into the ONNX export ({export})")]
    MaxNumPatchesMismatch { opt: u32, export: u32 },

    /// Returned when a batch input exceeds `BatchOptions::batch_size_max`.
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
```

L2-normalization tolerance is ε = 5e-4, matching textclap's convention. Embeddings produced by
`siglip2`'s own encoders are renormalized in-place if they fall within ε but not at exactly 1.0;
embeddings supplied via `TryFrom<Vec<f32>>` are rejected (no silent renormalization of caller-owned
data). All output tensors are hard-checked to be rank-2 with shape `[batch, 768]`; rank 3 (e.g. an
accidental `last_hidden_state` re-export) is a hard error rather than a "fall back to first token"
silent corruption.

### 3.6 Options

All three config structs follow the same pattern: private fields, getter
methods, builder methods (`with_*`, consume `self`) for chained construction,
and setter methods (`set_*`, take `&mut self`) for in-place mutation. All three
implement `Default`, `Clone`, `Copy`, and `Debug`. **`Options::default()` is the
canonical entry point**; `Options::new()` is retained as an alias that calls
through to `default()` for callers who prefer that idiom.

```rust
#[derive(Clone, Copy, Debug)]
pub struct Options {
    graph_optimization_level: GraphOptimizationLevel,
    batch:                    BatchOptions,
    threads:                  ThreadOptions,
}

impl Options {
    pub fn new() -> Self { Self::default() }

    // Getters
    pub fn graph_optimization_level(&self) -> GraphOptimizationLevel { self.graph_optimization_level }
    pub fn batch(&self)   -> BatchOptions  { self.batch }
    pub fn threads(&self) -> ThreadOptions { self.threads }

    // Builder (consume self, return self)
    pub fn with_graph_optimization_level(mut self, l: GraphOptimizationLevel) -> Self {
        self.graph_optimization_level = l; self
    }
    pub fn with_batch(mut self, b: BatchOptions)   -> Self { self.batch = b; self }
    pub fn with_threads(mut self, t: ThreadOptions) -> Self { self.threads = t; self }

    // In-place setters
    pub fn set_graph_optimization_level(&mut self, l: GraphOptimizationLevel) -> &mut Self {
        self.graph_optimization_level = l; self
    }
    pub fn set_batch(&mut self, b: BatchOptions)   -> &mut Self { self.batch = b; self }
    pub fn set_threads(&mut self, t: ThreadOptions) -> &mut Self { self.threads = t; self }
}

impl Default for Options {
    /// `graph_optimization_level: Level1` (matching the existing
    /// `findit-siglip2-vision` service at `lib.rs:91`; the 0.99917 cosine floor
    /// was established under Level1, and Level3's extended optimizations apply
    /// layout transforms and normalization-layer fusions that may produce
    /// numerically different outputs on the SigLIP2 vision tower's
    /// LayerNorm/MatMul subgraphs — CI runs §8.2 fixtures at both Level1 and
    /// Level3, and both must clear the floor before a release).
    /// `batch: BatchOptions::default()`, `threads: ThreadOptions::default()`.
    fn default() -> Self { /* ... */ }
}

#[derive(Clone, Copy, Debug)]
pub struct BatchOptions {
    max_num_patches: u32,
    batch_size:      usize,
    batch_size_max:  usize,
}

impl BatchOptions {
    pub fn new() -> Self { Self::default() }

    pub fn max_num_patches(&self) -> u32   { self.max_num_patches }
    pub fn batch_size(&self)      -> usize { self.batch_size }
    pub fn batch_size_max(&self)  -> usize { self.batch_size_max }

    pub fn with_max_num_patches(mut self, n: u32)   -> Self { self.max_num_patches = n; self }
    pub fn with_batch_size(mut self, n: usize)      -> Self { self.batch_size = n; self }
    pub fn with_batch_size_max(mut self, n: usize)  -> Self { self.batch_size_max = n; self }

    pub fn set_max_num_patches(&mut self, n: u32)   -> &mut Self { self.max_num_patches = n; self }
    pub fn set_batch_size(&mut self, n: usize)      -> &mut Self { self.batch_size = n; self }
    pub fn set_batch_size_max(&mut self, n: usize)  -> &mut Self { self.batch_size_max = n; self }
}

impl Default for BatchOptions {
    /// `max_num_patches: 256` (must equal the value baked into the ONNX export
    /// — 0.1.0 ships only the `max_num_patches=256` export).
    /// `batch_size: 8` (inference micro-batch for `embed_pixels_batch` /
    /// `embed_batch`).
    /// `batch_size_max: 1024` (hard ceiling — `Error::BatchTooLarge` above
    /// this). The arithmetic: a single preprocessed sample is
    /// `256 patches × 768 channels × 4 B = 786 432 B ≈ 0.75 MB`; 1024 of those
    /// is ~770 MB of `pixel_values` buffer. Combined with §7.3's ~1.55 GB
    /// resident weights per worker, peak per-worker RAM at max batch is
    /// ~2.3 GB. On the typical 8-worker thread-per-core deployment that's
    /// ~18 GB; lower `batch_size_max` if your per-worker memory budget is
    /// tighter.
    fn default() -> Self { /* ... */ }
}

#[derive(Clone, Copy, Debug)]
pub struct ThreadOptions {
    intra_threads:      usize,
    inter_threads:      usize,
    parallel_execution: bool,
}

impl ThreadOptions {
    pub fn new() -> Self { Self::default() }

    pub fn intra_threads(&self)      -> usize { self.intra_threads }
    pub fn inter_threads(&self)      -> usize { self.inter_threads }
    pub fn parallel_execution(&self) -> bool  { self.parallel_execution }

    pub fn with_intra_threads(mut self, n: usize)      -> Self { self.intra_threads = n; self }
    pub fn with_inter_threads(mut self, n: usize)      -> Self { self.inter_threads = n; self }
    pub fn with_parallel_execution(mut self, p: bool)  -> Self { self.parallel_execution = p; self }

    pub fn set_intra_threads(&mut self, n: usize)      -> &mut Self { self.intra_threads = n; self }
    pub fn set_inter_threads(&mut self, n: usize)      -> &mut Self { self.inter_threads = n; self }
    pub fn set_parallel_execution(&mut self, p: bool)  -> &mut Self { self.parallel_execution = p; self }
}

impl Default for ThreadOptions {
    /// `intra_threads: 1`, `inter_threads: 1`, `parallel_execution: false`.
    /// The single-thread defaults match the existing service's thread-per-core
    /// deployment and avoid N×N oversubscription when the caller already owns
    /// an outer worker pool.
    fn default() -> Self { /* ... */ }
}
```

Constructing an `ImageEncoder` whose `BatchOptions::max_num_patches` does not match the value
inferred from the loaded ONNX graph's `pixel_values` shape (`[batch, 256, 768]`) returns
`Error::MaxNumPatchesMismatch` at load time.

`Options` is `Copy`; all `*_with_options` constructors take it by value, consistent with
`Preprocessor::new(opts: Options)`.

### 3.7 `Preprocessor` (low-level zero-copy path)

`embed_pixels_batch` is convenient but allocates the batch buffer internally on every call. For
keyframe pipelines that decode thousands of frames per chunk, that's an unnecessary alloc on the hot
path (~24 MB for a chunk of 32 keyframes). The low-level surface lets callers preallocate once and
write directly into pre-sliced sub-ranges:

```rust
/// Stateless wrapper around the NaFlex preprocessing pipeline. Carries no
/// persistent scratch space in 0.1.0 (`image::imageops::resize` allocates
/// per call, and the patchify/normalize step is small enough that pooling
/// it isn't worth the API cost). `&self` rather than `&mut self` so a
/// single `Preprocessor` can be shared across worker threads.
///
/// **`Preprocessor: Send + Sync`** — guaranteed by the auto-derives because
/// the inner is a `Copy`-friendly POD config with no interior mutability.
/// `tests/integration.rs` carries a compile-time assertion (§8.4) so a
/// future field that breaks `Send + Sync` would fail the build, not just
/// silently regress the documented contract.
pub struct Preprocessor { /* opaque; small POD config */ }

impl Preprocessor {
    pub fn new(opts: Options) -> Self;

    /// Per-image stride sizes (multiply by batch_size to size a chunk buffer).
    /// The `BASE_NAFLEX_` prefix makes the variant-specificity explicit so
    /// 0.2.0 can introduce e.g. `LARGE_NAFLEX_PIXEL_VALUES_STRIDE` without
    /// renaming.
    pub const BASE_NAFLEX_PIXEL_VALUES_STRIDE:   usize = 256 * 768; // = 196_608
    pub const BASE_NAFLEX_ATTENTION_MASK_STRIDE: usize = 256;
    pub const BASE_NAFLEX_SPATIAL_SHAPES_STRIDE: usize = 2;

    /// Writes preprocessed tensors for one image into the supplied buffers.
    /// Buffer lengths must equal the per-image strides above; otherwise returns
    /// `Error::PreprocBufferLength { which: ... }`.
    pub fn preprocess_into(
        &self,
        view:               ImageView<'_>,
        pixel_values_out:   &mut [f32],
        attention_mask_out: &mut [i32],
        spatial_shapes_out: &mut [i32],
    ) -> Result<()>;
}
```

Typical caller (using shorter local aliases for readability):

```rust
const PVS: usize = Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE;
const AMS: usize = Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE;
const SSS: usize = Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE;

let mut pix  = vec![0.0f32; PVS * chunk_size];
let mut mask = vec![0i32;   AMS * chunk_size];
let mut spat = vec![0i32;   SSS * chunk_size];

let mut count = 0;
for view in chunk.iter() {
    let p = &mut pix [count * PVS..][..PVS];
    let m = &mut mask[count * AMS..][..AMS];
    let s = &mut spat[count * SSS..][..SSS];
    if pre.preprocess_into(*view, p, m, s).is_ok() { count += 1; }
}
let embeddings = encoder.embed_preprocessed(
    &pix [..count * PVS],
    &mask[..count * AMS],
    &spat[..count * SSS],
    count,
)?;
```

This is the API the existing `findit-siglip2-vision` worker actually wants and the migration target
for that service. `embed_pixels` and `embed_pixels_batch` remain as the convenient front door for
everyone else.

## 4. NaFlex preprocessing

The vision tower expects three parallel inputs per image:

```text
pixel_values         f32 [batch, 256, 768]    # 256 patches, each 16*16*3 = 768 channels
pixel_attention_mask i32 [batch, 256]         # 1 = real patch, 0 = padding
spatial_shapes       i32 [batch, 2]           # (H_p, W_p) — patch grid before padding
```

### 4.1 Patch-grid sizing — binary search on scale (eps-terminated, upstream-equivalent)

Direct port of upstream
`transformers.models.siglip2.image_processing_siglip2_fast.Siglip2ImageProcessorFast.get_image_size_for_max_num_patches`.
**Do not "tighten" the eps termination** — the small safety margin
(`eps = 1e-5`) is what keeps `s` clearly below the boundary at which the
two `ceil()` calls flip. A tighter (e.g. fixed 64-iteration) loop crashes
into f64 noise at that boundary and produces silently-wrong grids on
thousands of `(H, W)` pairs (per the Codex review's regression case
`(3, 39) → (4, 53) instead of (4, 52)`). See
`SCALE_EPS` in `src/preproc/naflex.rs`.

```text
INPUT:
  H, W  : input pixel dimensions (u32)
  P     : patch_size = 16
  M     : max_num_patches = 256
  eps   : 1e-5 (matches upstream)

HELPER:
  scaled_pixel_size(s, x, P) = max(P, ceil(s * x / P) * P)
    # Round up to a multiple of P, but never below P (so a 1-pixel
    # axis still gets one full patch).

ALGORITHM:
  scale_min, scale_max := eps / 10, 100
  while (scale_max - scale_min) >= eps:
      s := (scale_min + scale_max) / 2
      H_res := scaled_pixel_size(s, H, P)
      W_res := scaled_pixel_size(s, W, P)
      if (H_res * W_res) / (P * P) <= M:
          scale_min := s   # feasible — try larger
      else:
          scale_max := s   # infeasible — try smaller

  H_res := scaled_pixel_size(scale_min, H, P)
  W_res := scaled_pixel_size(scale_min, W, P)

OUTPUT:
  H_p   = max(1, H_res / P)   # always feasible, never overshoots
  W_p   = max(1, W_res / P)
```

The loop invariant — `scale_min` is always feasible, since it's only
updated when the budget check passes — means the returned `(H_p, W_p)`
trivially satisfies `H_p * W_p ≤ M`. No defensive post-clamp is needed.

Reference behavior on representative inputs (verified against the
upstream Python reference; must match in golden tests):

| input H × W   | (H_p, W_p) | resized H × W |
|---|---|---|
| 16 × 16       | (16, 16)   | 256 × 256     |
| 100 × 100     | (16, 16)   | 256 × 256     |
| 224 × 224     | (16, 16)   | 256 × 256     |
| 1080 × 1920   | (12, 21)   | 192 × 336     |
| 1920 × 1080   | (21, 12)   | 336 × 192     |
| 2160 × 4096   | (12, 21)   | 192 × 336     |
| 1024 × 1      | (256, 1)   | 4096 × 16     |
| 3 × 39        | (4, 52)    | 64 × 832      |

The `(3, 39) → (4, 52)` row is the regression case from the Codex
adversarial review: the rev-8 64-iteration binary search returned
`(4, 53)`. It is pinned in
`tests/preproc::naflex::tests::reference_table_matches`.

### 4.2 Resize, normalize, patchify

1. **Resize.** Bilinear-resize the RGB image from `(H, W)` to `(H_res, W_res)` using
   `image::imageops::resize(&img, W_res, H_res, FilterType::Triangle)`. This is the same call the
   existing service uses; switching to a different resize kernel (e.g. `fast_image_resize`) would
   break the validated 0.99917 cosine floor and is explicitly out of scope (§6).
2. **Normalize.** Convert `u8` → `f32 / 255.0`, then apply `(x − 0.5) / 0.5` per channel. This is
   the SigLIP convention (`mean = std = 0.5`) and is the constant the export script uses; the
   existing parity test at median cosine 0.99997 already proves it.
3. **Patchify.** For each patch grid cell `(py, px)` in row-major order
   `(0,0), (0,1), …, (0, W_p-1), (1, 0), …, (H_p-1, W_p-1)`, extract the `P × P × 3` block
   starting at pixel `(py * P, px * P)` and flatten it as
   **`(row_within_patch, col_within_patch, channel)` with channel innermost** — i.e. interleaved
   R, G, B from the resized RGB image, no axis transposition. Each patch becomes 768 contiguous
   `f32`s; the full `pixel_values[batch_i, :H_p*W_p, :]` is `H_p*W_p` such patches stacked.
4. **Right-pad** the patch axis with zeros to `[256, 768]`.
5. **`pixel_attention_mask`** of length 256: `1` for the first `H_p*W_p` slots, `0` otherwise.
6. **`spatial_shapes`**: `[H_p, W_p]` as i32.
7. **Stack** across the batch dimension. Per-image `H_p*W_p` varies, but every image is right-padded
   to 256 patches, so the batch tensor is rectangular.

### 4.3 Edge cases

- `H == 0 || W == 0` → `Error::InvalidImage`.
- `rgb.len() != H * W * 3` → `Error::RgbLength`.
- A grid that resolves to `H_p == 0` or `W_p == 0` (e.g. a 1-pixel-tall input) is clamped to a
  minimum of 1; the attention mask correctly marks the (1 × W_p) or (H_p × 1) valid region.

The `preproc::naflex` module's algorithm is private to the crate; only `Preprocessor` is publicly
visible.

## 5. Tokenization & calibration (text side)

### 5.1 Tokenizer

`tokenizer.json` is the **text encoder's** tokenizer (loaded directly via
`tokenizers = "0.22"`). It is a Gemma-style SentencePiece-trained vocabulary wrapped in HF
Tokenizers JSON form — the wrapper file embeds the vocab, so neither `tokenizer.model` nor the
`tokenizer_config.json` / `special_tokens_map.json` files are needed at runtime (they are archival,
not loaded — see §1's runtime/archival column).

**The vision encoder has no tokenizer.** SigLIP2's vision tower is a Vision Transformer over
NaFlex-patchified RGB pixels (§4); the only "tokenization" on the image side is patchification, and
that's pure pixel arithmetic. No tokenizer asset, JSON or otherwise, is bundled or loaded for
`ImageEncoder`.

**Rust ↔ Python tokenizer parity.** The HF Rust `tokenizers` crate and Python
`transformers`'s `GemmaTokenizer` share the JSON spec but are not byte-exact
on all unicode-edge inputs (NFC vs NFKC normalization corner cases, BPE merge
ties, special-token handling). The upstream verify script established the
0.99917 cosine floor using Python; the multilingual fixture set in §8.2 is
the explicit regression net for this divergence. If a tokenizer-version bump
in this crate breaks parity on any fixture, the bump is rejected.

### 5.2 Sequence length and dtypes

- **Sequence length.** Pad / truncate to `seq_len = 64` tokens. The text graph's `input_ids` axis
  is verified at load time against the ONNX graph's input shape — if the export used a different
  value, the constant in this crate must be updated and text fixtures regenerated.
- **Padding.** Use `tokenizers`'s `PaddingParams` with `strategy: BatchLongest` is **wrong** for a
  static-seq-len graph; we set `strategy: Fixed(64)` with `pad_id = 0` (per the release body —
  `pad_token_id = 0`). This guarantees every input is exactly 64 tokens regardless of batch
  contents.
- **Input dtype.** `input_ids: i64[batch, 64]`. (Verified against the export with
  `np.int64`; passing i32 yields a runtime ORT error.)
- **No `attention_mask` input.** SigLIP2's text tower learns padding through a dedicated PAD-token
  embedding and does not accept an explicit attention mask.
- **Output.** `[batch, 768]`, rank-checked, then L2-normalized into `Embedding`.

### 5.3 Calibration (`calibration.json`)

The release ships a 303-byte JSON file:

```json
{
  "logit_scale": 4.747554302215576,
  "logit_bias":  -16.776988983154297
}
```

These are the trained sigmoid parameters of the SigLIP2 contrastive head. **`logit_scale` is the raw
learned parameter** (matching HuggingFace `Siglip2Model.logit_scale`); the model exponentiates it at
inference time. For the pinned values, the effective scale used in the logit is
`exp(4.7476) ≈ 115.36`, not 4.7476 directly. SigLIP2's raw cosines are
narrow (a confident match might score ~0.18 against ~0.05 for non-matches), so raw cosine values
are **not** interpretable as probabilities. `Siglip2::classify` applies

```text
score(image, label) = sigmoid(exp(logit_scale) · cos(image_embedding, text_embedding) + logit_bias)
```

returning `LabeledScore` values in `[0, 1]`. Callers that only need *ranking* (not calibrated
probabilities) can use `cos(...)` directly via `image.cosine(&text_embedding)`; ranking is order-
preserving under the affine-then-sigmoid transform.

```rust
// Deliberately no `Serialize` / `Deserialize` derives on Calibration —
// see the prose below. JSON parsing is routed through a private raw struct
// so every Calibration construction path passes through `validate`.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    logit_scale: f32,
    logit_bias:  f32,
}

impl Calibration {
    /// Const constructor for tests and callers with hard-coded values.
    /// `new` does **not** validate — `logit_scale` may be 0, negative, or NaN.
    /// Production paths should use [`Calibration::from_path`] or
    /// [`Calibration::from_bytes`], which both run the validation pipeline.
    /// If a `Calibration` built via `new` is passed to
    /// [`Siglip2::from_parts`], that constructor re-runs validation, so
    /// the unchecked path can't reach `classify` undetected.
    pub const fn new(logit_scale: f32, logit_bias: f32) -> Self {
        Self { logit_scale, logit_bias }
    }

    pub fn logit_scale(&self) -> f32 { self.logit_scale }
    pub fn logit_bias(&self)  -> f32 { self.logit_bias }

    /// Parses and validates the JSON. Validation rejects:
    /// - non-finite `logit_scale` or `logit_bias` (NaN, ±∞) →
    ///   `Error::InvalidCalibration { reason: "logit_scale is not finite" | "logit_bias is not finite" }`
    /// - non-positive `logit_scale` →
    ///   `Error::InvalidCalibration { reason: "logit_scale must be positive" }`
    /// `logit_bias` may be negative (the actual export is ≈ -16.78); only finite-ness is required.
    pub fn from_path(path: &Path) -> Result<Self>;

    /// Path-less variant. Parse errors surface as
    /// `Error::LoadCalibration { path: None, source }`; validation errors as
    /// `Error::InvalidCalibration` (same reasons as `from_path`).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self>;
}

// Private. Used internally by from_path/from_bytes and never exposed.
#[derive(serde::Deserialize)]
struct CalibrationRaw {
    logit_scale: f32,
    logit_bias:  f32,
}

impl Calibration {
    /// Crate-internal — called by `from_path` / `from_bytes` and re-run by
    /// `Siglip2::from_parts` to close the unchecked-`new` gap.
    pub(crate) fn validate(logit_scale: f32, logit_bias: f32) -> Result<Self> {
        if !logit_scale.is_finite() {
            return Err(Error::InvalidCalibration { reason: "logit_scale is not finite" });
        }
        if logit_scale <= 0.0 {
            return Err(Error::InvalidCalibration { reason: "logit_scale must be positive" });
        }
        if !logit_bias.is_finite() {
            return Err(Error::InvalidCalibration { reason: "logit_bias is not finite" });
        }
        Ok(Self { logit_scale, logit_bias })
    }
}
```

Calibration is a value object — no `with_*` / `set_*` methods, no public `Serialize` /
`Deserialize`. The serde-bypass risk is the same one as for `Embedding`: a public auto-derived
`Deserialize` would let a corrupted or hand-edited JSON file produce a `Calibration` with NaN or
negative scale that silently poisons every `classify` score. By keeping the public type without
serde derives and routing JSON parsing through `CalibrationRaw` + `validate`, every construction
path has identical guarantees.

If a caller genuinely wants to serialize a `Calibration` for cross-process transport, they can
build a JSON object explicitly (e.g.
`serde_json::json!({ "logit_scale": cal.logit_scale(), "logit_bias": cal.logit_bias() })`) and
reconstruct via `Calibration::from_bytes`, which validates.

## 6. Image resize policy

Image resize is pinned to `image::imageops::resize(..., FilterType::Triangle)` — the same kernel
the existing service uses to hit the 0.99917 cosine floor. `Triangle` is `image`'s pure-f32
bilinear filter, empirically close to PIL `BILINEAR`.

**Do not switch resize kernels.** Faster libraries (`fast_image_resize`, `libvips`) use
fixed-point i16/i32 weights and may differ in antialiasing convention; either change re-opens the
parity question against the validated 99-frame corpus. If a future revision wants
`fast_image_resize` for performance, it must (a) be feature-flagged off-by-default, (b) regenerate
the golden fixtures from the upstream PyTorch processor, and (c) re-prove ≥0.99917 cosine in CI.

The patchify-and-normalize inner loop after resize is scalar in 0.1.0; SIMD acceleration of that
step is a follow-up.

## 7. Model loading

### 7.1 ONNX with external-data sidecar (vision and text)

Both vision and text graphs are external-data exports. The single supported public path is
`from_files(graph_path)`: ORT auto-discovers the `.onnx.data` sidecar by relative filename in the
same directory.

- `from_files(graph)` → `ort::Session::builder().commit_from_file(graph)`. ORT reads the sidecar
  from the same directory automatically. Caller MUST keep both files together; this is documented
  in rustdoc and we do not validate it (ORT's error already names both files).
- `from_ort_session(session)` (single-encoder) /
  `Siglip2::from_parts(image, text, tokenizer, calibration)` → caller-built sessions, for custom
  execution-provider tuning.

There is **no `from_memory` for ONNX in 0.1.0.** The `ort = "2.0.0-rc.12"` Rust binding does not
expose a public way to bind external initializer data from memory (the underlying ORT C API has
`AddExternalInitializersFromFilesInMemory` and `AddExternalInitializerFromArray`, but the Rust
crate does not surface either as a stable safe API). The temp-file workaround is unsafe under
concurrent loads (two threads writing the same temp filename race) and inappropriate for a
public-API guarantee. When a future ORT release stabilizes a safe in-memory path, `from_memory`
returns as a non-breaking minor.

### 7.2 Tokenizer

- `from_files(graph, tokenizer)` — `tokenizers::Tokenizer::from_file(tokenizer)`.
- `bundled(graph)` (gated on `bundled`) — `Tokenizer::from_bytes(BUNDLED_TOKENIZER)`.

The tokenizer is a single self-contained file with no external sidecar, so `from_memory` for the
tokenizer side is well-defined and remains available implicitly via `Tokenizer::from_bytes` in the
caller's own code; it is not exposed as a separate public constructor.

### 7.3 Memory & threading model

Each `ImageEncoder` / `TextEncoder` owns its own `ort::Session`. Mutable encoders mean callers
serialize calls per encoder; for parallelism, instantiate one encoder per worker thread.

Per-worker resident memory at f32 weights, approximate:

| asset | resident |
|---|---|
| vision weights (`.onnx.data`) | 358 MB |
| text weights (`.onnx.data`)   | 1.08 GB |
| tokenizer                     | 32 MB |
| ORT runtime overhead          | ~100 MB |
| **total per worker**          | **~1.55 GB** |

On an 8-core thread-per-core deployment that is ~12 GB of weights resident. Deployments with tight
memory budgets should consider sharing one `ImageEncoder` and one `TextEncoder` across worker
threads with explicit serialization (a `Mutex<Siglip2>` in the caller's pool), trading throughput
for footprint.

`ThreadOptions` (§3.6) defaults `intra_threads = 1` and `parallel_execution = false` to prevent
ORT's intra-op thread pool from oversubscribing the CPU when N outer worker threads each own an
encoder.

## 8. Testing

### 8.1 Integration tests

Gated on the `SIGLIP2_MODELS_DIR` env var. When unset, integration tests are marked
`#[ignore]` (cargo reports them honestly as ignored, not falsely as passed).

When set, the directory must contain the runtime-mandatory artifacts:

```
$SIGLIP2_MODELS_DIR/
  vision_model_naflex_256.onnx
  vision_model_naflex_256.onnx.data
  text_model_naflex.onnx
  text_model_naflex.onnx.data
  tokenizer.json
  calibration.json
```

Tests verify the SHA256 of each file against `tests/fixtures/MODELS.sha256` before running, and
skip with a clear message on mismatch. `MODELS.sha256` is the **single source of truth for
runtime-asset checksums** — it lists hashes for every runtime-mandatory file in
`$SIGLIP2_MODELS_DIR` *and* for the bundled `models/tokenizer.json`, so callers using the `bundled`
feature have one place to verify the bytes they ship match the bytes pinned in the release.

`MODELS.sha256` lives under `tests/fixtures/` and ships in the GitHub repo for development and CI;
it is **not** in the crate's `include` list, so crates.io users do not see it. That's intentional:
the bundled tokenizer bytes ship in the tarball directly, so tarball users trust them by reference,
not by hash. The hash file exists for the development workflow (running integration tests against
a freshly downloaded `$SIGLIP2_MODELS_DIR`) and for CI's bundled-tokenizer drift detection.

### 8.2 Golden fixtures

Under `tests/fixtures/`:

- 10–20 RGB images of varying aspect ratios (square, wide, tall, extreme widescreen) saved as
  lossless `.png`, sourced from the same kind of stock-footage keyframes the upstream export was
  validated against.
- For each image, the corresponding 768-dim PyTorch-reference embedding saved as `.npy` (read at
  test time via `npyz`).
- Per-image assertion: cosine similarity between `siglip2`'s output and the reference is ≥ 0.99917.
- **Parity gate runs at both `Level1` and `Level3`.** The default is `Level1` (§3.6), but CI
  re-runs every fixture under `GraphOptimizationLevel::Level3` as well. Both passes must clear the
  0.99917 floor before a release ships. This catches regressions where a future ORT release alters
  Level3's fusion list and silently widens the parity gap on the SigLIP2 vision tower's
  LayerNorm/MatMul subgraphs.

For text fixtures, **multilingual coverage is required**: at least 10 prompts spanning English,
Chinese, and Japanese (matching the upstream verify script's coverage), each with a PyTorch-reference
embedding. SigLIP2 was trained multilingually and an English-only fixture set would miss tokenizer-
specific bugs (see §5.1 — the multilingual fixture set is the parity net for Rust↔Python tokenizer
divergence).

### 8.3 Cross-modal sanity

A handful of (image, matching text, distractor text) triples; assert
`cos(image, matching) > cos(image, distractor)` strictly. Catches L2-normalization sign errors and
patch-ordering bugs that golden-fixture cosines would still pass. Includes at least one triple
where the matching text is in Chinese or Japanese.

### 8.4 NaFlex preprocessing tests

- **Reference table (§4.1).** For each row, assert `(H_p, W_p)` and `(H_res, W_res)` match exactly.
- **Property tests** over a coarse grid of `(H, W)` (random + corner cases like `1×1`, `1×2048`,
  `2048×1`, `4096×4096`):
  - `H_p * W_p ≤ 256` and `H_p, W_p ≥ 1`.
  - The number of `1`s in `pixel_attention_mask` equals `H_p * W_p`.
  - The first `H_p * W_p` rows of `pixel_values` are not identically zero (assuming non-zero input);
    rows `H_p*W_p ..` are exactly zero.
- **Patch-byte-order test.** A constructed image whose pixel values are a known per-channel pattern
  (e.g. R=0, G=128, B=255 in stripes) has its `pixel_values[0, 0, ..]` byte sequence asserted to be
  the explicit interleaved (row, col, channel) ordering — protects against silent byte-layout
  regressions of the kind described in §4.2 step 3.
- **Trait-bound assertions.** A non-test module compiles a stub function whose body
  is `fn req<T: Send + Sync>() {}` followed by `req::<Preprocessor>()`,
  `req::<Embedding>()`, `req::<Calibration>()`. If a future field breaks any
  of those auto-derives, the build fails immediately with a clear pointer at
  the regressing trait — cheaper than discovering it at first multi-threaded
  deploy.

### 8.5 Benchmarks (`benches/`)

Criterion harness, mirroring textclap:

- `bench_naflex` — pure preprocessing, varying input sizes.
- `bench_image_encode` — full preprocessing + ONNX, single image and batch=8.
- `bench_text_encode` — tokenize + ONNX, single text and batch=8.

Benches require `SIGLIP2_MODELS_DIR` and exit with a clear message when unset.

## 9. Crate layout & feature flags

```
src/
  lib.rs           # public re-exports, BUNDLED_TOKENIZER (feature-gated)
  siglip2.rs       # Siglip2 wrapper
  image_enc.rs     # ImageEncoder
  text_enc.rs      # TextEncoder
  preproc/
    mod.rs         # Preprocessor (public)
    naflex.rs      # private: patch-grid sizing, resize, normalize, patchify, mask, spatial_shapes
  embedding.rs     # Embedding, LabeledScore[Owned], TryFrom<Vec<f32>>
  calibration.rs   # Calibration, JSON loading
  error.rs         # Error, Result
  options.rs       # Options, BatchOptions, ThreadOptions, GraphOptimizationLevel
models/
  tokenizer.json   # text-tower tokenizer; bundled when `bundled` is on (~33 MB)
  MODELS.md        # provenance documentation (no checksums; see tests/fixtures/MODELS.sha256)
THIRD_PARTY_NOTICES.md  # Apache-2.0 attribution for bundled tokenizer.json
examples/
  embed_keyframes.rs    # disk JPEGs → ImageEncoder → Vec<Embedding>
  index_and_search.rs   # parallel to textclap's, sketches caller-side lancedb usage
benches/
  bench_naflex.rs
  bench_image_encode.rs
  bench_text_encode.rs
tests/
  integration.rs        # gated on SIGLIP2_MODELS_DIR
  fixtures/
```

`Cargo.toml`:

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
# `Embedding` no longer derives `Deserialize` (§3.5 — would bypass the
# L2-norm and dim invariants), so the `rc` feature is not needed: the
# only types this crate deserializes are the private `CalibrationRaw`
# struct and downstream user types. `derive` covers everything else.
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
# on `LabeledScore` (Serialize only — borrowed) and `LabeledScoreOwned`
# (both). `Embedding` and `Calibration` deliberately do NOT carry the
# cfg_attr derives — see §3.5 and §5.3 for the bypass-risk reasoning.
# `smol_str/serde` activates that crate's serde derives (smol_str is a
# hard dep, so the `?` syntax is unnecessary).
serde       = ["smol_str/serde"]

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
```

`serde` and `serde_json` are now hard dependencies (not optional) because `Calibration::from_path`
must always be able to parse `calibration.json`. `image` and `bundled` are *on by
default* — first-time users running `cargo add siglip2` get a working `Siglip2::bundled` without
hunting for a tokenizer file. Callers who need a minimum-footprint build use
`default-features = false`.

Execution providers (CUDA, CoreML, etc.) are not enabled by default. Users who want GPU acceleration
must enable the appropriate `ort` features in their own `Cargo.toml` and pass a custom session via
`from_ort_session` / `Siglip2::from_parts`. This is documented prominently in the README;
**ANE-on-Mac surprises will land otherwise.**

**Crate tarball size.** The bundled `models/tokenizer.json` is included in the crate tarball
unconditionally (the `include` list is feature-agnostic), so `cargo package` produces a ~33 MB
tarball regardless of whether `bundled` is on. Most users will keep the default features,
so this is the right trade-off; the `CHANGELOG.md` calls it out for users who turn the feature off
and may be surprised by the tarball size.

**Third-party attribution.** `models/tokenizer.json` is derived from
`google/siglip2-base-patch16-naflex` (Apache-2.0). `THIRD_PARTY_NOTICES.md` carries the required
attribution; the file is shipped in the crate tarball via the `include` list above.

## 10. Out of scope (deliberate, restated)

- **Storage layer.** `lancedb` and any other vector-store integration are caller-owned. The
  `index_and_search` example sketches the caller-side wiring but does not pull `lancedb` into the
  crate's `[dependencies]`.
- **Image decoding** beyond the optional `embed_path` helper (feature `decoders`; JPEG and PNG only).
- **CLI binary.**
- **Quantized / int8 export.** Note that `calibration.json` is *not* an int8-quantization artifact —
  it is the trained sigmoid head's logit-scale and bias and is loaded at runtime for `classify`
  (§5.3).
- **Re-exports at non-256 NaFlex patch budgets.**
- **SIMD acceleration of the patchify-and-normalize inner loop** (scalar only in 0.1.0).
- **Multi-variant SigLIP2 support** (large, so400m). The `Arc<[f32]>` embedding shape leaves room
  for a future minor (slice length, not a fixed-size array).
- **In-memory ONNX construction for the vision/text towers** — see §7.1 and §11.
- **Async support.**

## 11. Open implementation questions (for the writing-plans phase)

These are not blockers for the spec; they're items the implementation plan must answer up front:

1. **Confirm `text_model_naflex.onnx` `seq_len` axis.** Expected static at 64; the implementation
   plan's first integration test loads the graph and asserts the shape. If dynamic, document and
   keep `seq_len = 64` as a runtime constant.
2. **Confirm pad-token id and padding strategy.** Use `tokenizers::Tokenizer::pad_token_id()` if
   exposed; otherwise read it from `tokenizer.json` directly. Validate that `Fixed(64)` padding
   produces the same byte-for-byte tensor as the upstream verify script for a fixed prompt set.
3. **Reference-table regeneration.** The §4.1 table values are derived by hand and must be
   regenerated by running the upstream `Siglip2ImageProcessorFast` on each input row. Pin the
   transformers commit used for regeneration in `tests/fixtures/MODELS.md`.
4. **Calibration sanity.** Add a one-off test that loads `calibration.json`, computes
   `sigmoid(scale · 0 + bias)` and asserts it equals `5.174e-8` to within 1% relative
   tolerance (`assert!((s - 5.174e-8).abs() / 5.174e-8 < 1e-2)`). This catches both
   parsing errors and the more common bug of forgetting to apply the bias. Verified:
   `sigmoid(-16.776988983154297) = 5.174236e-8` (Python f64).
5. **ORT `intra_threads = 0` semantics.** ORT treats `0` as "auto"; we forbid that default but
   should let callers opt in by passing 0 explicitly, surfacing it as a `Some(usize)` /
   `Auto` enum if API clarity warrants it. Decide during implementation.
