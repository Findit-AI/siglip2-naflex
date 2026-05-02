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

### Parity-against-reference (CI-enforced when secret is configured)

The CI workflow at `.github/workflows/parity.yml` is **two-stage**:

1. `model-load-smoke` — proves the runtime can load the released
   ONNX + tokenizer + calibration. Requires the `FINDIT_INDEXER_TOKEN`
   repo secret (PAT with read access to `Findit-AI/indexer`). **NOT a
   parity gate.**
2. `parity-against-pytorch` — proves cosine-floor parity (≥ 0.99917)
   against precomputed PyTorch reference embeddings under
   `tests/fixtures/`. Requires the same secret. Fixtures are committed
   in-tree (12 synthetic keyframes spanning aspect ratios + 12
   multilingual prompts), so this gate runs and enforces parity on
   every push and PR in any environment that has the secret.

The fixtures are reproducible end-to-end via
`scripts/generate_synthetic_keyframes.py` and
`scripts/generate_parity_fixtures.py`; see `tests/fixtures/README.md`.
Forks without `FINDIT_INDEXER_TOKEN` skip the workflow entirely
(forks are not expected to hold the release-repo PAT).

### Notes for users disabling default features

The crate tarball is ~33 MB regardless of whether the `bundled` feature
is enabled; the bundled `tokenizer.json` is included unconditionally so
the package contents match the source tree. Disabling `bundled` only
removes the `BUNDLED_TOKENIZER` constant from the public API.
