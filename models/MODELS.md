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
SIGLIP2_MODELS_DIR=models/siglip2 cargo test --all-features -- --ignored
```

Without `SIGLIP2_MODELS_DIR`, integration tests are marked `#[ignore]` and
skipped by default.
