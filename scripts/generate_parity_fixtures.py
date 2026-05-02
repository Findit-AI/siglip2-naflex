#!/usr/bin/env python3
"""Generate the SigLIP2 NaFlex parity fixtures used by the integration tests.

Background
----------
The `image_parity_against_pytorch_reference` and
`text_parity_against_pytorch_reference` tests in `tests/integration.rs`
compare this crate's encoder outputs to PyTorch reference embeddings at
a 0.99917 cosine floor. Those reference embeddings are not in-tree —
generating them requires running the upstream `transformers` SigLIP2
model. This script is the canonical one-shot recipe.

Run once per upstream model bump (or whenever the in-tree reference
should be regenerated). Commit the resulting `tests/fixtures/`
artifacts. The CI parity workflow (`.github/workflows/parity.yml`)
fails closed until those artifacts exist.

Inputs
------
- `transformers >= 4.X`, `torch`, `pillow`, `numpy` installed
- 10–20 representative RGB images on disk (pass via `--images-dir`)
- 10+ multilingual text prompts (English, Chinese, Japanese — see spec
  §5.1; pass via `--prompts-json` pointing at a list)
- network access to `huggingface.co` for the upstream SigLIP2 model
  (`google/siglip2-base-patch16-naflex`)

Outputs
-------
- `tests/fixtures/images/<stem>.png` (lossless re-encode of each input)
- `tests/fixtures/embeddings/<stem>.npy` (768-dim float32 reference)
- `tests/fixtures/text_prompts.json` (the prompt list, copied verbatim)
- `tests/fixtures/text_embeddings.npy` (`[N, 768]` float32 array, one
  row per prompt in order)

Usage
-----
::

    pip install transformers torch pillow numpy
    python scripts/generate_parity_fixtures.py \\
        --images-dir path/to/raw_keyframes/ \\
        --prompts-json path/to/prompts.json \\
        --out-dir tests/fixtures/

Tolerance
---------
The crate's parity tests assert per-fixture cosine ≥ 0.99917 against
these references — the same floor the upstream `Findit-AI/indexer`
release validated. If a future re-export tightens or loosens that
floor, update both this script's reference and the cosine assertion in
`tests/integration.rs` together.

This script is intentionally simple: it has no test coverage, no error
recovery beyond a clean exit on missing dependencies, and runs single-
threaded against CPU. It exists to be run once per upstream bump, not
to be a production tool.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# These imports are deferred to give a clean error if the user runs the
# script without the upstream toolchain installed.
try:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
except ImportError as e:
    sys.stderr.write(
        f"Missing dependency: {e}.\n"
        "Install with: pip install transformers torch pillow numpy\n"
    )
    sys.exit(1)


MODEL_ID = "google/siglip2-base-patch16-naflex"
EMBEDDING_DIM = 768


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate SigLIP2 NaFlex parity fixtures for tests/fixtures/"
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory of representative .png/.jpg keyframes (10–20 recommended; "
             "use a mix of square, wide, tall, and extreme-aspect-ratio frames).",
    )
    p.add_argument(
        "--prompts-json",
        type=Path,
        required=True,
        help="JSON file containing a list of multilingual prompts (English, "
             "Chinese, Japanese; 10+ recommended per spec §5.1).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/fixtures"),
        help="Where to write the fixtures (default: tests/fixtures).",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Cap on the number of images encoded (default: 20).",
    )
    return p.parse_args()


def load_image_paths(images_dir: Path, cap: int) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg"}
    paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not paths:
        sys.stderr.write(f"No images found under {images_dir}\n")
        sys.exit(1)
    return paths[:cap]


def main() -> int:
    args = parse_args()
    out = args.out_dir
    images_out = out / "images"
    embeddings_out = out / "embeddings"
    images_out.mkdir(parents=True, exist_ok=True)
    embeddings_out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} (this downloads ~600 MB on first run)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval()

    # ---- Image fixtures ----
    image_paths = load_image_paths(args.images_dir, args.max_images)
    print(f"Encoding {len(image_paths)} images...")
    for src in image_paths:
        stem = src.stem
        # Copy to fixtures dir as a lossless PNG (re-encode if input
        # was JPEG to ensure deterministic per-pixel state).
        dst_image = images_out / f"{stem}.png"
        if src.suffix.lower() == ".png":
            shutil.copy2(src, dst_image)
        else:
            Image.open(src).convert("RGB").save(dst_image, "PNG")

        with torch.no_grad():
            inputs = processor(images=Image.open(dst_image).convert("RGB"), return_tensors="pt")
            features = model.get_image_features(**inputs)
            # transformers >= 5 returns BaseModelOutputWithPooling; older
            # releases return the pooler tensor directly.
            pooled = features.pooler_output if hasattr(features, "pooler_output") else features
            embedding = torch.nn.functional.normalize(pooled, dim=-1)
            embedding_np = embedding.squeeze(0).cpu().numpy().astype(np.float32)

        assert embedding_np.shape == (EMBEDDING_DIM,), (
            f"unexpected embedding shape for {stem}: {embedding_np.shape}"
        )
        np.save(embeddings_out / f"{stem}.npy", embedding_np)
        print(f"  {stem}: shape={embedding_np.shape} norm={np.linalg.norm(embedding_np):.6f}")

    # ---- Text fixtures ----
    prompts: list[str] = json.loads(args.prompts_json.read_text())
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        sys.stderr.write(f"{args.prompts_json} must be a JSON list of strings\n")
        return 1
    print(f"Encoding {len(prompts)} prompts...")
    with torch.no_grad():
        text_inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        text_features = model.get_text_features(**text_inputs)
        text_pooled = (
            text_features.pooler_output if hasattr(text_features, "pooler_output") else text_features
        )
        text_embeddings = torch.nn.functional.normalize(text_pooled, dim=-1)
        text_embeddings_np = text_embeddings.cpu().numpy().astype(np.float32)

    assert text_embeddings_np.shape == (len(prompts), EMBEDDING_DIM), (
        f"unexpected text embeddings shape: {text_embeddings_np.shape}"
    )
    np.save(out / "text_embeddings.npy", text_embeddings_np)
    (out / "text_prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
    print(f"  text_embeddings: shape={text_embeddings_np.shape}")

    print(
        f"\nDone. {len(image_paths)} image fixtures + {len(prompts)} text fixtures "
        f"written to {out}/"
    )
    print(
        "Next steps: commit `tests/fixtures/` and verify "
        "`cargo test --all-features --test integration -- --ignored` passes "
        "with `SIGLIP2_MODELS_DIR` set."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
