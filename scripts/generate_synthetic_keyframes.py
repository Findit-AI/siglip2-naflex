#!/usr/bin/env python3
"""Generate procedurally-varied synthetic keyframes for the parity-fixture pipeline.

Why this exists
---------------
`scripts/generate_parity_fixtures.py` requires `--images-dir <dir>` pointing
at 10-20 representative PNG keyframes. The upstream SigLIP2 parity validation
set is not redistributable, and depending on private corpora makes the
fixture set non-reproducible. This script produces a deterministic synthetic
substitute that spans the aspect ratios NaFlex preprocessing actually
exercises (square, wide, tall, extreme), so anyone can regenerate the
parity fixtures end-to-end with the in-tree tooling.

Run once to populate an inputs directory; pass that directory to
`generate_parity_fixtures.py --images-dir`.

Usage
-----
::

    python scripts/generate_synthetic_keyframes.py --out-dir /tmp/keyframes
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError as e:  # pragma: no cover
    sys.stderr.write(f"Missing dependency: {e}.\nInstall with: pip install pillow\n")
    sys.exit(1)


# Twelve canonical sizes that span the aspect-ratio space NaFlex resizes
# through. Square, common landscape/portrait, extreme aspects, and small
# sizes that already fit the patch budget without a resize step.
SIZES = [
    ("01_square_512", 512, 512),
    ("02_landscape_640x480", 640, 480),
    ("03_landscape_1280x720", 1280, 720),
    ("04_portrait_480x640", 480, 640),
    ("05_portrait_720x1280", 720, 1280),
    ("06_wide_1920x720", 1920, 720),
    ("07_tall_360x1280", 360, 1280),
    ("08_extreme_wide_1600x400", 1600, 400),
    ("09_extreme_tall_400x1600", 400, 1600),
    ("10_small_224x224", 224, 224),
    ("11_medium_900x600", 900, 600),
    ("12_landscape_2k_2048x1080", 2048, 1080),
]


# Twelve multilingual prompts (English + Japanese + simplified Chinese) that
# cover photo, screenshot, diagram, and abstract content; the multilingual
# subset exercises the SentencePiece tokenizer's non-ASCII paths.
PROMPTS = [
    "a photograph of a sunset over the ocean",
    "a screenshot of source code on a dark background",
    "a close-up of a flower with morning dew",
    "an aerial view of a city at night",
    "a street scene with people walking",
    "a diagram showing a neural network architecture",
    "海辺の夕日の写真",
    "夜の都市の航空写真",
    "源代码的屏幕截图",
    "盛开的花朵特写",
    "a cat sitting on a windowsill",
    "abstract geometric shapes in primary colors",
]


def gradient(w: int, h: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    c1 = (rng.randint(40, 220), rng.randint(40, 220), rng.randint(40, 220))
    c2 = (rng.randint(40, 220), rng.randint(40, 220), rng.randint(40, 220))
    img = Image.new("RGB", (w, h), c1)
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        c = tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))
        draw.line([(0, y), (w, y)], fill=c)
    rng2 = random.Random(seed + 17)
    for _ in range(8):
        x0, y0 = rng2.randint(0, w - 1), rng2.randint(0, h - 1)
        x1, y1 = rng2.randint(x0, w), rng2.randint(y0, h)
        col = (rng2.randint(0, 255), rng2.randint(0, 255), rng2.randint(0, 255))
        if rng2.random() < 0.5:
            draw.rectangle([x0, y0, x1, y1], outline=col, width=2)
        else:
            draw.ellipse([x0, y0, x1, y1], outline=col, width=2)
    return img


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, w, h) in enumerate(SIZES):
        img = gradient(w, h, seed=i + 1)
        if i % 3 == 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        out = img_dir / f"{name}.png"
        img.save(out, "PNG", optimize=True)
        print(f"  {out.name}: {w}x{h} ({out.stat().st_size} bytes)")
    prompts_path = args.out_dir / "prompts.json"
    prompts_path.write_text(json.dumps(PROMPTS, ensure_ascii=False, indent=2))
    print(f"  {prompts_path.name}: {len(PROMPTS)} prompts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
