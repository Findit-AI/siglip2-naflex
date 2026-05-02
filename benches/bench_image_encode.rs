//! End-to-end image encode (preprocess + ORT). Requires SIGLIP2_MODELS_DIR.

use std::{hint::black_box, path::PathBuf};

use criterion::{Criterion, criterion_group, criterion_main};
use siglip2_naflex::{ImageEncoder, ImageView};

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
