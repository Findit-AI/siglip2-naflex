//! End-to-end text encode (tokenize + ORT). Requires SIGLIP2_MODELS_DIR.

use std::{hint::black_box, path::PathBuf};

use criterion::{Criterion, criterion_group, criterion_main};
use siglip2_naflex::TextEncoder;

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
