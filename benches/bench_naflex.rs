//! NaFlex preprocessing throughput. Standalone — does not require ORT models.
use criterion::{Criterion, criterion_group, criterion_main};
use siglip2_naflex::{ImageView, Options, Preprocessor};
use std::hint::black_box;

fn bench_naflex(c: &mut Criterion) {
  let opts = Options::default();
  let pre = Preprocessor::new(opts).unwrap();

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
      pre
        .preprocess_into(black_box(view), &mut pv, &mut am, &mut ss)
        .unwrap();
    });
  });
}

criterion_group!(benches, bench_naflex);
criterion_main!(benches);
