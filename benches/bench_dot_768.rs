//! Microbenchmark for the cosine hot path: `dot_768` scalar vs the
//! best SIMD backend the dispatcher selects on the host.
//!
//! Hits two functions exposed for benchmarks:
//! - `siglip2_naflex::__bench_internal::dot_768_scalar` — four-accumulator
//!   scalar reference (what runs on architectures without a SIMD
//!   backend).
//! - `siglip2_naflex::__bench_internal::dot_768_dispatch` — runtime-selected
//!   backend (NEON on aarch64, AVX2+FMA on x86_64, scalar elsewhere).
//!   This is the same path `Embedding::cosine` calls.
//!
//! Run: `cargo bench --bench bench_dot_768`. Each iteration computes
//! exactly one 768-dim dot product, so the criterion µs/iter number is
//! the per-cosine cost — multiply by N to estimate retrieval over N
//! candidates.
//!
//! To force the scalar fallback through the dispatcher (useful for
//! verifying the runtime detection path itself isn't where the cost
//! is): `RUSTFLAGS='--cfg siglip2_force_scalar' cargo bench --bench bench_dot_768`.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn deterministic_pair() -> (Vec<f32>, Vec<f32>) {
  // Pseudo-random values squashed to roughly [-1, 1]. Identical to the
  // generator in the SIMD module's tests so bench inputs match what
  // those tests exercise.
  let mut a = Vec::with_capacity(768);
  let mut b = Vec::with_capacity(768);
  for i in 0..768u32 {
    let xa = i.wrapping_mul(2_654_435_761) as f32;
    let xb = i.wrapping_mul(40_503).wrapping_add(13) as f32;
    a.push((xa * 1e-9).sin());
    b.push((xb * 1e-9).cos());
  }
  (a, b)
}

fn bench_dot_768(c: &mut Criterion) {
  let (a, b) = deterministic_pair();

  let mut group = c.benchmark_group("dot_768");
  group.bench_function("scalar", |bencher| {
    bencher.iter(|| {
      let r = siglip2_naflex::__bench_internal::dot_768_scalar(black_box(&a), black_box(&b));
      black_box(r);
    });
  });
  group.bench_function("dispatch", |bencher| {
    bencher.iter(|| {
      let r = siglip2_naflex::__bench_internal::dot_768_dispatch(black_box(&a), black_box(&b));
      black_box(r);
    });
  });
  group.finish();
}

fn bench_normalize_patchify_row(c: &mut Criterion) {
  // 16-pixel RGB row = 48 contiguous bytes per call.
  let src: Vec<u8> = (0..48).map(|i| (i as u8).wrapping_mul(17)).collect();
  let mut dst = vec![0.0f32; 48];

  let mut group = c.benchmark_group("normalize_patchify_row");
  group.bench_function("scalar", |bencher| {
    bencher.iter(|| {
      siglip2_naflex::__bench_internal::normalize_patchify_row_scalar(black_box(&src), &mut dst);
      black_box(&dst);
    });
  });
  group.bench_function("dispatch", |bencher| {
    bencher.iter(|| {
      siglip2_naflex::__bench_internal::normalize_patchify_row_dispatch(black_box(&src), &mut dst);
      black_box(&dst);
    });
  });
  group.finish();
}

fn bench_scale_768_inplace(c: &mut Criterion) {
  let template: Vec<f32> = (0..768).map(|i| ((i as f32) * 0.001).sin()).collect();

  let mut group = c.benchmark_group("scale_768_inplace");
  group.bench_function("scalar", |bencher| {
    bencher.iter_batched_ref(
      || template.clone(),
      |v| {
        siglip2_naflex::__bench_internal::scale_768_inplace_scalar(v, 0.5);
        black_box(&v);
      },
      criterion::BatchSize::SmallInput,
    );
  });
  group.bench_function("dispatch", |bencher| {
    bencher.iter_batched_ref(
      || template.clone(),
      |v| {
        siglip2_naflex::__bench_internal::scale_768_inplace_dispatch(v, 0.5);
        black_box(&v);
      },
      criterion::BatchSize::SmallInput,
    );
  });
  group.finish();
}

criterion_group!(
  benches,
  bench_dot_768,
  bench_normalize_patchify_row,
  bench_scale_768_inplace
);
criterion_main!(benches);
