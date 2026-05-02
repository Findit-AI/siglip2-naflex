//! Direct ort A/B between the EP stack siglip2 would register today
//! and a strict CPU-only session. Mirrors `crate::session::
//! collect_execution_providers`'s cfg ladder: with no GPU feature
//! active the two stacks are identical (CPU); with `--features coreml`
//! / `cuda` / `directml` / `tensorrt` / `rocm` the corresponding EP
//! gets registered ahead of the implicit CPU fallback.
//!
//! The benchmark bypasses `siglip2::ImageEncoder` so it works against
//! whatever NaFlex export you have on disk — including the static-
//! batch=1 release build that `validate_image_session` rejects today.
//!
//! Usage:
//!
//! ```bash
//! # CPU-only baseline (default — no GPU feature active):
//! SIGLIP2_VISION_GRAPH=/path/to/vision_model_naflex_256.onnx \
//!     cargo run --release --example bench_ep
//!
//! # With CoreML on macOS:
//! SIGLIP2_VISION_GRAPH=/path/to/vision_model_naflex_256.onnx \
//!     cargo run --release --example bench_ep --features coreml
//! ```
//!
//! Reports cold (first inference after each session is built) and
//! warm (median of 30 subsequent runs) latencies for both stacks.

use std::{
  path::{Path, PathBuf},
  time::{Duration, Instant},
};

use ort::{
  ep::ExecutionProviderDispatch,
  session::{Session, builder::GraphOptimizationLevel},
  value::TensorRef,
};

const WARM_RUNS: usize = 30;
const MAX_PATCHES: usize = 256;
const PATCH_FEATURES: usize = 3 * 16 * 16; // 768 (channels * P * P)
// Override via SIGLIP2_BATCH=N to test scaling behaviour (e.g.
// N=8 to see if CoreML wins at larger batches).
fn batch_size() -> usize {
  std::env::var("SIGLIP2_BATCH")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(1)
}

fn main() {
  let graph = match std::env::var_os("SIGLIP2_VISION_GRAPH") {
    Some(v) => PathBuf::from(v),
    None => {
      eprintln!("error: set SIGLIP2_VISION_GRAPH to the vision_model ONNX path");
      std::process::exit(1);
    }
  };

  let batch = batch_size();
  println!("=== siglip2 EP A/B benchmark (vision tower, batch={batch}) ===");
  println!("graph: {}", graph.display());
  println!("default-stack EPs on this host: {}", default_stack_label());
  println!();

  // ---- Variant B (run first as a sanity check): CPU-only ------------
  let mut sess_cpu = match build_session(&graph, vec![]) {
    Ok(s) => s,
    Err(e) => {
      eprintln!("CPU-only session build failed: {e}");
      std::process::exit(2);
    }
  };
  let (cold_b, warm_b) = measure(&mut sess_cpu, "CPU-only", batch);

  // ---- Variant A: default stack -------------------------------------
  // CoreML on macOS, CPU elsewhere. Same registration order as
  // `crate::session::collect_execution_providers` in production.
  let providers: Vec<ExecutionProviderDispatch> = default_providers();
  let mut sess_default = match build_session(&graph, providers) {
    Ok(s) => s,
    Err(e) => {
      eprintln!("default session build failed: {e}");
      std::process::exit(2);
    }
  };
  let stack_label = format!("default ({})", default_stack_label());
  let (cold_a, warm_a) = measure(&mut sess_default, &stack_label, batch);

  println!();
  println!("--- summary ---");
  println!(
    "cold (first call):   default {:>7.2} ms   CPU-only {:>7.2} ms   Δ {:+7.2} ms",
    ms(cold_a),
    ms(cold_b),
    ms(cold_a) - ms(cold_b)
  );
  println!(
    "warm (median × {WARM_RUNS}): default {:>7.2} ms   CPU-only {:>7.2} ms   Δ {:+7.2} ms",
    ms(warm_a),
    ms(warm_b),
    ms(warm_a) - ms(warm_b),
  );
  println!();
  let speedup = warm_b.as_secs_f64() / warm_a.as_secs_f64();
  if speedup >= 1.0 {
    println!("warm speedup of default vs CPU-only: {speedup:.2}x");
  } else {
    println!(
      "warm SLOWDOWN of default vs CPU-only: {:.2}x ({:.0}% of CPU baseline)",
      1.0 / speedup,
      speedup * 100.0
    );
  }
}

/// Mirror `crate::session::collect_execution_providers` so the bench
/// reflects exactly what the crate would register at session build.
#[allow(clippy::vec_init_then_push)]
fn default_providers() -> Vec<ExecutionProviderDispatch> {
  #[allow(unused_mut)]
  let mut providers: Vec<ExecutionProviderDispatch> = Vec::new();
  #[cfg(feature = "tensorrt")]
  providers.push(ort::ep::TensorRT::default().build());
  #[cfg(feature = "cuda")]
  providers.push(ort::ep::CUDA::default().build());
  #[cfg(feature = "directml")]
  providers.push(ort::ep::DirectML::default().build());
  #[cfg(feature = "rocm")]
  providers.push(ort::ep::ROCm::default().build());
  #[cfg(feature = "coreml")]
  providers.push(ort::ep::CoreML::default().build());
  providers
}

/// Human-readable label of the default stack under the current
/// feature configuration. Matches the EP order in `default_providers`.
fn default_stack_label() -> String {
  let mut parts = Vec::new();
  if cfg!(feature = "tensorrt") {
    parts.push("TensorRT");
  }
  if cfg!(feature = "cuda") {
    parts.push("CUDA");
  }
  if cfg!(feature = "directml") {
    parts.push("DirectML");
  }
  if cfg!(feature = "rocm") {
    parts.push("ROCm");
  }
  if cfg!(feature = "coreml") {
    parts.push("CoreML");
  }
  if parts.is_empty() {
    "CPU only".into()
  } else {
    parts.push("CPU fallback");
    parts.join(" → ")
  }
}

fn build_session(graph: &Path, providers: Vec<ExecutionProviderDispatch>) -> ort::Result<Session> {
  let mut b = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level1)?
    .with_intra_threads(1)?
    .with_inter_threads(1)?
    .with_parallel_execution(false)?;
  if !providers.is_empty() {
    b = b.with_execution_providers(providers)?;
  }
  b.commit_from_file(graph)
}

fn measure(session: &mut Session, label: &str, batch: usize) -> (Duration, Duration) {
  // Synthesize a NaFlex input matching the graph's input contract:
  // `pixel_values [batch, 256, 768]`, `pixel_attention_mask
  // [batch, 256]`, `spatial_shapes [batch, 2]`. Values are
  // deterministic so repeated runs hit the same code paths in
  // CoreML's planner. The graph's pooler_output is statically
  // declared `[1, 768]`; ort overrides this with the runtime batch
  // dimension (verified by inspection of the run output below).
  let pixel_values: Vec<f32> = (0..batch * MAX_PATCHES * PATCH_FEATURES)
    .map(|i| ((i as f32) * 0.001).sin())
    .collect();
  let attention_mask: Vec<i32> = vec![1; batch * MAX_PATCHES];
  let mut spatial_shapes: Vec<i32> = Vec::with_capacity(batch * 2);
  for _ in 0..batch {
    spatial_shapes.extend_from_slice(&[16, 16]);
  }

  let pv = TensorRef::from_array_view((
    vec![batch as i64, MAX_PATCHES as i64, PATCH_FEATURES as i64],
    pixel_values.as_slice(),
  ))
  .expect("pixel_values tensor");
  let am = TensorRef::from_array_view((
    vec![batch as i64, MAX_PATCHES as i64],
    attention_mask.as_slice(),
  ))
  .expect("attention_mask tensor");
  let ss = TensorRef::from_array_view((vec![batch as i64, 2_i64], spatial_shapes.as_slice()))
    .expect("spatial_shapes tensor");

  // Cold: first inference. CoreML compiles its ANE program here.
  // Drop the output explicitly so the next run can borrow `session`
  // again — `SessionOutputs<'_>` keeps the session pinned otherwise.
  let t0 = Instant::now();
  {
    let _out = session
      .run(ort::inputs![
        "pixel_values" => pv.clone(),
        "pixel_attention_mask" => am.clone(),
        "spatial_shapes" => ss.clone(),
      ])
      .expect("first run");
  }
  let cold = t0.elapsed();

  // Warm: median of WARM_RUNS subsequent runs.
  let mut samples = Vec::with_capacity(WARM_RUNS);
  for _ in 0..WARM_RUNS {
    let t = Instant::now();
    let _ = session
      .run(ort::inputs![
        "pixel_values" => pv.clone(),
        "pixel_attention_mask" => am.clone(),
        "spatial_shapes" => ss.clone(),
      ])
      .expect("warm run");
    samples.push(t.elapsed());
  }
  samples.sort();
  let warm = samples[samples.len() / 2];
  let warm_min = samples[0];
  let warm_max = samples[samples.len() - 1];

  println!("[{label}]");
  println!("  cold: {:>8.2} ms", ms(cold));
  println!(
    "  warm: median {:>5.2} ms (min {:.2}, max {:.2}) over {} runs",
    ms(warm),
    ms(warm_min),
    ms(warm_max),
    WARM_RUNS,
  );

  (cold, warm)
}

fn ms(d: Duration) -> f64 {
  d.as_secs_f64() * 1000.0
}
