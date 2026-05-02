//! Text-tower variant of `bench_ep`. Same A/B (CoreML+CPU vs CPU-only)
//! but for the text encoder, which has a different op profile (small
//! token batches, sequence-length 64).

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
const SEQ_LEN: usize = 64;

fn main() {
  let graph = match std::env::var_os("SIGLIP2_TEXT_GRAPH") {
    Some(v) => PathBuf::from(v),
    None => {
      eprintln!("error: set SIGLIP2_TEXT_GRAPH to the text_model ONNX path");
      std::process::exit(1);
    }
  };

  println!("=== siglip2 EP A/B benchmark (text tower, batch=1, seq=64) ===");
  println!("graph: {}", graph.display());
  println!();

  let mut sess_cpu = build_session(&graph, vec![]).expect("CPU session");
  let (cold_b, warm_b) = measure(&mut sess_cpu, "CPU-only");

  let providers: Vec<ExecutionProviderDispatch> = default_providers();
  let mut sess_default = build_session(&graph, providers).expect("default session");
  let label = format!(
    "default ({})",
    if default_providers().is_empty() {
      "CPU only".into()
    } else if cfg!(feature = "coreml")
      && !cfg!(any(
        feature = "cuda",
        feature = "tensorrt",
        feature = "directml",
        feature = "rocm"
      ))
    {
      "CoreML → CPU fallback".into()
    } else {
      "GPU EP → CPU fallback".to_string()
    }
  );
  let (cold_a, warm_a) = measure(&mut sess_default, &label);

  println!();
  println!("--- summary ---");
  println!(
    "cold:               default {:>7.2} ms   CPU-only {:>7.2} ms   Δ {:+7.2} ms",
    ms(cold_a),
    ms(cold_b),
    ms(cold_a) - ms(cold_b)
  );
  println!(
    "warm (median × {WARM_RUNS}): default {:>7.2} ms   CPU-only {:>7.2} ms   Δ {:+7.2} ms",
    ms(warm_a),
    ms(warm_b),
    ms(warm_a) - ms(warm_b)
  );
  let speedup = warm_b.as_secs_f64() / warm_a.as_secs_f64();
  if speedup >= 1.0 {
    println!("warm speedup of default vs CPU-only: {speedup:.2}x");
  } else {
    println!(
      "warm SLOWDOWN of default vs CPU-only: {:.2}x",
      1.0 / speedup
    );
  }
}

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

fn measure(session: &mut Session, label: &str) -> (Duration, Duration) {
  // Synthesize batch=1, seq=64 input_ids (the SigLIP2 NaFlex text
  // graph takes only `input_ids`; padding handled internally).
  let input_ids: Vec<i64> = (0..SEQ_LEN as i64).map(|i| (i * 31) % 32_000).collect();
  let ids = TensorRef::from_array_view((vec![1i64, SEQ_LEN as i64], input_ids.as_slice()))
    .expect("input_ids tensor");

  let t0 = Instant::now();
  {
    let _out = session
      .run(ort::inputs!["input_ids" => ids.clone()])
      .expect("first run");
  }
  let cold = t0.elapsed();

  let mut samples = Vec::with_capacity(WARM_RUNS);
  for _ in 0..WARM_RUNS {
    let t = Instant::now();
    let _ = session
      .run(ort::inputs!["input_ids" => ids.clone()])
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
    WARM_RUNS
  );

  (cold, warm)
}

fn ms(d: Duration) -> f64 {
  d.as_secs_f64() * 1000.0
}
