//! Shared `ort::Session` constructor for `ImageEncoder` and
//! `TextEncoder`.
//!
//! Both encoders need an identical session-building pipeline (graph
//! optimization level + thread config + execution-provider
//! registration), so the wiring lives in one place. Prior to this
//! split, `image_enc.rs` and `text_enc.rs` each carried a
//! byte-identical `build_session` — keeping both in sync as the GPU
//! EP cfg ladder grows would just be duplicated maintenance.
//!
//! Gated on `feature = "inference"` because every type touched here
//! (`ort::Session`, `ort::ep::*`) only exists when ort is in the
//! dependency graph.

use std::path::Path;

use crate::{
  error::{Error, Result},
  options::Options,
};

/// Build an `ort::Session` from the graph at `path` with the
/// caller-supplied `Options`. Registers any execution providers the
/// caller opted into (`cuda` / `tensorrt` / `directml` / `rocm` /
/// `coreml` Cargo features) before committing the graph file. The
/// implicit CPU EP is always available as the final fallback when
/// none of the registered EPs claim a given op. With no GPU feature
/// active, the session runs CPU-only — which is the default and
/// (per `examples/bench_ep.rs` measurements) the fastest path on
/// Apple Silicon for the current released fp32 NaFlex model.
pub(crate) fn build_session(graph: &Path, opts: Options) -> Result<ort::session::Session> {
  use ort::session::Session;

  let level = opts.optimization_level();

  // Session::builder() returns ort::Result<SessionBuilder, ort::Error<()>>.
  // The with_* methods return BuilderResult = Result<SessionBuilder, ort::Error<SessionBuilder>>.
  // ort::Error::from converts Error<SessionBuilder> → Error<()>.
  let mut builder = Session::builder()
    .map_err(|source| Error::LoadGraph {
      path: graph.to_path_buf(),
      source,
    })?
    .with_optimization_level(level)
    .map_err(|source| Error::LoadGraph {
      path: graph.to_path_buf(),
      source: ort::Error::from(source),
    })?
    .with_intra_threads(opts.threads().intra_threads())
    .map_err(|source| Error::LoadGraph {
      path: graph.to_path_buf(),
      source: ort::Error::from(source),
    })?
    .with_inter_threads(opts.threads().inter_threads())
    .map_err(|source| Error::LoadGraph {
      path: graph.to_path_buf(),
      source: ort::Error::from(source),
    })?
    .with_parallel_execution(opts.threads().parallel_execution())
    .map_err(|source| Error::LoadGraph {
      path: graph.to_path_buf(),
      source: ort::Error::from(source),
    })?;

  let providers = collect_execution_providers();
  if !providers.is_empty() {
    builder = builder
      .with_execution_providers(providers)
      .map_err(|source| Error::LoadGraph {
        path: graph.to_path_buf(),
        source: ort::Error::from(source),
      })?;
  }

  builder
    .commit_from_file(graph)
    .map_err(|source| Error::LoadGraph {
      path: graph.to_path_buf(),
      source,
    })
}

/// Collect the execution-provider dispatchers active under the
/// current target + feature configuration. Order matters: ort tries
/// each in the supplied list before falling back to the implicit
/// CPU EP, so the first registered EP gets first refusal on each op.
///
/// Cfg ladder rationale: we only push an EP whose underlying ort
/// sub-feature was actually compiled in. Pushing without the feature
/// would make `register()` return `RegisterError::MissingFeature` and
/// fail session creation. Pairing each push with its enabling cfg
/// keeps the build and runtime gates aligned.
fn collect_execution_providers() -> Vec<ort::ep::ExecutionProviderDispatch> {
  // `unused_mut` fires when every cfg-gated push below is excluded
  // i.e. when no opt-in EP feature is active, which is the default.
  // The `Vec::new` is the only reachable statement; the `mut` then
  // becomes formally unused. ort's implicit CPU EP picks up every op
  // when this Vec is empty.
  #[allow(unused_mut)]
  let mut providers: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();

  // TensorRT before CUDA when both are enabled: TensorRT typically
  // beats raw CUDA on supported ops, and the unsupported ones fall
  // back to CUDA's general execution path. With only the `cuda`
  // feature, CUDA is registered alone.
  #[cfg(feature = "tensorrt")]
  {
    providers.push(ort::ep::TensorRT::default().build());
  }
  #[cfg(feature = "cuda")]
  {
    providers.push(ort::ep::CUDA::default().build());
  }
  #[cfg(feature = "directml")]
  {
    providers.push(ort::ep::DirectML::default().build());
  }
  #[cfg(feature = "rocm")]
  {
    providers.push(ort::ep::ROCm::default().build());
  }
  // CoreML — opt-in like the others. The macOS auto-on policy was
  // reverted after `examples/bench_ep.rs` measured a 1.6× (vision)
  // / 2.7× (text) latency regression vs CPU-only on Apple Silicon
  // at batch=1 with the current released fp32 NaFlex model. A caller
  // who has fp16 weights, or a batch>1 export, or a workload that
  // amortizes the first-call graph-compile cost can opt in via
  // `siglip2 = { features = ["coreml"] }`.
  #[cfg(feature = "coreml")]
  {
    providers.push(ort::ep::CoreML::default().build());
  }

  providers
}
