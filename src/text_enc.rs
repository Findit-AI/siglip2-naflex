//! Text encoder. 4 and §5.

use std::path::Path;

use tokenizers::{
  PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
  TruncationParams, TruncationStrategy,
  normalizers::{Lowercase, NormalizerWrapper, Sequence as NormalizerSequence},
};

use crate::{
  embedding::Embedding,
  error::{Error, Result},
  options::Options,
};

const SEQ_LEN: usize = 64;
const PAD_TOKEN_ID: u32 = 0;

/// SigLIP2 NaFlex text-tower inference. Owns one `ort::Session` and one
/// `tokenizers::Tokenizer`.
///
/// `TextEncoder: Send + !Sync` — `ort::Session` is `!Sync`. Workers wanting
/// parallelism instantiate one `TextEncoder` per thread, or share one behind
/// a `Mutex<TextEncoder>`.
pub struct TextEncoder {
  session: ort::session::Session,
  tokenizer: Tokenizer,
  opts: Options,
  /// Reusable `input_ids` scratch for `embed_batch`. Without this,
  /// every chunk allocates a fresh `Vec<i64>` of `chunk_size * SEQ_LEN`
  /// = up to 1024 × 64 × 8 bytes (512 KiB at the cap). Reuse cuts
  /// per-chunk allocation churn to zero.
  input_ids_scratch: Vec<i64>,
}

impl TextEncoder {
  /// **Not available on wasm32.** See [`crate::image_enc::ImageEncoder::from_files`]
  /// for rationale; same workaround applies — construct an
  /// `ort::session::Session` via the wasm-specific async APIs and
  /// pass it to [`Self::from_ort_session`].
  #[cfg(not(target_arch = "wasm32"))]
  pub fn from_files(graph: &Path, tokenizer: &Path) -> Result<Self> {
    Self::from_files_with_options(graph, tokenizer, Options::default())
  }

  /// Same wasm32 caveat as [`Self::from_files`].
  #[cfg(not(target_arch = "wasm32"))]
  pub fn from_files_with_options(graph: &Path, tokenizer: &Path, opts: Options) -> Result<Self> {
    let session = crate::session::build_session(graph, opts)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| Error::Tokenizer(e.to_string()))?;
    let tokenizer = configure_padding(tokenizer)?;
    Self::from_ort_session_with_options(session, tokenizer, opts)
  }

  /// Same wasm32 caveat as [`Self::from_files`].
  #[cfg(all(feature = "bundled", not(target_arch = "wasm32")))]
  pub fn bundled(graph: &Path) -> Result<Self> {
    Self::bundled_with_options(graph, Options::default())
  }

  /// Same wasm32 caveat as [`Self::from_files`].
  #[cfg(all(feature = "bundled", not(target_arch = "wasm32")))]
  pub fn bundled_with_options(graph: &Path, opts: Options) -> Result<Self> {
    let session = crate::session::build_session(graph, opts)?;
    let tokenizer = Tokenizer::from_bytes(crate::BUNDLED_TOKENIZER)
      .map_err(|e| Error::Tokenizer(e.to_string()))?;
    let tokenizer = configure_padding(tokenizer)?;
    Self::from_ort_session_with_options(session, tokenizer, opts)
  }

  /// Construct from a caller-built `ort::Session` and `Tokenizer`,
  /// using crate-default [`Options`]. On wasm32 this is the supported
  /// entry point because `ort 2.0.0-rc.12` cfg-gates `commit_from_file`
  /// out of wasm builds — wasm callers must build the session
  /// themselves and pass it in.
  pub fn from_ort_session(session: ort::session::Session, tokenizer: Tokenizer) -> Result<Self> {
    Self::from_ort_session_with_options(session, configure_padding(tokenizer)?, Options::default())
  }

  fn from_ort_session_with_options(
    session: ort::session::Session,
    tokenizer: Tokenizer,
    opts: Options,
  ) -> Result<Self> {
    validate_text_session(&session)?;
    // Mirror `Preprocessor::new` — the image and text paths share
    // `Options`, so a `BatchOptions` config that the image path would
    // reject must not silently work on the text path. Without this,
    // `batch_size = 0` was previously coerced to `1` inside
    // `embed_batch`.
    opts.batch().validate()?;
    Ok(Self {
      session,
      tokenizer,
      opts,
      input_ids_scratch: Vec::new(),
    })
  }

  /// Encode a single string and return its 768-dim L2-normalized
  /// [`Embedding`]. Empty input is rejected with [`Error::EmptyText`].
  /// For multiple inputs, prefer [`Self::embed_batch`] — it amortizes
  /// the per-call ORT overhead across the batch.
  pub fn embed(&mut self, text: &str) -> Result<Embedding> {
    if text.is_empty() {
      return Err(Error::EmptyText);
    }
    let mut out = self.embed_batch(&[text])?;
    Ok(out.remove(0))
  }

  /// Returns `Ok(vec![])` for an empty input slice (no ORT call).
  /// Returns `Error::BatchTooLarge` when `texts.len() > opts.batch.max_batch_size`.
  /// Internally chunks `texts` into groups of size `BatchOptions::batch_size`
  /// and runs one ORT inference per chunk; the returned `Vec` preserves
  /// input order and has the same length as `texts` on success.
  ///
  /// **Failure semantics.** Aborts on the first failing input and returns
  /// `Error::Batch { index, source }` carrying the offending zero-based
  /// index — symmetric with `ImageEncoder::embed_pixels_batch`. Already-
  /// computed embeddings from earlier chunks are dropped.
  pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
    if texts.is_empty() {
      return Ok(Vec::new());
    }
    let max = self.opts.batch().max_batch_size();
    if texts.len() > max {
      return Err(Error::BatchTooLarge {
        got: texts.len(),
        max,
      });
    }
    // Surface the offending index — `Error::Batch { index, source }` is
    // the documented batched-failure shape, so a `classify` call with
    // 100 labels where one is `""` can identify the bad record for
    // retry / cleanup.
    if let Some((index, _)) = texts.iter().enumerate().find(|(_, t)| t.is_empty()) {
      return Err(Error::Batch {
        index,
        source: Box::new(Error::EmptyText),
      });
    }
    // `batch_size >= 1` is guaranteed by `BatchOptions::validate` at
    // construction (`from_ort_session_with_options`). The previous
    // `.max(1)` silent-coercion fallback is gone — a misconfigured
    // `batch_size == 0` would now fail at construction, not here.
    let chunk = self.opts.batch().batch_size();
    // Take the scratch out so we can hold `&mut self.session` and
    // `&mut input_ids` simultaneously inside the inner loop without
    // aliasing on `self`. Put back unconditionally before returning
    // so the next call reuses the buffer's capacity.
    let mut input_ids = std::mem::take(&mut self.input_ids_scratch);
    let result = embed_batch_inner(
      &mut self.session,
      &self.tokenizer,
      &mut input_ids,
      chunk,
      texts,
    );
    self.input_ids_scratch = input_ids;
    result
  }

  /// Run a single throwaway inference to amortize ORT's first-call
  /// graph-compilation cost. Subsequent `embed` / `embed_batch` calls
  /// avoid the cold-start latency the first real call would otherwise
  /// pay.
  pub fn warmup(&mut self) -> Result<()> {
    let _ = self.embed("warmup")?;
    Ok(())
  }
}

fn validate_text_session(session: &ort::session::Session) -> Result<()> {
  use crate::image_enc::check_outlet;
  use ort::value::TensorElementType;

  let inputs = session.inputs();
  let outputs = session.outputs();

  check_outlet(
    inputs,
    "input_ids",
    TensorElementType::Int64,
    &[-1, SEQ_LEN as i64],
  )?;
  check_outlet(
    outputs,
    "pooler_output",
    TensorElementType::Float32,
    &[-1, 768],
  )?;
  // Tighten: assert exact input/output counts. The released SigLIP2 NaFlex
  // text export takes ONLY `input_ids` (no separate `attention_mask`) — the
  // graph handles padding via the `pad_token_id = 0` sentinel internally,
  // verified bit-exact at cosine 1.00000 against the PyTorch reference.
  // If a future re-export adds `attention_mask` as a required input, our
  // `Session::run(inputs!["input_ids" => val])` call would fail with a
  // confusing missing-input error; this check surfaces it cleanly at
  // construction time.
  if inputs.len() != 1 {
    return Err(Error::SessionShapeMismatch {
      input: "<input count>",
      expected: "1 input (input_ids only — see release contract)",
      got: vec![inputs.len() as i64],
    });
  }
  if outputs.len() != 1 {
    return Err(Error::SessionShapeMismatch {
      input: "<output count>",
      expected: "1 output (pooler_output)",
      got: vec![outputs.len() as i64],
    });
  }
  Ok(())
}

fn configure_padding(mut tokenizer: Tokenizer) -> Result<Tokenizer> {
  // The text ONNX graph takes ONLY `input_ids` (no separate
  // `attention_mask`); the model handles padding internally via the
  // `pad_token_id = 0` sentinel that's bit-exact-validated against the
  // PyTorch reference. If the loaded tokenizer's `<pad>` id isn't 0,
  // every padded prompt would produce length-correct but
  // semantically-wrong embeddings — there's no `attention_mask`
  // input to mask the wrong sentinel out. Reject the mismatch
  // upfront so a swapped tokenizer surfaces at construction, not as
  // silently degraded retrieval.
  let pad_id = tokenizer
    .token_to_id("<pad>")
    .ok_or_else(|| Error::Tokenizer("loaded tokenizer has no `<pad>` token".to_string()))?;
  if pad_id != PAD_TOKEN_ID {
    return Err(Error::Tokenizer(format!(
      "loaded tokenizer's `<pad>` token id is {pad_id}, expected {PAD_TOKEN_ID}; \
       the SigLIP2 NaFlex text graph has no `attention_mask` input and relies \
       on `pad_token_id = 0` as the padding sentinel — a mismatched tokenizer \
       silently corrupts every padded-prompt embedding"
    )));
  }

  // Prepend a `Lowercase` normalizer to whatever the loaded tokenizer.json
  // already carries (the bundled JSON has `Replace(" ", "▁")` for the
  // SentencePiece marker — see normalizer field of models/tokenizer.json).
  // Upstream `transformers.models.siglip2.tokenization_siglip2.Siglip2Tokenizer`
  // does the same wrap at runtime in its `__init__` (lines 95-96 of that
  // file): `backend.normalizer = Sequence([Lowercase(), backend.normalizer])`.
  // The `Lowercase` step is NOT serialized into the exported tokenizer.json,
  // so a Rust caller loading the JSON directly (without going through the
  // Python `Siglip2Tokenizer` class) would silently encode mixed-case input
  // differently from upstream — different token ids → different embeddings
  // → wrong retrieval / `classify` rankings.
  let existing = tokenizer.get_normalizer().cloned();
  let mut wrapped: Vec<NormalizerWrapper> = vec![Lowercase.into()];
  if let Some(n) = existing {
    wrapped.push(n);
  }
  // `with_normalizer` returns `Result` because it calls
  // `refresh_normalized_tokens`, which iterates added tokens with
  // `normalized = true` and re-applies the new normalizer; the
  // normalizer's own `normalize()` is fallible. The bundled SigLIP2
  // tokenizer never trips this path, but `from_files` /
  // `from_ort_session` accept arbitrary caller-supplied tokenizers,
  // so `expect` here would turn a bad asset into a process panic
  // during construction. Surface as `Error::Tokenizer` instead, in
  // line with every other fallible step on this loader path.
  tokenizer
    .with_normalizer(Some(NormalizerSequence::new(wrapped)))
    .map_err(|e| Error::Tokenizer(e.to_string()))?;

  // Pad short inputs to SEQ_LEN. `Fixed` only pads — long inputs are not
  // truncated by padding alone, so we also configure truncation below.
  tokenizer.with_padding(Some(PaddingParams {
    strategy: PaddingStrategy::Fixed(SEQ_LEN),
    direction: PaddingDirection::Right,
    pad_id: PAD_TOKEN_ID,
    pad_token: "<pad>".to_string(),
    pad_type_id: 0,
    pad_to_multiple_of: None,
  }));
  // Truncate long inputs to SEQ_LEN. The bundled `tokenizer.json` already
  // carries a truncation config, but `from_ort_session` accepts a caller-
  // built tokenizer that may not — without this call, an over-64-token
  // query would surface as `Error::Batch { source: Tokenizer("…produced N
  // ids; expected 64") }` instead of being truncated to the graph's static
  // `[batch, 64]` `input_ids` axis. `with_truncation` returns
  // `Result<&mut Self>` and only fails when `stride > effective_max_length`;
  // with `stride = 0` and `max_length = 64` this is infallible.
  tokenizer
    .with_truncation(Some(TruncationParams {
      direction: TruncationDirection::Right,
      max_length: SEQ_LEN,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
    }))
    .expect("with_truncation infallible at stride=0, max_length>0");
  Ok(tokenizer)
}

/// Inner workhorse for `embed_batch`, pulled out so the caller can
/// pass disjoint `&mut self.session` and `&mut self.input_ids_scratch`
/// borrows that the borrow checker can't see through methods.
/// `input_ids` is cleared and refilled per chunk; the buffer's
/// capacity is preserved across calls when the caller hands the
/// scratch back to `self.input_ids_scratch`.
fn embed_batch_inner(
  session: &mut ort::session::Session,
  tokenizer: &Tokenizer,
  input_ids: &mut Vec<i64>,
  chunk: usize,
  texts: &[&str],
) -> Result<Vec<Embedding>> {
  let mut out = Vec::with_capacity(texts.len());
  for (chunk_idx, group) in texts.chunks(chunk).enumerate() {
    let encodings = tokenizer
      .encode_batch(group.to_vec(), true)
      .map_err(|e| Error::Batch {
        index: chunk_idx * chunk,
        source: Box::new(Error::Tokenizer(e.to_string())),
      })?;
    input_ids.clear();
    input_ids.reserve(group.len() * SEQ_LEN);
    for (i, enc) in encodings.iter().enumerate() {
      let ids = enc.get_ids();
      if ids.len() != SEQ_LEN {
        return Err(Error::Batch {
          index: chunk_idx * chunk + i,
          source: Box::new(Error::Tokenizer(format!(
            "tokenizer produced {} ids; expected {} (Fixed padding misconfigured)",
            ids.len(),
            SEQ_LEN
          ))),
        });
      }
      input_ids.extend(ids.iter().map(|&u| u as i64));
    }
    let chunk_emb = run_text_session(session, input_ids, group.len())?;
    out.extend(chunk_emb);
  }
  Ok(out)
}

// `build_session` was moved to the shared `crate::session` module so
// the EP-registration cfg ladder lives in one place. Both
// `ImageEncoder` and `TextEncoder` now call into it.

fn run_text_session(
  session: &mut ort::session::Session,
  input_ids: &[i64],
  batch_size: usize,
) -> Result<Vec<Embedding>> {
  use ort::value::TensorRef;

  let shape: [usize; 2] = [batch_size, SEQ_LEN];
  let val = TensorRef::from_array_view((shape, input_ids)).map_err(Error::Ort)?;

  let outputs = session
    .run(ort::inputs!["input_ids" => val])
    .map_err(Error::Ort)?;

  let pooler = outputs
    .get("pooler_output")
    .ok_or(Error::MissingOnnxOutput {
      name: "pooler_output",
    })?;
  let (shape, data) = pooler.try_extract_tensor::<f32>().map_err(Error::Ort)?;

  if shape.len() != 2 {
    return Err(Error::OutputRank {
      rank: shape.len(),
      shape: shape.to_vec(),
    });
  }
  if shape[0] != batch_size as i64 || shape[1] != 768 {
    return Err(Error::SessionShapeMismatch {
      input: "pooler_output",
      expected: "[batch, 768]",
      got: shape.to_vec(),
    });
  }

  let mut embeddings = Vec::with_capacity(batch_size);
  for i in 0..batch_size {
    embeddings.push(Embedding::from_model_output(&data[i * 768..(i + 1) * 768])?);
  }
  Ok(embeddings)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn constants_are_correct() {
    assert_eq!(SEQ_LEN, 64);
    assert_eq!(PAD_TOKEN_ID, 0);
  }

  /// SigLIP2 expects lowercased input. The
  /// bundled `tokenizer.json` only carries `Replace(" ", "▁")` for the
  /// SentencePiece marker — Python's `Siglip2Tokenizer.__init__` wraps it
  /// in `Sequence([Lowercase, Replace])` at runtime, but that wrap is NOT
  /// serialized into the JSON. `configure_padding` must reapply the wrap
  /// in our Rust path, otherwise mixed-case queries tokenize differently
  /// from upstream.
  ///
  /// Verified end-to-end: `"HELLO"` and `"hello"` must encode to the
  /// same token IDs after `configure_padding`. Without the fix this test
  /// fails (uppercase produces `<unk>` runs because the SPM model only
  /// has lowercase merges).
  #[cfg(feature = "bundled")]
  #[test]
  fn configure_padding_applies_lowercase() {
    let tok =
      Tokenizer::from_bytes(crate::BUNDLED_TOKENIZER).expect("bundled tokenizer must parse");
    let tok = configure_padding(tok).expect("bundled tokenizer must satisfy pad-id contract");

    let hi_upper = tok.encode("HELLO WORLD", true).expect("encode HELLO WORLD");
    let hi_lower = tok.encode("hello world", true).expect("encode hello world");
    assert_eq!(
      hi_upper.get_ids(),
      hi_lower.get_ids(),
      "configure_padding must lowercase: HELLO→{:?}, hello→{:?}",
      hi_upper.get_ids(),
      hi_lower.get_ids()
    );
  }

  /// the text ONNX graph has no
  /// `attention_mask` input — it uses `pad_token_id = 0` as the
  /// padding sentinel, validated bit-exact against the PyTorch
  /// reference. A swapped tokenizer whose `<pad>` is at a different
  /// id (e.g. fine-tunes that re-vocabulary'd, BERT-style tokenizers
  /// re-purposed by mistake) would produce length-correct but
  /// semantically wrong embeddings for every padded prompt.
  /// `configure_padding` must reject the mismatch upfront.
  ///
  /// Sanity baseline: bundled tokenizer's `<pad>` is at id 0, so
  /// the bundled-tokenizer path passes — proven by
  /// `configure_padding_applies_lowercase` above. Here we synthesize
  /// the failure case by parsing a minimal tokenizer.json whose
  /// `<pad>` ends up at a non-zero id and verify rejection.
  #[test]
  fn configure_padding_rejects_wrong_pad_id() {
    // Minimal in-memory `Tokenizer` whose vocabulary places `<pad>` at
    // id 7, not 0 — a contract violation.
    let json = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": null,
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {"<unk>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "<pad>": 7},
        "unk_token": "<unk>"
      }
    }"#;
    let tok = Tokenizer::from_bytes(json.as_bytes()).expect("test tokenizer must parse");
    let err = configure_padding(tok).expect_err("non-zero pad id must be rejected");
    match err {
      Error::Tokenizer(msg) => {
        assert!(
          msg.contains("<pad>") && msg.contains("7"),
          "expected pad-id mismatch message, got {msg:?}"
        );
      }
      _ => panic!("expected Error::Tokenizer, got {err}"),
    }
  }
}
