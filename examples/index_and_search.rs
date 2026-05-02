//! Index a directory of images by their SigLIP2 vision embedding, then run a
//! text query and print the top-K matches with calibrated sigmoid scores.
//!
//! Usage:
//!     cargo run --release --example index_and_search --features decoders -- \
//!         <models-dir> <images-dir> "<query text>" [top_k]
//!
//! Demonstrates the spec §1.1 pipeline. Storage is in-memory; for a real
//! deployment, write the (image_path, embedding) pairs to lancedb or similar
//! at index time.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut args = std::env::args().skip(1);
  let models_dir: PathBuf = args.next().ok_or("usage error")?.into();
  let images_dir: PathBuf = args.next().ok_or("usage error")?.into();
  let query: String = args.next().ok_or("usage error")?;
  let top_k: usize = args.next().map(|s| s.parse().unwrap_or(5)).unwrap_or(5);

  let mut s = siglip2_naflex::Siglip2::from_files(
    &models_dir.join("vision_model_naflex_256.onnx"),
    &models_dir.join("text_model_naflex.onnx"),
    &models_dir.join("tokenizer.json"),
    &models_dir.join("calibration.json"),
  )?;

  // Index: collect (path, embedding) pairs.
  let mut entries: Vec<_> = std::fs::read_dir(&images_dir)?
    .filter_map(|e| e.ok())
    .filter(|e| {
      e.path()
        .extension()
        .is_some_and(|ext| ext == "jpg" || ext == "jpeg" || ext == "png")
    })
    .collect();
  entries.sort_by_key(|e| e.file_name());

  let mut index: Vec<(PathBuf, siglip2_naflex::Embedding)> = Vec::with_capacity(entries.len());
  for entry in entries {
    let path = entry.path();
    let emb = s.image().embed_path(&path)?;
    index.push((path, emb));
  }
  eprintln!("indexed {} images", index.len());

  // Query.
  let q = s.text().embed(&query)?;

  // Rank.
  let mut scored: Vec<(&PathBuf, f32)> = index.iter().map(|(p, e)| (p, e.cosine(&q))).collect();
  scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

  println!("query: {query:?}");
  for (path, cos) in scored.into_iter().take(top_k) {
    println!("  cos={cos:.4}  {}", path.display());
  }
  Ok(())
}
