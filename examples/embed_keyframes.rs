//! Embed a directory of JPEG/PNG keyframes and print one row per file:
//! `<filename>\t<dim0>,<dim1>,...,<dim767>`.
//!
//! Usage:
//!     cargo run --release --example embed_keyframes -- \
//!         <models-dir> <keyframes-dir>
//!
//! Where `<models-dir>` contains `vision_model_naflex_256.onnx` (+ sidecar)
//! and `<keyframes-dir>` contains `*.jpg` / `*.png` files.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut args = std::env::args().skip(1);
  let models_dir: PathBuf = args
    .next()
    .ok_or("usage: embed_keyframes <models-dir> <keyframes-dir>")?
    .into();
  let keyframes_dir: PathBuf = args
    .next()
    .ok_or("usage: embed_keyframes <models-dir> <keyframes-dir>")?
    .into();

  let mut enc =
    siglip2_naflex::ImageEncoder::from_files(&models_dir.join("vision_model_naflex_256.onnx"))?;

  let mut entries: Vec<_> = std::fs::read_dir(&keyframes_dir)?
    .filter_map(|e| e.ok())
    .filter(|e| {
      e.path()
        .extension()
        .is_some_and(|ext| ext == "jpg" || ext == "jpeg" || ext == "png")
    })
    .collect();
  entries.sort_by_key(|e| e.file_name());

  for entry in entries {
    let path = entry.path();
    let emb = enc.embed_path(&path)?;
    let mut line = String::with_capacity(8 * 768 + 256);
    line.push_str(path.file_name().unwrap().to_str().unwrap_or("?"));
    line.push('\t');
    for (i, x) in emb.as_slice().iter().enumerate() {
      if i > 0 {
        line.push(',');
      }
      use std::fmt::Write;
      let _ = write!(line, "{x:.6}");
    }
    println!("{line}");
  }
  Ok(())
}
