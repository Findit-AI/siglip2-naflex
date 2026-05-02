//! `ImageView` — borrowed RGB pixel buffer with validating constructor.
//!
//! Lives outside `image_enc` so it's available without the
//! `inference` feature (i.e. on wasm32 builds, which can't pull in
//! `ort`). The preprocessor consumes `ImageView` to produce a
//! [`crate::PreprocessedBatch`]; that batch is then either embedded
//! by [`crate::ImageEncoder`] (inference feature) or shipped to a
//! caller-supplied inference engine.
//!
//! Split out from `image_enc.rs` for the wasm
//! feature-gate refactor.

use crate::error::{Error, Result};

/// View over decoded RGB pixels. `Copy` is safe because all fields are `Copy`
/// and the validating `new` constructor is the only construction path.
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
  rgb: &'a [u8],
  width: u32,
  height: u32,
}

impl<'a> ImageView<'a> {
  /// Constructs a view over RGB pixels. `rgb` must be exactly
  /// `width * height * 3` bytes, row-major, no row padding. Returns
  /// `Error::RgbLength` on length mismatch and `Error::InvalidImage` on
  /// zero dimensions.
  ///
  /// **Caller owns EXIF orientation.** The pixels handed in here are
  /// embedded as-is; this constructor does not read or apply any
  /// orientation metadata. JPEGs from phone cameras (and other
  /// EXIF-capable sources) commonly store pixels in the sensor's
  /// native landscape grid with an `Orientation` tag that displays
  /// correctly only after the viewer rotates / flips on the way out.
  /// `ImageEncoder::embed_path` handles this for you (it decodes via
  /// `image::ImageDecoder::orientation` and applies the transform
  /// before constructing the `ImageView`); callers who decode their
  /// own pixels — wasm consumers, custom decoder paths, callers with
  /// already-loaded buffers — must apply orientation themselves
  /// before calling `ImageView::new`, or accept that misoriented
  /// inputs will produce embeddings for the wrong rotation.
  /// (counterpart to the `embed_path`
  /// auto-orientation note in `image_enc.rs`).
  pub fn new(rgb: &'a [u8], width: u32, height: u32) -> Result<Self> {
    if width == 0 || height == 0 {
      return Err(Error::InvalidImage { width, height });
    }
    // `(w as usize) * (h as usize) * 3` can wrap on 64-bit release builds
    // for legal `u32` inputs near `u32::MAX` (e.g. w=h≈sqrt(u64::MAX/3)).
    // A wrap-to-small product would let a small rgb slice satisfy the
    // length check that follows. Use `checked_mul` to surface this as
    // `Error::DimensionsOverflow` instead.
    let expected = (width as usize)
      .checked_mul(height as usize)
      .and_then(|wh| wh.checked_mul(3))
      .ok_or(Error::DimensionsOverflow { width, height })?;
    if rgb.len() != expected {
      return Err(Error::RgbLength {
        got: rgb.len(),
        expected,
      });
    }
    Ok(Self { rgb, width, height })
  }

  /// The borrowed interleaved RGB byte slice (`width * height * 3` bytes).
  pub fn rgb(&self) -> &'a [u8] {
    self.rgb
  }
  /// Image width in pixels.
  pub fn width(&self) -> u32 {
    self.width
  }
  /// Image height in pixels.
  pub fn height(&self) -> u32 {
    self.height
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn image_view_validates_length() {
    let bad = vec![0u8; 10];
    let err = ImageView::new(&bad, 4, 4).unwrap_err();
    match err {
      Error::RgbLength {
        got: 10,
        expected: 48,
      } => {}
      _ => panic!("expected RgbLength, got {err}"),
    }
  }

  #[test]
  fn image_view_rejects_zero_dim() {
    let bad = vec![];
    let err = ImageView::new(&bad, 0, 480).unwrap_err();
    match err {
      Error::InvalidImage {
        width: 0,
        height: 480,
      } => {}
      _ => panic!("expected InvalidImage, got {err}"),
    }
  }

  #[test]
  fn image_view_is_copy() {
    fn _require_copy<T: Copy>() {}
    _require_copy::<ImageView<'_>>();
  }

  /// dimensions near `u32::MAX` whose
  /// `width * height * 3` overflows `usize` would silently wrap to a
  /// small value in release builds, letting a tiny rgb slice pass the
  /// length check. Reject with `DimensionsOverflow` instead.
  #[test]
  fn image_view_rejects_dimension_overflow() {
    let bad = vec![0u8; 4079];
    let err = ImageView::new(&bad, 2_479_008_847, 2_480_392_395).unwrap_err();
    match err {
      Error::DimensionsOverflow {
        width: 2_479_008_847,
        height: 2_480_392_395,
      } => {}
      _ => panic!("expected DimensionsOverflow, got {err}"),
    }
  }
}
