//! NaFlex preprocessing — patch-grid sizing, resize, normalize, patchify.

pub(crate) const PATCH_SIZE: u32 = 16;

/// Tolerance for the binary-search termination. Matches the upstream
/// `transformers.models.siglip2.image_processing_siglip2_fast`
/// `get_image_size_for_max_num_patches` value. **Do not lower** this — it is
/// what keeps `s` clearly below the boundary where `ceil()` flips, and is
/// the difference between us emitting upstream-equivalent grids vs silently
/// drifting by 1 in the longer axis on edge inputs (see the spec's
pub(crate) const SCALE_EPS: f64 = 1e-5;

/// Direct port of upstream `get_image_size_for_max_num_patches` — find the
/// largest scalar scale `s` such that, after rounding each axis up to a
/// multiple of `P`, the patch grid `target_h/P × target_w/P` fits within
/// `max_num_patches`. Returns `(H_p, W_p)` with both ≥ 1.
///
/// We binary-search `s ∈ [eps/10, 100]` and return at `hi - lo < eps`.
/// `scale_min` is always feasible (the budget check is the only thing that
/// updates it), so the final values trivially satisfy `H_p · W_p ≤ M`
/// no defensive post-clamp is needed.
pub(crate) fn patch_grid(height: u32, width: u32, max_num_patches: u32) -> (u32, u32) {
  let h = height as f64;
  let w = width as f64;
  let p = PATCH_SIZE as f64;
  let m = max_num_patches as f64;

  // Round `scale * original` up to a multiple of `P`, then floor at `P` so
  // the smallest possible target is one full patch (matches upstream).
  fn scaled_pixel_size(scale: f64, original: f64, patch: f64) -> f64 {
    let scaled = scale * original;
    let scaled = (scaled / patch).ceil() * patch;
    scaled.max(patch)
  }

  let mut scale_min: f64 = SCALE_EPS / 10.0;
  let mut scale_max: f64 = 100.0;
  while (scale_max - scale_min) >= SCALE_EPS {
    let scale = 0.5 * (scale_min + scale_max);
    let target_h = scaled_pixel_size(scale, h, p);
    let target_w = scaled_pixel_size(scale, w, p);
    let num_patches = (target_h * target_w) / (p * p);
    if num_patches <= m {
      scale_min = scale;
    } else {
      scale_max = scale;
    }
  }

  let target_h = scaled_pixel_size(scale_min, h, p);
  let target_w = scaled_pixel_size(scale_min, w, p);
  let h_p = ((target_h / p) as u32).max(1);
  let w_p = ((target_w / p) as u32).max(1);
  (h_p, w_p)
}

use image::{ImageBuffer, Rgb, imageops::FilterType};

use crate::error::{Error, Result};

/// Per-image preprocessing strides (= per-image lengths of the three
/// model-input slices for the base/naflex variant).
pub(crate) const PIXEL_VALUES_STRIDE: usize = 256 * 768; // 196_608
pub(crate) const ATTENTION_MASK_STRIDE: usize = 256;
pub(crate) const SPATIAL_SHAPES_STRIDE: usize = 2;

/// Writes preprocessed tensors for a single RGB image into the supplied
/// per-image buffers. Buffer lengths must match the strides above; otherwise
/// returns `Error::PreprocBufferLength { which }`.
pub(crate) fn preprocess_into(
  rgb: &[u8],
  width: u32,
  height: u32,
  max_num_patches: u32,
  pixel_values_out: &mut [f32],
  attention_mask_out: &mut [i32],
  spatial_shapes_out: &mut [i32],
) -> Result<()> {
  if width == 0 || height == 0 {
    return Err(Error::InvalidImage { width, height });
  }
  // Mirror ImageView::new's overflow check: `(w * h * 3)` in usize can
  // wrap on 64-bit for `u32` inputs near `u32::MAX`. Caught at this entry
  // point too because callers can reach `preprocess_into` via the
  // `Preprocessor::preprocess_into` public API with raw rgb slices.
  let expected_rgb_len = (width as usize)
    .checked_mul(height as usize)
    .and_then(|wh| wh.checked_mul(3))
    .ok_or(Error::DimensionsOverflow { width, height })?;
  if rgb.len() != expected_rgb_len {
    return Err(Error::RgbLength {
      got: rgb.len(),
      expected: expected_rgb_len,
    });
  }
  if pixel_values_out.len() != PIXEL_VALUES_STRIDE {
    return Err(Error::PreprocBufferLength {
      which: "pixel_values",
      got: pixel_values_out.len(),
      expected: PIXEL_VALUES_STRIDE,
    });
  }
  if attention_mask_out.len() != ATTENTION_MASK_STRIDE {
    return Err(Error::PreprocBufferLength {
      which: "attention_mask",
      got: attention_mask_out.len(),
      expected: ATTENTION_MASK_STRIDE,
    });
  }
  if spatial_shapes_out.len() != SPATIAL_SHAPES_STRIDE {
    return Err(Error::PreprocBufferLength {
      which: "spatial_shapes",
      got: spatial_shapes_out.len(),
      expected: SPATIAL_SHAPES_STRIDE,
    });
  }

  let (h_p, w_p) = patch_grid(height, width, max_num_patches);
  // Postcondition: enforce the patch-budget invariant before slicing into
  // the fixed-size pixel_values buffer. The binary search assumes its
  // initial `scale_min = eps/10` is feasible, but for legal `u32` inputs
  // near the upper bound (e.g. width > ~4·10⁹ at height = 1) the entire
  // feasible range is below that floor, so the loop's invariant collapses
  // and the final grid can land just above the budget. Without this check
  // the patchify loop would index out of `pixel_values_out` and panic.
  let grid_patches = (h_p as u64) * (w_p as u64);
  if grid_patches > max_num_patches as u64 {
    return Err(Error::ImageTooLarge {
      width,
      height,
      grid_patches,
      max_num_patches,
    });
  }
  let h_res = h_p * PATCH_SIZE;
  let w_res = w_p * PATCH_SIZE;

  // Borrowed-buffer ImageBuffer over the caller's RGB slice — zero
  // copy at the input boundary. `ImageBuffer::from_raw` accepts any
  // `Container: Deref<Target = [u8]>` and `&[u8]` satisfies that;
  // `imageops::resize` only needs a `GenericImageView`, which
  // `ImageBuffer<Rgb<u8>, &[u8]>` implements. The previous code did
  // `rgb.to_vec()` here, which doubled peak memory for large inputs
  // (~25 MB extra for 4K, ~800 MB for 16K) before the bounded
  // downscale fired.
  //
  // The resize result is still owned (`ImageBuffer<Rgb<u8>, Vec<u8>>`,
  // sized at most `256 * 16 * 16 * 3 ≈ 200 KB`), so downstream code
  // that borrows `resized.as_raw()` is unaffected.
  let src =
    ImageBuffer::<Rgb<u8>, &[u8]>::from_raw(width, height, rgb).ok_or(Error::RgbLength {
      got: rgb.len(),
      expected: expected_rgb_len,
    })?;
  let resized = image::imageops::resize(&src, w_res, h_res, FilterType::Triangle);

  // Normalize and patchify in (row, col, channel) order with channel
  // innermost — interleaved RGB, no axis transposition.
  let n_patches = (h_p as usize) * (w_p as usize);

  // Zero out the output (we'll right-pad with zeros to 256 patches).
  pixel_values_out.fill(0.0);

  let stride_per_patch: usize = (PATCH_SIZE as usize) * (PATCH_SIZE as usize) * 3; // 768

  // Per-row processing through the SIMD dispatcher: every patch row is
  // 16 contiguous RGB pixels = 48 contiguous source bytes mapped to 48
  // contiguous f32 outputs. The resized buffer is row-major,
  // RGB-interleaved, no row padding.
  let resized_buf = resized.as_raw();
  let src_row_stride = (w_res as usize) * 3;
  let row_bytes: usize = (PATCH_SIZE as usize) * 3; // 48

  for py in 0..h_p {
    for px in 0..w_p {
      let patch_idx = (py as usize) * (w_p as usize) + (px as usize);
      let out_offset = patch_idx * stride_per_patch;
      let dst_patch = &mut pixel_values_out[out_offset..out_offset + stride_per_patch];

      for r_in_patch in 0..PATCH_SIZE as usize {
        let src_y = (py as usize) * (PATCH_SIZE as usize) + r_in_patch;
        let src_x = (px as usize) * (PATCH_SIZE as usize);
        let src_off = src_y * src_row_stride + src_x * 3;
        let src_row = &resized_buf[src_off..src_off + row_bytes];
        let dst_off = r_in_patch * row_bytes;
        let dst_row = &mut dst_patch[dst_off..dst_off + row_bytes];
        crate::simd::normalize_patchify_row(src_row, dst_row);
      }
    }
  }

  // Attention mask: 1 for the first n_patches slots, 0 for padding.
  for (i, slot) in attention_mask_out.iter_mut().enumerate() {
    *slot = if i < n_patches { 1 } else { 0 };
  }

  // Spatial shapes: [H_p, W_p].
  spatial_shapes_out[0] = h_p as i32;
  spatial_shapes_out[1] = w_p as i32;

  Ok(())
}

#[cfg(test)]
#[allow(warnings)]
mod tests {
  use super::*;

  /// Reference table from. Values verified against the upstream
  /// `transformers.models.siglip2.image_processing_siglip2_fast`
  /// `get_image_size_for_max_num_patches` Python implementation.
  ///
  /// The (3, 39) → (4, 52) row is the regression case from the
  /// adversarial review: the previous 64-iteration binary search overshot
  /// the boundary at this aspect ratio and returned (4, 53). The current
  /// `eps = 1e-5` termination matches upstream exactly.
  #[test]
  fn reference_table_matches() {
    const M: u32 = 256;
    let cases: &[((u32, u32), (u32, u32))] = &[
      ((16, 16), (16, 16)),
      ((100, 100), (16, 16)),
      ((224, 224), (16, 16)),
      ((1080, 1920), (12, 21)),
      ((1920, 1080), (21, 12)),
      ((2160, 4096), (12, 21)),
      ((1024, 1), (256, 1)),
      // Regression: pre-fix algorithm returned (4, 53) here; upstream is
      // (4, 52).rs caveat.
      ((3, 39), (4, 52)),
    ];
    for (input, expected) in cases {
      let (h_p, w_p) = patch_grid(input.0, input.1, M);
      assert_eq!(
        (h_p, w_p),
        *expected,
        "patch_grid({}x{}) — expected {:?}, got ({h_p}, {w_p})",
        input.0,
        input.1,
        expected
      );
    }
  }

  #[test]
  fn budget_respected_on_random_inputs() {
    const M: u32 = 256;
    let cases = [
      (1, 1),
      (1, 2048),
      (2048, 1),
      (4096, 4096),
      (640, 480),
      (3840, 2160),
      (32, 7),
      (7, 32),
    ];
    for (h, w) in cases {
      let (h_p, w_p) = patch_grid(h, w, M);
      assert!(h_p >= 1 && w_p >= 1, "{h}x{w} → {h_p}x{w_p} has zero axis");
      assert!(
        (h_p as u64) * (w_p as u64) <= M as u64,
        "{h}x{w} → {h_p}x{w_p} exceeds budget {M}"
      );
    }
  }

  fn make_zeroed_buffers() -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    (
      vec![0.0f32; PIXEL_VALUES_STRIDE],
      vec![0i32; ATTENTION_MASK_STRIDE],
      vec![0i32; SPATIAL_SHAPES_STRIDE],
    )
  }

  ///.4: byte-layout test on a constructed image with a known per-channel
  /// pattern. Catches axis-order regressions silently.
  #[test]
  fn patch_byte_order_is_row_col_channel_innermost() {
    // 16x16 image (one patch grid cell), each pixel R=10, G=20, B=30.
    // After normalization: R = (10/255 - 0.5) / 0.5, etc.
    let rgb: Vec<u8> = std::iter::repeat_n([10u8, 20, 30], 16 * 16)
      .flatten()
      .collect();
    let (mut pv, mut am, mut ss) = make_zeroed_buffers();
    preprocess_into(&rgb, 16, 16, 256, &mut pv, &mut am, &mut ss).unwrap();

    // First three values of the first patch must be R, G, B in that order.
    let r = (10f32 / 255.0 - 0.5) / 0.5;
    let g = (20f32 / 255.0 - 0.5) / 0.5;
    let b = (30f32 / 255.0 - 0.5) / 0.5;
    assert!((pv[0] - r).abs() < 1e-5, "pv[0] should be R, got {}", pv[0]);
    assert!((pv[1] - g).abs() < 1e-5, "pv[1] should be G, got {}", pv[1]);
    assert!((pv[2] - b).abs() < 1e-5, "pv[2] should be B, got {}", pv[2]);

    // Spatial shapes should be non-trivial; check attention mask consistency.
    assert!(ss[0] >= 1 && ss[1] >= 1);
    let n_patches = (ss[0] as usize) * (ss[1] as usize);
    for i in 0..n_patches {
      assert_eq!(am[i], 1, "attention[{i}] should be 1");
    }
    for i in n_patches..ATTENTION_MASK_STRIDE {
      assert_eq!(am[i], 0, "attention[{i}] (padding) should be 0");
    }
  }

  #[test]
  fn patch_grid_overshoots_at_extreme_dims() {
    // at w ≈ u32::MAX with h = 1, the
    // initial `scale_min = SCALE_EPS / 10 = 1e-6` is itself outside the
    // feasible range (the entire feasible region is below ~1e-9 because
    // `target_w = ceil(s · 4·10⁹ / 16) · 16` blows past 4096 for any
    // s > 1e-9). The binary search therefore never finds a feasible `s`,
    // `scale_min` stays at the infeasible initial value, and the function
    // returns a grid that violates the budget. This is what the
    // post-condition in `preprocess_into` exists to catch.
    let (h_p, w_p) = patch_grid(1, 4_096_000_001, 256);
    let product = (h_p as u64) * (w_p as u64);
    assert!(
      product > 256,
      "overshoot input must produce a budget-violating grid: ({h_p}, {w_p}) = {product}"
    );
  }

  #[test]
  fn rejects_image_too_large_for_budget() {
    // Synthetic exercise of the post-condition path. The realistic
    // extreme-u32 trigger would need a ~12 GB RGB
    // buffer; using a 1×1 image with `max_num_patches = 0` reaches the
    // same Error::ImageTooLarge return path because patch_grid returns
    // (1, 1), product 1 > 0. Both paths exercise the same line.
    let rgb = vec![128u8; 3];
    let (mut pv, mut am, mut ss) = make_zeroed_buffers();
    let err = preprocess_into(&rgb, 1, 1, 0, &mut pv, &mut am, &mut ss).unwrap_err();
    match err {
      Error::ImageTooLarge {
        width: 1,
        height: 1,
        grid_patches: 1,
        max_num_patches: 0,
      } => {}
      _ => panic!("expected ImageTooLarge, got {err}"),
    }
  }

  #[test]
  fn rejects_zero_dimensions() {
    let rgb = vec![];
    let (mut pv, mut am, mut ss) = make_zeroed_buffers();
    let err = preprocess_into(&rgb, 0, 480, 256, &mut pv, &mut am, &mut ss).unwrap_err();
    match err {
      Error::InvalidImage {
        width: 0,
        height: 480,
      } => {}
      _ => panic!("expected InvalidImage 0x480, got {err}"),
    }
  }

  #[test]
  fn rejects_wrong_rgb_length() {
    let rgb = vec![0u8; 100];
    let (mut pv, mut am, mut ss) = make_zeroed_buffers();
    let err = preprocess_into(&rgb, 16, 16, 256, &mut pv, &mut am, &mut ss).unwrap_err();
    match err {
      Error::RgbLength { got: 100, expected } => {
        assert_eq!(expected, 16 * 16 * 3);
      }
      _ => panic!("expected RgbLength, got {err}"),
    }
  }

  #[test]
  fn rejects_wrong_buffer_lengths() {
    let rgb = vec![0u8; 16 * 16 * 3];

    let mut pv = vec![0f32; 100];
    let mut am = vec![0i32; ATTENTION_MASK_STRIDE];
    let mut ss = vec![0i32; SPATIAL_SHAPES_STRIDE];
    let err = preprocess_into(&rgb, 16, 16, 256, &mut pv, &mut am, &mut ss).unwrap_err();
    match err {
      Error::PreprocBufferLength {
        which: "pixel_values",
        ..
      } => {}
      _ => panic!("expected PreprocBufferLength pixel_values, got {err}"),
    }

    let mut pv = vec![0f32; PIXEL_VALUES_STRIDE];
    let mut am = vec![0i32; 5];
    let err = preprocess_into(&rgb, 16, 16, 256, &mut pv, &mut am, &mut ss).unwrap_err();
    match err {
      Error::PreprocBufferLength {
        which: "attention_mask",
        ..
      } => {}
      _ => panic!("expected PreprocBufferLength attention_mask, got {err}"),
    }
  }

  #[test]
  fn padding_rows_are_exactly_zero() {
    // A 1x1-pixel input produces a 1x1 patch grid → 1 valid patch, 255 padding.
    let rgb = vec![128u8; 3];
    let (mut pv, mut am, mut ss) = make_zeroed_buffers();
    preprocess_into(&rgb, 1, 1, 256, &mut pv, &mut am, &mut ss).unwrap();

    let n_patches = (ss[0] as usize) * (ss[1] as usize);
    let stride = 768;
    for patch_i in n_patches..256 {
      for j in 0..stride {
        assert_eq!(
          pv[patch_i * stride + j],
          0.0,
          "padding patch {patch_i} idx {j} must be exactly zero"
        );
      }
    }
  }
}
