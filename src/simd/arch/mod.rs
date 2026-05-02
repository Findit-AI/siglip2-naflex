//! Architecture-specific SIMD backends for the siglip2 hot paths.
//!
//! Each submodule is gated on the target architecture it targets. The
//! dispatcher in [`super`] picks among them at call boundaries via
//! runtime CPU feature detection (compile-time on wasm32, where
//! features are fixed at module produce time).

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_avx2;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_avx512;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_sse2;

#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm_simd128;
