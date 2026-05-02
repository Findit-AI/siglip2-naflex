#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Error: TARGET is not provided"
  exit 1
fi

TARGET="$1"

# Install cross-compilation toolchain on Linux
if [ "$(uname)" = "Linux" ]; then
  case "$TARGET" in
    aarch64-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-aarch64-linux-gnu
      ;;
    i686-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-multilib
      ;;
    powerpc64-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-powerpc64-linux-gnu
      ;;
    s390x-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-s390x-linux-gnu
      ;;
    riscv64gc-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-riscv64-linux-gnu
      ;;
  esac
fi

rustup toolchain install nightly --component miri
rustup override set nightly
cargo miri setup

export MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-disable-isolation -Zmiri-symbolic-alignment-check -Zmiri-tree-borrows"

# Scope and configuration:
#
# 1. Test filter `simd::` — every `unsafe` block in this crate lives
#    under `src/simd/` (verified by `grep -rn unsafe src/`). The rest
#    is safe Rust, so miri adds no signal there.
#
# 2. `--cfg siglip2_force_scalar` — miri can't evaluate foreign LLVM
#    intrinsics like `llvm.aarch64.neon.faddv.f32.v4f32` (NEON) or
#    `llvm.x86.avx2.*`. Without this cfg, the dispatcher hits its
#    arch-specific path and miri errors `unsupported operation`.
#    With this cfg every `*_available()` helper short-circuits to
#    `false` and the dispatcher falls through to the scalar reference.
#    The intrinsic paths themselves are exercised natively under SDE
#    (AVX-512), wasmtime (wasm simd128), and the regular test job
#    (NEON / AVX2 / SSE2 on hosts that have them).
#
# 3. `--no-default-features --features serde` — skips `inference`
#    (ort + tokenizers) and `decoders`. Half of the miri matrix
#    (`{i686,powerpc64,s390x,riscv64gc}-unknown-linux-gnu`) is not in
#    the `ort-sys` prebuilt distribution, so the ort build script
#    would fail before miri even starts. Inference FFI also can't be
#    evaluated under miri anyway.
#
# Codex round-32 CI sweep.
export RUSTFLAGS="${RUSTFLAGS:-} --cfg siglip2_force_scalar"
cargo miri test --lib --target "$TARGET" --no-default-features --features serde simd::
