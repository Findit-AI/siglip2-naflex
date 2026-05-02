#!/bin/bash
set -ex

export ASAN_OPTIONS="detect_odr_violation=0 detect_leaks=0"

TARGET="x86_64-unknown-linux-gnu"

# Scope: SIMD module only.
#
# Every `unsafe` block in this crate lives under `src/simd/` (verified
# by `grep -rn unsafe src/`). The rest of the codebase is safe Rust,
# so sanitizers add no signal there. Limiting to `simd::` tests:
#
#   1. Removes the false-positive churn from tokenizers' C/C++ deps
#      (`onig_sys`, `esaxx-rs`) — those aren't sanitizer-instrumented,
#      so MSAN reports `use-of-uninitialized-value` inside them on
#      every run. Not our bug, not fixable in our code.
#   2. Cuts wall time and surface area: the SIMD test set is small,
#      deterministic, and exercises every load/store/intrinsic path.
#   3. `--no-default-features --features serde` skips `inference`
#      (ort + tokenizers) and `decoders` (image/jpeg, image/png) for
#      the same reason — they're external C/C++/FFI surface that
#      sanitizers don't instrument.
#
# Codex round-32 CI sweep.

# Run address sanitizer
RUSTFLAGS="-Z sanitizer=address" \
cargo test --lib --target "$TARGET" --no-default-features --features serde simd::

# Run leak sanitizer
RUSTFLAGS="-Z sanitizer=leak" \
cargo test --lib --target "$TARGET" --no-default-features --features serde simd::

# Run memory sanitizer (requires -Zbuild-std for instrumented std)
RUSTFLAGS="-Z sanitizer=memory" \
cargo -Zbuild-std test --lib --target "$TARGET" --no-default-features --features serde simd::

# Run thread sanitizer (requires -Zbuild-std for instrumented std).
# Note: SIMD code in this crate has no concurrency primitives — TSAN
# is kept here for symmetry with the colconv template and to catch
# any future regression that introduces shared state. Cheap to run.
RUSTFLAGS="-Z sanitizer=thread" \
cargo -Zbuild-std test --lib --target "$TARGET" --no-default-features --features serde simd::
