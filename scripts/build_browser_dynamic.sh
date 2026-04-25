#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMSDK_DIR="${EMSDK_DIR:-/tmp/emsdk}"
EMSDK_ENV="${EMSDK_DIR}/emsdk_env.sh"
EMDAWNWEBGPU_DIR_DEFAULT="/tmp/emdawnwebgpu/emdawnwebgpu_pkg"
EMDAWNWEBGPU_DIR="${EMDAWNWEBGPU_DIR:-$EMDAWNWEBGPU_DIR_DEFAULT}"
THREADS="${THREADS:-8}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
if command -v nproc >/dev/null 2>&1; then
  BUILD_JOBS="$(nproc)"
else
  BUILD_JOBS="$(getconf _NPROCESSORS_ONLN)"
fi

if [[ ! -f "$EMSDK_ENV" ]]; then
  echo "missing emsdk env: $EMSDK_ENV" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$EMSDK_ENV" >/dev/null

cd "$ROOT_DIR"

common_flags="-sALLOW_MEMORY_GROWTH=1 -sASSERTIONS=0 -sEXIT_RUNTIME=1"
pthread_flags="${common_flags} -sUSE_PTHREADS=1 -sPTHREAD_POOL_SIZE=${THREADS} -pthread"

rm -rf build-wasm-web-dyn build-wasm-web-pthread-dyn build-wasm-webgpu-browser-dyn

emcmake "$CMAKE_BIN" -S . -B build-wasm-web-dyn \
  -DGGML_WEBGPU=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS="$common_flags"
"$CMAKE_BIN" --build build-wasm-web-dyn --target embedding_wasm_model_bench -j"$BUILD_JOBS"

emcmake "$CMAKE_BIN" -S . -B build-wasm-web-pthread-dyn \
  -DGGML_WEBGPU=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-sUSE_PTHREADS=1 -pthread" \
  -DCMAKE_CXX_FLAGS="-sUSE_PTHREADS=1 -pthread" \
  -DCMAKE_EXE_LINKER_FLAGS="$pthread_flags"
"$CMAKE_BIN" --build build-wasm-web-pthread-dyn --target embedding_wasm_model_bench -j"$BUILD_JOBS"

webgpu_args=(
  -DGGML_WEBGPU=ON
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_EXE_LINKER_FLAGS="$common_flags"
)

if [[ -d "$EMDAWNWEBGPU_DIR" ]]; then
  webgpu_args+=("-DEMDAWNWEBGPU_DIR=${EMDAWNWEBGPU_DIR}")
fi

emcmake "$CMAKE_BIN" -S . -B build-wasm-webgpu-browser-dyn "${webgpu_args[@]}"
"$CMAKE_BIN" --build build-wasm-webgpu-browser-dyn --target embedding_wasm_model_bench -j"$BUILD_JOBS"
