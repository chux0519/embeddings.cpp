#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-192.168.1.101}"
REMOTE_DIR="${REMOTE_DIR:-~/repos/embeddings.cpp-macmini}"
MODEL_NAME="${MODEL_NAME:-snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf}"
MODEL_SRC="${MODEL_SRC:-models/${MODEL_NAME}}"
MODEL_DST="${MODEL_DST:-models/${MODEL_NAME}}"
PORT="${PORT:-18081}"
REMOTE_EMDAWNWEBGPU_DIR="${REMOTE_EMDAWNWEBGPU_DIR:-$REMOTE_DIR/third_party/emdawnwebgpu}"

if [[ ! -f "${MODEL_SRC}" ]]; then
  echo "missing model: ${MODEL_SRC}" >&2
  exit 1
fi

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
)

RSYNC_RSH="ssh ${SSH_OPTS[*]}"
export RSYNC_RSH

echo "[1/5] probing ${HOST}"
REMOTE_DIR_ABS="$(
  ssh "${SSH_OPTS[@]}" "${HOST}" "python3 - <<'PY'
import os
print(os.path.expanduser(${REMOTE_DIR@Q}))
PY"
)"
ssh "${SSH_OPTS[@]}" "${HOST}" "hostname; uname -a; mkdir -p ${REMOTE_DIR_ABS@Q}"

LOCAL_EMDAWNWEBGPU_DIR=""
if [[ -f /tmp/emdawnwebgpu/emdawnwebgpu_pkg/emdawnwebgpu.port.py ]]; then
  LOCAL_EMDAWNWEBGPU_DIR="/tmp/emdawnwebgpu/emdawnwebgpu_pkg"
fi

echo "[2/5] syncing repo"
rsync -az --delete \
  --exclude .git \
  --exclude .venv \
  --exclude .cache \
  --exclude __pycache__ \
  --exclude node_modules \
  --exclude 'models' \
  --exclude 'build' \
  --exclude 'build-*' \
  --exclude '*.pyc' \
  ./ "${HOST}:${REMOTE_DIR_ABS}/"

echo "[3/5] syncing model"
ssh "${SSH_OPTS[@]}" "${HOST}" "mkdir -p ${REMOTE_DIR_ABS@Q}/models"
rsync -az "${MODEL_SRC}" "${HOST}:${REMOTE_DIR_ABS}/${MODEL_DST}"
if [[ -n "${LOCAL_EMDAWNWEBGPU_DIR}" ]]; then
  ssh "${SSH_OPTS[@]}" "${HOST}" "mkdir -p ${REMOTE_EMDAWNWEBGPU_DIR@Q}"
  rsync -az "${LOCAL_EMDAWNWEBGPU_DIR}/" "${HOST}:${REMOTE_EMDAWNWEBGPU_DIR}/"
fi

echo "[4/5] building wasm targets on remote"
ssh "${SSH_OPTS[@]}" "${HOST}" "bash -lc '
  set -euo pipefail
  CMAKE_BIN=/opt/homebrew/bin/cmake
  if [[ ! -x \"\$CMAKE_BIN\" ]]; then
    CMAKE_BIN=cmake
  fi
  cd ${REMOTE_DIR_ABS@Q}
  if [[ ! -d /tmp/emsdk ]]; then
    git clone https://github.com/emscripten-core/emsdk.git /tmp/emsdk
  fi
  cd /tmp/emsdk
  ./emsdk install 5.0.6
  ./emsdk activate 5.0.6
  source /tmp/emsdk/emsdk_env.sh >/dev/null 2>&1
  NODE_BIN=\"\$EMSDK_NODE\"
  NPM_BIN=\"\$(dirname \"\$EMSDK_NODE\")/npm\"
  cd ${REMOTE_DIR_ABS@Q}
  \"\$NPM_BIN\" install --no-save playwright
  rm -rf build-wasm-web build-wasm-web-pthread build-wasm-webgpu-browser
  emcmake \"\$CMAKE_BIN\" -S . -B build-wasm-web \
    -DGGML_WEBGPU=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_BLAS=OFF \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS=\"-O3 -msimd128\" -DCMAKE_CXX_FLAGS=\"-O3 -msimd128\" \
    -DCMAKE_EXE_LINKER_FLAGS=\"-sALLOW_MEMORY_GROWTH=1 -sASSERTIONS=0 -sEXIT_RUNTIME=1 --preload-file ${REMOTE_DIR_ABS}/models/${MODEL_NAME}@/models/${MODEL_NAME} --preload-file ${REMOTE_DIR_ABS}/scripts/data/snowflake_wasm_batch8.txt@/snowflake_wasm_batch.txt --preload-file ${REMOTE_DIR_ABS}/scripts/data/snowflake_wasm_batch1.txt@/snowflake_wasm_batch1.txt --preload-file ${REMOTE_DIR_ABS}/scripts/data/snowflake_wasm_short8.txt@/snowflake_wasm_short8.txt\"
  \"\$CMAKE_BIN\" --build build-wasm-web --target embedding_wasm_model_bench -j\$(sysctl -n hw.ncpu)
  emcmake \"\$CMAKE_BIN\" -S . -B build-wasm-web-pthread \
    -DGGML_WEBGPU=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_BLAS=OFF \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS=\"-O3 -msimd128 -pthread\" -DCMAKE_CXX_FLAGS=\"-O3 -msimd128 -pthread\" \
    -DCMAKE_EXE_LINKER_FLAGS=\"-sALLOW_MEMORY_GROWTH=1 -sASSERTIONS=0 -sEXIT_RUNTIME=1 -sUSE_PTHREADS=1 -sPTHREAD_POOL_SIZE=8 -pthread --preload-file ${REMOTE_DIR_ABS}/models/${MODEL_NAME}@/models/${MODEL_NAME} --preload-file ${REMOTE_DIR_ABS}/scripts/data/snowflake_wasm_batch8.txt@/snowflake_wasm_batch.txt --preload-file ${REMOTE_DIR_ABS}/scripts/data/snowflake_wasm_batch1.txt@/snowflake_wasm_batch1.txt --preload-file ${REMOTE_DIR_ABS}/scripts/data/snowflake_wasm_short8.txt@/snowflake_wasm_short8.txt\"
  \"\$CMAKE_BIN\" --build build-wasm-web-pthread --target embedding_wasm_model_bench -j\$(sysctl -n hw.ncpu)
  WEBGPU_ARGS=(
    -DGGML_WEBGPU=ON
    -DGGML_METAL=OFF
    -DGGML_VULKAN=OFF
    -DGGML_CUDA=OFF
    -DGGML_BLAS=OFF
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS=\"-O3 -msimd128\" -DCMAKE_CXX_FLAGS=\"-O3 -msimd128\" \
    -DCMAKE_EXE_LINKER_FLAGS=\"-sALLOW_MEMORY_GROWTH=1 -sASSERTIONS=0 -sEXIT_RUNTIME=1 --preload-file ${REMOTE_DIR_ABS}/models/${MODEL_NAME}@/models/${MODEL_NAME} --preload-file ${REMOTE_DIR_ABS}/scripts/data/snowflake_wasm_batch8.txt@/snowflake_wasm_batch.txt --preload-file ${REMOTE_DIR_ABS}/scripts/data/snowflake_wasm_batch1.txt@/snowflake_wasm_batch1.txt --preload-file ${REMOTE_DIR_ABS}/scripts/data/snowflake_wasm_short8.txt@/snowflake_wasm_short8.txt\"
  )
  if [[ -f ${REMOTE_EMDAWNWEBGPU_DIR@Q}/emdawnwebgpu.port.py ]]; then
    WEBGPU_ARGS+=( -DEMDAWNWEBGPU_DIR=${REMOTE_EMDAWNWEBGPU_DIR@Q} )
  fi
  emcmake \"\$CMAKE_BIN\" -S . -B build-wasm-webgpu-browser \"\${WEBGPU_ARGS[@]}\"
  \"\$CMAKE_BIN\" --build build-wasm-webgpu-browser --target embedding_wasm_model_bench -j\$(sysctl -n hw.ncpu)
  ./scripts/build_browser_dynamic.sh
'"

echo "[5/5] running browser bench on remote"
ssh "${SSH_OPTS[@]}" "${HOST}" "bash -lc '
  set -euo pipefail
  source /tmp/emsdk/emsdk_env.sh >/dev/null 2>&1
  NODE_BIN=\"\$EMSDK_NODE\"
  cd ${REMOTE_DIR_ABS@Q}
  python3 scripts/browser_wasm_bench_server.py --host 127.0.0.1 --port ${PORT} --root ${REMOTE_DIR_ABS@Q} >/tmp/embeddings-browser-bench.log 2>&1 &
  server_pid=\$!
  trap \"kill \$server_pid >/dev/null 2>&1 || true\" EXIT
  sleep 2
  EXECUTABLE_PATH=\"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome\" MODE=bundled \"\$NODE_BIN\" scripts/run_browser_cases.mjs
  EXECUTABLE_PATH=\"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome\" MODE=downloaded MODEL_URL=\"http://127.0.0.1:${PORT}/models/${MODEL_NAME}\" BATCH_URL=\"http://127.0.0.1:${PORT}/scripts/data/snowflake_wasm_batch1.txt\" \"\$NODE_BIN\" scripts/run_browser_cases.mjs
'"
