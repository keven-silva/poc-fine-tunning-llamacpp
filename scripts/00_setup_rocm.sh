#!/usr/bin/env bash
# Stage 0 -- verify and configure the AMD ROCm/HIP environment for llama.cpp build.
#
# Unlike CUDA, ROCm relies on the host's ROCm SDK/drivers (typically in /opt/rocm).
# This script ensures hipcc is accessible and checks kernel driver permissions.
set -euo pipefail

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# 1. Verifica se o hipcc ja esta no PATH ou no diretório padrão do ROCm
if ! command -v hipcc >/dev/null 2>&1; then
  if [ -x "$ROCM_PATH/bin/hipcc" ]; then
    export PATH="$ROCM_PATH/bin:$PATH"
    echo ">> added $ROCM_PATH/bin to PATH"
  fi
fi

# 2. Valida a presença do compilador hipcc
if command -v hipcc >/dev/null 2>&1; then
  HIPCC_BIN="$(command -v hipcc)"
  echo ">> ROCm/HIP compiler found at $HIPCC_BIN"
  echo ">> ok: $($HIPCC_BIN --version | head -1)"
else
  echo "FATAL: hipcc not found on PATH or in $ROCM_PATH/bin." >&2
  echo "Please ensure ROCm dev tools are installed (e.g. rocm-dev / hip-dev)." >&2
  exit 1
fi

# 3. Valida se o usuario tem permissao no /dev/kfd
if [ -r /dev/kfd ] && [ -w /dev/kfd ]; then
  echo ">> /dev/kfd is accessible (read-write ok)"
else
  echo "WARNING: /dev/kfd is not readable/writable by current user." >&2
  echo "         Ensure user is in 'render'/'video' groups or run: sudo chmod 666 /dev/kfd" >&2
fi

echo ">> ROCm setup ready"