#!/usr/bin/env bash
# Evaluate base and SFT checkpoints on held-out init states 40..49.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/dongyicheng/miniconda3}"
RPENT_CONDA_ENV="${RPENT_CONDA_ENV:-rpent}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/home/dongyicheng/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT}"
POST_CHECKPOINT="${POST_CHECKPOINT:-${REPO_ROOT}/logs/distillation/sft/libero_harness_pi05_sft/deploy_step_200}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${REPO_ROOT}/logs/distillation/sft/libero_harness_pi05_sft/standalone_eval}"
VLA_GPU="${VLA_GPU:-3}"
VLA_HOST="${VLA_HOST:-127.0.0.1}"
VLA_PORT="${VLA_PORT:-18082}"
VLA_ENDPOINT="http://${VLA_HOST}:${VLA_PORT}"
VLA_READY_TIMEOUT_S="${VLA_READY_TIMEOUT_S:-600}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${RPENT_CONDA_ENV}"
unset __EGL_VENDOR_LIBRARY_DIRS
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

VLA_PID=""
VLA_FIFO=""
cleanup() {
  if [[ -n "${VLA_PID}" ]] && kill -0 "${VLA_PID}" 2>/dev/null; then
    kill -INT "${VLA_PID}" 2>/dev/null || true
    wait "${VLA_PID}" 2>/dev/null || true
  fi
  if [[ -n "${VLA_FIFO}" && -p "${VLA_FIFO}" ]]; then
    rm -f "${VLA_FIFO}"
  fi
}
trap cleanup EXIT INT TERM

run_eval() {
  local label=$1
  local checkpoint=$2
  local report="${OUTPUT_DIR}/${label}.json"
  [[ -d "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 1; }
  VLA_FIFO="${OUTPUT_DIR}/${label}_vla.fifo"
  if [[ -p "${VLA_FIFO}" ]]; then
    rm -f "${VLA_FIFO}"
  elif [[ -e "${VLA_FIFO}" ]]; then
    echo "non-FIFO path blocks VLA service: ${VLA_FIFO}" >&2
    exit 1
  fi
  mkfifo "${VLA_FIFO}"
  exec 8<>"${VLA_FIFO}"
  CUDA_VISIBLE_DEVICES="${VLA_GPU}" "${PYTHON_BIN}" robots/libero/vla_server.py \
    --transport http --host "${VLA_HOST}" --port "${VLA_PORT}" \
    --model-path "${checkpoint}" < "${VLA_FIFO}" \
    > "${OUTPUT_DIR}/${label}_vla.log" 2>&1 &
  VLA_PID=$!
  local ready=0
  local deadline=$(( $(date +%s) + VLA_READY_TIMEOUT_S ))
  while (( $(date +%s) < deadline )); do
    if ! kill -0 "${VLA_PID}" 2>/dev/null; then break; fi
    if "${PYTHON_BIN}" scripts/baseline/baseline_results.py wait-rpc \
        --url "${VLA_ENDPOINT}" --timeout-s 2 --interval-s 0.5 >/dev/null 2>&1; then
      ready=1
      break
    fi
  done
  [[ "${ready}" == "1" ]] || { tail -n 100 "${OUTPUT_DIR}/${label}_vla.log" >&2; exit 1; }
  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_standalone_vla.py" \
    --endpoint "${VLA_ENDPOINT}" --checkpoint "${checkpoint}" --output "${report}" \
    --suite libero_object_lan --task-id 0 --seed-start 40 --seed-stop-exclusive 50
  kill -INT "${VLA_PID}" 2>/dev/null || true
  wait "${VLA_PID}" 2>/dev/null || true
  VLA_PID=""
  exec 8>&-
  rm -f "${VLA_FIFO}"
  VLA_FIFO=""
}

run_eval pre "${BASE_CHECKPOINT}"
run_eval post "${POST_CHECKPOINT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_standalone_evals.py" \
  --pre "${OUTPUT_DIR}/pre.json" --post "${OUTPUT_DIR}/post.json" \
  --output "${OUTPUT_DIR}/comparison.json"
