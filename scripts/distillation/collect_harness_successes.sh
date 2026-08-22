#!/usr/bin/env bash
# Collect exactly SUCCESS_TARGET authoritative Harness successes for VLA SFT.
# Edit the quick configuration section or override values through the environment.

set -Eeuo pipefail

# =============================================================================
# Quick configuration
# =============================================================================
EXPERIMENT_NAME="${EXPERIMENT_NAME:-libero_object_lan_t0_harness_sft_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/dongyicheng/rpent/logs/distillation}"
SUITE="${SUITE:-libero_object_lan}"
TASK_ID="${TASK_ID:-0}"
TASK_INSTRUCTION="${TASK_INSTRUCTION:-grab alphabet soup and put it into basket}"
SUCCESS_TARGET="${SUCCESS_TARGET:-20}"
INIT_STATE_START="${INIT_STATE_START:-0}"
INIT_STATE_STOP_EXCLUSIVE="${INIT_STATE_STOP_EXCLUSIVE:-40}"
MAX_PLANNER_SEEDS_PER_INIT="${MAX_PLANNER_SEEDS_PER_INIT:-3}"
PLANNER_SEED_BASE="${PLANNER_SEED_BASE:-200000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-120}"

MEMORY_DIR="${MEMORY_DIR:-/home/dongyicheng/rpent/resources/libero/memory}"
PI05_CHECKPOINT="${PI05_CHECKPOINT:-/home/dongyicheng/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT}"
VLA_GPU="${VLA_GPU:-3}"
START_SHARED_VLA="${START_SHARED_VLA:-1}"
VLA_ENDPOINT="${VLA_ENDPOINT:-http://127.0.0.1:18081}"
VLA_HOST="${VLA_HOST:-127.0.0.1}"
VLA_PORT="${VLA_PORT:-18081}"
VLA_READY_TIMEOUT_S="${VLA_READY_TIMEOUT_S:-600}"

PLANNER="${PLANNER:-api}"
if [[ ! -v PLANNER_MODEL ]]; then
  if [[ "${PLANNER}" == "api" ]]; then
    PLANNER_MODEL="qwen-vl:Qwen3.5-9B"
  else
    PLANNER_MODEL=""
  fi
fi
PLANNER_BASE_URL="${PLANNER_BASE_URL:-http://114.212.227.193:8000/v1}"
QWEN_VL_API_KEY="${QWEN_VL_API_KEY:-EMPTY}"
CHECK_PLANNER_SERVER="${CHECK_PLANNER_SERVER:-1}"
PLANNER_TIMEOUT_S="${PLANNER_TIMEOUT_S:-}"
MAX_TOKENS="${MAX_TOKENS:-24576}"
MAX_TURNS="${MAX_TURNS:-40}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-10000}"
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-3600}"
HIRES_RETENTION_STEPS="${HIRES_RETENTION_STEPS:-5}"
LIBERO_TYPE="${LIBERO_TYPE:-pro}"

CONDA_ROOT="${CONDA_ROOT:-/home/dongyicheng/miniconda3}"
RPENT_CONDA_ENV="${RPENT_CONDA_ENV:-rpent}"
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MANAGER="${SCRIPT_DIR}/manage_dataset.py"
CONDA_INIT="${CONDA_ROOT}/etc/profile.d/conda.sh"

die() {
  echo "[collect] ERROR: $*" >&2
  exit 1
}

[[ -f "${CONDA_INIT}" ]] || die "missing Conda init: ${CONDA_INIT}"
# shellcheck source=/dev/null
source "${CONDA_INIT}"
conda activate "${RPENT_CONDA_ENV}"
unset __EGL_VENDOR_LIBRARY_DIRS
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

for value in SUCCESS_TARGET INIT_STATE_START INIT_STATE_STOP_EXCLUSIVE MAX_PLANNER_SEEDS_PER_INIT MAX_ATTEMPTS; do
  [[ "${!value}" =~ ^[0-9]+$ ]] || die "${value} must be a non-negative integer"
done
(( SUCCESS_TARGET > 0 )) || die "SUCCESS_TARGET must be positive"
(( INIT_STATE_STOP_EXCLUSIVE > INIT_STATE_START )) || die "empty init-state range"
(( MAX_PLANNER_SEEDS_PER_INIT > 0 )) || die "MAX_PLANNER_SEEDS_PER_INIT must be positive"
(( MAX_ATTEMPTS > 0 && MAX_ATTEMPTS <= 120 )) || die "MAX_ATTEMPTS must be in 1..120"
[[ -d "${MEMORY_DIR}" ]] || die "memory directory not found: ${MEMORY_DIR}"
[[ -d "${PI05_CHECKPOINT}" ]] || die "checkpoint not found: ${PI05_CHECKPOINT}"

EXPERIMENT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
RUNS_DIR="${EXPERIMENT_DIR}/runs"
ACCEPTED_DIR="${EXPERIMENT_DIR}/accepted"
SERVICES_DIR="${EXPERIMENT_DIR}/services"
MEMORY_SNAPSHOT="${EXPERIMENT_DIR}/memory_snapshot"
MANIFEST="${EXPERIMENT_DIR}/success_manifest.json"
DATASET_DIR="${EXPERIMENT_DIR}/dataset"
mkdir -p "${RUNS_DIR}" "${ACCEPTED_DIR}" "${SERVICES_DIR}"

command -v flock >/dev/null 2>&1 || die "flock is required"
exec 9> "${EXPERIMENT_DIR}/collector.lock"
flock -n 9 || die "another collector owns ${EXPERIMENT_DIR}"

export PI05_CHECKPOINT_PATH="${PI05_CHECKPOINT}"
export CUDA_VISIBLE_DEVICES="${VLA_GPU}"
export LIBERO_TYPE QWEN_VL_API_KEY
export QWEN_VL_BASE_URL="${PLANNER_BASE_URL}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${MANAGER}" snapshot \
  --memory-dir "${MEMORY_DIR}" --output "${MEMORY_SNAPSHOT}"
"${PYTHON_BIN}" "${MANAGER}" init \
  --manifest "${MANIFEST}" --experiment "${EXPERIMENT_NAME}" \
  --suite "${SUITE}" --task-id "${TASK_ID}" --task "${TASK_INSTRUCTION}" \
  --success-target "${SUCCESS_TARGET}" --checkpoint "${PI05_CHECKPOINT}" \
  --memory-snapshot "${MEMORY_SNAPSHOT}"

if [[ "${CHECK_PLANNER_SERVER}" == "1" && "${PLANNER_MODEL}" == qwen-vl:* ]]; then
  "${PYTHON_BIN}" scripts/qwen_vl/check_server.py \
    --base-url "${PLANNER_BASE_URL}" --api-key "${QWEN_VL_API_KEY}" \
    --model "${PLANNER_MODEL#qwen-vl:}"
fi

VLA_PID=""
VLA_FIFO=""
VLA_FD=""
cleanup() {
  if [[ -n "${VLA_PID}" ]] && kill -0 "${VLA_PID}" 2>/dev/null; then
    kill -INT "${VLA_PID}" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "${VLA_PID}" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "${VLA_PID}" 2>/dev/null; then
      kill -TERM "${VLA_PID}" 2>/dev/null || true
      for _ in {1..10}; do
        kill -0 "${VLA_PID}" 2>/dev/null || break
        sleep 0.5
      done
    fi
    if kill -0 "${VLA_PID}" 2>/dev/null; then
      kill -KILL "${VLA_PID}" 2>/dev/null || true
    fi
    wait "${VLA_PID}" 2>/dev/null || true
  fi
  if [[ -n "${VLA_FD}" ]]; then
    exec {VLA_FD}>&- || true
    VLA_FD=""
  fi
  if [[ -n "${VLA_FIFO}" && -p "${VLA_FIFO}" ]]; then
    rm -f "${VLA_FIFO}"
  fi
}
trap cleanup EXIT INT TERM

if [[ "${START_SHARED_VLA}" == "1" ]]; then
  VLA_FIFO="${SERVICES_DIR}/vla_stdin.fifo"
  if [[ -p "${VLA_FIFO}" ]]; then
    rm -f "${VLA_FIFO}"
  elif [[ -e "${VLA_FIFO}" ]]; then
    die "non-FIFO path blocks VLA service: ${VLA_FIFO}"
  fi
  mkfifo "${VLA_FIFO}"
  exec {VLA_FD}<>"${VLA_FIFO}"
  CUDA_VISIBLE_DEVICES="${VLA_GPU}" \
    "${PYTHON_BIN}" robots/libero/vla_server.py \
      --transport http --host "${VLA_HOST}" --port "${VLA_PORT}" \
      --model-path "${PI05_CHECKPOINT}" < "${VLA_FIFO}" \
      > "${SERVICES_DIR}/vla_server.log" 2>&1 &
  VLA_PID=$!
fi

ready=0
deadline=$(( $(date +%s) + VLA_READY_TIMEOUT_S ))
while (( $(date +%s) < deadline )); do
  if [[ -n "${VLA_PID}" ]] && ! kill -0 "${VLA_PID}" 2>/dev/null; then
    break
  fi
  if "${PYTHON_BIN}" scripts/baseline/baseline_results.py wait-rpc \
      --url "${VLA_ENDPOINT}" --timeout-s 2 --interval-s 0.5 >/dev/null 2>&1; then
    ready=1
    break
  fi
done
if [[ "${ready}" != "1" ]]; then
  tail -n 100 "${SERVICES_DIR}/vla_server.log" >&2 2>/dev/null || true
  die "VLA service was not ready: ${VLA_ENDPOINT}"
fi

for (( init_state=INIT_STATE_START; init_state<INIT_STATE_STOP_EXCLUSIVE; init_state++ )); do
  for (( planner_repeat=0; planner_repeat<MAX_PLANNER_SEEDS_PER_INIT; planner_repeat++ )); do
    successes="$("${PYTHON_BIN}" "${MANAGER}" success-count --manifest "${MANIFEST}")"
    if (( successes >= SUCCESS_TARGET )); then
      break 2
    fi
    attempts="$("${PYTHON_BIN}" "${MANAGER}" attempt-count --manifest "${MANIFEST}")"
    if (( attempts >= MAX_ATTEMPTS )); then
      break 2
    fi
    planner_seed=$(( PLANNER_SEED_BASE + init_state * MAX_PLANNER_SEEDS_PER_INIT + planner_repeat ))
    attempt_key="init_${init_state}__planner_${planner_seed}"
    if "${PYTHON_BIN}" "${MANAGER}" attempt-recorded \
        --manifest "${MANIFEST}" --attempt-key "${attempt_key}"; then
      continue
    fi

    run_dir="${RUNS_DIR}/${attempt_key}"
    trajectory="${run_dir}/trajectory_raw.hdf5"
    mkdir -p "${run_dir}"
    echo "[collect] ${attempt_key}: ${successes}/${SUCCESS_TARGET} successes"
    cmd=(
      "${PYTHON_BIN}" -m rpent.cli.main
      --env libero --suite "${SUITE}" --task "${TASK_ID}" --seed "${init_state}"
      --libero-type "${LIBERO_TYPE}"
      --planner "${PLANNER}"
      --planner-sampling-seed "${planner_seed}"
      --max-tokens "${MAX_TOKENS}" --max-turns "${MAX_TURNS}"
      --max-episode-steps "${MAX_EPISODE_STEPS}"
      --hires-retention-steps "${HIRES_RETENTION_STEPS}"
      --cuda-device "${VLA_GPU}" --vla-endpoint "${VLA_ENDPOINT}"
      --memory-snapshot "${MEMORY_SNAPSHOT}"
      --trajectory-output "${trajectory}"
      --output-dir "${run_dir}"
    )
    if [[ -n "${PLANNER_MODEL}" ]]; then
      cmd+=(--model "${PLANNER_MODEL}")
    fi
    if [[ -n "${PLANNER_BASE_URL}" ]]; then
      cmd+=(--base-url "${PLANNER_BASE_URL}")
    fi
    if [[ -n "${PLANNER_TIMEOUT_S}" ]]; then
      cmd+=(--planner-timeout-s "${PLANNER_TIMEOUT_S}")
    fi
    set +e
    timeout --signal=INT --kill-after=120s "${RUN_TIMEOUT_S}s" \
      "${cmd[@]}" 2>&1 | tee "${run_dir}/console.log"
    status=("${PIPESTATUS[@]}")
    set -e
    process_exit_code="${status[0]}"
    "${PYTHON_BIN}" "${MANAGER}" record-attempt \
      --manifest "${MANIFEST}" --attempt-key "${attempt_key}" \
      --trajectory "${trajectory}" --run-dir "${run_dir}" \
      --accepted-dir "${ACCEPTED_DIR}" --init-state "${init_state}" \
      --planner-seed "${planner_seed}" --process-exit-code "${process_exit_code}"
  done
done

successes="$("${PYTHON_BIN}" "${MANAGER}" success-count --manifest "${MANIFEST}")"
if (( successes != SUCCESS_TARGET )); then
  die "collection exhausted with ${successes}/${SUCCESS_TARGET} successes"
fi
"${PYTHON_BIN}" "${MANAGER}" export \
  --manifest "${MANIFEST}" --output "${DATASET_DIR}" \
  --repo-id "rpent/${EXPERIMENT_NAME}" --expected-episodes "${SUCCESS_TARGET}"
echo "[collect] complete: ${DATASET_DIR}"
