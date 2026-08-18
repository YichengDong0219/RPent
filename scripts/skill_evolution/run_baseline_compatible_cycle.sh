#!/usr/bin/env bash
# One-click serial batch of baseline-compatible skill evolution cycles.
# Edit only the quick configuration section, then run:
#   bash scripts/skill_evolution/run_baseline_compatible_cycle.sh

set -Eeuo pipefail

# =============================================================================
# Quick configuration
# =============================================================================
EXPERIMENT_NAME="libero_object_t0_failure_fix_evolution_v1"
# Base LIBERO checkout. Standard suites use its assets directly. PRO suites
# use the installed liberopro assets selected by LIBERO_TYPE while retaining
# this checkout for the shared LIBERO import surface.
LIBERO_CHECKOUT="/home/dongyicheng/LIBERO"
LIBERO_TYPE="pro"

# Evolution targets, run serially in the listed order. Each entry uses
# SUITE:TASK syntax and gets an independent S000 -> S001 cycle/output tree.
EVAL_TASKS=(
  "libero_object_lan:0"
)
# Causal task-local evolution stream. A proposal batch learns a Failure/Fix
# contrast; forward and retention run only after the target causal gate passes.
SEED_START=0
SEED_STOP_EXCLUSIVE=50
PROPOSAL_SEED_COUNT=5
REPEATS_PER_SEED=2
PROPOSAL_TOPUP_SEED_COUNT=2
MAX_PROPOSAL_ROLLOUTS=18
MIN_FAILURE_SUPPORT=2
MIN_SUCCESS_REFERENCES=1
FORWARD_SEED_COUNT=2
FORWARD_REPEATS=2
RETENTION_CASES=4
MAX_CANDIDATE_ROUNDS_PER_CLUSTER=2
DIAGNOSER_MAX_IMAGES=6
MAX_CYCLES_PER_RUN=1
MAX_CONSECUTIVE_NO_GAIN=3
RESET_STALLED=0

# Execution planner (same values as the baseline experiment).
PLANNER="api"
PLANNER_MODEL="qwen-vl:Qwen3.5-9B"
QWEN_BASE_URL="http://114.212.227.193:8000/v1"
QWEN_API_KEY="EMPTY"
MAX_TOKENS=24576
MAX_TURNS=40
PLANNER_SEED_BASE=100000

# Independent stronger multimodal skill optimizer (OpenAI-compatible API).
SKILL_OPTIMIZER_BASE_URL="http://114.212.227.193:8000/v1"
SKILL_OPTIMIZER_API_KEY="EMPTY"
SKILL_OPTIMIZER_MODEL="Qwen3.5-9B"
SKILL_OPTIMIZER_MAX_TOKENS=24576
SKILL_OPTIMIZER_TIMEOUT_S=600
EVIDENCE_MAX_IMAGES_PER_ROLLOUT=6
SKILL_DIAGNOSER_PATH="scripts/skill_evolution/skill_diagnoser/SKILL.md"
SKILL_PATCH_WRITER_PATH="scripts/skill_evolution/skill_patch_writer/SKILL.md"
MAX_PATCH_LINES=24
MAX_PATCH_NEW_CHARS=2000
MAX_PATCH_GROWTH_CHARS=1000

# Pi0.5: same checkpoint, endpoint and unrestricted baseline tool schema.
PI05_CHECKPOINT="/home/dongyicheng/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT"
VLA_GPU="3"
START_SHARED_VLA=1
VLA_ENDPOINT="http://127.0.0.1:18081"
VLA_HOST="127.0.0.1"
VLA_PORT=18081
VLA_READY_TIMEOUT_S=600

# Runtime and output.
CONDA_ROOT="/home/dongyicheng/miniconda3"
CONDA_ENV="rpent"
MAX_EPISODE_STEPS=10000
HIRES_RETENTION_STEPS=5
RUN_TIMEOUT_S=3600
MAX_ATTEMPTS=3
OUTPUT_ROOT="/home/dongyicheng/rpent/logs/skill_evolution"
# =============================================================================

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
unset __EGL_VENDOR_LIBRARY_DIRS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
EXPERIMENT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
SERVICES_DIR="${EXPERIMENT_DIR}/services"
mkdir -p "${SERVICES_DIR}"

# Resolve and validate the complete batch before starting either shared service.
EVAL_SUITES=()
EVAL_TASK_IDS=()
EVAL_TASK_DIRS=()
declare -A SEEN_EVAL_TASKS=()
for task_spec in "${EVAL_TASKS[@]}"; do
  if [[ ! "${task_spec}" =~ ^([^:[:space:]]+):([0-9]+)$ ]]; then
    echo "[skill-evolve] ERROR: invalid EVAL_TASKS entry '${task_spec}'; expected SUITE:TASK" >&2
    exit 1
  fi
  eval_suite="${BASH_REMATCH[1]}"
  eval_task_id=$((10#${BASH_REMATCH[2]}))
  eval_task_key="${eval_suite}:$eval_task_id"
  if [[ -n "${SEEN_EVAL_TASKS[${eval_task_key}]:-}" ]]; then
    echo "[skill-evolve] ERROR: duplicate EVAL_TASKS entry '${eval_task_key}'" >&2
    exit 1
  fi
  SEEN_EVAL_TASKS["${eval_task_key}"]=1
  EVAL_SUITES+=("${eval_suite}")
  EVAL_TASK_IDS+=("${eval_task_id}")
  printf -v eval_task_dir '%s/tasks/%s__t%03d' \
    "${EXPERIMENT_DIR}" "${eval_suite}" "${eval_task_id}"
  EVAL_TASK_DIRS+=("${eval_task_dir}")
done
if (( ${#EVAL_SUITES[@]} == 0 )); then
  echo "[skill-evolve] ERROR: EVAL_TASKS must contain at least one SUITE:TASK entry" >&2
  exit 1
fi

# Pin imports to the user-selected LIBERO checkout without editing that repo.
export PYTHONPATH="${LIBERO_CHECKOUT}/libero:${PYTHONPATH:-}"
export LIBERO_TYPE
export QWEN_VL_BASE_URL="${QWEN_BASE_URL}"
export QWEN_VL_API_KEY="${QWEN_API_KEY}"
export HF_HUB_OFFLINE=1
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

VLA_PID=""
VLA_STDIN_FIFO=""
VLA_STDIN_FD=""
VLA_STDIN_OWNED=0
cleanup() {
  if [[ -n "${VLA_PID}" ]]; then
    "${PYTHON_BIN}" scripts/baseline/baseline_results.py shutdown-rpc \
      --url "${VLA_ENDPOINT}" --timeout-s 10 >/dev/null 2>&1 || true
    wait "${VLA_PID}" 2>/dev/null || true
  fi
  if [[ -n "${VLA_STDIN_FD}" ]]; then
    exec {VLA_STDIN_FD}>&-
  fi
  if [[ "${VLA_STDIN_OWNED}" == "1" && -n "${VLA_STDIN_FIFO}" \
      && -p "${VLA_STDIN_FIFO}" ]]; then
    rm -f -- "${VLA_STDIN_FIFO}"
  fi
}
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/qwen_vl/check_server.py \
  --base-url "${QWEN_BASE_URL}" \
  --api-key "${QWEN_API_KEY}" \
  --model "${PLANNER_MODEL#qwen-vl:}"
"${PYTHON_BIN}" -m rpent.evolution.cli check-optimizer \
  --base-url "${SKILL_OPTIMIZER_BASE_URL}" \
  --api-key "${SKILL_OPTIMIZER_API_KEY}" \
  --model "${SKILL_OPTIMIZER_MODEL}" \
  --timeout-s 60

if [[ "${START_SHARED_VLA}" == "1" ]]; then
  VLA_STDIN_FIFO="${SERVICES_DIR}/vla_stdin.fifo"
  if [[ -e "${VLA_STDIN_FIFO}" ]]; then
    echo "[skill-evolve] ERROR: stale VLA stdin FIFO: ${VLA_STDIN_FIFO}" >&2
    exit 1
  fi
  mkfifo "${VLA_STDIN_FIFO}"
  VLA_STDIN_OWNED=1
  # RpcFacade watches stdin for EOF as its parent-death signal. Keep both
  # ends of this FIFO open for the complete cycle so a Bash background job
  # does not receive /dev/null and shut itself down immediately.
  exec {VLA_STDIN_FD}<>"${VLA_STDIN_FIFO}"
  CUDA_VISIBLE_DEVICES="${VLA_GPU}" \
    "${PYTHON_BIN}" robots/libero/vla_server.py \
      --transport http --host "${VLA_HOST}" --port "${VLA_PORT}" \
      --model-path "${PI05_CHECKPOINT}" \
      < "${VLA_STDIN_FIFO}" \
      > "${SERVICES_DIR}/vla_server.log" 2>&1 &
  VLA_PID=$!
fi
VLA_READY=0
VLA_READY_DEADLINE=$(( $(date +%s) + VLA_READY_TIMEOUT_S ))
while (( $(date +%s) < VLA_READY_DEADLINE )); do
  if [[ -n "${VLA_PID}" ]] && ! kill -0 "${VLA_PID}" 2>/dev/null; then
    echo "[skill-evolve] ERROR: Pi0.5 service exited during startup" >&2
    tail -n 100 "${SERVICES_DIR}/vla_server.log" >&2 || true
    exit 1
  fi
  if "${PYTHON_BIN}" scripts/baseline/baseline_results.py wait-rpc \
      --url "${VLA_ENDPOINT}" --timeout-s 2 --interval-s 0.5 \
      >/dev/null 2>&1; then
    VLA_READY=1
    break
  fi
done
if [[ "${VLA_READY}" != "1" ]]; then
  echo "[skill-evolve] ERROR: Pi0.5 service was not ready within ${VLA_READY_TIMEOUT_S}s" >&2
  tail -n 100 "${SERVICES_DIR}/vla_server.log" >&2 || true
  exit 1
fi
echo "[skill-evolve] Pi0.5 service ready: ${VLA_ENDPOINT}"

BATCH_EXIT_CODE=0
TASK_EXIT_CODES=()
for task_index in "${!EVAL_SUITES[@]}"; do
  eval_suite="${EVAL_SUITES[task_index]}"
  eval_task_id="${EVAL_TASK_IDS[task_index]}"
  eval_task_dir="${EVAL_TASK_DIRS[task_index]}"
  echo "[skill-evolve] BATCH $((task_index + 1))/${#EVAL_SUITES[@]}: ${eval_suite}:$eval_task_id"
  echo "[skill-evolve] task output: ${eval_task_dir}"

  task_exit_code=0
  stream_args=(
    "${PYTHON_BIN}" scripts/skill_evolution/run_stream.py
    --repo-root "${REPO_ROOT}" \
    --libero-root "${LIBERO_CHECKOUT}" \
    --experiment-dir "${eval_task_dir}" \
    --memory-dir "${REPO_ROOT}/resources/libero/memory" \
    --suite "${eval_suite}" --task "${eval_task_id}" \
    --seed-start "${SEED_START}" \
    --seed-stop-exclusive "${SEED_STOP_EXCLUSIVE}" \
    --proposal-seed-count "${PROPOSAL_SEED_COUNT}" \
    --repeats-per-seed "${REPEATS_PER_SEED}" \
    --proposal-topup-seed-count "${PROPOSAL_TOPUP_SEED_COUNT}" \
    --max-proposal-rollouts "${MAX_PROPOSAL_ROLLOUTS}" \
    --min-failure-support "${MIN_FAILURE_SUPPORT}" \
    --min-success-references "${MIN_SUCCESS_REFERENCES}" \
    --forward-seed-count "${FORWARD_SEED_COUNT}" \
    --forward-repeats "${FORWARD_REPEATS}" \
    --retention-cases "${RETENTION_CASES}" \
    --max-candidate-rounds-per-cluster "${MAX_CANDIDATE_ROUNDS_PER_CLUSTER}" \
    --diagnoser-max-images "${DIAGNOSER_MAX_IMAGES}" \
    --max-cycles-per-run "${MAX_CYCLES_PER_RUN}" \
    --max-consecutive-no-gain "${MAX_CONSECUTIVE_NO_GAIN}" \
    --planner "${PLANNER}" --model "${PLANNER_MODEL}" \
    --qwen-base-url "${QWEN_BASE_URL}" --qwen-api-key "${QWEN_API_KEY}" \
    --optimizer-base-url "${SKILL_OPTIMIZER_BASE_URL}" \
    --optimizer-api-key "${SKILL_OPTIMIZER_API_KEY}" \
    --optimizer-model "${SKILL_OPTIMIZER_MODEL}" \
    --optimizer-max-tokens "${SKILL_OPTIMIZER_MAX_TOKENS}" \
    --optimizer-timeout-s "${SKILL_OPTIMIZER_TIMEOUT_S}" \
    --evidence-max-images-per-rollout "${EVIDENCE_MAX_IMAGES_PER_ROLLOUT}" \
    --diagnoser-skill-path "${REPO_ROOT}/${SKILL_DIAGNOSER_PATH}" \
    --patch-writer-skill-path "${REPO_ROOT}/${SKILL_PATCH_WRITER_PATH}" \
    --max-patch-lines "${MAX_PATCH_LINES}" \
    --max-patch-new-chars "${MAX_PATCH_NEW_CHARS}" \
    --max-patch-growth-chars "${MAX_PATCH_GROWTH_CHARS}" \
    --vla-endpoint "${VLA_ENDPOINT}" --libero-type "${LIBERO_TYPE}" \
    --cuda-device "${VLA_GPU}" --max-tokens "${MAX_TOKENS}" \
    --planner-seed-base "${PLANNER_SEED_BASE}" \
    --max-turns "${MAX_TURNS}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --hires-retention-steps "${HIRES_RETENTION_STEPS}" \
    --run-timeout-s "${RUN_TIMEOUT_S}" --max-attempts "${MAX_ATTEMPTS}" \
    --python "${PYTHON_BIN}"
  )
  if [[ "${RESET_STALLED}" == "1" ]]; then
    stream_args+=(--reset-stalled)
  fi
  "${stream_args[@]}" || task_exit_code=$?

  TASK_EXIT_CODES+=("${task_exit_code}")
  if (( task_exit_code != 0 )); then
    BATCH_EXIT_CODE=1
    echo "[skill-evolve] WARN: ${eval_suite}:$eval_task_id exited with status ${task_exit_code}; continuing batch" >&2
  fi
done

echo "[skill-evolve] Batch summary:"
for task_index in "${!EVAL_SUITES[@]}"; do
  echo "[skill-evolve]   ${EVAL_SUITES[task_index]}:${EVAL_TASK_IDS[task_index]} -> exit ${TASK_EXIT_CODES[task_index]} (${EVAL_TASK_DIRS[task_index]})"
done
exit "${BATCH_EXIT_CODE}"
