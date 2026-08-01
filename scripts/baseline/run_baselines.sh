#!/usr/bin/env bash
# RPent / LIBERO-PRO 稳定批量评测脚本。
#
# 使用方法：
#   1. 修改下面的“用户快速配置区”
#   2. 预览任务：bash scripts/baseline/run_baselines.sh --dry-run
#   3. 正式运行：bash scripts/baseline/run_baselines.sh
#   4. 中断后再次执行相同命令，即可断点续跑

set -Eeuo pipefail

# Activate the complete RPent Conda environment before resolving any runtime
# paths. Calling envs/rpent/bin/python directly is not equivalent to activation:
# Conda's activation/deactivation hooks also remove base's Mesa-only
# __EGL_VENDOR_LIBRARY_DIRS override so MuJoCo can discover the system NVIDIA
# EGL provider.
CONDA_ROOT="${CONDA_ROOT:-/home/dongyicheng/miniconda3}"
RPENT_CONDA_ENV="${RPENT_CONDA_ENV:-rpent}"
CONDA_INIT="${CONDA_ROOT}/etc/profile.d/conda.sh"

if [[ ! -f "${CONDA_INIT}" ]]; then
  echo "[baseline] ERROR: Conda initialization script not found: ${CONDA_INIT}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_INIT}"
conda activate "${RPENT_CONDA_ENV}"
unset __EGL_VENDOR_LIBRARY_DIRS

if [[ "${CONDA_DEFAULT_ENV:-}" != "${RPENT_CONDA_ENV}" ]]; then
  echo "[baseline] ERROR: failed to activate Conda environment ${RPENT_CONDA_ENV}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULT_HELPER="${SCRIPT_DIR}/baseline_results.py"

# =============================================================================
# 用户快速配置区：通常只需要修改这一段
# =============================================================================

# -----------------------------------------------------------------------------
# 1. 实验名称与评测任务
# -----------------------------------------------------------------------------

# 实验名称会成为输出目录名。
# 更换模型、prompt、任务集合或关键参数后，请换一个新名称，避免结果混合。
EXPERIMENT_NAME="801_qwen35_9b_pi05_libero_pro_baseline"

# 任务配置模式，可选：
#   "cartesian"：最常用，运行 EVAL_SUITES × EVAL_TASK_IDS 的所有组合。
#   "list"：     精确指定若干 suite/task，适合不同 suite 跑不同 task。
#   "full"：     完整 LIBERO-PRO：16 个 suite × 10 个 task = 160 项。
EVAL_MODE="${EVAL_MODE:-full}"

# cartesian 模式配置。
# 当前示例只运行冒烟任务 libero_object_swap task 2。
# LIBERO-PRO suite 命名规则：
#   libero_{spatial|object|goal|10}_{swap|task|lan|object}
# 多 suite、多 task 示例：
#   EVAL_SUITES=("libero_object_swap" "libero_spatial_swap")
#   EVAL_TASK_IDS=(0 1 2 3)
# 运行 task 0～9 可写成：EVAL_TASK_IDS=({0..9})
EVAL_SUITES=(
  "libero_object_swap"
)
EVAL_TASK_IDS=(2)

# list 模式配置，每行严格写成 "<suite> <task_id>"。
# 仅当 EVAL_MODE="list" 时使用这里的内容。
EXACT_TASKS=(
  "libero_object_swap 2"
  # "libero_spatial_swap 0"
  # "libero_goal_task 5"
)

# 每个任务使用哪些随机种子，以及每个 seed 重复多少次。
# 例如 SEEDS=(0 1 2)、REPEATS=3：每个 suite/task 共运行 9 次。
SEEDS=(0)
REPEATS=3

# -----------------------------------------------------------------------------
# 2. Planner 与 Qwen-VL 服务
# -----------------------------------------------------------------------------

# 当前配置与已经启动的 Qwen3.5-9B OpenAI 兼容服务对齐。
PLANNER="api"
MODEL="qwen-vl:Qwen3.5-9B"
QWEN_VL_BASE_URL="${QWEN_VL_BASE_URL:-http://127.0.0.1:8000/v1}"
QWEN_VL_API_KEY="${QWEN_VL_API_KEY:-EMPTY}"

# 正式运行前检查 Qwen 的图片输入与工具调用，建议保持为 1。
CHECK_QWEN_SERVER=1

# -----------------------------------------------------------------------------
# 3. Pi0.5、GPU 与运行预算
# -----------------------------------------------------------------------------

# Pi0.5 checkpoint，以及运行 Pi0.5 的物理 GPU 编号。
PI05_CHECKPOINT_PATH="${PI05_CHECKPOINT_PATH:-/home/dongyicheng/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"

# 保持 1：脚本只加载一次 Pi0.5，后续任务复用该服务。
# 如果已经手动启动 Pi0.5 服务，可以填写：
# VLA_ENDPOINT="http://127.0.0.1:18081"
START_SHARED_VLA=1
VLA_ENDPOINT="${VLA_ENDPOINT:-}"
VLA_HOST="127.0.0.1"
VLA_PORT=18081
VLA_READY_TIMEOUT_S=600

# 当前 Qwen3.5-9B 服务的 max_model_len=131072（128K）。
# MAX_TOKENS=4096 是单次回复上限，可为 RPent 的长输入和多轮工具调用留足空间。
# RUN_TIMEOUT_S 是一个逻辑任务允许的最长实际运行时间，单位为秒。
MAX_TOKENS=4096
MAX_TURNS=40
MAX_EPISODE_STEPS=10000
RUN_TIMEOUT_S=3600

# -----------------------------------------------------------------------------
# 4. 结果留存、资源模式与失败重试
# -----------------------------------------------------------------------------

# 高清轨迹留存：
#   0：保留全部步骤的高清 RGB 和高清 world map，适合研究但占空间较大。
#   5：只保留最后 5 步高清数据；完整视频和逐步低清数据仍会保留。
HIRES_RETENTION_STEPS=5

# 任务结束后的最终留存策略：
#   "video_and_structured_logs"：删除逐步图片/深度/world map，只保留完整视频和日志。
#   "all"：保留上述全部逐步视觉文件。
# 视觉文件只在任务完全结束后删除，不会影响 Agent 运行中的观察和定位。
ARTIFACT_RETENTION="video_and_structured_logs"

# 资源模式：
#   "offline"：使用已经下载好的完整 memory/assets，不访问 Hugging Face。
#   "sync"：   允许从 Hugging Face 同步资源。
#   "ablation"：刻意禁用 memory/resources，用于消融实验。
RESOURCE_MODE="offline"
LIBERO_TYPE="pro"

# 只对超时、服务异常、Agent 异常、文件不完整进行重试。
# 正常执行但 benchmark 失败属于有效结果，不会重复尝试“刷成功”。
MAX_ATTEMPTS=3
RETRY_DELAY_S=20

# 剩余空间低于此阈值时停止启动新任务，避免写坏结果文件。
MIN_FREE_DISK_GB=20

# 输出目录结构：
# <OUTPUT_ROOT>/<EXPERIMENT_NAME>/runs/<run_id>/attempts/attempt_NN/
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/baselines}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"

# 后台运行：
#   1：直接执行本脚本时自动通过 nohup + setsid 转入后台，可安全断开 SSH。
#   0：始终在当前终端前台运行。
# 后台总日志固定写入：<OUTPUT_ROOT>/<EXPERIMENT_NAME>/runner.log
AUTO_DETACH="${AUTO_DETACH:-1}"

# 如需给所有 rpent 命令追加参数，在这里填写，例如：
# EXTRA_RPENT_ARGS=(--verbose)
EXTRA_RPENT_ARGS=()

# =============================================================================
# 用户快速配置区结束：以下通常不需要修改
# =============================================================================

# 根据快速配置生成内部任务矩阵。
TASK_MATRIX=()
case "${EVAL_MODE}" in
  cartesian)
    for suite_name in "${EVAL_SUITES[@]}"; do
      for task_id in "${EVAL_TASK_IDS[@]}"; do
        TASK_MATRIX+=("${suite_name} ${task_id}")
      done
    done
    ;;
  list)
    TASK_MATRIX=("${EXACT_TASKS[@]}")
    ;;
  full)
    for family in spatial object goal 10; do
      for perturbation in swap task lan object; do
        for task_id in {0..9}; do
          TASK_MATRIX+=("libero_${family}_${perturbation} ${task_id}")
        done
      done
    done
    ;;
esac

DRY_RUN=0
SUMMARIZE_ONLY=0
FOREGROUND=0
SHARED_VLA_PID=""
SHARED_VLA_STDIN_FD=""
SHARED_VLA_FIFO=""
OWN_SHARED_VLA=0
ACTIVE_RUN_ID=""
ACTIVE_ATTEMPT_DIR=""
ACTIVE_CANONICAL_RESULT=""
ACTIVE_START_EPOCH=""
RUNNER_LOG_FILE=""
RUNNER_PID_FILE=""

usage() {
  printf '%s\n' \
    "用法：bash scripts/baseline/run_baselines.sh [选项]" \
    "" \
    "默认：自动进入后台，终端立即返回。" \
    "" \
    "选项：" \
    "  --dry-run         在前台预览任务，不启动模型或实验" \
    "  --summarize-only  在前台重新生成汇总文件" \
    "  --foreground      强制在当前终端前台运行" \
    "  -h, --help        显示帮助"
}

die() {
  echo "[baseline] ERROR: $*" >&2
  exit 1
}

log() {
  echo "[baseline] $*"
}

cleanup() {
  local exit_code=$?
  local cleanup_end_epoch
  local cleanup_elapsed
  local recorded_runner_pid
  trap - EXIT INT TERM
  if [[ "${exit_code}" != "0" \
      && -n "${ACTIVE_ATTEMPT_DIR}" \
      && -f "${ACTIVE_ATTEMPT_DIR}/launch.json" \
      && ! -f "${ACTIVE_ATTEMPT_DIR}/result.json" ]]; then
    cleanup_end_epoch="$(date +%s)"
    cleanup_elapsed=0
    if [[ -n "${ACTIVE_START_EPOCH}" ]]; then
      cleanup_elapsed=$(( cleanup_end_epoch - ACTIVE_START_EPOCH ))
    fi
    "${PYTHON_BIN}" "${RESULT_HELPER}" finalize \
      --attempt-dir "${ACTIVE_ATTEMPT_DIR}" \
      --canonical-result "${ACTIVE_CANONICAL_RESULT}" \
      --exit-code "${exit_code}" \
      --timed-out 0 \
      --wall-elapsed-s "${cleanup_elapsed}" >/dev/null 2>&1 || true
    "${PYTHON_BIN}" "${RESULT_HELPER}" summarize \
      --experiment-dir "${EXPERIMENT_DIR}" >/dev/null 2>&1 || true
  fi
  if [[ "${OWN_SHARED_VLA}" == "1" && -n "${SHARED_VLA_PID}" ]]; then
    log "stopping shared Pi0.5 VLA service (pid=${SHARED_VLA_PID})"
    "${PYTHON_BIN}" "${RESULT_HELPER}" shutdown-rpc \
      --url "http://${VLA_HOST}:${VLA_PORT}" \
      --timeout-s 10 >/dev/null 2>&1 || true
    if [[ -n "${SHARED_VLA_STDIN_FD}" ]]; then
      exec {SHARED_VLA_STDIN_FD}>&- || true
    fi
    if kill -0 "${SHARED_VLA_PID}" 2>/dev/null; then
      for _ in $(seq 1 20); do
        kill -0 "${SHARED_VLA_PID}" 2>/dev/null || break
        sleep 0.5
      done
    fi
    if kill -0 "${SHARED_VLA_PID}" 2>/dev/null; then
      kill "${SHARED_VLA_PID}" 2>/dev/null || true
    fi
    wait "${SHARED_VLA_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SHARED_VLA_FIFO}" && -p "${SHARED_VLA_FIFO}" ]]; then
    rm -f "${SHARED_VLA_FIFO}"
  fi
  if [[ "${RPENT_BASELINE_BACKGROUND:-0}" == "1" \
      && -n "${RUNNER_PID_FILE}" \
      && -f "${RUNNER_PID_FILE}" ]]; then
    recorded_runner_pid="$(<"${RUNNER_PID_FILE}")"
    if [[ "${recorded_runner_pid}" == "$$" ]]; then
      rm -f "${RUNNER_PID_FILE}"
    fi
  fi
  if [[ "${exit_code}" != "0" && -n "${ACTIVE_RUN_ID}" ]]; then
    echo "[baseline] interrupted while processing ${ACTIVE_RUN_ID}" >&2
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      DRY_RUN=1
      ;;
    --summarize-only)
      SUMMARIZE_ONLY=1
      ;;
    --foreground)
      FOREGROUND=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: ${arg}"
      ;;
  esac
done

[[ "${AUTO_DETACH}" =~ ^[01]$ ]] || die "AUTO_DETACH must be 0 or 1"

BACKGROUND_EXPERIMENT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
RUNNER_LOG_FILE="${BACKGROUND_EXPERIMENT_DIR}/runner.log"
RUNNER_PID_FILE="${BACKGROUND_EXPERIMENT_DIR}/runner.pid"

# 直接运行时创建一个与 SSH 会话脱离的后台子进程。dry-run、汇总和显式
# foreground 请求仍留在当前终端，方便观察输出。
if [[ "${AUTO_DETACH}" == "1" \
    && "${RPENT_BASELINE_BACKGROUND:-0}" != "1" \
    && "${DRY_RUN}" == "0" \
    && "${SUMMARIZE_ONLY}" == "0" \
    && "${FOREGROUND}" == "0" ]]; then
  command -v nohup >/dev/null 2>&1 || die "the 'nohup' command is required"
  command -v setsid >/dev/null 2>&1 || die "the 'setsid' command is required"

  mkdir -p "${BACKGROUND_EXPERIMENT_DIR}"

  if [[ -f "${RUNNER_PID_FILE}" ]]; then
    existing_runner_pid="$(<"${RUNNER_PID_FILE}")"
    if [[ "${existing_runner_pid}" =~ ^[0-9]+$ ]] \
        && kill -0 "${existing_runner_pid}" 2>/dev/null; then
      echo "[baseline] 后台任务已经运行，PID=${existing_runner_pid}"
      echo "[baseline] 全局日志：${RUNNER_LOG_FILE}"
      echo "[baseline] 查看进度：tail -f ${RUNNER_LOG_FILE}"
      exit 0
    fi
    rm -f "${RUNNER_PID_FILE}"
  fi

  nohup setsid env RPENT_BASELINE_BACKGROUND=1 \
    bash "${SCRIPT_DIR}/run_baselines.sh" "$@" \
    >> "${RUNNER_LOG_FILE}" 2>&1 < /dev/null &
  background_pid=$!

  echo "[baseline] 已转入后台运行，PID=${background_pid}"
  echo "[baseline] 可以安全断开 SSH。"
  echo "[baseline] 全局日志：${RUNNER_LOG_FILE}"
  echo "[baseline] 查看进度：tail -f ${RUNNER_LOG_FILE}"
  echo "[baseline] 结果目录：${BACKGROUND_EXPERIMENT_DIR}"
  exit 0
fi

[[ -f "${RESULT_HELPER}" ]] || die "missing result helper: ${RESULT_HELPER}"
[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || die "python not executable: ${PYTHON_BIN}"
[[ "${EXPERIMENT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "EXPERIMENT_NAME may contain only letters, digits, dot, underscore, dash"
[[ "${EVAL_MODE}" =~ ^(cartesian|list|full)$ ]] \
  || die "EVAL_MODE must be cartesian, list, or full"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be >= 1"
[[ "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_ATTEMPTS must be >= 1"
[[ "${RUN_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]] || die "RUN_TIMEOUT_S must be >= 1"
[[ "${MIN_FREE_DISK_GB}" =~ ^[0-9]+$ ]] || die "MIN_FREE_DISK_GB must be >= 0"
[[ "${HIRES_RETENTION_STEPS}" =~ ^[0-9]+$ ]] \
  || die "HIRES_RETENTION_STEPS must be >= 0"
[[ "${ARTIFACT_RETENTION}" =~ ^(all|video_and_structured_logs)$ ]] \
  || die "ARTIFACT_RETENTION must be all or video_and_structured_logs"
[[ "${#TASK_MATRIX[@]}" -gt 0 ]] || die "TASK_MATRIX is empty"
[[ "${#SEEDS[@]}" -gt 0 ]] || die "SEEDS is empty"
for task_spec in "${TASK_MATRIX[@]}"; do
  read -r suite task extra <<< "${task_spec}"
  [[ -n "${suite:-}" && -n "${task:-}" && -z "${extra:-}" ]] \
    || die "invalid TASK_MATRIX row (expected '<suite> <task>'): ${task_spec}"
  [[ "${suite}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "invalid suite name: ${suite}"
  [[ "${task}" =~ ^[0-9]+$ ]] \
    || die "task must be a non-negative integer: ${task}"
done
for seed in "${SEEDS[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] \
    || die "seed must be a non-negative integer: ${seed}"
done

case "${RESOURCE_MODE}" in
  offline)
    export HF_HUB_OFFLINE=1
    unset RPENT_ABLATE_RESOURCES || true
    ;;
  sync)
    unset HF_HUB_OFFLINE || true
    unset RPENT_ABLATE_RESOURCES || true
    ;;
  ablation)
    export RPENT_ABLATE_RESOURCES=1
    unset HF_HUB_OFFLINE || true
    ;;
  *)
    die "RESOURCE_MODE must be offline, sync, or ablation"
    ;;
esac

export PI05_CHECKPOINT_PATH
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export LIBERO_TYPE
export QWEN_VL_BASE_URL
export QWEN_VL_API_KEY
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

EXPERIMENT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
RUNS_DIR="${EXPERIMENT_DIR}/runs"
SERVICES_DIR="${EXPERIMENT_DIR}/services"
mkdir -p "${RUNS_DIR}" "${SERVICES_DIR}"
command -v flock >/dev/null 2>&1 || die "the 'flock' command is required"
exec 9> "${EXPERIMENT_DIR}/runner.lock"
flock -n 9 || die "another runner already owns experiment ${EXPERIMENT_NAME}"

if [[ "${RPENT_BASELINE_BACKGROUND:-0}" == "1" ]]; then
  printf '%s\n' "$$" > "${RUNNER_PID_FILE}"
  log "============================================================"
  log "后台 Runner 启动：PID=$$"
  log "全局日志：${RUNNER_LOG_FILE}"
  log "============================================================"
fi

GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
RUNNER_SHA256="$(sha256sum "$0" | awk '{print $1}')"
RESOLVED_CONFIG_TMP="$(mktemp "${EXPERIMENT_DIR}/.resolved_config.XXXXXX")"
{
  printf 'experiment_name=%s\n' "${EXPERIMENT_NAME}"
  printf 'planner=%s\n' "${PLANNER}"
  printf 'model=%s\n' "${MODEL}"
  printf 'qwen_vl_base_url=%s\n' "${QWEN_VL_BASE_URL}"
  printf 'pi05_checkpoint_path=%s\n' "${PI05_CHECKPOINT_PATH}"
  printf 'cuda_device=%s\n' "${CUDA_DEVICE}"
  printf 'libero_type=%s\n' "${LIBERO_TYPE}"
  printf 'max_tokens=%s\n' "${MAX_TOKENS}"
  printf 'max_turns=%s\n' "${MAX_TURNS}"
  printf 'max_episode_steps=%s\n' "${MAX_EPISODE_STEPS}"
  printf 'run_timeout_s=%s\n' "${RUN_TIMEOUT_S}"
  printf 'hires_retention_steps=%s\n' "${HIRES_RETENTION_STEPS}"
  printf 'artifact_retention=%s\n' "${ARTIFACT_RETENTION}"
  printf 'resource_mode=%s\n' "${RESOURCE_MODE}"
  printf 'python_bin=%s\n' "${PYTHON_BIN}"
  printf 'repeats=%s\n' "${REPEATS}"
  printf 'seeds=%s\n' "${SEEDS[*]}"
  printf 'eval_mode=%s\n' "${EVAL_MODE}"
  printf 'max_attempts=%s\n' "${MAX_ATTEMPTS}"
  printf 'min_free_disk_gb=%s\n' "${MIN_FREE_DISK_GB}"
  printf 'auto_detach=%s\n' "${AUTO_DETACH}"
  printf 'start_shared_vla=%s\n' "${START_SHARED_VLA}"
  printf 'external_vla_endpoint=%s\n' "${VLA_ENDPOINT}"
  printf 'git_commit=%s\n' "${GIT_COMMIT}"
  printf 'runner_sha256=%s\n' "${RUNNER_SHA256}"
  printf 'task_matrix_begin\n'
  printf '%s\n' "${TASK_MATRIX[@]}"
  printf 'task_matrix_end\n'
} > "${RESOLVED_CONFIG_TMP}"

if [[ -f "${EXPERIMENT_DIR}/resolved_config.txt" ]]; then
  if ! cmp -s "${RESOLVED_CONFIG_TMP}" "${EXPERIMENT_DIR}/resolved_config.txt"; then
    rm -f "${RESOLVED_CONFIG_TMP}"
    die "resolved config differs from the existing experiment; choose a new EXPERIMENT_NAME"
  fi
  rm -f "${RESOLVED_CONFIG_TMP}"
else
  mv "${RESOLVED_CONFIG_TMP}" "${EXPERIMENT_DIR}/resolved_config.txt"
  cp "$0" "${EXPERIMENT_DIR}/runner_snapshot.sh"
fi

summarize() {
  "${PYTHON_BIN}" "${RESULT_HELPER}" summarize \
    --experiment-dir "${EXPERIMENT_DIR}"
}

if [[ "${SUMMARIZE_ONLY}" == "1" ]]; then
  summarize
  exit 0
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -c "import rpent, robots.libero" \
  || die "the selected Python environment cannot import RPent"

if [[ "${START_SHARED_VLA}" == "1" || -z "${VLA_ENDPOINT}" ]]; then
  [[ -d "${PI05_CHECKPOINT_PATH}" ]] \
    || die "Pi0.5 checkpoint directory not found: ${PI05_CHECKPOINT_PATH}"
fi

if [[ "${MODEL}" == qwen-vl:* && "${CHECK_QWEN_SERVER}" == "1" && "${DRY_RUN}" == "0" ]]; then
  log "checking Qwen-VL image and tool-call compatibility"
  "${PYTHON_BIN}" scripts/qwen_vl/check_server.py \
    --base-url "${QWEN_VL_BASE_URL}" \
    --api-key "${QWEN_VL_API_KEY}" \
    --model "${MODEL#qwen-vl:}"
fi

EFFECTIVE_VLA_ENDPOINT="${VLA_ENDPOINT}"
if [[ -z "${EFFECTIVE_VLA_ENDPOINT}" && "${START_SHARED_VLA}" == "1" && "${DRY_RUN}" == "0" ]]; then
  SHARED_VLA_FIFO="${SERVICES_DIR}/vla_stdin.fifo"
  [[ ! -e "${SHARED_VLA_FIFO}" ]] \
    || die "stale VLA stdin path exists: ${SHARED_VLA_FIFO}"
  mkfifo "${SHARED_VLA_FIFO}"
  exec {SHARED_VLA_STDIN_FD}<>"${SHARED_VLA_FIFO}"

  log "starting one shared Pi0.5 VLA service on GPU ${CUDA_DEVICE}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    "${PYTHON_BIN}" robots/libero/vla_server.py \
      --transport http \
      --host "${VLA_HOST}" \
      --port "${VLA_PORT}" \
      --model-path "${PI05_CHECKPOINT_PATH}" \
      < "${SHARED_VLA_FIFO}" \
      > "${SERVICES_DIR}/vla_server.log" 2>&1 &
  SHARED_VLA_PID=$!
  OWN_SHARED_VLA=1
  EFFECTIVE_VLA_ENDPOINT="http://${VLA_HOST}:${VLA_PORT}"
  vla_ready=0
  vla_ready_deadline=$(( $(date +%s) + VLA_READY_TIMEOUT_S ))
  while (( $(date +%s) < vla_ready_deadline )); do
    if ! kill -0 "${SHARED_VLA_PID}" 2>/dev/null; then
      break
    fi
    if "${PYTHON_BIN}" "${RESULT_HELPER}" wait-rpc \
        --url "${EFFECTIVE_VLA_ENDPOINT}" \
        --timeout-s 2 \
        --interval-s 0.5 >/dev/null 2>&1; then
      vla_ready=1
      break
    fi
  done
  if [[ "${vla_ready}" != "1" ]]; then
    tail -n 100 "${SERVICES_DIR}/vla_server.log" >&2 || true
    die "shared Pi0.5 VLA service did not become ready"
  fi
  log "shared Pi0.5 VLA service ready: ${EFFECTIVE_VLA_ENDPOINT}"
elif [[ -n "${EFFECTIVE_VLA_ENDPOINT}" && "${DRY_RUN}" == "0" ]]; then
  log "checking external Pi0.5 VLA service: ${EFFECTIVE_VLA_ENDPOINT}"
  "${PYTHON_BIN}" "${RESULT_HELPER}" wait-rpc \
    --url "${EFFECTIVE_VLA_ENDPOINT}" \
    --timeout-s 30 \
    --interval-s 1
elif [[ -z "${EFFECTIVE_VLA_ENDPOINT}" && "${START_SHARED_VLA}" == "1" ]]; then
  # Dry-run representation of the endpoint that the real run will create.
  EFFECTIVE_VLA_ENDPOINT="http://${VLA_HOST}:${VLA_PORT}"
fi

TOTAL_RUNS=$(( ${#TASK_MATRIX[@]} * ${#SEEDS[@]} * REPEATS ))
log "experiment: ${EXPERIMENT_NAME}"
log "scheduled logical runs: ${TOTAL_RUNS}"
log "outputs: ${EXPERIMENT_DIR}"
if [[ "${HIRES_RETENTION_STEPS}" == "0" ]]; then
  log "high-resolution retention: complete trajectory (monitor disk usage)"
else
  log "high-resolution retention: latest ${HIRES_RETENTION_STEPS} states"
fi

run_one() {
  local suite=$1
  local task=$2
  local seed=$3
  local repeat=$4
  local run_id
  local run_dir
  local canonical_result
  local attempt_dirs=()
  local attempt_no
  local attempt_dir
  local started_at
  local finished_at
  local start_epoch
  local end_epoch
  local wall_elapsed
  local cmd_rc
  local tee_rc
  local timed_out
  local final_status
  local pipeline_status=()
  local available_kb
  local required_kb
  local cmd=()

  printf -v run_id '%s__t%03d__s%06d__r%03d' \
    "${suite}" "${task}" "${seed}" "${repeat}"
  run_dir="${RUNS_DIR}/${run_id}"
  canonical_result="${run_dir}/result.json"
  mkdir -p "${run_dir}/attempts"
  ACTIVE_RUN_ID="${run_id}"

  if [[ -f "${canonical_result}" ]] \
      && "${PYTHON_BIN}" "${RESULT_HELPER}" check-status \
        --result "${canonical_result}" --kind valid; then
    log "SKIP valid result: ${run_id}"
    ACTIVE_RUN_ID=""
    return 0
  fi

  shopt -s nullglob
  attempt_dirs=("${run_dir}"/attempts/attempt_*)
  shopt -u nullglob
  for attempt_dir in "${attempt_dirs[@]}"; do
    if [[ -f "${attempt_dir}/launch.json" && ! -f "${attempt_dir}/result.json" ]]; then
      log "recovering interrupted attempt record: ${attempt_dir}"
      "${PYTHON_BIN}" "${RESULT_HELPER}" finalize \
        --attempt-dir "${attempt_dir}" \
        --canonical-result "${canonical_result}" \
        --exit-code 255 \
        --timed-out 0 \
        --wall-elapsed-s 0 >/dev/null
    fi
  done
  attempt_no=$(( ${#attempt_dirs[@]} + 1 ))

  if (( attempt_no > MAX_ATTEMPTS )); then
    log "NO RETRIES LEFT: ${run_id} (see ${canonical_result})"
    ACTIVE_RUN_ID=""
    return 0
  fi

  while (( attempt_no <= MAX_ATTEMPTS )); do
    required_kb=$(( MIN_FREE_DISK_GB * 1024 * 1024 ))
    available_kb="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR==2 {print $4}')"
    [[ "${available_kb}" =~ ^[0-9]+$ ]] \
      || die "could not determine free disk space for ${OUTPUT_ROOT}"
    if (( available_kb < required_kb )); then
      die "free disk space is below MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB}"
    fi

    printf -v attempt_dir '%s/attempts/attempt_%02d' "${run_dir}" "${attempt_no}"

    cmd=(
      "${PYTHON_BIN}" -m rpent.cli.main
      --env libero
      --suite "${suite}"
      --task "${task}"
      --seed "${seed}"
      --libero-type "${LIBERO_TYPE}"
      --planner "${PLANNER}"
      --model "${MODEL}"
      --base-url "${QWEN_VL_BASE_URL}"
      --max-tokens "${MAX_TOKENS}"
      --max-turns "${MAX_TURNS}"
      --max-episode-steps "${MAX_EPISODE_STEPS}"
      --hires-retention-steps "${HIRES_RETENTION_STEPS}"
      --cuda-device "${CUDA_DEVICE}"
      --output-dir "${attempt_dir}"
    )
    if [[ -n "${EFFECTIVE_VLA_ENDPOINT}" ]]; then
      cmd+=(--vla-endpoint "${EFFECTIVE_VLA_ENDPOINT}")
    fi
    if (( ${#EXTRA_RPENT_ARGS[@]} > 0 )); then
      cmd+=("${EXTRA_RPENT_ARGS[@]}")
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[baseline] DRY-RUN %s: ' "${run_id}"
      printf '%q ' "${cmd[@]}"
      printf '\n'
      ACTIVE_RUN_ID=""
      return 0
    fi

    mkdir -p "${attempt_dir}"
    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "${PYTHON_BIN}" "${RESULT_HELPER}" write-launch \
      --attempt-dir "${attempt_dir}" \
      --repo-root "${REPO_ROOT}" \
      --run-id "${run_id}" \
      --experiment "${EXPERIMENT_NAME}" \
      --attempt "${attempt_no}" \
      --started-at "${started_at}" \
      --suite "${suite}" \
      --task "${task}" \
      --seed "${seed}" \
      --repeat "${repeat}" \
      --libero-type "${LIBERO_TYPE}" \
      --planner "${PLANNER}" \
      --model "${MODEL}" \
      --base-url "${QWEN_VL_BASE_URL}" \
      --max-tokens "${MAX_TOKENS}" \
      --max-turns "${MAX_TURNS}" \
      --run-timeout-s "${RUN_TIMEOUT_S}" \
      --max-episode-steps "${MAX_EPISODE_STEPS}" \
      --hires-retention-steps "${HIRES_RETENTION_STEPS}" \
      --artifact-retention "${ARTIFACT_RETENTION}" \
      --pi05-checkpoint-path "${PI05_CHECKPOINT_PATH}" \
      --vla-endpoint "${EFFECTIVE_VLA_ENDPOINT}" \
      --resource-mode "${RESOURCE_MODE}" \
      -- "${cmd[@]}"

    log "RUN ${run_id}, attempt ${attempt_no}/${MAX_ATTEMPTS}"
    start_epoch="$(date +%s)"
    ACTIVE_ATTEMPT_DIR="${attempt_dir}"
    ACTIVE_CANONICAL_RESULT="${canonical_result}"
    ACTIVE_START_EPOCH="${start_epoch}"
    set +e
    timeout --signal=INT --kill-after=120s "${RUN_TIMEOUT_S}s" \
      "${cmd[@]}" 2>&1 | tee "${attempt_dir}/console.log"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    cmd_rc="${pipeline_status[0]}"
    tee_rc="${pipeline_status[1]}"
    if [[ "${tee_rc}" != "0" && "${cmd_rc}" == "0" ]]; then
      cmd_rc=74
    fi
    end_epoch="$(date +%s)"
    wall_elapsed=$(( end_epoch - start_epoch ))
    finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    timed_out=0
    if [[ "${cmd_rc}" == "124" || "${cmd_rc}" == "137" ]]; then
      timed_out=1
    fi

    if [[ "${ARTIFACT_RETENTION}" == "video_and_structured_logs" ]]; then
      "${PYTHON_BIN}" "${RESULT_HELPER}" prune-attempt \
        --attempt-dir "${attempt_dir}" >/dev/null
    fi

    final_status="$("${PYTHON_BIN}" "${RESULT_HELPER}" finalize \
      --attempt-dir "${attempt_dir}" \
      --canonical-result "${canonical_result}" \
      --exit-code "${cmd_rc}" \
      --timed-out "${timed_out}" \
      --wall-elapsed-s "${wall_elapsed}" \
      --finished-at "${finished_at}")"
    ACTIVE_ATTEMPT_DIR=""
    ACTIVE_CANONICAL_RESULT=""
    ACTIVE_START_EPOCH=""
    summarize >/dev/null
    log "RESULT ${run_id}: ${final_status} (${wall_elapsed}s)"

    if "${PYTHON_BIN}" "${RESULT_HELPER}" check-status \
        --result "${canonical_result}" --kind valid; then
      ACTIVE_RUN_ID=""
      return 0
    fi
    if ! "${PYTHON_BIN}" "${RESULT_HELPER}" check-status \
        --result "${canonical_result}" --kind retryable; then
      ACTIVE_RUN_ID=""
      return 0
    fi

    attempt_no=$(( attempt_no + 1 ))
    if (( attempt_no <= MAX_ATTEMPTS )); then
      log "retrying ${run_id} in ${RETRY_DELAY_S}s"
      sleep "${RETRY_DELAY_S}"
    fi
  done

  ACTIVE_RUN_ID=""
}

for task_spec in "${TASK_MATRIX[@]}"; do
  read -r suite task extra <<< "${task_spec}"
  for seed in "${SEEDS[@]}"; do
    for (( repeat=0; repeat<REPEATS; repeat++ )); do
      run_one "${suite}" "${task}" "${seed}" "${repeat}"
    done
  done
done

ACTIVE_RUN_ID=""
log "all scheduled runs processed"
summarize
