#!/usr/bin/env bash
# Train Pi0.5 on the collected Harness-success LeRobot dataset using RLinf.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RLINF_ROOT="${RLINF_ROOT:-/home/dongyicheng/RLinf}"
RLINF_PYTHON="${RLINF_PYTHON:-${RLINF_ROOT}/.venv/bin/python}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/rlinf_config/libero_harness_pi05_sft.yaml}"
DATASET="${DATASET:-${REPO_ROOT}/logs/distillation/libero_object_lan_t0_harness_sft_v1/dataset}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/home/dongyicheng/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT}"
NORM_STATS="${NORM_STATS:-${BASE_CHECKPOINT}/physical-intelligence/libero/norm_stats.json}"
OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-${REPO_ROOT}/logs/distillation/sft}"
EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-libero_harness_pi05_sft}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_LOADER_ONLY="${PREFLIGHT_LOADER_ONLY:-0}"
SFT_PREFLIGHT_ONLY="${SFT_PREFLIGHT_ONLY:-0}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-20}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-200}"
SFT_SAVE_INTERVAL="${SFT_SAVE_INTERVAL:-${SFT_MAX_STEPS}}"
SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-2}"
SFT_GLOBAL_BATCH_SIZE="${SFT_GLOBAL_BATCH_SIZE:-16}"
SFT_NUM_WORKERS="${SFT_NUM_WORKERS:-2}"
SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS:-20}"
SFT_GPU="${SFT_GPU:-0}"
SFT_RAY_ADDRESS="${SFT_RAY_ADDRESS:-local}"
EXPORT_CHECKPOINT="${EXPORT_CHECKPOINT:-1}"
DEPLOY_OUTPUT="${DEPLOY_OUTPUT:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}/deploy_step_${SFT_MAX_STEPS}}"
RPENT_PYTHON="${RPENT_PYTHON:-/home/dongyicheng/miniconda3/envs/rpent/bin/python}"
WANDB_USERNAME="${WANDB_USERNAME:-YichengDong}"
WANDB_ENTITY="${WANDB_ENTITY:-ethan-dong-nanjing-university-org}"
WANDB_PROJECT="${WANDB_PROJECT:-harness-vla-sft}"
WANDB_REQUESTED_MODE="${WANDB_MODE:-auto}"
WANDB_VERIFY_TIMEOUT_S="${WANDB_VERIFY_TIMEOUT_S:-15}"

[[ -x "${RLINF_PYTHON}" ]] || { echo "missing RLinf uv interpreter: ${RLINF_PYTHON}" >&2; exit 1; }
[[ -d "${DATASET}" ]] || { echo "missing dataset: ${DATASET}" >&2; exit 1; }
[[ -d "${BASE_CHECKPOINT}" ]] || { echo "missing checkpoint: ${BASE_CHECKPOINT}" >&2; exit 1; }
[[ -f "${NORM_STATS}" ]] || { echo "missing norm stats: ${NORM_STATS}" >&2; exit 1; }

for value in EXPECTED_EPISODES SFT_MAX_STEPS SFT_SAVE_INTERVAL SFT_MICRO_BATCH_SIZE SFT_GLOBAL_BATCH_SIZE SFT_NUM_WORKERS SFT_WARMUP_STEPS; do
  [[ "${!value}" =~ ^[0-9]+$ ]] || { echo "${value} must be a non-negative integer" >&2; exit 1; }
done
(( EXPECTED_EPISODES > 0 )) || { echo "EXPECTED_EPISODES must be positive" >&2; exit 1; }
(( SFT_MAX_STEPS > 0 )) || { echo "SFT_MAX_STEPS must be positive" >&2; exit 1; }
(( SFT_SAVE_INTERVAL > 0 )) || { echo "SFT_SAVE_INTERVAL must be positive" >&2; exit 1; }
(( SFT_MICRO_BATCH_SIZE > 0 )) || { echo "SFT_MICRO_BATCH_SIZE must be positive" >&2; exit 1; }
(( SFT_GLOBAL_BATCH_SIZE > 0 )) || { echo "SFT_GLOBAL_BATCH_SIZE must be positive" >&2; exit 1; }
(( SFT_GLOBAL_BATCH_SIZE % SFT_MICRO_BATCH_SIZE == 0 )) || {
  echo "SFT_GLOBAL_BATCH_SIZE must be divisible by SFT_MICRO_BATCH_SIZE" >&2
  exit 1
}

"${RPENT_PYTHON}" "${SCRIPT_DIR}/manage_dataset.py" validate \
  --dataset "${DATASET}" --expected-episodes "${EXPECTED_EPISODES}"

export RLINF_OPENPI_USE_QUANTILE_NORM=0
export PYTHONPATH="${SCRIPT_DIR}/rlinf_shim:${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/sft"
export REPO_PATH="${RLINF_ROOT}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="${SFT_GPU}"
export RAY_ADDRESS="${SFT_RAY_ADDRESS}"
mkdir -p "${OUTPUT_ROOT}"

if ! "${RLINF_PYTHON}" -c "import openpi, rlinf" >/dev/null 2>&1; then
  echo "RLinf uv environment is incomplete (cannot import openpi and rlinf)." >&2
  echo "No dependency was installed or modified; finish configuring ${RLINF_ROOT}/.venv and rerun." >&2
  exit 1
fi

if ! WANDB_RESOLVED_MODE="$(
  "${RLINF_PYTHON}" "${SCRIPT_DIR}/resolve_wandb_mode.py" \
    --mode "${WANDB_REQUESTED_MODE}" --username "${WANDB_USERNAME}" \
    --entity "${WANDB_ENTITY}" --timeout "${WANDB_VERIFY_TIMEOUT_S}"
)"; then
  exit 1
fi
export WANDB_MODE="${WANDB_RESOLVED_MODE}"
export RPENT_WANDB_ENABLED=1
if [[ "${WANDB_MODE}" == "online" ]]; then
  export RPENT_WANDB_AUTO_FALLBACK=1
else
  unset RPENT_WANDB_AUTO_FALLBACK
fi
echo "[wandb] mode=${WANDB_MODE} entity=${WANDB_ENTITY} project=${WANDB_PROJECT} user=${WANDB_USERNAME}"

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  preflight_args=(
    --config "${CONFIG}" --dataset "${DATASET}"
    --checkpoint "${BASE_CHECKPOINT}" --norm-stats "${NORM_STATS}"
    --micro-batch-size "${SFT_MICRO_BATCH_SIZE}"
    --num-workers "${SFT_NUM_WORKERS}"
  )
  if [[ "${PREFLIGHT_LOADER_ONLY}" == "1" ]]; then
    preflight_args+=(--loader-only)
  fi
  (
    cd "${RLINF_ROOT}"
    "${RLINF_PYTHON}" "${SCRIPT_DIR}/preflight_rlinf_sft.py" "${preflight_args[@]}"
  )
fi
if [[ "${SFT_PREFLIGHT_ONLY}" == "1" ]]; then
  echo "[sft] preflight-only mode complete; optimizer step and export were not run"
  exit 0
fi

(
  cd "${RLINF_ROOT}"
  "${RLINF_PYTHON}" examples/sft/train_vla_sft.py \
    --config-path "$(dirname "${CONFIG}")" \
    --config-name "$(basename "${CONFIG}" .yaml)" \
    "data.train_data_paths=${DATASET}" \
    "actor.model.model_path=${BASE_CHECKPOINT}" \
    "actor.model.openpi_data.norm_stats_path=${NORM_STATS}" \
    "runner.logger.log_path=${OUTPUT_ROOT}" \
    "runner.logger.experiment_name=${EXPERIMENT_NAME}" \
    "runner.logger.project_name=${WANDB_PROJECT}" \
    "runner.logger.wandb_entity=${WANDB_ENTITY}" \
    "runner.max_steps=${SFT_MAX_STEPS}" \
    "runner.save_interval=${SFT_SAVE_INTERVAL}" \
    "data.num_workers=${SFT_NUM_WORKERS}" \
    "actor.micro_batch_size=${SFT_MICRO_BATCH_SIZE}" \
    "actor.global_batch_size=${SFT_GLOBAL_BATCH_SIZE}" \
    "actor.optim.lr_warmup_steps=${SFT_WARMUP_STEPS}" \
    "actor.optim.total_training_steps=${SFT_MAX_STEPS}"
)

SAVED_CHECKPOINT="${OUTPUT_ROOT}/${EXPERIMENT_NAME}/checkpoints/global_step_${SFT_MAX_STEPS}"
if [[ "${EXPORT_CHECKPOINT}" == "1" ]]; then
  "${RLINF_PYTHON}" "${SCRIPT_DIR}/export_rlinf_openpi_checkpoint.py" \
    --checkpoint "${SAVED_CHECKPOINT}" --reference "${BASE_CHECKPOINT}" \
    --output "${DEPLOY_OUTPUT}" --global-step "${SFT_MAX_STEPS}"
  echo "[sft] deploy checkpoint: ${DEPLOY_OUTPUT}"
else
  echo "[sft] checkpoint: ${SAVED_CHECKPOINT} (deploy export disabled)"
fi
