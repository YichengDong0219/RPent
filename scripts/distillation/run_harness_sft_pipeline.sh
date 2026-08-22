#!/usr/bin/env bash
# Run environment checks, Harness success collection, and RLinf Pi0.5 SFT.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROFILE="${1:-${PIPELINE_PROFILE:-full}}"
PIPELINE_STAGE="${PIPELINE_STAGE:-all}"
case "${PROFILE}" in
  full|smoke) ;;
  check)
    PIPELINE_STAGE=check
    PROFILE=full
    ;;
  *)
    echo "usage: bash $0 [full|smoke|check]" >&2
    exit 2
    ;;
esac
case "${PIPELINE_STAGE}" in
  all|check|collect|sft) ;;
  *)
    echo "PIPELINE_STAGE must be one of: all, check, collect, sft" >&2
    exit 2
    ;;
esac

RLINF_ROOT="${RLINF_ROOT:-/home/dongyicheng/RLinf}"
RLINF_PYTHON="${RLINF_PYTHON:-${RLINF_ROOT}/.venv/bin/python}"
RPENT_PYTHON="${RPENT_PYTHON:-/home/dongyicheng/miniconda3/envs/rpent/bin/python}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/home/dongyicheng/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT}"
PIPELINE_OUTPUT_ROOT="${PIPELINE_OUTPUT_ROOT:-${REPO_ROOT}/logs/distillation}"

if [[ "${PROFILE}" == "smoke" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-libero_object_lan_t0_harness_sft_smoke}"
  SUCCESS_TARGET="${SUCCESS_TARGET:-1}"
  INIT_STATE_START="${INIT_STATE_START:-0}"
  INIT_STATE_STOP_EXCLUSIVE="${INIT_STATE_STOP_EXCLUSIVE:-3}"
  MAX_PLANNER_SEEDS_PER_INIT="${MAX_PLANNER_SEEDS_PER_INIT:-1}"
  MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
  MAX_TURNS="${MAX_TURNS:-20}"
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1500}"
  RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-1200}"
  SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-pi05_sft_smoke}"
  SFT_MAX_STEPS="${SFT_MAX_STEPS:-1}"
  SFT_SAVE_INTERVAL="${SFT_SAVE_INTERVAL:-1}"
  SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-1}"
  SFT_GLOBAL_BATCH_SIZE="${SFT_GLOBAL_BATCH_SIZE:-1}"
  SFT_NUM_WORKERS="${SFT_NUM_WORKERS:-0}"
  SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS:-0}"
  WANDB_MODE="${WANDB_MODE:-offline}"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-libero_object_lan_t0_harness_sft_820}"
  SUCCESS_TARGET="${SUCCESS_TARGET:-20}"
  INIT_STATE_START="${INIT_STATE_START:-0}"
  INIT_STATE_STOP_EXCLUSIVE="${INIT_STATE_STOP_EXCLUSIVE:-40}"
  MAX_PLANNER_SEEDS_PER_INIT="${MAX_PLANNER_SEEDS_PER_INIT:-3}"
  MAX_ATTEMPTS="${MAX_ATTEMPTS:-120}"
  MAX_TURNS="${MAX_TURNS:-40}"
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-10000}"
  RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-3600}"
  SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-libero_harness_pi05_sft}"
  SFT_MAX_STEPS="${SFT_MAX_STEPS:-200}"
  SFT_SAVE_INTERVAL="${SFT_SAVE_INTERVAL:-200}"
  SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-2}"
  SFT_GLOBAL_BATCH_SIZE="${SFT_GLOBAL_BATCH_SIZE:-16}"
  SFT_NUM_WORKERS="${SFT_NUM_WORKERS:-2}"
  SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS:-20}"
  WANDB_MODE="${WANDB_MODE:-auto}"
fi

PIPELINE_RUN_DIR="${PIPELINE_OUTPUT_ROOT}/${EXPERIMENT_NAME}"
DATASET_WAS_SET="${DATASET+x}"
DATASET="${DATASET:-${PIPELINE_RUN_DIR}/dataset}"
SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-${PIPELINE_RUN_DIR}/sft}"
SFT_GPU="${SFT_GPU:-0}"
VLA_GPU="${VLA_GPU:-3}"
SFT_RAY_ADDRESS="${SFT_RAY_ADDRESS:-local}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-${SUCCESS_TARGET}}"
SFT_PREFLIGHT_ONLY="${SFT_PREFLIGHT_ONLY:-0}"
PREFLIGHT_LOADER_ONLY="${PREFLIGHT_LOADER_ONLY:-0}"
CHECK_PACKAGE_CONSISTENCY="${CHECK_PACKAGE_CONSISTENCY:-1}"
STRICT_PACKAGE_CONSISTENCY="${STRICT_PACKAGE_CONSISTENCY:-0}"
CHECK_CUDA="${CHECK_CUDA:-1}"
SMOKE_DATA_SOURCE="${SMOKE_DATA_SOURCE:-harness}"
OFFICIAL_SMOKE_DEMO="${OFFICIAL_SMOKE_DEMO:-/home/dongyicheng/LIBERO/libero/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5}"

mkdir -p \
  "${PIPELINE_RUN_DIR}/cache/numba" \
  "${PIPELINE_RUN_DIR}/cache/uv" \
  "${PIPELINE_RUN_DIR}/cache/huggingface" \
  "${PIPELINE_RUN_DIR}/cache/matplotlib"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${PIPELINE_RUN_DIR}/cache/numba}"
export HF_HOME="${HF_HOME:-${PIPELINE_RUN_DIR}/cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PIPELINE_RUN_DIR}/cache/matplotlib}"

die() {
  echo "[pipeline] ERROR: $*" >&2
  exit 1
}

case "${SMOKE_DATA_SOURCE}" in
  harness) ;;
  official_demo)
    [[ "${PROFILE}" == "smoke" ]] || die "SMOKE_DATA_SOURCE=official_demo is smoke-only"
    ;;
  *) die "SMOKE_DATA_SOURCE must be harness or official_demo" ;;
esac
if [[ "${SMOKE_DATA_SOURCE}" == "official_demo" && -z "${DATASET_WAS_SET}" ]]; then
  DATASET="${PIPELINE_RUN_DIR}/official_demo_fixture_dataset"
fi

check_environment() {
  echo "[pipeline] checking rpent and RLinf environments"
  [[ -x "${RPENT_PYTHON}" ]] || die "missing rpent interpreter: ${RPENT_PYTHON}"
  [[ -x "${RLINF_PYTHON}" ]] || die "missing RLinf interpreter: ${RLINF_PYTHON}"
  [[ -d "${BASE_CHECKPOINT}" ]] || die "missing checkpoint: ${BASE_CHECKPOINT}"
  [[ -f "${BASE_CHECKPOINT}/model.safetensors" ]] || die "checkpoint has no model.safetensors"
  [[ -f "${BASE_CHECKPOINT}/metadata.pt" ]] || die "checkpoint has no metadata.pt"
  [[ -f "${BASE_CHECKPOINT}/physical-intelligence/libero/norm_stats.json" ]] || \
    die "checkpoint has no LIBERO norm_stats.json"

  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${RPENT_PYTHON}" -c '
import h5py, lerobot, libero, robosuite, rpent
from libero.libero import benchmark
print("[pipeline] rpent imports: ok")
'
  PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${RLINF_PYTHON}" -c '
import openpi, ray, rlinf, safetensors, torch, wandb
from rlinf.data.datasets.openpi_rlinf import build_official_openpi_sft_dataloader
print("[pipeline] RLinf/OpenPI imports: ok")
'

  if [[ "${CHECK_CUDA}" == "1" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    CUDA_VISIBLE_DEVICES="${SFT_GPU}" "${RLINF_PYTHON}" -c '
import torch
assert torch.cuda.is_available(), "RLinf torch cannot access CUDA"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
print("[pipeline] RLinf CUDA:", torch.cuda.get_device_name(0))
'
  fi

  if [[ "${CHECK_PACKAGE_CONSISTENCY}" == "1" ]] && command -v uv >/dev/null 2>&1; then
    set +e
    UV_CACHE_DIR="${PIPELINE_RUN_DIR}/cache/uv" uv pip check \
      --python "${RLINF_PYTHON}"
    package_status=$?
    set -e
    if (( package_status != 0 )); then
      if [[ "${STRICT_PACKAGE_CONSISTENCY}" == "1" ]]; then
        die "uv pip check reported incompatible packages"
      fi
      echo "[pipeline] WARNING: uv pip check reported incompatibilities; runtime smoke will be authoritative" >&2
    fi
  fi
}

run_collection() {
  if [[ "${SMOKE_DATA_SOURCE}" == "official_demo" ]]; then
    echo "[pipeline] WARNING: building an official-demo smoke fixture; this is not Harness teacher data" >&2
    "${RPENT_PYTHON}" "${SCRIPT_DIR}/build_official_demo_smoke_dataset.py" \
      --demo "${OFFICIAL_SMOKE_DEMO}" --output "${DATASET}"
    "${RPENT_PYTHON}" "${SCRIPT_DIR}/manage_dataset.py" validate \
      --dataset "${DATASET}" --expected-episodes 1
    return
  fi
  echo "[pipeline] collecting ${SUCCESS_TARGET} successful episode(s)"
  env \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    OUTPUT_ROOT="${PIPELINE_OUTPUT_ROOT}" \
    SUCCESS_TARGET="${SUCCESS_TARGET}" \
    INIT_STATE_START="${INIT_STATE_START}" \
    INIT_STATE_STOP_EXCLUSIVE="${INIT_STATE_STOP_EXCLUSIVE}" \
    MAX_PLANNER_SEEDS_PER_INIT="${MAX_PLANNER_SEEDS_PER_INIT}" \
    MAX_ATTEMPTS="${MAX_ATTEMPTS}" \
    MAX_TURNS="${MAX_TURNS}" \
    MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
    RUN_TIMEOUT_S="${RUN_TIMEOUT_S}" \
    VLA_GPU="${VLA_GPU}" \
    PI05_CHECKPOINT="${BASE_CHECKPOINT}" \
    bash "${SCRIPT_DIR}/collect_harness_successes.sh"
}

run_sft() {
  echo "[pipeline] training action expert for ${SFT_MAX_STEPS} optimizer step(s)"
  env \
    DATASET="${DATASET}" \
    EXPECTED_EPISODES="${EXPECTED_EPISODES}" \
    BASE_CHECKPOINT="${BASE_CHECKPOINT}" \
    SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT}" \
    SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME}" \
    SFT_MAX_STEPS="${SFT_MAX_STEPS}" \
    SFT_SAVE_INTERVAL="${SFT_SAVE_INTERVAL}" \
    SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE}" \
    SFT_GLOBAL_BATCH_SIZE="${SFT_GLOBAL_BATCH_SIZE}" \
    SFT_NUM_WORKERS="${SFT_NUM_WORKERS}" \
    SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS}" \
    SFT_GPU="${SFT_GPU}" \
    SFT_RAY_ADDRESS="${SFT_RAY_ADDRESS}" \
    SFT_PREFLIGHT_ONLY="${SFT_PREFLIGHT_ONLY}" \
    PREFLIGHT_LOADER_ONLY="${PREFLIGHT_LOADER_ONLY}" \
    WANDB_MODE="${WANDB_MODE}" \
    bash "${SCRIPT_DIR}/run_rlinf_sft.sh"
}

check_environment
case "${PIPELINE_STAGE}" in
  check) ;;
  collect) run_collection ;;
  sft) run_sft ;;
  all)
    run_collection
    run_sft
    ;;
esac

echo "[pipeline] complete"
echo "[pipeline] dataset: ${DATASET}"
if [[ ( "${PIPELINE_STAGE}" == "all" || "${PIPELINE_STAGE}" == "sft" ) && "${SFT_PREFLIGHT_ONLY}" != "1" ]]; then
  echo "[pipeline] deploy checkpoint: ${SFT_OUTPUT_ROOT}/${SFT_EXPERIMENT_NAME}/deploy_step_${SFT_MAX_STEPS}"
fi
