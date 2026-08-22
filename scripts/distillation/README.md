# Harness success trajectories → RLinf Pi0.5 SFT

This directory implements the task-0 experiment for
`libero_object_lan`: **grab alphabet soup and put it into basket**. Collection
uses the `rpent` Conda environment. Training only points at the existing RLinf
uv environment; none of the scripts install or upgrade dependencies.

## Data contract

The environment server records the observation immediately before each action
actually passed to `env.step()`. Chunk execution retains every real intermediate
observation, includes the action that terminates the episode, and excludes the
unused tail of that chunk.

| dataset field | stored value |
| --- | --- |
| `image` | agent-view RGB, `uint8[256,256,3]`, with RLinf/LIBERO's canonical orientation |
| `wrist_image` | wrist-view RGB, `uint8[256,256,3]`, canonical orientation |
| `state` | `float32[8]`: EEF world xyz + EEF axis-angle + two gripper qpos |
| `actions` | `float32[7]`: the OSC_POSE delta xyz, delta rotation, gripper command sent to the simulator |
| `task` | the complete original instruction on every frame |

Franka has seven arm joints and a two-finger gripper, but the policy interface
is Cartesian: 8D proprioception and 7D control. The RLinf OpenPI loader performs
32D padding and constructs a `10×7` action target (represented as `10×32` after
padding); padding is not stored in the dataset.

The simulator still executes at 20 Hz. As required by the Pi0.5 LIBERO training
contract, LeRobot metadata is `fps=10`, while every executed control action is
stored—there is no 2:1 subsampling. Export applies the official no-op rule:
arm-only no-ops below `1e-4` are removed, while gripper transitions are retained.

Only `terminated=true` is accepted. Truncations, planner completion, primitive
local success, crashes, and all other attempts are failures. Their raw sensor
and action artifacts are removed; their seed, frame count, exit code, and reason
remain in `success_manifest.json`.

## 0. Integrated entry and smoke profile

The integrated entry checks both environments, collects data, validates the
dataset, runs the RLinf preflight, trains, saves, and exports the checkpoint:

```bash
bash scripts/distillation/run_harness_sft_pipeline.sh full
```

The real end-to-end smoke profile uses one successful Harness episode, one
optimizer step, micro/global batch 1, an isolated local Ray runtime, and offline
W&B:

```bash
VLA_GPU=2 SFT_GPU=2 \
  bash scripts/distillation/run_harness_sft_pipeline.sh smoke
```

Environment checking or a single stage can be selected without editing code:

```bash
bash scripts/distillation/run_harness_sft_pipeline.sh check
PIPELINE_STAGE=collect bash scripts/distillation/run_harness_sft_pipeline.sh smoke
PIPELINE_STAGE=sft bash scripts/distillation/run_harness_sft_pipeline.sh smoke
```

Important integrated parameters are:

| parameter | meaning | full / smoke default |
| --- | --- | --- |
| `PIPELINE_STAGE` | `all`, `check`, `collect`, or `sft` | `all` |
| `PIPELINE_OUTPUT_ROOT` | collection and training root | `logs/distillation` |
| `EXPERIMENT_NAME` | collection manifest/dataset namespace | task v1 / smoke |
| `SUCCESS_TARGET` | exact number of accepted `terminated=true` episodes | 20 / 1 |
| `EXPECTED_EPISODES` | episode count required by the SFT dataset validator | `SUCCESS_TARGET` |
| `INIT_STATE_START`, `INIT_STATE_STOP_EXCLUSIVE` | collection init-state half-open range | 0–40 / 0–3 |
| `MAX_PLANNER_SEEDS_PER_INIT`, `MAX_ATTEMPTS` | retry budget | 3,120 / 1,3 |
| `PLANNER`, `PLANNER_MODEL`, `PLANNER_BASE_URL` | Harness planner backend/model/endpoint | API Qwen defaults |
| `PLANNER_TIMEOUT_S`, `RUN_TIMEOUT_S` | planner and whole-attempt timeout | unset,3600 / unset,1200 |
| `VLA_GPU`, `VLA_ENDPOINT`, `VLA_PORT` | collection inference device/service | GPU 0, localhost:18081 |
| `SFT_GPU`, `SFT_RAY_ADDRESS` | training device and Ray selection | GPU 0, isolated `local` |
| `SFT_MAX_STEPS`, `SFT_SAVE_INTERVAL` | optimizer and checkpoint cadence | 200,200 / 1,1 |
| `SFT_MICRO_BATCH_SIZE`, `SFT_GLOBAL_BATCH_SIZE` | per-step and effective batch | 2,16 / 1,1 |
| `SFT_NUM_WORKERS`, `SFT_WARMUP_STEPS` | loader workers and LR warmup | 2,20 / 0,0 |
| `SFT_PREFLIGHT_ONLY` | stop before optimizer/checkpoint after preflight | 0 |
| `PREFLIGHT_LOADER_ONLY` | preflight checks transformed batch without loading CUDA model | 0 |
| `WANDB_MODE` | `auto`, `online`, `offline`, or `disabled` | auto / offline |
| `CHECK_PACKAGE_CONSISTENCY` | run `uv pip check` and report conflicts | 1 |
| `STRICT_PACKAGE_CONSISTENCY` | turn package conflicts into a hard failure | 0 |

`SMOKE_DATA_SOURCE=official_demo` is a smoke-only diagnostic escape hatch. It
converts one successful official LIBERO demonstration into the same LeRobot
schema so the loader/training half can be tested when no Harness attempt
succeeds. It prints a warning, never enters `success_manifest.json`, and must
not be treated as Harness teacher data or used for the formal SFT experiment.

## 1. Collect exactly 20 successes

Review the quick-configuration block at the top of the standalone entry, then
run:

```bash
bash scripts/distillation/collect_harness_successes.sh
```

Defaults cover init states 0–39, three planner seeds per state, and at most 120
attempts. The script freezes `resources/libero/memory` once, never enables skill
evolution/candidate admission, resumes from the manifest, and stops accepting at
exactly 20 successes. It exports the LeRobot v2 dataset under:

```text
logs/distillation/libero_object_lan_t0_harness_sft_v1/dataset
```

Important settings—including output, checkpoint/GPU, VLA endpoint, planner
endpoint/model/key, memory, ranges, and attempt limits—can be overridden with
environment variables named in the script header.

## 2. Train with RLinf

After RLinf's existing uv environment contains its declared OpenPI dependencies:

```bash
bash scripts/distillation/run_rlinf_sft.sh
```

The run logs to both W&B and TensorBoard. Defaults are W&B user
`YichengDong`, entity `ethan-dong-nanjing-university-org`, and project
`harness-vla-sft`. Supply credentials only at runtime:

```bash
WANDB_API_KEY=... bash scripts/distillation/run_rlinf_sft.sh
```

`WANDB_MODE=auto` is the default: valid credentials select online mode, while
missing/invalid credentials or a network failure select offline mode. Explicit
`online`, `offline`, and `disabled` are also supported. Offline runs are kept
under the SFT output's `wandb/` directory for a later manual `wandb sync`.
`WANDB_USERNAME`, `WANDB_ENTITY`, and `WANDB_PROJECT` can override the defaults;
the API key is never added to Hydra config or command-line arguments.

The launcher uses `/home/dongyicheng/RLinf/.venv/bin/python` from
`/home/dongyicheng/RLinf`, validates exactly 20 episodes, and performs one CUDA
forward/backward preflight before launching FSDP. The preflight checks the
`10×32` target, finite loss, frozen VLM, and gradients limited to the action
expert/projections.

Training inherits `pi05_libero` and the original checkpoint normalization,
using mean/std (`use_quantile_norm=false`), 200 optimizer steps, global batch 16
(micro batch 2), LR `2.5e-5`, 20 warmup steps, cosine decay, and weight decay
`1e-10`. FSDP uses bf16 forward parameters/buffers with fp32 reductions and
fp32 optimizer/master parameters. RLinf's current OpenPI SFT implementation
explicitly disables gradient checkpointing in `sft_forward`; the config keeps it
off instead of claiming unsupported checkpointing is active.

The mean/std override is implemented by `rlinf_shim/sitecustomize.py`. It is
enabled only by this launcher and inherited by Ray workers; it does not edit the
RLinf source tree or uv environment. The step-200 checkpoint is converted to the
current server layout with exact key/shape checks:

```text
model.safetensors
metadata.pt
physical-intelligence/libero/norm_stats.json
```

## 3. Standalone acceptance evaluation

Run the base and post-SFT models without Harness, planner, memory, or analytic
primitives:

```bash
bash scripts/distillation/evaluate_pre_post_sft.sh
```

Both models use task-0 init states 40–49. The underlying model horizon is 10 and
the server executes the first five actions per prediction. Only LIBERO
`terminated=true` counts. `comparison.json` contains every init-state result and
returns nonzero unless post-SFT successes are strictly greater than pre-SFT.

LeRobot is deliberately only the dataset container/compatibility layer here.
Switching training backends remains gated on exact checkpoint-key loading,
normalization parity, and same-input action parity.
