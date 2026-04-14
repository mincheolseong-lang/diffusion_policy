#!/bin/bash
# Lift vision: robomimic 0.2 공식 미러에는 image.hdf5 만 있음 (image_abs.hdf5 URL 없음 → 404).
# skip_env_runner: 일부 GPU 노드에서 robosuite+EGL 다중 시뮬 시 Segmentation fault → 스모크는 rollout 생략.
# Rollout만 짧게 검증: sbatch slurm_lift_rollout_smoke.sh (training.skip_env_runner=false, n_envs=1 등).
#
# 장애물 A/B 전략:
#   Baseline: Training/Val = 깨끗한 오프라인 이미지 (CLI로 task.dataset.image_occlusion.enabled=false).
#   Our method (slide 2): TRAIN_STOCH_OCC=1 → task=lift_image_train_stoch_occ, train에서 p=0.5로
#     agentview 고정 블록 + wrist tracking 블록(롤아웃과 동일), val은 깨끗.
#   Test (rollout): OBSTACLE=1이면 policy 관측에 image-space occlusion (OBSTACLE_ALPHA 기본 1.0 = 불투명)
#                          - agentview_image:            고정 사각형
#                          - robot0_eye_in_hand_image:   tracking 사각형 (red target 추적)
#   sbatch --export=ALL,OBSTACLE=0,ROLLOUT=1,FULL_TRAIN=1 --job-name=dp_lift_noobs slurm_train_lift.sh
#   sbatch --export=ALL,OBSTACLE=1,ROLLOUT=1,FULL_TRAIN=1 --job-name=dp_lift_obs   slurm_train_lift.sh
#   Our method + rollout 가림: --export=ALL,TRAIN_STOCH_OCC=1,OBSTACLE=1,OBSTACLE_ALPHA=1.0,ROLLOUT=1,FULL_TRAIN=1
# 공정 비교(가림만 vs 가림+빨간 타깃 XYZ): RED_TARGET=1 → task=lift_image_red_target (low_dim red_target_pos).
#   sbatch --export=ALL,OBSTACLE=1,RED_TARGET=1,ROLLOUT=1,FULL_TRAIN=1 --job-name=dp_lift_obs_rt slurm_train_lift.sh
# 그래프: logs.json.txt → scripts/plot_training_log.py (--triple: train|val|test success, --labels e.g. no_pos pos)
#
# Epoch / 스모크:
#   기본(FULL_TRAIN 미설정): training.debug=True → workspace 가 num_epochs=8, max_train/val_steps=12 로 짧게(스모크).
#   본 실험: FULL_TRAIN=1 → training.debug=false, training.num_epochs=NUM_EPOCHS (기본 150; Hydra 기본 8000 무시).
#   롤아웃 주기: training.rollout_every=30 → workspace에서 1-based 에폭 1,30,60,…,150마다 평가.
#   평가 에피소드: task.env_runner.n_test=50 (lift_image.yaml 기본과 동일, 명시).
#   예: sbatch --export=ALL,OBSTACLE=1,ROLLOUT=1,FULL_TRAIN=1,NUM_EPOCHS=150 --job-name=dp_lift_obs ...
# 짧은 테스트만 하려면: sbatch --time=0:30:00 ... (아래 기본 time 은 풀런용)
# dataloader.num_workers=0: 메인에서 CUDA/MuJoCo 등 로드 후 fork된 DataLoader 워커가 SIGABRT 로 죽는 경우 방지(HPC).
# 데이터 (한 번):
#   mkdir -p data/robomimic/datasets/lift/ph
#   wget -O data/robomimic/datasets/lift/ph/image.hdf5 \
#     http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/image.hdf5
# 또는:
#   python /path/to/robomimic/scripts/download_datasets.py \
#     --tasks lift --dataset_types ph --hdf5_types image \
#     --download_dir "$PWD/data/robomimic/datasets"
# 나중에 goal 좌표까지 쓰려면 abs 데모(HF 등)로 image_abs 맞추거나 변환 스크립트 검토.

#SBATCH --job-name=dp_train_lift_img
#SBATCH --partition=gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# SBATCH time = 상한. 풀 학습(FULL_TRAIN=1)에 맞춰 기본 12h; 스모크만이면 sbatch --time=0:30:00 로 덮어쓰기.
# training.debug=True(기본) → workspace: num_epochs=8, max_train/val_steps=12.
#SBATCH --time=72:00:00
#SBATCH --output=/mnt/shared-scratch/Hou_I/mincheolseong/diffusion_policy/logs/%x-%j.out
#SBATCH --error=/mnt/shared-scratch/Hou_I/mincheolseong/diffusion_policy/logs/%x-%j.err

set -euo pipefail

mkdir -p /mnt/shared-scratch/Hou_I/mincheolseong/diffusion_policy/logs

DP_ROOT="/mnt/shared-scratch/Hou_I/mincheolseong/diffusion_policy"
ROBODIFF_PREFIX="/mnt/shared-scratch/Hou_I/mincheolseong/conda/envs/robodiff"
SCRATCH_CONDA="/mnt/shared-scratch/Hou_I/mincheolseong/conda"
cd "$DP_ROOT"

if [[ -f /home/grads/m/mincheolseong/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/grads/m/mincheolseong/miniconda3/etc/profile.d/conda.sh
  set +u
  if [[ -d "$ROBODIFF_PREFIX" ]]; then
    conda activate "$ROBODIFF_PREFIX"
  elif conda env list | awk '{print $1}' | grep -qx robodiff; then
    conda activate robodiff
  else
    conda activate base
  fi
  set -u
fi

if [[ -f "$SCRATCH_CONDA/env_robodiff_scratch.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRATCH_CONDA/env_robodiff_scratch.sh"
fi

# EGL path: NVIDIA libs + glew headers (conda glew); patchelf in PATH for mujoco_py post-step
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="${CONDA_PREFIX}/include:${CPATH:-}"
# Hydra 기본 메시지는 원인 예외를 숨김 (RobomimicImageRunner / mujoco_py 등)
export HYDRA_FULL_ERROR=1
# wandb 내부 service 가 HPC 노드에서 포트를 못 찾아 assert ports_found 로 죽는 문제 방지.
# logging.mode=disabled 를 이미 넘기지만, 새 wandb 버전은 mode 확인 전에 service 를 먼저 띄움.
export WANDB_DISABLE_SERVICE=true
export WANDB_MODE=disabled

# --- Rollout / EGL workarounds ---
# skip_env_runner 기본값: ROLLOUT=1 이면 rollout 켬, 아니면 끔
ROLLOUT="${ROLLOUT:-0}"
if [[ "$ROLLOUT" == "1" ]]; then
  # subprocess rollout: EGL env vars inherited by child process
  export MUJOCO_GL=egl
  export DP_ROBOMIMIC_NO_EGL_PROBE=1
  export DP_ROBOSUITE_RENDER_GPU_DEVICE_ID=0
  [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=0
  SKIP_ENV_RUNNER=true          # training process never touches MuJoCo
  SUBPROCESS_ROLLOUT=true       # rollout spawned as fresh child
else
  SKIP_ENV_RUNNER=true
  SUBPROCESS_ROLLOUT=false
fi

# mujoco_py ignores MUJOCO_GL; it uses OSMesa unless get_nvidia_lib_dir() finds
# /usr/lib/nvidia-NNN. HPC nodes often only have /usr/lib64 → patch + one rebuild.
set +e
python "$DP_ROOT/scripts/ensure_mujoco_py_egl.py"
_mjpy_patch=$?
set -e
if [[ "$_mjpy_patch" -eq 1 ]]; then exit 1; fi
if [[ "$_mjpy_patch" -eq 2 ]]; then export MUJOCO_PY_FORCE_REBUILD=1; fi

# Rollout 비활성화 시 robomimic env_runner 불필요 — import 프리플라이트 생략.
# Rollout 검증은 slurm_lift_rollout_smoke.sh 사용.

# Baseline: force clean training pixels. TRAIN_STOCH_OCC=1 leaves task YAML (stochastic occlusion) intact.
# Rollout (test) occlusion: training.rollout_obs_occlusion when OBSTACLE=1.
if [[ "${TRAIN_STOCH_OCC:-0}" =~ ^(1|true|yes|on)$ ]]; then
  OCC_ARGS=()
else
  OCC_ARGS=(task.dataset.image_occlusion.enabled=false)
fi

# When OBSTACLE=1: enable obs-space occlusion in the rollout subprocess.
#   agentview_image  → fixed rectangle covering target region
#   wrist camera     → tracking rectangle that follows the red target each frame
#   alpha: overlay opacity [0,1]; 1=fully opaque (old). Semi-transparent: OBSTACLE_ALPHA=0.5~0.85 in sbatch --export
OBSTACLE_ALPHA="${OBSTACLE_ALPHA:-1.0}"
ROLLOUT_OCC_ARGS=()
if [[ "${OBSTACLE:-0}" =~ ^(1|true|yes|on)$ ]]; then
  ROLLOUT_OCC_ARGS=(
    training.rollout_obs_occlusion.enabled=true
    "training.rollout_obs_occlusion.agentview_rect_norm=[0.25,0.25,0.75,0.75]"
    "training.rollout_obs_occlusion.wrist_rect_norm=[0.45,0.45,0.85,0.85]"
    training.rollout_obs_occlusion.wrist_tracking=true
    "training.rollout_obs_occlusion.alpha=${OBSTACLE_ALPHA}"
  )
else
  ROLLOUT_OCC_ARGS=(training.rollout_obs_occlusion.enabled=false)
fi

# Hydra run dir collision guard:
# two sbatch jobs can start in the same second and collide on ${now:%H.%M.%S}.
# Include obstacle label + SLURM job id so each run dir is always unique.
TASK_YAML="lift_image"
if [[ "${TRAIN_STOCH_OCC:-0}" =~ ^(1|true|yes|on)$ ]]; then
  TASK_YAML="lift_image_train_stoch_occ"
elif [[ "${RED_TARGET:-0}" =~ ^(1|true|yes|on)$ ]]; then
  TASK_YAML="lift_image_red_target"
fi

if [[ "${OBSTACLE:-0}" =~ ^(1|true|yes|on)$ ]]; then
  if [[ "${TRAIN_STOCH_OCC:-0}" =~ ^(1|true|yes|on)$ ]]; then
    RUN_VARIANT="obs_stoch"
  elif [[ "${RED_TARGET:-0}" =~ ^(1|true|yes|on)$ ]]; then
    RUN_VARIANT="obs_rt"
  else
    RUN_VARIANT="obs"
  fi
else
  if [[ "${TRAIN_STOCH_OCC:-0}" =~ ^(1|true|yes|on)$ ]]; then
    RUN_VARIANT="noobs_stoch"
  else
    RUN_VARIANT="noobs"
  fi
fi
RUN_JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_DIR_OVERRIDE="data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}_${RUN_VARIANT}_job${RUN_JOB_ID}"

# FULL_TRAIN=1: train_diffusion_unet_image_workspace 의 debug 분기(num_epochs=8) 끔 → 긴 학습
FULL_TRAIN="${FULL_TRAIN:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-150}"
DEBUG_ARGS=(training.debug=True)
if [[ "$FULL_TRAIN" == "1" ]]; then
  DEBUG_ARGS=(
    training.debug=false
    "training.num_epochs=${NUM_EPOCHS}"
  )
fi

# FULL_TRAIN=1 일 때만 num_epochs=NUM_EPOCHS(기본 150) 는 DEBUG_ARGS 에서 설정됨. 스모크는 workspace debug(8 ep).
# rollout 30에폭마다(1-based 1,30,…,150), n_test=50 명시
python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  "task=${TASK_YAML}" \
  "${DEBUG_ARGS[@]}" \
  "training.skip_env_runner=${SKIP_ENV_RUNNER}" \
  "training.use_subprocess_rollout=${SUBPROCESS_ROLLOUT}" \
  training.resume=False \
  logging.mode=disabled \
  training.device=cuda:0 \
  training.rollout_every=30 \
  task.env_runner.n_test=50 \
  task.env_runner.n_envs=14 \
  dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.batch_size=8 \
  val_dataloader.num_workers=0 \
  hydra.run.dir="${RUN_DIR_OVERRIDE}" \
  "${OCC_ARGS[@]}" \
  "${ROLLOUT_OCC_ARGS[@]}"
