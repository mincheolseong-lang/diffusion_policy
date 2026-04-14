#!/bin/bash
# Rollout smoke test: env_runner ON, minimal sims (1 env at a time), short episodes, no train/test videos.
# Submit: sbatch slurm_lift_rollout_smoke.sh
#
# Success: logs/*.out 에 "Eval LiftImage" / tqdm 진행, logs.json.txt 에 test_mean_score 또는 train/mean_score 등 runner 키.
# Failure: segfault → 노드/드라이버 이슈; n_envs=1 유지, 다른 파티션 시도
# train 없이 env 만: python scripts/robomimic_lift_env_smoke.py (같은 export 권장).

#SBATCH --job-name=dp_lift_roll_smoke
#SBATCH --partition=gpu-research-sh
#SBATCH --qos=olympus-research-gpu-sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0:25:00
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

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="${CONDA_PREFIX}/include:${CPATH:-}"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl
# Linux 기본 SyncVectorEnv; 명시 (async+spawn 은 Slurm 에서 더 잘 깨짐)
export DP_ROBOMIMIC_SYNC_VECTOR=1
# PyTorch CUDA 후 robomimic egl_probe 가 render GPU 를 덮어쓸 때 segfault 나는 노드 대응
# (diffusion_policy/env_runner/robomimic_image_runner.py 의 _apply_slurm_egl_workarounds)
export DP_ROBOMIMIC_NO_EGL_PROBE=1
# Slurm 단일 GPU: 0 이 더 안정적인 경우가 많음 (-1 이 segfault 이면 0 유지)
export DP_ROBOSUITE_RENDER_GPU_DEVICE_ID=0

set +e
python "$DP_ROOT/scripts/ensure_mujoco_py_egl.py"
_mjpy_patch=$?
set -e
if [[ "$_mjpy_patch" -eq 1 ]]; then exit 1; fi
if [[ "$_mjpy_patch" -eq 2 ]]; then export MUJOCO_PY_FORCE_REBUILD=1; fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

_runner_import() {
  python -c "from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner; print('env_runner import ok')"
}
if ! _runner_import; then
  rm -f "${CONDA_PREFIX}"/lib/python3.*/site-packages/mujoco_py/generated/cymj_*.so 2>/dev/null || true
  rm -rf "${CONDA_PREFIX}"/lib/python3.*/site-packages/mujoco_py/generated/_pyxbld_* 2>/dev/null || true
  export MUJOCO_PY_FORCE_REBUILD=1
  _runner_import
fi

python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=lift_image \
  training.debug=True \
  training.skip_env_runner=false \
  training.resume=False \
  logging.mode=disabled \
  training.device=cuda:0 \
  task.env_runner.n_train=1 \
  task.env_runner.n_test=1 \
  task.env_runner.n_envs=1 \
  task.env_runner.n_train_vis=0 \
  task.env_runner.n_test_vis=0 \
  task.env_runner.max_steps=120 \
  task.env_runner.tqdm_interval_sec=1.0 \
  dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.batch_size=8 \
  val_dataloader.num_workers=0

echo "slurm_lift_rollout_smoke: train.py exited $?"
echo "Check latest run logs.json.txt for keys like test_mean_score (slashes -> underscores in checkpoint)."
