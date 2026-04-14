#!/bin/bash
# 로그인 노드에서: sbatch slurm_dump_lift_sim_views.sh
# (GPU 1개 할당 후 dump_lift_sim_obstacle_views.py 실행 — mujoco_py EGL 경로)
#
# 인자 전달: sbatch slurm_dump_lift_sim_views.sh -- --both --seed 0 --render-gpu-device-id -1
# (-- 뒤를 주면 전체 인자를 대체함; Slurm EGL 세그폴트 시 --render-gpu-device-id -1 유지 권장)

#SBATCH --job-name=dp_dump_lift_sim
#SBATCH --partition=gpu-research-sh
#SBATCH --qos=olympus-research-gpu-sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0:15:00
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
# Offscreen EGL + some driver stacks segfault with fixed device id; -1 lets robosuite infer.
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

set +e
python "$DP_ROOT/scripts/ensure_mujoco_py_egl.py"
_mjpy_patch=$?
set -e
if [[ "$_mjpy_patch" -eq 1 ]]; then exit 1; fi
if [[ "$_mjpy_patch" -eq 2 ]]; then export MUJOCO_PY_FORCE_REBUILD=1; fi

# Slurm은 GPU를 할당하므로 CUDA_VISIBLE_DEVICES는 보통 자동; 없으면 0
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

DUMP_ARGS=(--both --seed 0 --render-gpu-device-id -1)
if [[ "${1:-}" == "--" ]]; then
  shift
  DUMP_ARGS=("$@")
fi

python "$DP_ROOT/scripts/dump_lift_sim_obstacle_views.py" "${DUMP_ARGS[@]}"
echo "slurm_dump_lift_sim_views: done"
