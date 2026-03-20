#!/bin/bash
#SBATCH --job-name=lift_check
#SBATCH --output=log/lift_out_%j.txt
#SBATCH --error=log/lift_err_%j.txt
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail
mkdir -p log

ml Miniconda3/23.10.0-1
module load WebProxy || true

cd /scratch/user/mincheolseong/VISNAV-project/diffusion_policy

# conda init 없이 안정적으로 활성화
eval "$(/sw/eb/sw/Miniconda3/23.10.0-1/bin/conda shell.bash hook)"
conda activate /scratch/user/mincheolseong/.conda/envs/robodiff

export MUJOCO_GL=osmesa
export CPATH="$CONDA_PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "python: $(which python)"
python -u lift_check.py