#!/bin/bash

#SBATCH --job-name=dp_gpu_probe
#SBATCH --partition=gpu-research-sh
#SBATCH --qos=olympus-research-gpu-sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
#SBATCH --output=/mnt/shared-scratch/Hou_I/mincheolseong/diffusion_policy/logs/%x-%j.out
#SBATCH --error=/mnt/shared-scratch/Hou_I/mincheolseong/diffusion_policy/logs/%x-%j.err

set -euo pipefail

mkdir -p /mnt/shared-scratch/Hou_I/mincheolseong/diffusion_policy/logs

echo "=== Job info ==="
echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "PWD will be: /mnt/shared-scratch/Hou_I/mincheolseong/diffusion_policy"

echo ""
echo "=== nvidia-smi ==="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi
else
  echo "nvidia-smi not in PATH"
  exit 1
fi

echo ""
echo "=== CUDA env (if any) ==="
env | grep -E '^(CUDA|SLURM|GPU)' | sort || true

DP_ROOT="/mnt/shared-scratch/Hou_I/mincheolseong/diffusion_policy"
cd "$DP_ROOT"

echo ""
echo "=== Python / PyTorch (conda if available) ==="
if [[ -f /home/grads/m/mincheolseong/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/grads/m/mincheolseong/miniconda3/etc/profile.d/conda.sh
  set +u
  if conda env list | awk '{print $1}' | grep -qx robodiff; then
    conda activate robodiff
    echo "Activated conda env: robodiff"
  else
    conda activate base
    echo "robodiff env not found; using base"
  fi
  set -u
else
  echo "No miniconda conda.sh at ~/miniconda3; using job default python"
fi

python - <<'PY'
import sys
print("executable:", sys.executable)
print("version:", sys.version.split()[0])
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device[0]:", torch.cuda.get_device_name(0))
        x = torch.zeros(1, device="cuda")
        print("cuda tensor ok:", x.device)
except Exception as e:
    print("torch import/check failed:", repr(e))
PY

echo ""
echo "=== diffusion_policy import (optional) ==="
python - <<'PY'
import sys
sys.path.insert(0, "/mnt/shared-scratch/Hou_I/mincheolseong/diffusion_policy")
try:
    import diffusion_policy
    print("import diffusion_policy: OK", diffusion_policy.__file__)
except Exception as e:
    print("import diffusion_policy: skipped or failed:", repr(e))
PY

echo ""
echo "=== Probe finished OK ==="
