#!/bin/bash
# diffusion_policy conda_environment.yaml 설치 전에 한 번 실행
# 반드시 현재 셸에 변수를 남기려면:  source setup_mujoco_path.sh
# ( ./setup_mujoco_path.sh 만 하면 MUJOCO_PATH가 부모 셸로 안 넘어감 )
# pip가 dm-control 의존성으로 mujoco를 빌드할 때 MUJOCO_PATH가 필요함.

set -euo pipefail

MUJOCO_VER="${MUJOCO_VER:-2.3.7}"
INSTALL_DIR="${MUJOCO_INSTALL_DIR:-$HOME/.mujoco}"
TARBALL="mujoco-${MUJOCO_VER}-linux-x86_64.tar.gz"
URL="https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VER}/${TARBALL}"
EXTRACTED="${INSTALL_DIR}/mujoco-${MUJOCO_VER}"

mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

if [[ ! -d "${EXTRACTED}/lib" ]]; then
  echo "Downloading ${URL}"
  wget -nc "${URL}" -O "${TARBALL}"
  tar -xzf "${TARBALL}"
fi

export MUJOCO_PATH="${EXTRACTED}"
export LD_LIBRARY_PATH="${MUJOCO_PATH}/lib:${LD_LIBRARY_PATH:-}"

echo "MUJOCO_PATH=${MUJOCO_PATH}"
echo "LD_LIBRARY_PATH (prefix): ${MUJOCO_PATH}/lib"
echo ""
echo "Next (same shell):"
echo "  conda env remove -n robodiff   # 이전 실패 환경이 있으면"
echo "  conda env create -f conda_environment.yaml"
