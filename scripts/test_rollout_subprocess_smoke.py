#!/usr/bin/env python3
"""
학습 루프 없이 `TrainDiffusionUnetImageWorkspace._run_rollout_subprocess` 만 돌려
rollout_eval 서브프로세스 + result.json + runner 메트릭이 나오는지 검증한다.

전제:
  - repo 루트에서 실행 (`diffusion_policy/` 에 train.py 가 있는 디렉터리)
  - `data/robomimic/datasets/lift/ph/image.hdf5` 존재
  - robodiff(또는 동일 의존성) conda
  - GPU 노드 권장: `training.device=cuda:0` (CPU는 매우 느림)

예:
  cd /path/to/diffusion_policy
  export WANDB_MODE=disabled
  python scripts/test_rollout_subprocess_smoke.py
  python scripts/test_rollout_subprocess_smoke.py --task lift_image_red_target --occlusion
  python scripts/test_rollout_subprocess_smoke.py --device cpu   # GPU 없을 때만
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        default="lift_image",
        help="lift_image | lift_image_red_target",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--occlusion",
        action="store_true",
        help="slurm OBSTACLE=1 과 유사한 rollout 이미지 가림 켜기",
    )
    ap.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="기본: results/rollout_subprocess_smoke",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=24,
        help="에피소드당 최대 env step (짧을수록 스모크 빠름; 기본 24)",
    )
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    import hydra
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    cfg_dir = (REPO_ROOT / "diffusion_policy" / "config").resolve()
    overrides = [
        f"task={args.task}",
        "task.env_runner.n_train=1",
        "task.env_runner.n_test=1",
        "task.env_runner.n_envs=2",
        "task.env_runner.n_train_vis=0",
        "task.env_runner.n_test_vis=0",
        f"task.env_runner.max_steps={args.max_steps}",
        f"training.device={args.device}",
        "logging.mode=disabled",
    ]
    if args.occlusion:
        overrides += [
            "training.rollout_obs_occlusion.enabled=true",
            "training.rollout_obs_occlusion.agentview_rect_norm=[0.25,0.25,0.75,0.75]",
            "training.rollout_obs_occlusion.wrist_rect_norm=[0.45,0.45,0.85,0.85]",
            "training.rollout_obs_occlusion.wrist_tracking=true",
        ]
    else:
        overrides.append("training.rollout_obs_occlusion.enabled=false")

    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(
            config_name="train_diffusion_unet_image_workspace",
            overrides=overrides,
        )

    out = (args.out_dir or (REPO_ROOT / "results" / "rollout_subprocess_smoke")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
        TrainDiffusionUnetImageWorkspace,
    )

    print(f"[smoke] repo={REPO_ROOT}", flush=True)
    print(f"[smoke] out_dir={out}", flush=True)
    print(f"[smoke] overrides={overrides}", flush=True)

    ws = TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(out))
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    ws.model.set_normalizer(dataset.get_normalizer())
    dev = torch.device(cfg.training.device)
    ws.model.to(dev)
    ws.model.eval()

    print("[smoke] _run_rollout_subprocess …", flush=True)
    log = ws._run_rollout_subprocess(cfg, ws.model)

    assert isinstance(log, dict) and len(log) > 0, f"empty log: {log!r}"
    assert any(
        k.startswith("test/") or k.startswith("train/") for k in log
    ), f"no runner keys in {list(log.keys())}"

    rj = out / ".rollout_tmp" / "result.json"
    assert rj.is_file(), f"missing {rj}"
    saved = json.loads(rj.read_text())
    assert saved, f"empty json {rj}"
    assert "rollout_fatal" not in saved, saved

    print("[smoke] OK", flush=True)
    print(json.dumps(log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
