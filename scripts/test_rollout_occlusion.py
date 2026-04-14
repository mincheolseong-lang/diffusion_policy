#!/usr/bin/env python3
"""
Visually verify rollout-time image occlusion BEFORE full training.

Creates a Lift robosuite environment with OSMesa (same setup as rollout_eval.py),
steps through a few random actions, and saves side-by-side PNG images:
  left  = clean observation (what the env returns)
  right = occluded observation (what the policy will see during test rollout)

Both cameras are verified:
  agentview_image         – fixed rectangle covering target area
  robot0_eye_in_hand_image – tracking rectangle that follows red target

Output: results/rollout_occlusion_test/step_NNN.png
"""
from __future__ import annotations

import os
import sys
import pathlib
import argparse

# Must be set BEFORE any GL/MuJoCo import — mirrors rollout_eval.py exactly.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("MUJOCO_PY_FORCE_CPU", "1")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "330")
os.environ.setdefault("DP_ROBOMIMIC_NO_EGL_PROBE", "1")

import faulthandler
faulthandler.enable()

print(f"MUJOCO_GL            = {os.environ['MUJOCO_GL']}", flush=True)
print(f"MUJOCO_PY_FORCE_CPU  = {os.environ['MUJOCO_PY_FORCE_CPU']}", flush=True)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_policy.common.image_occlusion import (
    apply_rectangle_occlusion_hwc_uint8,
    _find_red_target_center_norm,
)


# --------------------------------------------------------------------------- #
# Occlusion params (must match slurm_train_lift.sh ROLLOUT_OCC_ARGS)
# --------------------------------------------------------------------------- #
AGENTVIEW_RECT  = (0.25, 0.25, 0.75, 0.75)   # fixed, normalized (x0,y0,x1,y1)
WRIST_RECT      = (0.45, 0.45, 0.85, 0.85)   # tracking box size = (0.40, 0.40)
WRIST_BW        = float(WRIST_RECT[2] - WRIST_RECT[0])   # 0.40
WRIST_BH        = float(WRIST_RECT[3] - WRIST_RECT[1])   # 0.40


def occlude_agentview(hwc: np.ndarray) -> np.ndarray:
    return apply_rectangle_occlusion_hwc_uint8(hwc, AGENTVIEW_RECT)


def occlude_wrist(hwc: np.ndarray) -> np.ndarray:
    """Tracking: find red target centroid, place box there."""
    center = _find_red_target_center_norm(hwc)
    if center is not None:
        cx, cy = center
    else:
        cx, cy = 0.5, 0.5   # fallback to image centre
    x0 = float(np.clip(cx - WRIST_BW * 0.5, 0.0, 1.0))
    y0 = float(np.clip(cy - WRIST_BH * 0.5, 0.0, 1.0))
    x1 = float(np.clip(x0 + WRIST_BW, 0.0, 1.0))
    y1 = float(np.clip(y0 + WRIST_BH, 0.0, 1.0))
    detected = center is not None
    return apply_rectangle_occlusion_hwc_uint8(hwc, (x0, y0, x1, y1)), detected


def make_env(img_size: int = 84):
    import robosuite as suite
    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=img_size,
        camera_widths=img_size,
        render_gpu_device_id=-1,   # -1 = OSMesa (CPU renderer)
        ignore_done=True,
        reward_shaping=True,
    )
    return env


def save_comparison(
    step: int,
    agentview_clean: np.ndarray,
    agentview_occ: np.ndarray,
    wrist_clean: np.ndarray,
    wrist_occ: np.ndarray,
    wrist_detected: bool,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(
        f"Rollout Occlusion Verification  –  step {step:03d}",
        fontsize=13, fontweight="bold"
    )

    imgs = [agentview_clean, agentview_occ, wrist_clean, wrist_occ]
    titles = [
        "agentview  CLEAN\n(env output)",
        "agentview  OCCLUDED\n(what policy sees)",
        f"wrist  CLEAN\n(env output)",
        f"wrist  OCCLUDED  {'[tracking ✓]' if wrist_detected else '[fallback centre]'}\n(what policy sees)",
    ]
    colors = ["#2ecc71", "#e74c3c", "#2ecc71", "#e74c3c"]

    for ax, img, title, col in zip(axes.flat, imgs, titles, colors):
        ax.imshow(img)
        ax.set_title(title, fontsize=9, color=col, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    out = out_dir / f"step_{step:03d}.png"
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir",  default="results/rollout_occlusion_test",
                   type=pathlib.Path)
    p.add_argument("--n-steps",  default=8, type=int,
                   help="Number of environment steps to visualise")
    p.add_argument("--img-size", default=84, type=int)
    args = p.parse_args()

    out_dir = pathlib.Path(_ROOT) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Creating Lift environment …", flush=True)
    env = make_env(args.img_size)
    obs = env.reset()
    print("Environment ready.", flush=True)

    saved = []
    for step in range(args.n_steps):
        # Small random action (just enough to move the arm slightly)
        lo, hi = env.action_spec
        action = np.random.uniform(lo * 0.15, hi * 0.15)
        obs, reward, done, _ = env.step(action)

        # robosuite returns (H, W, 3) uint8 images
        av_clean = obs["agentview_image"]
        wr_clean  = obs["robot0_eye_in_hand_image"]

        av_occ = occlude_agentview(av_clean)
        wr_occ, wr_det = occlude_wrist(wr_clean)

        out = save_comparison(step, av_clean, av_occ, wr_clean, wr_occ, wr_det, out_dir)
        saved.append(out)
        print(f"  step {step:03d}: reward={reward:.3f}  wrist_target_detected={wr_det}  → {out}", flush=True)

    env.close()

    print(f"\nAll {len(saved)} images saved to: {out_dir}", flush=True)
    print("Files:", flush=True)
    for p in saved:
        print(f"  {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
