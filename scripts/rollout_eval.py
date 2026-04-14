#!/usr/bin/env python3
"""
Isolated rollout evaluator – spawned as a subprocess by the training loop.

Why subprocess: the training process has a live PyTorch CUDA context that
conflicts with MuJoCo EGL initialisation on some HPC nodes. A fresh child
process has no CUDA context, so MuJoCo EGL can initialise first and CUDA
only comes up afterwards when we move the policy to GPU.

Called by TrainDiffusionUnetImageWorkspace._run_rollout_subprocess().
Do NOT import torch / CUDA here at module level – let argparse finish first.
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import pathlib
import subprocess
import time
import signal
import atexit

# --------------------------------------------------------------------------- #
# Xvfb virtual display – start BEFORE any GL import so Mesa EGL can bind to
# the X11 surface instead of the NVIDIA device-based EGL that segfaults.
# Only started when DISPLAY is not already set.
# --------------------------------------------------------------------------- #
_xvfb_proc = None

def _start_xvfb() -> None:
    global _xvfb_proc
    if os.environ.get("DISPLAY"):
        print(f"rollout_eval: reusing existing DISPLAY={os.environ['DISPLAY']}", flush=True)
        return
    disp = f":{os.getpid() % 200 + 100}"  # unique per-process display number
    try:
        _xvfb_proc = subprocess.Popen(
            ["Xvfb", disp, "-screen", "0", "1280x960x24", "-ac", "+extension", "GLX"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)  # wait for Xvfb to be ready
        if _xvfb_proc.poll() is not None:
            print("rollout_eval: Xvfb exited immediately – continuing without virtual display", flush=True)
            _xvfb_proc = None
            return
        os.environ["DISPLAY"] = disp
        print(f"rollout_eval: Xvfb started on DISPLAY={disp} (pid={_xvfb_proc.pid})", flush=True)
    except FileNotFoundError:
        print("rollout_eval: Xvfb not found – continuing without virtual display", flush=True)

def _stop_xvfb() -> None:
    if _xvfb_proc is not None:
        try:
            _xvfb_proc.send_signal(signal.SIGTERM)
        except Exception:
            pass

atexit.register(_stop_xvfb)
_start_xvfb()

# Enable Python fault handler – prints C-level stack trace on segfault/SIGABRT.
import faulthandler as _fh
_fh.enable()

# Use OSMesa (CPU software renderer) – avoids device-based NVIDIA EGL that
# segfaults on HPC nodes without /dev/dri/renderD*.
# Fall back to egl only if MUJOCO_GL is already explicitly set to something else.
os.environ.setdefault("MUJOCO_GL", "osmesa")
# Force CPU (OSMesa) extension: mujoco_py picks GPU extension on any node with NVIDIA drivers,
# regardless of MUJOCO_GL.  MUJOCO_PY_FORCE_CPU=1 overrides this selection.
os.environ.setdefault("MUJOCO_PY_FORCE_CPU", "1")
# OSMesa needs explicit OpenGL version hints
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "330")
print(f"rollout_eval: MUJOCO_GL={os.environ.get('MUJOCO_GL')}", flush=True)
os.environ.setdefault("DP_ROBOMIMIC_NO_EGL_PROBE", "1")
os.environ.setdefault("DP_ROBOSUITE_RENDER_GPU_DEVICE_ID", "0")
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class ObsOcclusionPolicy:
    """
    Wraps a BaseImagePolicy and applies image-space occlusion to camera
    observations *before* the policy sees them during rollout.

    Training/validation data are always clean; occlusion is only applied
    here, at test (rollout) time, simulating an obstacle blocking the camera.

    obs_dict tensors at predict_action time:
        shape  (n_envs, T_obs, H, W, C)  dtype uint8  device=policy.device
    """

    def __init__(self, policy, cfg: dict):
        self._policy = policy
        self._agentview_rect = tuple(cfg.get("agentview_rect_norm", [0.25, 0.25, 0.75, 0.75]))
        wrist_rn = cfg.get("wrist_rect_norm", [0.45, 0.45, 0.85, 0.85])
        self._wrist_rect = tuple(wrist_rn)
        self._wrist_tracking = bool(cfg.get("wrist_tracking", True))
        self._alpha = float(cfg.get("alpha", 1.0))
        c = cfg.get("color", [55, 55, 60])
        self._color = tuple(int(x) for x in c)

    # ---- proxy mandatory BaseImagePolicy interface -------------------------
    @property
    def device(self):
        return self._policy.device

    @property
    def dtype(self):
        return self._policy.dtype

    def reset(self):
        return self._policy.reset()

    def predict_action(self, obs_dict):
        import torch
        import numpy as np
        from diffusion_policy.common.image_occlusion import (
            apply_rectangle_occlusion_thwc_uint8,
            apply_tracking_occlusion_thwc_uint8,
        )

        occluded = dict(obs_dict)

        def _apply(key, fn):
            if key not in occluded:
                return
            t = occluded[key]   # (n_envs, T, C, H, W) float32 [0,1]  — channels-first
            dev = t.device
            # float [0,1] → uint8 [0,255], then transpose to (n_envs, T, H, W, C)
            arr_f = t.detach().cpu().float().numpy()    # (n_envs, T, C, H, W)
            arr_u8 = np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
            arr_hwc = arr_u8.transpose(0, 1, 3, 4, 2)  # (n_envs, T, H, W, C)

            result_hwc = np.empty_like(arr_hwc)
            for i in range(arr_hwc.shape[0]):           # per env
                result_hwc[i] = fn(arr_hwc[i])          # fn: (T, H, W, C) → (T, H, W, C)

            # back to (n_envs, T, C, H, W) float [0,1]
            result_chw = result_hwc.transpose(0, 1, 4, 2, 3).astype(np.float32) / 255.0
            occluded[key] = torch.from_numpy(result_chw).to(dev)

        # agentview: fixed rectangle
        _apply(
            "agentview_image",
            lambda thwc: apply_rectangle_occlusion_thwc_uint8(
                thwc,
                self._agentview_rect,
                color=self._color,
                alpha=self._alpha,
            ),
        )

        # wrist: tracking (center follows red target) or fixed
        if self._wrist_tracking:
            bw = float(self._wrist_rect[2] - self._wrist_rect[0])
            bh = float(self._wrist_rect[3] - self._wrist_rect[1])
            _apply(
                "robot0_eye_in_hand_image",
                lambda thwc: apply_tracking_occlusion_thwc_uint8(
                    thwc,
                    rect_size_norm=(bw, bh),
                    color=self._color,
                    alpha=self._alpha,
                ),
            )
        else:
            _apply(
                "robot0_eye_in_hand_image",
                lambda thwc: apply_rectangle_occlusion_thwc_uint8(
                    thwc,
                    self._wrist_rect,
                    color=self._color,
                    alpha=self._alpha,
                ),
            )

        return self._policy.predict_action(occluded)

    def __getattr__(self, name):
        # Fall through for any other attribute (e.g. normalizer, noise_scheduler …)
        return getattr(self._policy, name)


def _atomic_write_result_json(path: pathlib.Path, payload: dict) -> None:
    """POSIX에서 replace로 원자적 기록 — 쓰기 도중 부모가 읽지 않도록."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    data = json.dumps(payload, indent=2)
    tmp.write_text(data)
    tmp.replace(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"result json not written: {path}")


def _main_impl(args: argparse.Namespace) -> int:
    args.output_dir = args.output_dir.resolve()
    args.result_json = args.result_json.resolve()
    args.runner_cfg = args.runner_cfg.resolve()
    args.policy_cfg = args.policy_cfg.resolve()
    args.policy_state = args.policy_state.resolve()

    # ------------------------------------------------------------------ #
    # 1. Imports (after env vars are set)
    # ------------------------------------------------------------------ #
    import torch
    import numpy as np
    import hydra
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    runner_cfg = OmegaConf.create(json.loads(args.runner_cfg.read_text()))
    policy_cfg = OmegaConf.create(json.loads(args.policy_cfg.read_text()))

    # ------------------------------------------------------------------ #
    # 2. Create env_runner FIRST – MuJoCo/EGL init here, CUDA not yet up
    # ------------------------------------------------------------------ #
    print("rollout_eval: instantiating env_runner …", flush=True)
    env_runner = hydra.utils.instantiate(
        runner_cfg, output_dir=str(args.output_dir))

    # ------------------------------------------------------------------ #
    # 3. Load policy (CUDA initialises here, after MuJoCo EGL)
    # ------------------------------------------------------------------ #
    print(f"rollout_eval: loading policy state from {args.policy_state} …",
          flush=True)
    device = torch.device(args.device)

    policy = hydra.utils.instantiate(policy_cfg)
    state = torch.load(str(args.policy_state), map_location="cpu")
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()

    # ------------------------------------------------------------------ #
    # 4. Optionally wrap policy with obs-space occlusion
    #    (training/val use clean data; occlusion is applied here at test time)
    # ------------------------------------------------------------------ #
    occ_cfg = json.loads(args.obs_occlusion_cfg)
    if occ_cfg.get("enabled", False):
        print(
            f"rollout_eval: obs occlusion ENABLED – "
            f"agentview rect={occ_cfg.get('agentview_rect_norm')}, "
            f"wrist rect={occ_cfg.get('wrist_rect_norm')} "
            f"(tracking={occ_cfg.get('wrist_tracking', True)}), "
            f"alpha={occ_cfg.get('alpha', 1.0)} (1=opaque overlay)",
            flush=True,
        )
        policy = ObsOcclusionPolicy(policy, occ_cfg)
    else:
        print("rollout_eval: obs occlusion disabled (clean observations)", flush=True)

    # ------------------------------------------------------------------ #
    # 5. Run rollout
    # ------------------------------------------------------------------ #
    print("rollout_eval: running env_runner …", flush=True)
    runner_log = env_runner.run(policy)

    # ------------------------------------------------------------------ #
    # 6. Serialise results (only scalar values)
    # ------------------------------------------------------------------ #
    def _to_py(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, (int, float, bool, str)):
            return v
        return None

    clean = {k: _to_py(v) for k, v in runner_log.items()
             if _to_py(v) is not None}
    _atomic_write_result_json(args.result_json.resolve(), clean)
    print(f"rollout_eval: done → {clean}", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Rollout evaluation subprocess")
    p.add_argument("--runner-cfg",     required=True, type=pathlib.Path,
                   help="JSON file: OmegaConf-serialised task.env_runner config")
    p.add_argument("--policy-cfg",     required=True, type=pathlib.Path,
                   help="JSON file: OmegaConf-serialised policy config")
    p.add_argument("--policy-state",   required=True, type=pathlib.Path,
                   help=".pt file: policy.state_dict() (includes normalizer)")
    p.add_argument("--output-dir",     required=True, type=pathlib.Path,
                   help="Runner output dir (videos etc.)")
    p.add_argument("--result-json",    required=True, type=pathlib.Path,
                   help="Where to write runner_log dict as JSON")
    p.add_argument("--device",         default="cuda:0")
    p.add_argument("--obs-occlusion-cfg", default="{}", type=str,
                   help="JSON: rollout obs occlusion config (from training.rollout_obs_occlusion)")
    args = p.parse_args()

    try:
        return _main_impl(args)
    except BaseException as e:
        import traceback
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        try:
            args.result_json = args.result_json.resolve()
            _atomic_write_result_json(
                args.result_json, {"rollout_fatal": err})
        except Exception as w:
            print(f"rollout_eval: could not write result json: {w!r}", flush=True)
        print(f"rollout_eval: FATAL {e!r}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
