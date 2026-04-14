#!/usr/bin/env python3
"""
Minimal robosuite Lift + offscreen cameras (no train.py, no zarr).
Use on a GPU Slurm step to see if MuJoCo/EGL alone segfaults.

  export MUJOCO_GL=egl
  export DP_ROBOMIMIC_NO_EGL_PROBE=1
  export DP_ROBOSUITE_RENDER_GPU_DEVICE_ID=0
  python scripts/robomimic_lift_env_smoke.py

Exit 0 and prints obs keys → env path OK; segfault → driver/node, not training order.
"""
from __future__ import annotations

import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    hdf5 = pathlib.Path(
        os.environ.get(
            "ROBOMIMIC_LIFT_HDF5",
            str(_ROOT / "data/robomimic/datasets/lift/ph/image.hdf5"),
        )
    ).resolve()
    if not hdf5.is_file():
        print(f"missing {hdf5}", file=sys.stderr)
        return 1

    import numpy as np
    import robomimic.utils.file_utils as FileUtils

    from diffusion_policy.env_runner.robomimic_image_runner import (
        _apply_slurm_egl_workarounds,
        create_env,
    )

    shape_meta = {
        "obs": {
            "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
            "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
        }
    }
    env_meta = FileUtils.get_env_metadata_from_dataset(str(hdf5))
    env_meta["env_kwargs"]["use_object_obs"] = False
    _apply_slurm_egl_workarounds(env_meta)

    print("building env (robosuite.make)...", flush=True)
    env = create_env(env_meta, shape_meta, enable_render=True)
    print("reset...", flush=True)
    obs = env.reset()
    print("ok keys:", list(obs.keys())[:8], "...", flush=True)
    a = np.zeros(env.action_dimension, dtype=np.float32)
    env.step(a)
    print("step ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
