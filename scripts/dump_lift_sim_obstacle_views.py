#!/usr/bin/env python3
"""
Render robosuite Lift camera views (agentview + eye-in-hand) like the training setup,
optionally with a fixed tabletop obstacle box to check goal occlusion.

HDF5 dump (dump_robomimic_lift_views.py) shows dataset pixels only; this script needs a
working mujoco_py + GPU/EGL setup (same as robomimic rollouts).

Examples:
  cd /path/to/diffusion_policy
  # On clusters: use a GPU allocation (mujoco_py uses OSMesa on "CPU" path → needs GL/osmesa.h).
  srun --gres=gpu:1 --pty bash
  export MUJOCO_GL=egl
  python scripts/dump_lift_sim_obstacle_views.py --both --seed 0

  # If you have /dev/nvidia0 but no Slurm, the script sets CUDA_VISIBLE_DEVICES=0 unless
  # you pass --no-auto-cuda-visible.

  # Tune obstacle position (table frame, same as robosuite placement around table_offset):
  python scripts/dump_lift_sim_obstacle_views.py --variant obstacle --obs-xy 0.09 0.0 --seed 0

If import fails on osmesashim.c / GL/osmesa.h: you are on mujoco_py's CPU renderer path.
Either run on a GPU node, or export CUDA_VISIBLE_DEVICES=0 when a GPU exists, or
``python scripts/ensure_mujoco_py_egl.py`` (once) plus MUJOCO_GL=egl.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _prepare_mujoco_headless_gl(*, auto_cuda_visible: bool) -> None:
    """
    Before importing robosuite/mujoco_py: prefer EGL. Without a GPU signal, mujoco_py
    builds the OSMesa shim (needs system OSMesa headers) — usually missing on HPC.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    if not auto_cuda_visible or not sys.platform.startswith("linux"):
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return
    if os.environ.get("SLURM_STEP_GPUS", "").strip():
        return
    if os.path.exists("/dev/nvidia0"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _to_hwc_u8(arr) -> "object":
    import numpy as np

    x = np.asarray(arr)
    if x.ndim != 3:
        raise ValueError(f"expected 3D image array, got shape {x.shape}")
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    if x.dtype != np.uint8:
        xf = x.astype(np.float32)
        if xf.max() <= 1.0 + 1e-6:
            xf = xf * 255.0
        x = np.clip(xf, 0, 255).astype(np.uint8)
    return x


def _load_env_kwargs_from_hdf5(hdf5_path: pathlib.Path) -> dict:
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        raw = f["data"].attrs["env_args"]
        if isinstance(raw, bytes):
            raw = raw.decode()
        meta = json.loads(raw)
    return dict(meta["env_kwargs"])


def _normalize_robots_kw(env_kwargs: dict) -> None:
    r = env_kwargs.get("robots")
    if isinstance(r, list) and len(r) == 1:
        env_kwargs["robots"] = r[0]


_LIFT_WITH_OBSTACLE_CLS = None


def _define_lift_with_obstacle():
    from robosuite.environments.manipulation.lift import Lift
    from robosuite.models.arenas import TableArena
    from robosuite.models.objects import BoxObject
    from robosuite.models.tasks import ManipulationTask
    from robosuite.utils.mjcf_utils import CustomMaterial
    from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

    class LiftWithObstacleImpl(Lift):
        """
        Standard Lift plus one extra box (fixed xy on the table) for occlusion checks.
        Cube placement matches vanilla Lift (uniform on the table patch).
        """

        def __init__(
            self,
            obstacle_half_sizes=(0.015, 0.12, 0.06),
            obstacle_xy=(0.07, 0.0),
            **kwargs,
        ):
            self._obs_half = tuple(float(x) for x in obstacle_half_sizes)
            self._obs_xy = tuple(float(x) for x in obstacle_xy)
            super().__init__(**kwargs)

        def _load_model(self):
            from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

            SingleArmEnv._load_model(self)

            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)

            mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_friction=self.table_friction,
                table_offset=self.table_offset,
            )
            mujoco_arena.set_origin([0, 0, 0])

            tex_attrib = {"type": "cube"}
            mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
            redwood = CustomMaterial(
                texture="WoodRed",
                tex_name="redwood",
                mat_name="redwood_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            self.cube = BoxObject(
                name="cube",
                size_min=[0.020, 0.020, 0.020],
                size_max=[0.022, 0.022, 0.022],
                rgba=[1, 0, 0, 1],
                material=redwood,
            )

            hx, hy, hz = self._obs_half
            self.obstacle = BoxObject(
                name="obstacle",
                size_min=[hx, hy, hz],
                size_max=[hx, hy, hz],
                rgba=[0.25, 0.25, 0.28, 1],
            )

            cube_sampler = UniformRandomSampler(
                name="CubeSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            ox, oy = self._obs_xy
            obs_sampler = UniformRandomSampler(
                name="ObstacleSampler",
                mujoco_objects=self.obstacle,
                x_range=[ox, ox],
                y_range=[oy, oy],
                rotation=0.0,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            compound = SequentialCompositeSampler(name="LiftObstacleCompound")
            compound.append_sampler(cube_sampler)
            compound.append_sampler(obs_sampler)
            self.placement_initializer = compound

            self.model = ManipulationTask(
                mujoco_arena=mujoco_arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=[self.cube, self.obstacle],
            )

    return LiftWithObstacleImpl


def _make_env(variant: str, env_kwargs: dict, obstacle_half, obstacle_xy):
    global _LIFT_WITH_OBSTACLE_CLS
    _normalize_robots_kw(env_kwargs)
    if variant == "baseline":
        import robosuite as suite

        return suite.make("Lift", **env_kwargs)
    if _LIFT_WITH_OBSTACLE_CLS is None:
        _LIFT_WITH_OBSTACLE_CLS = _define_lift_with_obstacle()
    cls = _LIFT_WITH_OBSTACLE_CLS
    return cls(
        obstacle_half_sizes=obstacle_half,
        obstacle_xy=obstacle_xy,
        **env_kwargs,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dump Lift sim RGB (agentview + eye_in_hand), with optional tabletop obstacle",
    )
    p.add_argument(
        "--hdf5",
        type=pathlib.Path,
        default=pathlib.Path("data/robomimic/datasets/lift/ph/image.hdf5"),
        help="Read env_args from this robomimic HDF5 (camera size, controller, etc.)",
    )
    p.add_argument(
        "--variant",
        choices=("baseline", "obstacle"),
        default="baseline",
        help="baseline: vanilla Lift; obstacle: extra box on the table",
    )
    p.add_argument(
        "--both",
        action="store_true",
        help="Write baseline and obstacle with the same RNG seed (fair comparison)",
    )
    p.add_argument("--seed", type=int, default=0, help="numpy seed before each env.reset()")
    p.add_argument(
        "-o",
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/dump_lift_views"),
        help="Output directory (same tree style as dump_robomimic_lift_views)",
    )
    p.add_argument(
        "--timestep-tag",
        type=int,
        default=0,
        help="Index used in filenames (sim has no demo index; default 0)",
    )
    p.add_argument(
        "--obs-half",
        type=float,
        nargs=3,
        default=(0.015, 0.12, 0.06),
        metavar=("HX", "HY", "HZ"),
        help="Obstacle box half-extents (meters), robosuite BoxObject convention",
    )
    p.add_argument(
        "--obs-xy",
        type=float,
        nargs=2,
        default=(0.07, 0.0),
        metavar=("X", "Y"),
        help="Obstacle center (x,y) in table placement frame (see robosuite UniformRandomSampler)",
    )
    p.add_argument(
        "--no-auto-cuda-visible",
        action="store_true",
        help="Do not set CUDA_VISIBLE_DEVICES=0 when /dev/nvidia0 exists (default: auto-set)",
    )
    p.add_argument(
        "--render-gpu-device-id",
        type=int,
        default=None,
        metavar="ID",
        help="Override env_kwargs render_gpu_device_id (HDF5 default is often 0; try -1 on Slurm EGL segfault)",
    )
    args = p.parse_args()

    _prepare_mujoco_headless_gl(auto_cuda_visible=not args.no_auto_cuda_visible)

    hdf5_path = args.hdf5.expanduser().resolve()
    if not hdf5_path.is_file():
        print(f"dump_lift_sim_obstacle_views: HDF5 not found: {hdf5_path}", file=sys.stderr)
        return 1

    try:
        from PIL import Image
    except ImportError:
        print("dump_lift_sim_obstacle_views: need Pillow", file=sys.stderr)
        return 1

    env_kwargs = _load_env_kwargs_from_hdf5(hdf5_path)
    # Match robomimic runner: no object-state tensors in obs (images only)
    env_kwargs["use_object_obs"] = False
    if args.render_gpu_device_id is not None:
        env_kwargs["render_gpu_device_id"] = args.render_gpu_device_id

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_keys = ("agentview_image", "robot0_eye_in_hand_image")

    variants = ("baseline", "obstacle") if args.both else (args.variant,)

    for var in variants:
        import numpy as np

        np.random.seed(args.seed)
        env = _make_env(var, dict(env_kwargs), args.obs_half, tuple(args.obs_xy))
        obs = env.reset()
        for key in rgb_keys:
            if key not in obs:
                print(
                    f"dump_lift_sim_obstacle_views: missing obs key {key!r}; have {list(obs.keys())}",
                    file=sys.stderr,
                )
                return 1
            hwc = _to_hwc_u8(obs[key])
            fname = f"sim_{var}_t{args.timestep_tag:04d}_{key}.png"
            path = out_dir / fname
            Image.fromarray(hwc).save(path)
            print(path)
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
