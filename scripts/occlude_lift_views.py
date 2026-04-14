#!/usr/bin/env python3
"""
Offline-only: draw a solid (or semi-transparent) rectangle on Lift camera images.
No MuJoCo / robosuite — use for BC when you only need "something blocks the goal in pixel space".

Reads either robomimic image.hdf5 (same layout as dump_robomimic_lift_views.py) or existing PNG paths.

Examples:
  cd /path/to/diffusion_policy
  # HDF5 → PNG (default: agentview only; wrist camera unchanged / not written here)
  python scripts/occlude_lift_views.py \\
    data/robomimic/datasets/lift/ph/image.hdf5 \\
    --rect-norm 0.38 0.38 0.62 0.62

  # Mask wrist too (optional)
  python scripts/occlude_lift_views.py data/.../image.hdf5 \\
    --keys robot0_eye_in_hand_image --rect-norm ...

  # Already dumped PNGs
  python scripts/occlude_lift_views.py --png \\
    results/dump_lift_views/demo_0_t0000_agentview_image.png \\
    --rect-norm 0.35 0.35 0.65 0.65
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np

from diffusion_policy.common.image_occlusion import apply_rectangle_occlusion_hwc_uint8

def _to_hwc_u8(arr: np.ndarray) -> np.ndarray:
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


def main() -> int:
    p = argparse.ArgumentParser(description="Occlude Lift images with a rectangle (offline, no sim)")
    p.add_argument(
        "hdf5",
        type=pathlib.Path,
        nargs="?",
        default=None,
        help="Path to image.hdf5 (omit if using --png only)",
    )
    p.add_argument(
        "--png",
        type=pathlib.Path,
        nargs="*",
        default=[],
        help="PNG files to process (HWC RGB); writes <stem>_blocked.png next to default out-dir rules",
    )
    p.add_argument("--demo", type=int, default=0, help="demo index for HDF5 mode")
    p.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[0, 50, 100],
        help="Timestep indices for HDF5 mode",
    )
    p.add_argument(
        "--rect-norm",
        type=float,
        nargs=4,
        required=True,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Rectangle in normalized image coords [0,1] (x0,y0) top-left to (x1,y1) bottom-right",
    )
    p.add_argument(
        "--color",
        type=int,
        nargs=3,
        default=(55, 55, 60),
        metavar=("R", "G", "B"),
        help="Fill RGB 0-255",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="1.0 = opaque block; <1 blends with original",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/dump_lift_views"),
        help="Output directory for HDF5 mode; for --png, default is PNG's directory",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="_blocked",
        help="Inserted before .png (default _blocked)",
    )
    p.add_argument(
        "--keys",
        type=str,
        nargs="+",
        default=["agentview_image"],
        metavar="OBS_KEY",
        help="HDF5 mode: which obs keys to occlude (default: agentview_image only; wrist stays clear)",
    )
    args = p.parse_args()

    try:
        from PIL import Image
    except ImportError:
        print("occlude_lift_views: need Pillow", file=sys.stderr)
        return 1

    x0, y0, x1, y1 = args.rect_norm
    color = tuple(int(c) for c in args.color)

    written: list[pathlib.Path] = []

    for png_path in args.png:
        png_path = png_path.expanduser().resolve()
        if not png_path.is_file():
            print(f"occlude_lift_views: missing {png_path}", file=sys.stderr)
            return 1
        hwc = np.array(Image.open(png_path).convert("RGB"))
        out = apply_rectangle_occlusion_hwc_uint8(
            hwc, (x0, y0, x1, y1), color=color, alpha=args.alpha
        )
        stem = png_path.stem
        if stem.endswith(args.suffix):
            stem = stem[: -len(args.suffix)]
        out_path = png_path.parent / f"{stem}{args.suffix}.png"
        Image.fromarray(out).save(out_path)
        written.append(out_path)

    if args.hdf5 is not None:
        hdf5_path = args.hdf5.expanduser().resolve()
        if not hdf5_path.is_file():
            print(f"occlude_lift_views: HDF5 not found: {hdf5_path}", file=sys.stderr)
            return 1
        out_dir = args.out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf5_path, "r") as f:
            demo_name = f"demo_{args.demo}"
            if demo_name not in f["data"]:
                print(f"occlude_lift_views: no {demo_name}", file=sys.stderr)
                return 1
            demo = f["data"][demo_name]
            obs = demo["obs"]
            T = demo["actions"].shape[0]
            for key in args.keys:
                if key not in obs:
                    print(
                        f"occlude_lift_views: obs/{key} missing in {demo_name}",
                        file=sys.stderr,
                    )
                    return 1
            for t in args.timesteps:
                tt = min(max(t, 0), T - 1)
                for key in args.keys:
                    hwc = _to_hwc_u8(obs[key][tt])
                    out = apply_rectangle_occlusion_hwc_uint8(
                        hwc, (x0, y0, x1, y1), color=color, alpha=args.alpha
                    )
                    fname = f"{demo_name}_t{tt:04d}_{key}{args.suffix}.png"
                    path = out_dir / fname
                    Image.fromarray(out).save(path)
                    written.append(path)

    if args.hdf5 is None and not args.png:
        print("occlude_lift_views: provide image.hdf5 path and/or --png file(s)", file=sys.stderr)
        return 1

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
