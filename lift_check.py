import sys
import types
from pathlib import Path
import numpy as np
import imageio.v2 as iio
import mujoco_py  # noqa: F401

# ---- numba fallback (cluster workaround) ----
numba_stub = types.ModuleType("numba")

def _jit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    def deco(fn):
        return fn
    return deco

numba_stub.jit = _jit
numba_stub.njit = _jit
sys.modules["numba"] = numba_stub
# ---------------------------------------------

import robosuite as suite


def get_cube_pos(sim):
    """
    Robustly find Lift cube body position from mujoco model/data.
    """
    candidates = [
        "cube_main", "cube", "cubeA", "object", "obj", "milk", "can"
    ]
    for name in candidates:
        try:
            bid = sim.model.body_name2id(name)
            return sim.data.body_xpos[bid].copy(), name
        except Exception:
            pass

    # fallback: search body names containing cube/object
    for i in range(sim.model.nbody):
        name = sim.model.body_id2name(i)
        if name is None:
            continue
        low = name.lower()
        if ("cube" in low) or ("object" in low) or ("obj" in low):
            return sim.data.body_xpos[i].copy(), name

    raise RuntimeError("Could not find cube/object body in model.")


def world_to_pixel(sim, cam_name, point_world, img_h, img_w):
    """
    Project world point to image pixel for mujoco camera.
    """
    cam_id = sim.model.camera_name2id(cam_name)
    cam_pos = sim.data.cam_xpos[cam_id].copy()
    cam_rot = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()  # camera->world rotation

    # world -> camera
    rel = point_world - cam_pos
    rel_cam = cam_rot.T @ rel  # camera coordinates

    # MuJoCo camera looks along -Z
    x_c, y_c, z_c = rel_cam[0], rel_cam[1], rel_cam[2]
    z = -z_c
    if z <= 1e-6:
        return None  # behind camera or too close

    fovy_deg = sim.model.cam_fovy[cam_id]
    f = 0.5 * img_h / np.tan(np.deg2rad(fovy_deg) / 2.0)

    u = (img_w / 2.0) + f * (x_c / z)
    v = (img_h / 2.0) - f * (y_c / z)
    return np.array([u, v], dtype=np.float32)


def draw_opaque_cylinder(img, center_uv, radius_px=24, height_px=90):
    """
    Draw a simple opaque cylinder (body + top/bottom ellipse) on RGB uint8 image.
    """
    out = img.copy()
    h, w, _ = out.shape
    cx, cy = int(round(center_uv[0])), int(round(center_uv[1]))

    yy, xx = np.ogrid[:h, :w]

    # cylinder body
    x0 = max(0, cx - radius_px)
    x1 = min(w - 1, cx + radius_px)
    y0 = max(0, cy - height_px // 2)
    y1 = min(h - 1, cy + height_px // 2)
    body = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)

    # caps
    cap_ry = max(4, radius_px // 3)
    top = (((xx - cx) / max(1, radius_px)) ** 2 + ((yy - y0) / max(1, cap_ry)) ** 2) <= 1.0
    bot = (((xx - cx) / max(1, radius_px)) ** 2 + ((yy - y1) / max(1, cap_ry)) ** 2) <= 1.0

    # opaque dark can-like color
    out[body] = np.array([28, 28, 32], dtype=np.uint8)
    out[top] = np.array([58, 58, 64], dtype=np.uint8)
    out[bot] = np.array([18, 18, 22], dtype=np.uint8)

    # highlight stripe
    sx0 = max(0, cx - radius_px // 3)
    sx1 = min(w - 1, sx0 + 2)
    stripe = (xx >= sx0) & (xx <= sx1) & (yy >= y0) & (yy <= y1)
    out[stripe] = np.array([145, 145, 150], dtype=np.uint8)

    return out


def main():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
    )

    obs = env.reset()
    sim = env.sim

    out_dir = Path("log")
    out_dir.mkdir(parents=True, exist_ok=True)

    # images (flipud for offscreen output)
    agent = np.flipud(obs["agentview_image"])
    hand = np.flipud(obs["robot0_eye_in_hand_image"])
    h, w, _ = agent.shape

    cam_id = sim.model.camera_name2id("agentview")
    cam_pos = sim.data.cam_xpos[cam_id].copy()
    cube_pos, cube_name = get_cube_pos(sim)

    # put cylinder between camera and cube in world
    # alpha near 0 => near camera, near 1 => near object
    alpha = 0.65
    cyl_world = (1.0 - alpha) * cam_pos + alpha * cube_pos

    # project to image
    uv_cube = world_to_pixel(sim, "agentview", cube_pos, h, w)
    uv_cyl = world_to_pixel(sim, "agentview", cyl_world, h, w)

    if uv_cube is None or uv_cyl is None:
        raise RuntimeError("Projection failed: cube or cylinder behind camera.")

    # radius/height from image-space distance to ensure blocking
    dist = np.linalg.norm(uv_cube - uv_cyl)
    radius_px = int(max(18, min(42, 0.35 * dist + 18)))
    height_px = int(radius_px * 2.8)

    agent_occ = draw_opaque_cylinder(agent, uv_cyl, radius_px=radius_px, height_px=height_px)

    # if still not covering cube center, expand once
    u_obj, v_obj = int(round(uv_cube[0])), int(round(uv_cube[1]))
    if 0 <= v_obj < h and 0 <= u_obj < w:
        if np.all(agent_occ[v_obj, u_obj] == agent[v_obj, u_obj]):
            agent_occ = draw_opaque_cylinder(
                agent, uv_cyl, radius_px=radius_px + 10, height_px=height_px + 24
            )

    iio.imwrite(out_dir / "lift_agentview_image.png", agent)
    iio.imwrite(out_dir / "lift_agentview_occluded.png", agent_occ)
    iio.imwrite(out_dir / "lift_robot0_eye_in_hand_image.png", hand)

    print("saved", out_dir / "lift_agentview_image.png")
    print("saved", out_dir / "lift_agentview_occluded.png")
    print("saved", out_dir / "lift_robot0_eye_in_hand_image.png")

    print("agentview cam_pos(world):", cam_pos)
    print("cube body:", cube_name, "cube_pos(world):", cube_pos)
    print("cyl_pos(world):", cyl_world)
    print("uv_cube:", uv_cube, "uv_cyl:", uv_cyl, "radius_px:", radius_px, "height_px:", height_px)

    env.close()
    print("done")


if __name__ == "__main__":
    main()