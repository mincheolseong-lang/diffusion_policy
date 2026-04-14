#!/usr/bin/env python3
"""Minimal mujoco_py osmesa offscreen render test on HPC GPU node."""
import os, sys, faulthandler
faulthandler.enable()
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["MUJOCO_PY_FORCE_CPU"] = "1"   # force CPU extension on GPU nodes (bypasses NVIDIA detection)
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"

print(f"MUJOCO_GL={os.environ['MUJOCO_GL']}", flush=True)
print(f"MUJOCO_PY_FORCE_CPU={os.environ['MUJOCO_PY_FORCE_CPU']}", flush=True)

import mujoco_py
import mujoco_py.cymj as cymj_mod
print(f"mujoco_py: {mujoco_py.__version__}", flush=True)
print(f"cymj: {cymj_mod.__file__}", flush=True)
if "gpu" in str(cymj_mod.__file__).lower():
    print("WARNING: still using GPU extension!", flush=True)
elif "cpu" in str(cymj_mod.__file__).lower():
    print("Good: CPU (osmesa) extension loaded.", flush=True)

xml = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam" pos="0 -1 1" xyaxes="1 0 0 0 1 1"/>
    <geom type="sphere" size="0.1"/>
  </worldbody>
</mujoco>
"""
model = mujoco_py.load_model_from_xml(xml)
sim = mujoco_py.MjSim(model)
sim.forward()
print("Sim OK", flush=True)

from mujoco_py import MjRenderContextOffscreen
print("Creating MjRenderContextOffscreen ...", flush=True)
ctx = MjRenderContextOffscreen(sim, device_id=0)
print("MjRenderContextOffscreen OK!", flush=True)

sim.step()
rgb, depth = ctx.read_pixels(64, 64, depth=True)
print(f"Render OK: rgb shape={rgb.shape}", flush=True)
print("ALL TESTS PASSED - OSMesa rollout should work!", flush=True)
