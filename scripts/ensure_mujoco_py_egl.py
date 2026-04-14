#!/usr/bin/env python3
"""
Patch mujoco_py for HPC GPU nodes: use EGL instead of OSMesa (avoids GL/osmesa.h).

1) builder.py — GPU detection + driver lib dirs (see markers in file).
2) builder.py — patchelf paths: stock code uses LD_LIBRARY_PATH + '/libfoo.so', which breaks
   when LD_LIBRARY_PATH has multiple entries (after prepend in __init__.py); use absolute paths.
3) __init__.py — stock code overwrites LD_LIBRARY_PATH and drops /usr/lib64;
   prepend mujoco bin instead so EGL check passes.
4) builder.py — _ensure_set_env_var: stock code raises unless get_nvidia_lib_dir() is already in
   LD_LIBRARY_PATH; GPU nodes often have /usr/lib64 but need /usr/lib64/nvidia prepended — prepend instead of raise.

Does not import mujoco_py. Exit: 0 ok, 2 patched something (rebuild once), 1 error.
"""
from __future__ import annotations

import os
import pathlib
import sys

MARKER_SMI = "HPC_NVIDIA_SMI_CHECK_V1"
MARKER_LIB = "HPC_EGL_FALLBACK_V1"
MARKER_LD = "HPC_LD_PRESERVE_V1"
MARKER_PATCHELF = "HPC_PATCHELF_LIB_PATH_V1"
MARKER_PREPEND = "HPC_LD_PREPEND_V1"

OLD_ENSURE = """def _ensure_set_env_var(var_name, lib_path):
    paths = os.environ.get(var_name, "").split(":")
    paths = [os.path.abspath(path) for path in paths]
    if lib_path not in paths:
        raise Exception("\\nMissing path to your environment variable. \\n"
                        "Current values %s=%s\\n"
                        "Please add following line to .bashrc:\\n"
                        "export %s=$%s:%s" % (var_name, os.environ.get(var_name, ""),
                                              var_name, var_name, lib_path))"""

NEW_ENSURE = """def _ensure_set_env_var(var_name, lib_path):
    # """ + MARKER_PREPEND + """: prepend lib_path instead of raising (Slurm: /usr/lib64 present but not /usr/lib64/nvidia).
    lib_path = os.path.abspath(lib_path)
    raw = os.environ.get(var_name, "") or ""
    paths = [os.path.abspath(p) for p in raw.split(":") if p]
    if lib_path not in paths:
        os.environ[var_name] = lib_path + (":" + raw if raw else "")"""

OLD_SMI = """    exists_nvidia_smi = subprocess.call("type nvidia-smi", shell=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0
    if not exists_nvidia_smi:
        return None"""

NEW_SMI = """    # """ + MARKER_SMI + """
    exists_nvidia_smi = (
        subprocess.call("command -v nvidia-smi >/dev/null 2>&1", shell=True) == 0
        or exists("/usr/bin/nvidia-smi")
    )
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    slurm_gpus = os.environ.get("SLURM_STEP_GPUS", "").strip()
    if not exists_nvidia_smi and not cuda_visible and not slurm_gpus:
        return None"""

OLD_LIB = """    paths = glob.glob('/usr/lib/nvidia-[0-9][0-9][0-9]')
    paths = sorted(paths)
    if len(paths) == 0:
        return None"""

NEW_LIB = """    paths = list(glob.glob('/usr/lib/nvidia-[0-9][0-9][0-9]'))
    paths.extend(glob.glob('/usr/lib64/nvidia*'))
    paths = sorted(set(paths))
    if len(paths) == 0:
        # """ + MARKER_LIB + """: Slurm / EL8 nodes: use system lib64 when GPU job
        if exists('/usr/lib64'):
            return '/usr/lib64'
        return None"""

OLD_INIT_LINUX = """if platform == "linux" or platform == "linux2":
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = os.path.join(mujoco_py.__path__[0], "binaries", "linux", "mujoco210")
    os.environ["LD_LIBRARY_PATH"] = os.path.join(mujoco_py.__path__[0], "binaries", "linux", "mujoco210", "bin")"""

NEW_INIT_LINUX = """if platform == "linux" or platform == "linux2":
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = os.path.join(mujoco_py.__path__[0], "binaries", "linux", "mujoco210")
    # """ + MARKER_LD + """: keep prior LD_LIBRARY_PATH (e.g. /usr/lib64 for NVIDIA EGL)
    _mj_ld = os.path.join(mujoco_py.__path__[0], "binaries", "linux", "mujoco210", "bin")
    os.environ["LD_LIBRARY_PATH"] = _mj_ld + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")"""

# Stock mujoco_py uses f'{LD_LIBRARY_PATH}/libfoo.so' in patchelf; that breaks when LD_LIBRARY_PATH
# has multiple entries (after HPC_LD_PRESERVE_V1 prepend). Use absolute paths instead.
OLD_PATCHELF_GAP = """    subprocess.check_call(['patchelf', '--add-needed', library_path, so_file])


def manually_link_libraries(mujoco_path, raw_cext_dll_path):"""

NEW_PATCHELF_GAP = """    subprocess.check_call(['patchelf', '--add-needed', library_path, so_file])


def _mj_bin_lib_for_patchelf(mujoco_path, name):
    # """ + MARKER_PATCHELF + """: patchelf --add-needed must receive a single absolute path.
    return join(mujoco_path, 'bin', name)


def _system_lib_for_patchelf(so_name):
    for d in ('/usr/lib64', '/usr/lib'):
        p = join(d, so_name)
        if exists(p):
            return p
    raise RuntimeError(
        'mujoco_py: could not find %s under /usr/lib64 or /usr/lib (EGL/OpenGL)' % so_name
    )


def manually_link_libraries(mujoco_path, raw_cext_dll_path):"""

OLD_CPU_BUILD = """    def _build_impl(self):
        so_file_path = super()._build_impl()
        # Removes absolute paths to libraries. Allows for dynamic loading.
        fix_shared_library(so_file_path, 'libmujoco210.so', f'{os.environ["LD_LIBRARY_PATH"]}/libmujoco210.so')
        fix_shared_library(so_file_path, 'libglewosmesa.so', f'{os.environ["LD_LIBRARY_PATH"]}/libglewosmesa.so')
        return so_file_path"""

NEW_CPU_BUILD = """    def _build_impl(self):
        so_file_path = super()._build_impl()
        # Removes absolute paths to libraries. Allows for dynamic loading.
        mj = self.mujoco_path
        fix_shared_library(so_file_path, 'libmujoco210.so', _mj_bin_lib_for_patchelf(mj, 'libmujoco210.so'))
        fix_shared_library(so_file_path, 'libglewosmesa.so', _mj_bin_lib_for_patchelf(mj, 'libglewosmesa.so'))
        return so_file_path"""

OLD_GPU_BUILD = """    def _build_impl(self):
        so_file_path = super()._build_impl()
        fix_shared_library(so_file_path, 'libOpenGL.so', f'{os.environ["LD_LIBRARY_PATH"]}/libOpenGL.so.0')
        fix_shared_library(so_file_path, 'libEGL.so', f'{os.environ["LD_LIBRARY_PATH"]}/libEGL.so.1')
        fix_shared_library(so_file_path, 'libmujoco210.so', f'{os.environ["LD_LIBRARY_PATH"]}/libmujoco210.so')
        fix_shared_library(so_file_path, 'libglewegl.so', f'{os.environ["LD_LIBRARY_PATH"]}/libglewegl.so')
        return so_file_path"""

NEW_GPU_BUILD = """    def _build_impl(self):
        so_file_path = super()._build_impl()
        mj = self.mujoco_path
        fix_shared_library(so_file_path, 'libOpenGL.so', _system_lib_for_patchelf('libOpenGL.so.0'))
        fix_shared_library(so_file_path, 'libEGL.so', _system_lib_for_patchelf('libEGL.so.1'))
        fix_shared_library(so_file_path, 'libmujoco210.so', _mj_bin_lib_for_patchelf(mj, 'libmujoco210.so'))
        fix_shared_library(so_file_path, 'libglewegl.so', _mj_bin_lib_for_patchelf(mj, 'libglewegl.so'))
        return so_file_path"""


def _site_mujoco_py() -> pathlib.Path | None:
    if "CONDA_PREFIX" in os.environ:
        p = (
            pathlib.Path(os.environ["CONDA_PREFIX"])
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "mujoco_py"
        )
        if p.is_dir():
            return p
    import sysconfig

    p = pathlib.Path(sysconfig.get_path("purelib")) / "mujoco_py"
    if p.is_dir():
        return p
    return None


def main() -> int:
    base = _site_mujoco_py()
    if base is None:
        print("ensure_mujoco_py_egl: site-packages/mujoco_py not found", file=sys.stderr)
        return 1
    builder = base / "builder.py"
    init_py = base / "__init__.py"
    changed = False

    if not builder.is_file() or not init_py.is_file():
        print("ensure_mujoco_py_egl: missing builder.py or __init__.py", file=sys.stderr)
        return 1

    text = builder.read_text()
    if MARKER_SMI not in text:
        if OLD_SMI not in text:
            print("ensure_mujoco_py_egl: nvidia-smi block not found in builder.py", file=sys.stderr)
            return 1
        text = text.replace(OLD_SMI, NEW_SMI, 1)
        changed = True
    if MARKER_LIB not in text:
        if OLD_LIB not in text:
            print("ensure_mujoco_py_egl: lib path block not found in builder.py", file=sys.stderr)
            return 1
        text = text.replace(OLD_LIB, NEW_LIB, 1)
        changed = True
    if MARKER_PATCHELF not in text:
        if OLD_PATCHELF_GAP not in text:
            print("ensure_mujoco_py_egl: patchelf gap (after fix_shared_library) not found", file=sys.stderr)
            return 1
        text = text.replace(OLD_PATCHELF_GAP, NEW_PATCHELF_GAP, 1)
        changed = True
        if OLD_CPU_BUILD not in text:
            print("ensure_mujoco_py_egl: LinuxCPU _build_impl block not found", file=sys.stderr)
            return 1
        text = text.replace(OLD_CPU_BUILD, NEW_CPU_BUILD, 1)
        if OLD_GPU_BUILD not in text:
            print("ensure_mujoco_py_egl: LinuxGPU _build_impl block not found", file=sys.stderr)
            return 1
        text = text.replace(OLD_GPU_BUILD, NEW_GPU_BUILD, 1)
    if MARKER_PREPEND not in text:
        if OLD_ENSURE not in text:
            print("ensure_mujoco_py_egl: _ensure_set_env_var block not found in builder.py", file=sys.stderr)
            return 1
        text = text.replace(OLD_ENSURE, NEW_ENSURE, 1)
        changed = True
    if changed:
        builder.write_text(text)

    itext = init_py.read_text()
    init_changed = False
    if MARKER_LD not in itext:
        if OLD_INIT_LINUX not in itext:
            print("ensure_mujoco_py_egl: linux LD block not found in __init__.py", file=sys.stderr)
            return 1
        init_py.write_text(itext.replace(OLD_INIT_LINUX, NEW_INIT_LINUX, 1))
        init_changed = True

    if changed or init_changed:
        print("ensure_mujoco_py_egl: patched", builder, "and/or", init_py)
        return 2
    print("ensure_mujoco_py_egl: already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
