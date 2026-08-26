import os
from pathlib import Path
try:
    import torch
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "The Opus shim requires a CUDA-enabled PyTorch installation in the active Python environment."
    ) from exc
from setuptools import setup
from torch.utils.cpp_extension import (
    CUDAExtension,
    CppExtension,
    BuildExtension,
    CUDA_HOME,
)

sources = ["src/opus.cpp"]

shim_root = Path(__file__).resolve().parent
repo_root = shim_root.parents[1]
local_include = str(shim_root / "include")
torch_includes = torch.utils.cpp_extension.include_paths()

if CUDA_HOME is None:
    raise RuntimeError("CUDA_HOME is not set; load a CUDA toolkit before building the Opus shim")
cuda_include = os.path.join(CUDA_HOME, "include")
cuda_lib = os.path.join(CUDA_HOME, "lib64")
NCCL_HOME = os.environ.get("NCCL_HOME", str(repo_root / "nccl" / "build"))
toml_include = os.environ.get(
    "OPUS_TOML_INCLUDE",
    str(repo_root / "third_party" / "tomlplusplus" / "include"),
)

include_dirs = [
    local_include,
    *torch_includes,
    cuda_include,
    os.path.join(NCCL_HOME, "include"),
    # "/opt/conda/lib/python3.11/site-packages/nvidia/nccl/include",
    # "/opt/udiImage/modules/nccl-2.18/include",
]
if not os.path.isdir(toml_include):
    raise RuntimeError(
        f"toml++ headers not found at {toml_include}. "
        "Restore third_party/tomlplusplus/include or set OPUS_TOML_INCLUDE."
    )
include_dirs.append(toml_include)

library_dirs = [
    cuda_lib,
    os.path.join(NCCL_HOME, "lib"),
    # "/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib",
    # "/opt/udiImage/modules/nccl-2.18/lib"
]

ext_modules = [
    CUDAExtension(
        name="opus",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["nccl"],
        extra_compile_args={
            "cxx": ["-std=c++17", "-DUSE_C10D_NCCL"]
        },
    )
]

setup(
    name="opus",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
