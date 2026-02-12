import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    CUDAExtension,
    CppExtension,
    BuildExtension,
    CUDA_HOME,
)

sources = ["src/opus.cpp"]

local_include = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")
torch_includes = torch.utils.cpp_extension.include_paths()

cuda_include = os.path.join(CUDA_HOME, "include")
cuda_lib = os.path.join(CUDA_HOME, "lib64")
NCCL_HOME = "/Opus/nccl/build"

include_dirs = [
    local_include,
    *torch_includes,
    cuda_include,
    os.path.join(NCCL_HOME, "include"),
    # "/opt/conda/lib/python3.11/site-packages/nvidia/nccl/include",
    # "/opt/udiImage/modules/nccl-2.18/include",
    os.path.expanduser("~/Opus/third_party/tomlplusplus/include"),
]

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
