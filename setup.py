from __future__ import annotations

import os
import platform

import numpy as np
from setuptools import Extension, setup


def openmp_flags() -> tuple[list[str], list[str]]:
    if os.environ.get("LWW_TRANSPORT_OPENMP", "").lower() not in {"1", "true", "yes", "on"}:
        return [], []

    if platform.system() == "Darwin":
        compile_args = ["-Xpreprocessor", "-fopenmp"]
        link_args = ["-lomp"]
    else:
        compile_args = ["-fopenmp"]
        link_args = ["-fopenmp"]

    if include_dir := os.environ.get("LWW_TRANSPORT_OPENMP_INCLUDE"):
        compile_args.append(f"-I{include_dir}")
    if lib_dir := os.environ.get("LWW_TRANSPORT_OPENMP_LIB"):
        link_args.append(f"-L{lib_dir}")
    return compile_args, link_args


openmp_compile_args, openmp_link_args = openmp_flags()

setup(
    ext_modules=[
        Extension(
            "lww_transport._cpp_kernels",
            sources=["src/lww_transport/_cpp_kernels.cpp"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17", "-O3", *openmp_compile_args],
            extra_link_args=openmp_link_args,
        )
    ]
)
