from setuptools import setup, Extension
from pathlib import Path

import numpy as np

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    pybind11_include = ""

opencv_include = ""
opencv_lib = ""
try:
    import cv2
    opencv_include = str(Path(cv2.__file__).parent / "opencv2")
except Exception:
    pass

cpp_dir = str(Path(__file__).parent / "src" / "wafer_srgan" / "cpp")

ext_modules = []
if pybind11_include:
    ext_modules.append(
        Extension(
            "wafer_srgan._edge_ops",
            sources=[
                str(Path(cpp_dir) / "edge_ops.cpp"),
                str(Path(cpp_dir) / "pybind_module.cpp"),
            ],
            include_dirs=[
                pybind11_include,
                cpp_dir,
                numpy.get_include(),
                opencv_include,
            ],
            language="c++",
            extra_compile_args=["/O2", "/std:c++17"] if __import__("sys").platform == "win32" else ["-O2", "-std=c++17"],
            libraries=["opencv_world4100"] if __import__("sys").platform == "win32" else ["opencv_core", "opencv_imgproc", "opencv_highgui"],
        )
    )

setup(ext_modules=ext_modules)
