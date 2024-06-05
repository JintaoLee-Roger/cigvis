# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

import sys
import os
from pathlib import Path

from setuptools import setup, find_packages

if sys.version_info[0] < 3:
    raise RuntimeError("CIGVis only supports Python 3")

# package version
version_path = Path(__file__).parent / "VERSION.txt"
if not version_path.exists():
    raise FileNotFoundError("VERSION.txt file not found")
version = version_path.read_text().strip()

if not version:
    raise RuntimeError("Failed to parse version from VERSION")

# requirements
requirements_path = Path(__file__).parent / 'requirements.txt'
if not requirements_path.exists():
    raise FileNotFoundError("requirements.txt file not found")
requirements = requirements_path.read_text().splitlines()

package_name = "cigvis"

description = "CIGVis is a tool for geophysical data visualization, " + \
      "which developed by Computational Interpretation Group (CIG)"

setup(
    name=package_name,
    version=version,
    author="Jintao Li, and others",
    author_email="lijintaobt@gmail.com",
    url="https://github.com/JintaoLee-Roger/cigvis",
    license='MIT',
    description=description,
    long_description=(Path(__file__).parent /
                      "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
)
