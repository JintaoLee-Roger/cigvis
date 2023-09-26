# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

import sys
import os
import re
from pathlib import Path

from setuptools import setup, find_packages

if sys.version_info[0] < 3:
    raise Exception("CIGVis only supports Python 3")

version = ""
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path + "/VERSION.txt", "r") as file:
    for line in file:
        version = line.strip()
        break

if not version:
    print("Fatal error: Failed to parse version from VERSION")
    exit(1)

package_name = "cigvis"

description = "CIGVis is a tool for geophysical data visualization, " + \
      "which developed by Computational Interpretation Group (CIG)"

setup(name=package_name,
      version=version,
      author="Jintao Li, and others",
      url="https://github.com/JintaoLee-Roger/cigvis",
      license='MIT',
      description=description,
      long_description=Path("README.md").read_text(encoding="utf-8"),
      long_description_content_type="text/markdown",
      python_requires=">=3.7",
      packages=find_packages())
