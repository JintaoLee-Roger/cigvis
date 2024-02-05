# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
geophysics data io, 

Including:
    - well logs (.las format)
    - vds file (need install openvds)
    - fault skin
    
"""

from .las import load_las
from .vds import VDSReader, create_vds_from_array
from .fault_skin import load_skins, load_one_skin
from . import horiz