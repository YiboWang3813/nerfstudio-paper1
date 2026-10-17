from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass(frozen=True)
class DatasetMeta:
    """
    描述一个具体数据集的固有物理属性 (Fact)。
    一旦数据生成，这些值就是宇宙真理，不可修改。
    """

    name: str

    # probe type 
    probe_type: str

    # image resolution
    H: int
    W: int

    # physical scale (mm)
    # for linear probe
    near_mm: float       
    far_mm: float        
    width_mm: float
    # for convex probe
    radius_mm: float
    fov_deg: float      

    # physical resolution (mm) 
    # lateral -> X, axial -> Y, elevational -> Z
    delta_lateral_mm: float 
    delta_axial_mm: float  
    delta_elevational_mm: float 

    # probe coordinate convention
    center_lateral: bool = True
    pose_init_type: str = "probe->world"
