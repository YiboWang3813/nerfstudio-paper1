from .synthetic_liver import SYNTHETIC_LIVER 
from .synthetic_liver_x2 import SYNTHETIC_LIVER_X2
from .spine_phantom import SPINE_PHANTOM
from .tus_rec_2024 import TUS_REC_2024
from .tdsc_abus_2023 import TDSC_ABUS_2023

DATASET_META_REGISTRY = {
    "synthetic_liver": SYNTHETIC_LIVER,
    "synthetic_liver_x2": SYNTHETIC_LIVER_X2,
    "spine_phantom": SPINE_PHANTOM,
    "tus_rec_2024": TUS_REC_2024,
    "tdsc_abus_2023": TDSC_ABUS_2023,
}
