import numpy as np

CFG = {
    "num_bands": 200,                    # matches your repo
    "target_size": (64, 64),             # from dataset.py
    "soil_properties": ["pH", "OM_pct", "N_ppm", "P_ppm", "K_ppm", "CEC", "Clay_pct", "Sand_pct"],
    "sampling_ratio": 1 / 8,             # 1 sample per 8 acres (SoilOptix rule)
    "min_samples": 3,
    "output_dir": "soiloptix_outputs",
    "random_seed": 42,
}
