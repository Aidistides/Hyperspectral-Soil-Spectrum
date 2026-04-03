import numpy as np
import torch
import pandas as pd
import geopandas as gpd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from pathlib import Path
from rasterio.transform import from_origin
import rasterio
from .config import CFG
from .preprocess_and_map import hyperspectral_normalize, create_variability_map, save_variability_tiff
from .sampling_strategy import suggest_sampling_locations

# Optional: import your repo's transform if you want full 3D-CNN augmentation
# from dataset import HyperspectralTransform   # uncomment after you copy dataset.py

def run_soiloptix_hsi_pipeline(
    hsi_path: str,                     # e.g. "field_01.hdr" or numpy .npy
    lab_data_csv: str,                 # columns: sample_id, pH, OM_pct, ...
    field_acres: float,
    crs: str = "EPSG:32615",           # example UTM
    pixel_size_m: float = 1.0          # drone resolution
):
    np.random.seed(CFG["random_seed"])
    Path(CFG["output_dir"]).mkdir(exist_ok=True)
    
    # 1. Load HSI cube (Bands, H, W) – supports .npy or rasterio
    if hsi_path.endswith('.npy'):
        cube = np.load(hsi_path)
    else:
        with rasterio.open(hsi_path) as src:
            cube = src.read().astype(np.float32)  # (bands, h, w)
            transform = src.transform
            crs = src.crs
    cube = hyperspectral_normalize(cube)   # exactly as in your repo
    
    # 2. Variability map (SoilOptix "countrate")
    var_map = create_variability_map(cube)
    save_variability_tiff(var_map, transform, crs, f"{CFG['output_dir']}/variability_map.tif")
    
    # 3. Strategic sampling (SODL suggestion tool)
    num_samples = max(CFG["min_samples"], int(field_acres * CFG["sampling_ratio"]))
    sample_idx = suggest_sampling_locations(var_map, num_samples, field_acres)
    print(f"✅ Suggested {len(sample_idx)} sampling locations (high/low variability zones)")
    
    # 4. Extract spectra at sample points + load lab ground-truth
    lab_df = pd.read_csv(lab_data_csv)
    X_samples = []
    y_samples = []
    for r, c in sample_idx:
        spectrum = cube[:, r, c]
        X_samples.append(spectrum)
    X_samples = np.stack(X_samples)
    
    # Assume lab_df rows match the order of suggested samples (or merge by ID)
    y_samples = lab_df[CFG["soil_properties"]].values[:len(X_samples)]
    
    # 5. Calibrate model – PLSR (industry standard for soil spectroscopy)
    # You can replace this with your SoilHSI3DCNN adapted for regression
    pls = PLSRegression(n_components=15)
    pls.fit(X_samples, y_samples)
    
    # Quick validation (R² on calibration set – in production use CV)
    y_pred_cal = pls.predict(X_samples)
    r2 = r2_score(y_samples, y_pred_cal, multioutput='uniform_average')
    print(f"Calibration R² = {r2:.3f}")
    
    # 6. Predict full-field maps
    H, W = cube.shape[1], cube.shape[2]
    flat_cube = cube.reshape(CFG["num_bands"], -1).T          # (pixels, bands)
    pred_flat = pls.predict(flat_cube)                        # (pixels, n_properties)
    pred_maps = pred_flat.T.reshape(len(CFG["soil_properties"]), H, W)
    
    # 7. Save as GeoTIFF + shapefile/CSV (exact SoilOptix output format)
    for i, prop in enumerate(CFG["soil_properties"]):
        out_tiff = f"{CFG['output_dir']}/{prop}_map.tif"
        with rasterio.open(out_tiff, 'w', driver='GTiff',
                           height=H, width=W, count=1,
                           dtype=pred_maps[i].dtype,
                           crs=crs, transform=transform) as dst:
            dst.write(pred_maps[i], 1)
        
        # Also export point shapefile for variable-rate prescriptions
        gdf = gpd.GeoDataFrame(
            lab_df,
            geometry=gpd.points_from_xy(lab_df['lon'], lab_df['lat']),  # add your coords
            crs=crs
        )
        gdf.to_file(f"{CFG['output_dir']}/prescription_{prop}.shp")
    
    print(f"✅ All SoilOptix-style outputs saved to ./{CFG['output_dir']}/")
    return pred_maps
