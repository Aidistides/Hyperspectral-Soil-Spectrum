# Or TIFF
import rasterio
with rasterio.open('data/.../patch_xxx.tif') as src:
    cube = src.read()  # (150, h, w)

from soiloptix_hsi.pipeline import run_soiloptix_hsi_pipeline

run_soil_hsi_pipeline(
    hsi_path="my_drone_scan.npy",      # or .tif / .hdr
    lab_data_csv="lab_results.csv",
    field_acres=80.0,
    crs="EPSG:32615"
)
