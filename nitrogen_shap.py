# nitrogen_shap.py
"""
Compute SHAP values for Nitrogen output in 3D CNN.
Generates exact band importance list (prioritizes SWIR).
"""

import torch
import shap
import numpy as np
import argparse
from model import Hyperspectral3DCNN  # your existing model
from dataset import HyperspectralDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .pth model")
    parser.add_argument("--data", required=True, help="Path to test HSI cube (.npy)")
    args = parser.parse_args()

    # Load model + data (reuses your pipeline)
    model = Hyperspectral3DCNN(num_classes=1)  # nitrogen regression
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    X_test = np.load(args.data)  # assume (1, bands, H, W) or adjust
    X_test = torch.from_numpy(X_test).unsqueeze(0).float()

    # Background for SHAP (use a small subset or mean)
    background = X_test[:1]  # or load small batch

    # DeepExplainer (works well with CNNs)
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_test)

    # Aggregate across spatial dims → per-band importance
    band_importance = np.abs(shap_values[0]).mean(axis=(2, 3))  # (1, bands)

    # Load wavelengths from dataset
    _, _, wavelengths = HyperspectralDataset(args.data, target='nitrogen').get_wavelengths()

    # Top bands
    top_idx = np.argsort(band_importance[0])[::-1][:20]
    print("\n=== TOP 20 BANDS BY SHAP IMPORTANCE FOR NITROGEN ===")
    for i, idx in enumerate(top_idx):
        print(f"{i+1:2d}. {wavelengths[idx]:.1f} nm  (importance: {band_importance[0, idx]:.4f})")

    # Save for visualization / reports
    np.save("configs/nitrogen_shap_importance.npy", band_importance[0])
    print("\nSHAP values saved to configs/nitrogen_shap_importance.npy")
    print("SWIR dominance:", (wavelengths[top_idx] > 1000).mean())
