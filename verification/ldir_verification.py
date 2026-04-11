from verification.ldir_verification import LDIRVerifier

"""
LDIR Verification Framework for Hyperspectral-Restruct
Cross-validates 3D CNN microplastics predictions against LDIR (Agilent 8700) ground-truth.
Compatible with Jia et al. (2022) "Automated identification and quantification of invisible microplastics in agricultural soils".
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, root_mean_squared_error
import seaborn as sns

class LDIRVerifier:
    """
    Framework to load LDIR CSV export and compare against HSI 3D-CNN predictions.
    Typical LDIR columns (from Agilent 8700 export): Polymer, Diameter_um, Area_um2,
    Match_Score, X_pos, Y_pos, etc.
    """
    
    def __init__(self, polymer_map: Optional[Dict[str, str]] = None):
        # Map common LDIR polymer names → your model's class names (PE, PP, PS, PET, etc.)
        self.polymer_map = polymer_map or {
            "Polyethylene": "PE", "PE": "PE",
            "Polypropylene": "PP", "PP": "PP",
            "Polystyrene": "PS", "PS": "PS",
            "Polyethylene Terephthalate": "PET", "PET": "PET",
            "Polyvinyl Chloride": "PVC", "PVC": "PVC",
            # Add more as needed from Jia et al. (26 polymers)
        }
    
    def load_ldir_data(self, csv_path: str, min_match_score: float = 65.0) -> pd.DataFrame:
        """Load Agilent LDIR export CSV and filter qualified particles."""
        df = pd.read_csv(csv_path)
        
        # Standard LDIR column cleaning (adjust column names if your export differs)
        df = df.rename(columns=lambda x: x.strip())
        
        # Filter by match score (Jia et al. used >65 %)
        if "Match_Score" in df.columns or "Hit Index" in df.columns:
            score_col = "Match_Score" if "Match_Score" in df.columns else "Hit Index"
            df = df[df[score_col] >= min_match_score]
        
        # Standardize polymer names
        if "Polymer" in df.columns:
            df["Polymer_std"] = df["Polymer"].map(self.polymer_map).fillna(df["Polymer"])
        
        print(f"Loaded {len(df)} qualified microplastic particles from LDIR")
        return df
    
    def aggregate_ldir_to_sample(self, ldir_df: pd.DataFrame, 
                                sample_id: str = None) -> Dict:
        """Aggregate LDIR particles to sample-level statistics (particles/kg, mass, polymer distribution)."""
        if "Polymer_std" not in ldir_df.columns:
            ldir_df["Polymer_std"] = ldir_df["Polymer"]
        
        stats = {
            "total_particles": len(ldir_df),
            "polymer_counts": ldir_df["Polymer_std"].value_counts().to_dict(),
            "polymer_fraction": (ldir_df["Polymer_std"].value_counts(normalize=True) * 100).to_dict(),
        }
        
        # Optional: if you have soil weight (kg) in metadata, compute particles/kg
        if "Sample_Weight_kg" in ldir_df.columns:
            stats["particles_per_kg"] = len(ldir_df) / ldir_df["Sample_Weight_kg"].iloc[0]
        
        return stats
    
    def compare_with_hsi_prediction(self, 
                                    hsi_prediction_map: np.ndarray,  # shape: (H, W, C) where C = num polymers + background
                                    ldir_stats: Dict,
                                    polymer_order: List[str],
                                    threshold: float = 0.5) -> Dict:
        """
        Compare aggregated HSI predictions (from predict.py) with LDIR ground-truth.
        hsi_prediction_map: probability or concentration map from your 3D CNN.
        """
        # Aggregate HSI over the sample area (or use whole map for field-scale)
        hsi_counts = {}
        for i, poly in enumerate(polymer_order):
            prob_map = hsi_prediction_map[..., i]
            detected = (prob_map > threshold).sum()
            hsi_counts[poly] = int(detected)
        
        # Metrics
        true_counts = [ldir_stats["polymer_counts"].get(p, 0) for p in polymer_order]
        pred_counts = [hsi_counts.get(p, 0) for p in polymer_order]
        
        results = {
            "mae": mean_absolute_error(true_counts, pred_counts),
            "rmse": root_mean_squared_error(true_counts, pred_counts),
            "classification_report": classification_report(true_counts, pred_counts, zero_division=0),
            "hsi_predicted_counts": hsi_counts,
            "ldir_ground_truth_counts": ldir_stats["polymer_counts"],
        }
        
        return results
    
    def plot_comparison(self, results: Dict, save_path: Optional[str] = None):
        """Visual comparison of LDIR vs HSI predictions."""
        polymers = list(results["ldir_ground_truth_counts"].keys())
        ldir_vals = [results["ldir_ground_truth_counts"].get(p, 0) for p in polymers]
        hsi_vals = [results["hsi_predicted_counts"].get(p, 0) for p in polymers]
        
        x = np.arange(len(polymers))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, ldir_vals, width, label="LDIR Ground Truth", alpha=0.8)
        plt.bar(x + width/2, hsi_vals, width, label="HSI 3D-CNN Prediction", alpha=0.8)
        plt.xlabel("Polymer")
        plt.ylabel("Particle Count")
        plt.title("LDIR vs HSI Microplastics Quantification")
        plt.xticks(x, polymers, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


# ============================
# Example usage (integrate with your existing predict.py / evaluate.py)
# ============================
if __name__ == "__main__":
    verifier = LDIRVerifier()
    
    # 1. Load LDIR data from the paper's sample (or your lab export)
    ldir_df = verifier.load_ldir_data("path/to/your_ldir_export.csv")
    ldir_stats = verifier.aggregate_ldir_to_sample(ldir_df)
    
    # 2. Load HSI prediction map from your model (output of predict.py)
    # Example: hsi_map = np.load("prediction_map.npy")  # or load from GeoTIFF
    # results = verifier.compare_with_hsi_prediction(hsi_map, ldir_stats, polymer_order=["PE", "PP", "PS", "PET", "PVC"])
    
    # 3. Visualize
    # verifier.plot_comparison(results, save_path="ldir_vs_hsi_comparison.png")
    
    print("✅ LDIR verification framework ready.")
