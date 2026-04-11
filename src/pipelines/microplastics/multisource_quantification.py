"""
Multisource Quantification Module for Soil Microplastics
Inspired exclusively by Li et al. (2021) - Chemosphere
"An effective method for the rapid detection of microplastics in soil"

Key ideas implemented:
- Minimal preprocessing (mirroring the paper's simple sample prep: drying/grinding/sieving)
- Local vs. Multisource modeling using Least Squares Support Vector Machine (LS-SVM)
- Regression for MP concentration (pollution degree) rather than pure classification
- Focus on generalization across different soil backgrounds (regions)

Target polymers in the paper: LDPE and PVC (extendable to PE, PP, PS, PET, etc.)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.svm import SVR  # Approximation of LS-SVM (radial basis kernel + regularization)
from sklearn.multioutput import MultiOutputRegressor
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class MultisourceMPQuantifier:
    """
    Implements local and multisource LS-SVM-style regression for MP concentration prediction.
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 100.0, gamma: str = 'scale'):
        self.scaler = StandardScaler()
        self.local_models: Dict[str, SVR] = {}      # One model per soil/region
        self.multisource_model: Optional[MultiOutputRegressor] = None
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.polymers = ["LDPE", "PVC"]  # As in the paper; extend as needed
    
    def preprocess(self, spectra: np.ndarray) -> np.ndarray:
        """Minimal preprocessing inspired by the paper (focus on raw spectral features)."""
        # Simple scaling + optional light smoothing could be added here
        return self.scaler.fit_transform(spectra)
    
    def train_local_model(self, region_id: str, X: np.ndarray, y: np.ndarray) -> SVR:
        """
        Train a region-specific (local) model.
        y shape: (n_samples,) or (n_samples, n_polymers) for multi-polymer.
        """
        X_scaled = self.preprocess(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        model = SVR(kernel=self.kernel, C=self.C, gamma=self.gamma)
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            model = MultiOutputRegressor(model)
        
        model.fit(X_train, y_train)
        self.local_models[region_id] = model
        
        # Evaluate
        y_pred = model.predict(X_test)
        print(f"=== Local Model - Region: {region_id} ===")
        print(f"R²: {r2_score(y_test, y_pred):.4f}")
        print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}\n")
        
        return model
    
    def train_multisource_model(self, X_list: List[np.ndarray], y_list: List[np.ndarray]) -> MultiOutputRegressor:
        """
        Multisource model: Combine datasets from multiple regions (as in Li et al.).
        This improves generalization across different soil types.
        """
        X_combined = np.vstack(X_list)
        y_combined = np.vstack(y_list) if len(y_list[0].shape) > 1 else np.concatenate(y_list)
        
        X_scaled = self.preprocess(X_combined)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_combined, test_size=0.2, random_state=42
        )
        
        base_svr = SVR(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.multisource_model = MultiOutputRegressor(base_svr)
        self.multisource_model.fit(X_train, y_train)
        
        y_pred = self.multisource_model.predict(X_test)
        
        print("=== Multisource Model (Combined Regions) ===")
        print(f"Overall R²: {r2_score(y_test, y_pred, multioutput='uniform_average'):.4f}")
        print(f"Overall RMSE: {root_mean_squared_error(y_test, y_pred, multioutput='uniform_average'):.4f}")
        print(f"Overall MAE: {mean_absolute_error(y_test, y_pred, multioutput='uniform_average'):.4f}\n")
        
        return self.multisource_model
    
    def predict_concentration(self, spectra: np.ndarray, region_id: str = None) -> np.ndarray:
        """Predict MP concentration. Use multisource by default if available."""
        X_scaled = self.preprocess(spectra)
        
        if region_id and region_id in self.local_models:
            model = self.local_models[region_id]
            print(f"Using local model for region: {region_id}")
        elif self.multisource_model is not None:
            model = self.multisource_model
            print("Using multisource model (recommended for new/variable soils)")
        else:
            raise ValueError("No trained model available.")
        
        return model.predict(X_scaled)
    
    def save_models(self, save_dir: str = "models/multisource_mp"):
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.local_models, path / "local_models.pkl")
        if self.multisource_model:
            joblib.dump(self.multisource_model, path / "multisource_model.pkl")
        joblib.dump(self.scaler, path / "scaler.pkl")
        print(f"Models saved to {save_dir}")
    
    def load_models(self, load_dir: str = "models/multisource_mp"):
        path = Path(load_dir)
        self.local_models = joblib.load(path / "local_models.pkl")
        if (path / "multisource_model.pkl").exists():
            self.multisource_model = joblib.load(path / "multisource_model.pkl")
        self.scaler = joblib.load(path / "scaler.pkl")


# ============================
# Example Usage (integrate with your HSI data loader)
# ============================
if __name__ == "__main__":
    # Simulate multiple regions (different soil backgrounds)
    # X_region1, y_region1 = load_hsi_spectra_and_concentrations(region="field_a")  # shape: (n_samples, n_bands), (n_samples, n_polymers)
    
    quantifier = MultisourceMPQuantifier()
    
    # Train local models for a few regions
    # quantifier.train_local_model("field_a", X_a, y_a)
    # quantifier.train_local_model("field_b", X_b, y_b)
    
    # Train multisource (recommended for generalization)
    # quantifier.train_multisource_model([X_a, X_b, X_c], [y_a, y_b, y_c])
    
    # Predict on new sample
    # pred = quantifier.predict_concentration(new_spectra, region_id=None)  # uses multisource
    
    print(" Multisource quantification module ready (Li et al. 2021 inspired)")
