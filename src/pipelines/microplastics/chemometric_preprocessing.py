"""
Chemometric Preprocessing & Analysis for Microplastics HSI
Inspired by Liu et al. (2023) - TrAC Trends in Analytical Chemistry
"Automated characterization and identification of microplastics through spectroscopy and chemical imaging in combination with chemometric"

This module implements common chemometric steps for HSI data:
- Ensemble preprocessing (multiple techniques combined)
- Dimensionality reduction (PCA, etc.)
- Baseline chemometric methods (PLS-DA, SVM on reduced features)
- Support for library matching and data fusion hints
- Designed to integrate with existing 3D CNN pipeline
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings("ignore")

class ChemometricMPProcessor:
    """
    Chemometric processor for hyperspectral microplastics data.
    Follows the review's emphasis on automated workflows combining
    spectroscopy/HSI with chemometrics for polymer identification.
    """
    
    def __init__(self, n_components: int = 10, sg_window: int = 11, sg_poly: int = 2):
        self.n_components = n_components
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.pca = None
        self.pls = None
        self.svm = None
    
    def ensemble_preprocess(self, spectra: np.ndarray, wavelengths: np.ndarray = None) -> np.ndarray:
        """
        Apply ensemble of preprocessing techniques recommended in the review.
        - Savitzky-Golay smoothing (already in your pipeline)
        - SNV (Standard Normal Variate)
        - Mean centering + scaling
        - Optional: MSC (Multiplicative Scatter Correction) or derivative
        """
        # Savitzky-Golay (already strong in your repo and in related HSI-MP papers)
        smoothed = savgol_filter(spectra, window_length=self.sg_window, polyorder=self.sg_poly, axis=1)
        
        # SNV
        mean = np.mean(smoothed, axis=1, keepdims=True)
        std = np.std(smoothed, axis=1, keepdims=True)
        snv = (smoothed - mean) / std
        
        # Mean centering
        processed = snv - np.mean(snv, axis=0)
        
        print(f"Applied ensemble preprocessing: SG({self.sg_window},{self.sg_poly}) + SNV + centering")
        return processed
    
    def reduce_dimensions(self, X: np.ndarray, y: np.ndarray = None, method: str = "pca") -> np.ndarray:
        """PCA or PLS for dimensionality reduction — core chemometric step."""
        X_scaled = StandardScaler().fit_transform(X)
        
        if method == "pca":
            self.pca = PCA(n_components=self.n_components)
            X_reduced = self.pca.fit_transform(X_scaled)
            explained = np.sum(self.pca.explained_variance_ratio_) * 100
            print(f"PCA: {self.n_components} components explain {explained:.2f}% variance")
            return X_reduced
        
        elif method == "pls" and y is not None:
            self.pls = PLSRegression(n_components=self.n_components)
            self.pls.fit(X_scaled, y)
            X_reduced = self.pls.transform(X_scaled)
            print(f"PLS-DA ready with {self.n_components} components")
            return X_reduced
        else:
            raise ValueError("Invalid method or missing labels for PLS")
    
    def train_baseline_classifier(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """Train a simple chemometric baseline (SVM on reduced features) for comparison with 3D CNN."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
        ])
        
        pipeline.fit(X_train, y_train)
        self.svm = pipeline
        
        y_pred = pipeline.predict(X_test)
        print("=== Chemometric Baseline (SVM) Performance ===")
        print(classification_report(y_test, y_pred, zero_division=0))
        return pipeline, y_test, y_pred
    
    def plot_spectral_comparison(self, original: np.ndarray, processed: np.ndarray, n_samples: int = 5):
        """Visualize effect of chemometric preprocessing (useful for documentation)."""
        plt.figure(figsize=(12, 6))
        for i in range(min(n_samples, len(original))):
            plt.subplot(2, 1, 1)
            plt.plot(original[i], alpha=0.7, label=f"Sample {i}" if i == 0 else "")
            plt.title("Original Spectra")
            plt.subplot(2, 1, 2)
            plt.plot(processed[i], alpha=0.7)
            plt.title("After Ensemble Chemometric Preprocessing")
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self):
        """Simple PCA loadings or PLS coefficients for interpretability (aligns with review's call for better understanding)."""
        if self.pca is not None:
            return np.abs(self.pca.components_).mean(axis=0)
        return None


# ============================
# Integration Example (add to your existing microplastics pipeline)
# ============================
if __name__ == "__main__":
    # Example usage with your HSI data loader
    processor = ChemometricMPProcessor(n_components=15)
    
    # Assume X_spectra is (n_pixels, n_bands) from your HSI cube
    # X_processed = processor.ensemble_preprocess(X_spectra)
    # X_reduced = processor.reduce_dimensions(X_processed)
    # model, y_test, y_pred = processor.train_baseline_classifier(X_reduced, y_labels)
    
    print(" Chemometric tools added — ready for hybrid CNN + chemometrics experiments.")
