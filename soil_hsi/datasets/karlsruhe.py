from .base_dataset import BaseSoilDataset
import numpy as np

class KarlsruheDataset(BaseSoilDataset):
    def load(self):
        # TEMP: replace with real loading later
        self.spectra = np.random.rand(100, 200)
        self.wavelengths = np.linspace(400, 2500, 200)
        self.labels = np.random.rand(100)
