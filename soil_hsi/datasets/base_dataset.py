class BaseSoilDataset:
    def __init__(self):
        self.spectra = None
        self.wavelengths = None
        self.labels = None

    def load(self):
        raise NotImplementedError("You must implement this method")

    def get_data(self):
        return {
            "spectra": self.spectra,
            "wavelengths": self.wavelengths,
            "labels": self.labels,
        }
