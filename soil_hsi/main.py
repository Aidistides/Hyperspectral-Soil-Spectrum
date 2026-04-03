from datasets.base_dataset import BaseSoilDataset

def main():
    print("Soil HSI Project Starting...")

if __name__ == "__main__":
    main()


from datasets.karlsruhe import KarlsruheDataset

def main():
    dataset = KarlsruheDataset()
    dataset.load()

    data = dataset.get_data()

    print(data["spectra"].shape)

if __name__ == "__main__":
    main()




from datasets.karlsruhe import KarlsruheDataset
from visualization.plots import (
    plot_spectrum,
    plot_multiple_spectra,
    plot_heatmap,
    plot_spectrum_with_peaks,
)

def main():
    dataset = KarlsruheDataset()
    dataset.load()
    data = dataset.get_data()

    spectra = data["spectra"]
    wavelengths = data["wavelengths"]

    plot_spectrum(wavelengths, spectra, sample_id=0)
    plot_multiple_spectra(wavelengths, spectra, [0, 1, 2])
    plot_heatmap(spectra)
    plot_spectrum_with_peaks(wavelengths, spectra, 0)

if __name__ == "__main__":
    main()
