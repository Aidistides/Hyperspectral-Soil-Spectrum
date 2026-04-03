import matplotlib.pyplot as plt

def plot_spectrum(wavelengths, spectra, sample_id):
    """
    Plot a single spectrum
    """
    plt.figure()
    plt.plot(wavelengths, spectra[sample_id])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(f"Spectrum - Sample {sample_id}")
    plt.show()


#plot_spectrum(wavelengths, spectra, sample_id=0)


def plot_multiple_spectra(wavelengths, spectra, sample_ids, labels=None):
    """
    Plot multiple spectra on same graph
    """
    plt.figure()

    for i, sid in enumerate(sample_ids):
        label = labels[i] if labels else f"Sample {sid}"
        plt.plot(wavelengths, spectra[sid], label=label)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Multiple Spectra Comparison")
    plt.legend()
    plt.show()


import seaborn as sns

def plot_heatmap(spectra):
    """
    Heatmap of all spectra
    """
    plt.figure()
    sns.heatmap(spectra, cmap="viridis")
    plt.title("Spectral Heatmap")
    plt.xlabel("Wavelength Index")
    plt.ylabel("Sample Index")
    plt.show()



from scipy.signal import find_peaks

def plot_spectrum_with_peaks(wavelengths, spectra, sample_id):
    spectrum = spectra[sample_id]

    peaks, _ = find_peaks(-spectrum)  # negative = dips (absorption)

    plt.figure()
    plt.plot(wavelengths, spectrum, label="Spectrum")
    plt.scatter(wavelengths[peaks], spectrum[peaks], color="red", label="Absorption")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(f"Absorption Features - Sample {sample_id}")
    plt.legend()
    plt.show()



import numpy as np

def plot_feature_importance(wavelengths, importances):
    """
    Plot importance vs wavelength
    """
    plt.figure()
    plt.plot(wavelengths, importances)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Importance")
    plt.title("Wavelength Importance")
    plt.show()
