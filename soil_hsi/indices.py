import numpy as np

def spectral_ratio(spectra, wavelengths, wl1, wl2):
    """
    Simple ratio index: R(wl1) / R(wl2)
    """
    i1 = np.argmin(np.abs(wavelengths - wl1))
    i2 = np.argmin(np.abs(wavelengths - wl2))

    return spectra[:, i1] / spectra[:, i2]


def normalized_difference(spectra, wavelengths, wl1, wl2):
    """
    NDVI-like index: (R1 - R2) / (R1 + R2)
    """
    i1 = np.argmin(np.abs(wavelengths - wl1))
    i2 = np.argmin(np.abs(wavelengths - wl2))

    r1 = spectra[:, i1]
    r2 = spectra[:, i2]

    return (r1 - r2) / (r1 + r2 + 1e-8)


def soil_index_example(spectra, wavelengths):
    """
    Example soil index using two wavelengths
    """
    return normalized_difference(spectra, wavelengths, 2200, 2100)
