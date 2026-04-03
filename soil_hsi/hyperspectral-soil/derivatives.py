import numpy as np

def first_derivative(spectra):
    """
    First derivative along spectral axis
    """
    return np.gradient(spectra, axis=1)


def second_derivative(spectra):
    """
    Second derivative along spectral axis
    """
    return np.gradient(np.gradient(spectra, axis=1), axis=1)
