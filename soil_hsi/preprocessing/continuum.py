import numpy as np
from scipy.spatial import ConvexHull

def continuum_removal(spectrum):
    """
    Apply continuum removal to a single spectrum
    """
    x = np.arange(len(spectrum))
    points = np.column_stack((x, spectrum))

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    hull_points = hull_points[np.argsort(hull_points[:, 0])]

    continuum = np.interp(x, hull_points[:, 0], hull_points[:, 1])

    return spectrum / continuum


def batch_continuum_removal(spectra):
    return np.array([continuum_removal(s) for s in spectra])
