import numpy as np
import torusgrid as tg


def is_liquid(psi: np.ndarray, tol: tg.FloatLike=1e-5):
    """
    Return whether max(psi) - min(psi) <= tol
    """
    return np.max(psi) - np.min(psi) <= tol

