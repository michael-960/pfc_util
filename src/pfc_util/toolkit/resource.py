import numpy as np
from torusgrid.fields import import_field, RealField2D
from numbers import Number

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import _res


def get_relaxed_minimized_coexistent_unit_cell_solid(eps: str) -> RealField2D:
    if not type(eps) is str:
        raise ValueError(f'eps should be specified as a string, not {type(eps)}')

    if eps == '0.1':
        with pkg_resources.path(_res, 'uc_eps0.1.npz') as pth:
            field = import_field(np.load(pth, allow_pickle=True))
        return field

    raise ValueError(f'no static resource for eps=\'{eps}\'')
