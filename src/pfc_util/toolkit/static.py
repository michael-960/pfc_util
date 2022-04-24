import numpy as np
from torusgrid.fields import import_field, RealField2D
from numbers import Number
from typing import Tuple

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import _res


_uc_path_map = {
        '0.1': 'uc_eps0.1.npz',
        '0.2': 'uc_eps0.2.npz',
        '0.3': 'uc_eps0.3.npz',
        '0.4': 'uc_eps0.4.npz',
}

_liq_path_map = {
        '0.1': 'uc_eps0.1_liq.npz',
        '0.2': 'uc_eps0.2_liq.npz',
        '0.3': 'uc_eps0.3_liq.npz',
        '0.4': 'uc_eps0.4_liq.npz',
}

def get_relaxed_minimized_coexistent_unit_cell(eps: str, liquid=False) -> RealField2D:
    if not type(eps) is str:
        raise ValueError(f'eps should be specified as a string, not {type(eps)}')

    if liquid:
        if not eps in _liq_path_map.keys():
            raise ValueError(f'no static resource for eps=\'{eps}\' liquid currently')

        with pkg_resources.path(_res, _liq_path_map[eps]) as pth:
            field = import_field(np.load(pth, allow_pickle=True))
        return field

    else:
        if not eps in _uc_path_map.keys():
            raise ValueError(f'no static resource for eps=\'{eps}\' solid currently')

        with pkg_resources.path(_res, _uc_path_map[eps]) as pth:
            field = import_field(np.load(pth, allow_pickle=True))
        return field


    raise ValueError(f'no static resource for eps=\'{eps}\' currently')


_coex_epsmu_bounds = {
        '0.1': (0.194970474, 0.194970476), 
        '0.2': (0.268182615, 0.268182617),
        '0.3': (0.320131817, 0.320131819),
        '0.4': (0.360511484, 0.360511486)
}

_coex_epsmu_final = {
        '0.1': 0.19497047539062498,
        '0.2': 0.268182616015625,
        '0.3': 0.320131818359375,
        '0.4': 0.360511484765625
}

def get_coexistent_mu_bounds(eps: str) -> Tuple[float, float]:
    if not eps in _coex_epsmu_bounds.keys():
        return ValueError(f'no coexistent mu for eps=\'{eps}\' currently')
    return _coex_epsmu_bounds[eps]

def get_coexistent_mu_final(eps: str) -> float:
    if not eps in _coex_epsmu_final.keys():
        return ValueError(f'no coexistent mu for eps=\'{eps}\' currently')
    return _coex_epsmu_final[eps]


def get_relaxed_unit_cell_size(eps: str, ratio=True) -> Tuple[float, float]:
    if not type(eps) is str:
        raise ValueError(f'eps should be specified as a string, not {type(eps)}')

    if not eps in _uc_path_map.keys():
        raise ValueError(f'no static resource for eps=\'{eps}\' liquid currently')

    with pkg_resources.path(_res, _uc_path_map[eps]) as pth:
        field = import_field(np.load(pth, allow_pickle=True))

    if ratio:
        return (field.Lx / np.pi / 4, field.Ly / np.pi / 4 * np.sqrt(3))
    else:
        return (field.Lx, field.Ly)






