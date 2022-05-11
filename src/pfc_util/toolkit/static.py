from numbers import Number
from typing import Tuple

import numpy as np
from torusgrid.fields import import_field, RealField2D

from . import _res

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources



_uc_path_map = {
    '0.04': 'uc_eps0.04.field',
    '0.06': 'uc_eps0.06.field',
    '0.08': 'uc_eps0.08.field',
    '0.1': 'uc_eps0.1.field',
    '0.2': 'uc_eps0.2.field',
    '0.3': 'uc_eps0.3.field',
    '0.4': 'uc_eps0.4.field'
}

_liq_path_map = {
    '0.04': 'uc_eps0.04_liq.field',
    '0.06': 'uc_eps0.06_liq.field',
    '0.08': 'uc_eps0.08_liq.field',
    '0.1': 'uc_eps0.1_liq.field',
    '0.2': 'uc_eps0.2_liq.field',
    '0.3': 'uc_eps0.3_liq.field',
    '0.4': 'uc_eps0.4_liq.field'
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
        '0.04': (0.125627867, 0.125627868),
        '0.06': (0.152877453, 0.152877454),
        '0.08': (0.175436493, 0.175436494),
        '0.1': (0.194970474, 0.194970476), 
        '0.2': (0.268182615, 0.268182617),
        '0.3': (0.320131817, 0.320131819),
        '0.4': (0.360511484, 0.360511486)
}

_coex_epsmu_final = {
    '0.04': 0.12562786787748337,
    '0.06': 0.152877453515625,
    '0.08': 0.1754364934563637,
    '0.1': 0.19497047539062498,
    '0.2': 0.268182616015625,
    '0.3': 0.320131818359375,
    '0.4': 0.360511484765625
}

_coex_epsucf = {
        '0.1': 1.00047017737,
        '0.2': 1.00174100766,
        '0.3': 1.00341955414,
        '0.4': 1.00521892348
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

    field = get_relaxed_minimized_coexistent_unit_cell(eps)
    if ratio:
        return (field.Lx / np.pi / 4, field.Ly / np.pi / 4 * np.sqrt(3))
    else:
        return (field.Lx, field.Ly)






