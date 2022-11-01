from typing import Tuple

import numpy as np
from torusgrid.fields import RealField2D
import torusgrid as tg

from . import _res

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources


_uc_path_map = {

    '0.01': 'uc_eps0.01.field',
    #'0.02': 'uc_eps0.02.field',
    '0.04': 'uc_eps0.04.field',
    '0.06': 'uc_eps0.06.field',
    '0.08': 'uc_eps0.08.field',
    '0.1': 'uc_eps0.1.field',
    '0.11': 'uc_eps0.11.field',
    '0.12': 'uc_eps0.12.field',
    '0.13': 'uc_eps0.13.field',
    '0.14': 'uc_eps0.14.field',
    '0.15': 'uc_eps0.15.field',
    '0.17': 'uc_eps0.17.field',
    '0.2': 'uc_eps0.2.field',
    '0.3': 'uc_eps0.3.field',
    '0.4': 'uc_eps0.4.field',
    '0.5': 'uc_eps0.5.field'
}

_liq_path_map = {
    '0.01': 'uc_eps0.01_liq.field',
    #'0.02': 'uc_eps0.02_liq.field',
    '0.04': 'uc_eps0.04_liq.field',
    '0.06': 'uc_eps0.06_liq.field',
    '0.08': 'uc_eps0.08_liq.field',
    '0.1': 'uc_eps0.1_liq.field',
    '0.11': 'uc_eps0.11_liq.field',
    '0.12': 'uc_eps0.12_liq.field',
    '0.13': 'uc_eps0.13_liq.field',
    '0.14': 'uc_eps0.14_liq.field',
    '0.15': 'uc_eps0.15_liq.field',
    '0.17': 'uc_eps0.17_liq.field',
    '0.2': 'uc_eps0.2_liq.field',
    '0.3': 'uc_eps0.3_liq.field',
    '0.4': 'uc_eps0.4_liq.field',
    '0.5': 'uc_eps0.5_liq.field'
}

def get_unit_cell(eps: str, liquid=False) -> RealField2D:
    """
    Retrieve a relaxed and minimized solid/liquid unit cell
    """
    if not type(eps) is str:
        raise ValueError(f'eps should be specified as a string, not {type(eps)}')

    if liquid:
        if not eps in _liq_path_map.keys():
            raise ValueError(f'no static resource for eps=\'{eps}\' liquid currently')

        with pkg_resources.path(_res, _liq_path_map[eps]) as pth:
            field = tg.load(tg.RealField2D, str(pth), autometa=True)
            # field = tg.proxies.RealField2DNPZ.read(str(pth))

        return field

    else:
        if not eps in _uc_path_map.keys():
            raise ValueError(f'no static resource for eps=\'{eps}\' solid currently')

        with pkg_resources.path(_res, _uc_path_map[eps]) as pth:
            field = tg.proxies.RealField2DNPZ.read(str(pth))
        return field

get_relaxed_minimized_coexistent_unit_cell = get_unit_cell # deprecated


_coex_epsmu_bounds = {
    '0.01': (0.063450057, 0.063450058),
    #'0.02': (0.080583046, 0.080583048),
    '0.04': (0.125627867, 0.125627868),
    '0.06': (0.152877453, 0.152877454),
    '0.08': (0.175436493, 0.175436494),
    '0.1': (0.194970474, 0.194970476), 
    '0.11': (0.203888060, 0.203888062), 
    '0.12': (0.212339532, 0.212339534), 
    '0.13': (0.220380763, 0.220380765),
    '0.14': (0.228057107, 0.228057109),
    '0.15': (0.235405973, 0.235405975),
    '0.17': (0.249241526, 0.249241528),
    '0.2': (0.268182615, 0.268182617),
    '0.3': (0.320131817, 0.320131819),
    '0.4': (0.360511484, 0.360511486),
    '0.5': (0.393053031, 0.393053032)
}

_coex_epsmu_final = {
    '0.01': 0.0634500576928258,
    #'0.02': 0.08058304741978646,
    '0.04': 0.12562786787748337,
    '0.06': 0.152877453515625,
    '0.08': 0.1754364934563637,
    '0.1': 0.19497047539062498,
    '0.11': 0.20388806097850207,
    '0.12': 0.21233953341841705,
    '0.13': 0.2203807639193684,
    '0.14': 0.2280571080446243,
    '0.15': 0.2354059738665819,
    '0.17': 0.2492415271699428,
    '0.2': 0.268182616015625,
    '0.3': 0.320131818359375,
    '0.4': 0.360511484765625,
    '0.5': 0.39305303167551753
}

'''
_coex_epsucf = {
    '0.1': 1.00047017737,
    '0.2': 1.00174100766,
    '0.3': 1.00341955414,
    '0.4': 1.00521892348
}
'''

def get_coexistent_mu_bounds(eps: str) -> Tuple[tg.FloatLike, tg.FloatLike]:
    """
    Return the bounds of solid-liquid coexistent chemical potential
    """
    if not eps in _coex_epsmu_bounds.keys():
        raise ValueError(f'no coexistent mu for eps=\'{eps}\' currently')
    return _coex_epsmu_bounds[eps]


def get_coexistent_mu_final(eps: str) -> tg.FloatLike:
    """
    Return the final value (during simulation) of solid-liquid coexistent chemical
    potential
    """
    if not eps in _coex_epsmu_final.keys():
        raise ValueError(f'no coexistent mu for eps=\'{eps}\' currently')
    return _coex_epsmu_final[eps]


def get_relaxed_unit_cell_size(eps: str, ratio=True) -> Tuple[tg.FloatLike, tg.FloatLike]:
    """
    Return the unit cell size at solid-liquid equilibrium.
    """
    if not type(eps) is str:
        raise ValueError(f'eps should be specified as a string, not {type(eps)}')

    if not eps in _uc_path_map.keys():
        raise ValueError(f'no static resource for eps=\'{eps}\' liquid currently')

    field = get_relaxed_minimized_coexistent_unit_cell(eps)
    if ratio:
        return (field.lx / np.pi / 4, field.ly / np.pi / 4 * np.sqrt(3))
    else:
        return (field.lx, field.ly)


