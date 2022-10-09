from __future__ import annotations
'''
A wrapper module for scipy.fft due to typing issues
'''
from scipy.fft import (
    fft as __fft,
    ifft as __ifft,
    rfft as __rfft,
    irfft as __irfft,
    fft2 as __fft2,
    ifft2 as __ifft2,
    rfft2 as __rfft2,
    irfft2 as __irfft2,

)

import numpy.typing as npt


def __convert_type(__f):
    def f(x: npt.NDArray) -> npt.NDArray:
        return __f(x)
    return f


fft = __convert_type(__fft)
ifft = __convert_type(__ifft)

rfft = __convert_type(__rfft)
irfft = __convert_type(__irfft)



fft2 = __convert_type(__fft2)
ifft2 = __convert_type(__ifft2)

rfft2 = __convert_type(__rfft2)
irfft2 = __convert_type(__irfft2)






