from __future__ import annotations

from typing import Protocol
import torusgrid as tg
from torusgrid.dynamics import FieldEvolver


class MinimizerSupplier(Protocol):
    def __call__(
        self, 
        field: tg.RealField2D, /) -> FieldEvolver[tg.RealField2D]: ...


class MuMinimizerSupplier(Protocol):
    def __call__(
        self, 
        field: tg.RealField2D, /, mu) -> FieldEvolver[tg.RealField2D]: ...

