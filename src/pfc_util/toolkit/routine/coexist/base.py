from __future__ import annotations
from typing import Dict, Generic, List, Literal, Optional, Tuple
import torusgrid as tg
from collections import OrderedDict
import numpy as np
import rich

console = rich.get_console()


class ZeroSearchRecord:
    """
    This interface handles the search of a zero of a **monotonically increasing
    function** that is expensive to evaluate on a given interval.

    There are currently two supported modes:

        - binary:
            Perform a binary search

        - interpolate:
            Really interpolate & extrapolate. This assumes that the function
            has definite concavity on the interval of interest.

    """
    def __init__(self, *, 
            initial_range: Optional[Tuple[tg.FloatLike, tg.FloatLike]]=None,
            search_method: Literal['binary', 'interpolate']='binary'
        ) -> None:
        self._x: List[tg.FloatLike] = []
        self._y: List[tg.FloatLike] = []

        self._record = OrderedDict[tg.FloatLike, tg.FloatLike]()

        self._zero: tg.FloatLike|None = None

        self._upper: List[tg.FloatLike] = []
        self._lower : List[tg.FloatLike] = []

        self.phase = 0

        if initial_range is not None:
            self._lower.append(initial_range[0])
            self._upper.append(initial_range[1])

        self.search_method = search_method

    @property
    def upper_bound(self):
        """
        Upper bound of zero
        """
        return self._upper[-1]

    @property
    def lower_bound(self):
        """
        Lower bound of zero
        """
        return self._lower[-1]

    @property
    def x(self):
        return list(self.record.keys())

    @property
    def y(self):
        return list(self.record.values())

    @property
    def zero(self):
        """
        Return the zero, if found
        """
        return self._zero

    @property
    def record(self):
        return self._record
    
    def update(self, x: tg.FloatLike, y: tg.FloatLike):
        self._record[x] = y

        if y > 0:
            if self.upper_bound is None:
                self._upper.append(x)
            elif self.upper_bound not in self.record.keys():
                # happens in the beginning
                self._upper.append(x)
            elif self.record[self.upper_bound] > y:
                self._upper.append(x)

        elif y < 0:
            if self.lower_bound is None:
                self._lower.append(x)
            elif self.lower_bound not in self.record.keys():
                self._lower.append(x)
            elif self.record[self.lower_bound] < y:
                self._lower.append(x)

        else:
            self._zero = x

    def polate_zero(
            self, 
            x1: tg.FloatLike, x2: tg.FloatLike, *, 
            rtol: tg.FloatLike=0):
        """
        Parameters: 
            a, b: record indices

        Return:
            The x coordinate the intersection betwee the x axis and
            the line passing through (x[a], f(x[a])) and (x[b], f(x[b])).

            If f(x[a]) and f(x[b]) is too close (w.r.t. rtol), a ValueError is raised
        """

        y1 = self.record[x1]
        y2 = self.record[x2]

        if np.abs(y1-y2) <= rtol * (np.abs(y1)+np.abs(y2)) / 2:
            raise ValueError(f'{y1} and {y2} are too close')
        return (y1*x2 - y2*x1) / (y1 - y2)

    def next(self, verbose: bool = True) -> tg.FloatLike:
        """
        Return the next point (x) to sample

        We want a balance between quick convergence and narrow bounds.

        The problem with binary search is that although the bounds are
        guaranteed to be narrow from both directions, the convergence is slow
        due to the method of iteration.

        The problem with linear inter/extrapolation is that it is easy to run
        into an endless loop of unilateral sampling. Assuming a monotonically
        increasing function with definite concavity within the interval of
        interest, then one iteration results in

        Convex:
            - - -> +
            - + -> -
            + + -> +

        Concave:
            - - -> -
            - + -> +
            + + -> -

        where -/+ means to the left/right of the zero. So if the function is
        concave, for example, one would keep sampling from the left of the zero.

        The algorithm provided here tackles this problem in the following way:
            
            - The sampling is done in four phases 

            - In phases 0 & 2, we use the current upper and lower bounds to obtain an
              inter/extrapolated zero.

            - In phase 1, the current and previous upper bounds are used

            - In phase 3, the current and previous lower bounds are used

            - If the above is not possible (due to lack of samples at the
              beginning or vanishing denominator), then fall back to binary
              sampling
        """

        if self.search_method == 'binary':
            x = (self.upper_bound + self.lower_bound) / 2
            
        else:
            try:

                if self.phase in [0,2]:
                    x1, x2 = self._upper[-1], self._lower[-1]
                elif self.phase == 1:
                    x1, x2 = self._upper[-1], self._upper[-2]
                else:
                    x1, x2 = self._lower[-1], self._lower[-2]

                x = self.polate_zero(x1, x2)
                console.log(f'Polate phase {self.phase}')
                console.log(f'[bold]Jumped[/bold]: {x1}, {x2} -> {x}')
                self.phase += 1
                self.phase %= 4

            except (ValueError, IndexError, KeyError):
                x = (self.upper_bound + self.lower_bound) / 2

        return x


class MuSearchRecord(ZeroSearchRecord):
    def __init__(self, *, initial_range: Tuple[tg.FloatLike, tg.FloatLike],
                 search_method: Literal['binary', 'interpolate'] = 'binary') -> None:
        super().__init__(initial_range=initial_range, search_method=search_method)
        self.omega_l: List[tg.FloatLike] = []
        self.omega_s: List[tg.FloatLike] = []

        self.mu_min_initial = initial_range[0]
        self.mu_max_initial = initial_range[1]

    def append(self, mu: tg.FloatLike, omega_l: tg.FloatLike, omega_s: tg.FloatLike):
        self.update(mu, omega_s - omega_l)
        self.omega_l.append(omega_l)
        self.omega_s.append(omega_s)

    @property
    def mu(self):
        return self.x

    def __getitem__(self, key: str):
        if key == 'mu':
            return self.mu
        if key == 'omega_s':
            return self.omega_s
        if key == 'omega_l':
            return self.omega_l
        if key == 'mu_min_initial':
            return self.mu_min_initial
        if key == 'mu_max_initial':
            return self.mu_max_initial
        if key == 'mu_min_final':
            return self.lower_bound
        if key == 'mu_max_final':
            return self.upper_bound



