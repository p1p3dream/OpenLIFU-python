from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.bf import delay_methods
from openlifu.geo import Point
from openlifu.xdc import Transducer


@dataclass
class DelayMethod(ABC):
    @abstractmethod
    def calc_delays(self, arr: Transducer, target: Point, params: xa.Dataset, transform:np.ndarray | None=None):
        pass

    def calc_delays_and_apod(
        self,
        arr: Transducer,
        target: Point,
        params: xa.Dataset,
        transform: np.ndarray | None = None,
    ):
        """Return ``(delays, apod)`` per element.

        The default implementation returns the scalar delays from
        :meth:`calc_delays` paired with ``None`` for the apodization. Subclasses
        that also want to contribute per-element amplitude weighting (for
        example a narrowband complex-weighted method) should override this
        method to return a numpy array for ``apod``. Downstream consumers such
        as :meth:`openlifu.plan.protocol.Protocol.beamform` check for a non-
        ``None`` ``apod`` and multiply it into the protocol's apodization
        method output.
        """
        delays = self.calc_delays(arr, target, params, transform=transform)
        return delays, None

    def to_dict(self):
        d = self.__dict__.copy()
        d['class'] = self.__class__.__name__
        return d

    @staticmethod
    def from_dict(d):
        d = d.copy()
        short_classname = d.pop("class")
        module_dict = delay_methods.__dict__
        class_constructor = module_dict[short_classname]
        return class_constructor(**d)

    @abstractmethod
    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the delay method parameters

        :returns: Pandas DataFrame of the delay method parameters
        """
        pass
