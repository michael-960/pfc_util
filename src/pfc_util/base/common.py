from abc import ABC
import numpy as np
from typing import Union
from numbers import Number

class IllegalActionError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ModifyingReadOnlyObjectError(IllegalActionError):
    def __init__(self, message, obj):
        super().__init__(message)
        self.obj = obj


# decorator for exporting objects
def with_type(type_name: str):
    def wrapper(export):
        def wrapped_export(self):
            state = export(self)
            state['type'] = type_name
            return state
        return wrapped_export
    return wrapper


def scalarize(data: Union[dict, list, np.ndarray, Number, str]):

    if isinstance(data, Number) or type(data) == str:
        return data

    if isinstance(data, np.ndarray):
        try:
            return scalarize(data.item())
        except ValueError:
            return data

    if type(data) == dict:
        r = dict()
        for key in data:
            r[key] = scalarize(data[key])
        return r
    
    if type(data) == list:
        return [scalarize(item) for item in data]
