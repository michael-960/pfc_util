from abc import ABC

class IllegalActionError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ModifyingReadOnlyObjectError(IllegalActionError):
    def __init__(self, message, obj):
        super().__init__(message)
        self.obj = obj


# decorator for exporting objects
def with_type(type_name):
    def wrapper(export):
        def wrapped_export(self):
            state = export(self)
            state['type'] = type_name
            return state
        return wrapped_export
    return wrapper

