

class IllegalActionError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ModifyingReadOnlyObjectError(IllegalActionError):
    def __init__(self, message, obj):
        super().__init__(message)
        self.obj = obj

