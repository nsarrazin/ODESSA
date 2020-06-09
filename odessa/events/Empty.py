import numba as nb


class Empty:
    """
    Creates an event function that never triggers
    """

    def __init__(self):
        self.terminal = True
        pass

    @property
    def event(self):
        @nb.njit
        def event(t, y):
            return 1

        event.terminal = self.terminal

        return event
