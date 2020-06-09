import numba as nb


class fixedTime:
    """
    Creates an event function that triggers at time t_event
    """

    def __init__(self):
        self.t_event = 0.
        self.terminal = True
    
    @property
    def event(self):
        @nb.njit
        def event_func(t, y, t_event):
            return t_event - t

        def event(t, y): return event_func(t, y, self.t_event)

        event.terminal = self.terminal
        return event
