import numba as nb
import numpy as np
from odessa.helpers.math_func import bisect

interpolator1d_spec = [("dx", nb.float64),
                       ("x", nb.float64[:]),
                       ("y" ,nb.float64[:]),
                       ("x_grid", nb.float64[:]),
                       ("y_grid", nb.float64[:])]

@nb.experimental.jitclass(interpolator1d_spec)
class Interpolator1DLinear(object):
    """A numba jitclass that can be evaluated to linearly interpolate y over x.
    """
    def __init__(self):
        self.x = np.zeros(2, dtype=np.float64) # assumes sorted
        self.y = np.zeros(2, dtype=np.float64)

    def build_grid(self):
        """Builds an evenly spaced grid at init time so that evaluating only costs O(1)
        """
        self.dx = np.amin(np.abs(self.x[:-1] - self.x[1:]))
        print(self.dx, self.x)
        n_points = int(np.ceil((self.x[-1]-self.x[0])/self.dx))+1

        self.dx = (self.x[-1]-self.x[0])/n_points # reset dx bc rounding

        self.x_grid = np.linspace(self.x[0], self.x[-1], n_points)
        self.y_grid = np.interp(self.x_grid, self.x, self.y)

    def eval(self, x_eval):
        """Evaluates x_eval on the grid.

        Args:
            x_eval ([float]): The value to evaluate.

        Returns:
            [float]: The value of y at x_eval
        """
        if x_eval <= self.x_grid[0]:
            return self.y_grid[0]
        elif x_eval >= self.x_grid[-1]:
            return self.y_grid[-1]

        i = int(np.floor_divide(x_eval, self.dx)+1)

        interp_port = (x_eval - self.x_grid[i-1]) / (self.x_grid[i] - self.x_grid[i-1])
        return self.y_grid[i-1] + (interp_port * (self.y_grid[i] - self.y_grid[i-1]))


if __name__ == "__main__":
    interp = Interpolator1DLinear()
    interp.x = np.array([0., 0.5, 3., 6.], order="C")
    interp.y = np.array([0., 2., 6., 12.], order="C")

    interp.build_grid()

    print(interp.eval(0))
    print(interp.eval(0.5))
    print(interp.eval(1))
    print(interp.eval(1.5))
    print(interp.eval(3))
    print(interp.eval(6))
    print(interp.eval(9999))
    print(interp.eval(-10))