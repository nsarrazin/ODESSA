import numpy as np
from odessa.helpers.interpolators.RegularGridInterpolator import RegularGridInterpolator

def test_grid_interpolator(plotting=False):
    def f(x, y):
        return 2 * x ** 3 + 3 * y ** 2


    x1 = np.linspace(-5., 10, 10)
    x2 = np.linspace(0, 20, 30)

    from itertools import product
    import matplotlib.pyplot as plt

    values = np.zeros((x1.size, x2.size))
    for (x1i, x1v), (x2i, x2v) in product(enumerate(x1), enumerate(x2)):
        values[x1i, x2i] = f(x1v, x2v)

    # print(values)
    test = RegularGridInterpolator()
    test.values = values
    test.d0x0 = x1[0]
    test.d1x0 = x2[0]
    test.d0dx = x1[1] - x1[0]
    test.d1dx = x2[1] - x2[0]

    x1q = np.linspace(-5.0, 10, 100)
    x2q = np.linspace(0, 20, 100)
    vqi = np.zeros(x1q.shape)

    for i, v in enumerate(x2q):
        vqi[i] = test.eval((x1q[i], v))

    x1w = np.linspace(-5., 10., 1000)
    x2w = np.linspace(0., 20., 1000)
    vwi = np.zeros(x1w.shape)

    for i, v in enumerate(x2w):
        vwi[i] = f(x1w[i], v)

    # TODO: add assert case and check error
    if plotting:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x1w, x2w, vwi)
        ax.scatter(x1q, x2q, vqi)

        plt.show()

if __name__ == "__main__":

    test_grid_interpolator()
