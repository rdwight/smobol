
import numpy as np
import pickle

from sparse import SparseGrid
import test_genz as genz
from quad_cc import QuadratureCC


if __name__ == '__main__':
    dim = 5
    Nrepeat = 100
    quadrature = QuadratureCC()

    results = {}
    for iter in range(Nrepeat):  # Random variation
        # Genz coefficients
        genz.w = np.random.random(5)
        # According to Laurent
        genz.c = np.random.random(5)
        genz.c *= 2.5 / np.linalg.norm(genz.c)
        exact = genz.f1_exact(dim)

        for lmax, fill in zip([10, 4], ['simplex', 'full_factor']):
            if fill not in results:
                results[fill] = np.zeros((lmax + 1, 5))
            print('%s Dimension = %d %s' % ('-' * 20, dim, '-' * 20))
            print('%10s %10s %16s %12s' % ('Level', '#nodes', 'Integral', 'Rel error'))
            for l in range(1, lmax + 1):
                sp = SparseGrid(dim, quadrature, level=l, fill=fill)
                fval = sp.sample_fn(genz.f1)
                approx = sp.integrate(fval)
                error = abs(approx - exact) / exact
                minerror = min(error, results[fill][l, 3])
                maxerror = max(error, results[fill][l, 4])
                results[fill][l, :3] += [len(fval), approx, error]
                results[fill][l, 3:] = [minerror, maxerror]
                print('%10d %10d %16.10g %12.1e' % (l, len(fval), approx, error))
            print('%10s %10s %20.10g' % ('Exact', '', exact))

    pickle.dump(results, open('results.pickle', 'wb'))

    import matplotlib.pyplot as plt

    for fill in ['simplex', 'full_factor']:
        res = results[fill]
        res[:, :3] /= Nrepeat
        plt.plot(np.log10(res[:, 0]), np.log10(res[:, 2]), '-+', label=fill)
        plt.legend()
    plt.savefig('results.pdf')
