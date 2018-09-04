import numpy as np

from sparse import SparseGrid
from quad_cc import QuadratureCC


def heavygas_barrier():
    """
    TNO heavy-gas (propane) transport with barrier wall.  Stijn's case.
    Inputs: U_{ABL}[m/s], U_{rel}[m/s], T_{rel}[m/s]  
    Output: Effect-distance[m]
    """
    ninput = 3  # First 3 columns inputs, last col output
    data = np.array(
        [
            [5., 20., 290., 180.04],
            [3., 20., 290., 226.67],
            [7., 20., 290., 161.04],
            [5., 18., 290., 166.23],
            [5., 22., 290., 193.1],
            [5., 20., 270., 175.09],
            [5., 20., 310., 186.11],
            [3.585786, 20., 290., 198.5],
            [6.414214, 20., 290., 167.14],
            [3., 18., 290., 204.35],
            [7., 18., 290., 149.08],
            [3., 22., 290., 250.17],
            [7., 22., 290., 172.94],
            [5., 18.58579, 290., 170.28],
            [5., 21.41421, 290., 189.29],
            [3., 20., 270., 262.36],
            [7., 20., 270., 162.29],
            [3., 20., 310., 215.01],
            [7., 20., 310., 159.2],
            [5., 18., 270., 160.9],
            [5., 22., 270., 188.37],
            [5., 18., 310., 172.13],
            [5., 22., 310., 199.47],
            [5., 20., 275.8579, 176.06],
            [5., 20., 304.1421, 184.59],
            [3.152241, 20., 290., 217.83],
            [4.234633, 20., 290., 184.57],
            [5.765367, 20., 290., 173.81],
            [6.847759, 20., 290., 162.67],
            [3.585786, 18., 290., 179.47],
            [6.414214, 18., 290., 154.59],
            [3.585786, 22., 290., 215.46],
            [6.414214, 22., 290., 179.48],
            [3., 18.58579, 290., 211.5],
            [7., 18.58579, 290., 152.78],
            [3., 21.41421, 290., 243.35],
            [7., 21.41421, 290., 169.32],
            [5., 18.15224, 290., 167.26],
            [5., 19.23463, 290., 174.91],
            [5., 20.76537, 290., 185.23],
            [5., 21.84776, 290., 192.1],
            [3.585786, 20., 270., 206.16],
            [6.414214, 20., 270., 167.],
            [3.585786, 20., 310., 200.23],
            [6.414214, 20., 310., 166.62],
            [3., 18., 270., 230.94],
            [7., 18., 270., 149.6],
            [3., 22., 270., 294.59],
            [7., 22., 270., 174.77],
            [3., 18., 310., 197.47],
            [7., 18., 310., 148.36],
            [3., 22., 310., 233.67],
            [7., 22., 310., 186.56],
            [5., 18.58579, 270., 165.05],
            [5., 21.41421, 270., 184.48],
            [5., 18.58579, 310., 176.28],
            [5., 21.41421, 310., 195.67],
            [3., 20., 275.8579, 249.58],
            [7., 20., 275.8579, 161.98],
            [3., 20., 304.1421, 217.17],
            [7., 20., 304.1421, 159.8],
            [5., 18., 275.8579, 161.58],
            [5., 22., 275.8579, 189.49],
            [5., 18., 304.1421, 170.56],
            [5., 22., 304.1421, 197.64],
            [5., 20., 271.5224, 175.36],
            [5., 20., 282.3463, 177.45],
            [5., 20., 297.6537, 182.6],
            [5., 20., 308.4776, 185.72],
        ]
    )
    varmin, varmax = data.min(0), data.max(0)
    for i in range(ninput):  # Transform to [0,1]^{ninput}
        data[:, i] = (data[:, i] - varmin[i]) / (varmax[i] - varmin[i])

        # Setup sparse grid - we know its form
        # for this data (level 4, simplex, CC)
    dim, level = 3, 4
    sp = SparseGrid(dim, QuadratureCC(), level=level, fill='simplex')
    xval = sp.get_nodes()
    fval = {}
    for k, x in xval.iteritems():
        for xref in data:
            if np.sqrt(np.sum((x - xref[:-1]) ** 2)) < 0.001:
                fval[k] = xref[-1]
                # Check
    # print sp.integrate(fval)           # Agrees with Stijn Desmedt MSc Table 4.1
    # print sp.compute_sobol_variances(fval, cardinality=3, levelrefine=2)
    return sp, fval


if __name__ == '__main__':

    sp, fval = heavygas_barrier()

    import adapt

    if False:
        adapt.sobol_adaptive(
            None,
            sp.dim,
            S_cutoff=0.95,
            max_samples=100,
            max_iters=6,
            max_level=4,
            max_k=4,
            fval=fval,
            plotting=True,
            labels=['$U_{ABL}$', '$U_{rel}$', '$T_{rel}$'],
        )

    adapt.gerstnerandgriebel_adaptive(
        None,
        sp.dim,
        max_samples=100,
        max_iters=20,
        max_level=4,
        max_k=4,
        fval=fval,
        plotting=True,
        labels=['$U_{ABL}$', '$U_{rel}$', '$T_{rel}$'],
    )
