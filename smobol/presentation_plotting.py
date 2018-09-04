import numpy as np

import sparse
from quad_patterson import QuadraturePatterson
from quad_cc import QuadratureCC


def levelset_plot_2d(highlight):
    dim = 2
    l = 5
    sp = sparse.SparseGrid(dim, QuadratureCC(), level=l, fill='simplex')
    sp.plot('sparse_CC_%d_%d.pdf' % highlight, highlight)


def levelset_adapt():
    ls = sparse.LevelSet(2, 1, fill='none')
    ls.I.append(np.array((2, 1)))
    ls.I.append(np.array((3, 1)))
    ls.I.append(np.array((4, 1)))
    ls.I.append(np.array((5, 1)))
    ls.I.append(np.array((1, 2)))
    ls.I.append(np.array((1, 3)))
    ls.I.append(np.array((2, 2)))
    ls.O = set(range(8))
    ls.A = set([])
    sp = sparse.SparseGrid(2, QuadratureCC(), levelset=ls)
    sp.plot('sparse_CC_adapt.pdf')


if __name__ == '__main__':

    levelset_adapt()

    if False:
        levelset_plot_2d((1, 1))
        levelset_plot_2d((2, 1))
        levelset_plot_2d((3, 1))
        levelset_plot_2d((4, 1))
        levelset_plot_2d((5, 1))

        levelset_plot_2d((1, 1))
        levelset_plot_2d((1, 2))
        levelset_plot_2d((1, 3))
        levelset_plot_2d((1, 4))
        levelset_plot_2d((1, 5))

        levelset_plot_2d((2, 2))

        levelset_plot_2d((2, 3))
        levelset_plot_2d((2, 4))

        levelset_plot_2d((3, 3))

        levelset_plot_2d((3, 2))
        levelset_plot_2d((4, 2))
