#   smobol - Sparse grid and Sobol approximations
#   This file is part of smobol.
#   Copyright 2013 Richard Dwight <richard.dwight@gmail.com>
#
#   smobol is free software: you can redistribute it and/or modify it under the
#   terms of the GNU Lesser General Public License as published by the Free
#   Software Foundation, either version 3 of the License, or (at your option)
#   any later version.
#
#   smobol is distributed in the hope that it will be useful, but WITHOUT ANY
#   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#   FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
#   more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with smobol.  If not, see <http://www.gnu.org/licenses/>.
import sys, copy, itertools

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d import Axes3D

    plt_loaded = True
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
except ImportError:
    plt_loaded = False

import mylib


# ------------------------------------------------------------------------------
# Operations on multi-indices, tuples eg. in 6-d (1,2,1,2,2,3)
def ei(i, dim):
    """Return unit multiindex increment in dimension i: (0,...,0,1,0,...,0)"""
    tmp = np.zeros(dim, dtype=np.int8)
    tmp[i] += 1
    return tmp


def isvalid(multiindex):
    """Are all entries in a multiindex > 0?  Ie. is multiindex "valid" """
    return all(multiindex)


def forward_neighbours(multiindex):
    """Return set of all forward-neighbours of a multiindex, e.g (2,1,1) =>
    {(3,1,1),(2,2,1),(2,1,2)}"""
    dim = len(multiindex)
    return set([tuple(np.array(multiindex) + ei(i, dim)) for i in range(dim)])


def backward_neighbours(multiindex):
    """Return set of all (valid) backward-neighbours of a multiindex, e.g 
    (2,2,1) => {(1,2,1),(2,1,1)}"""
    dim = len(multiindex)
    tmp = set([tuple(np.array(multiindex) - ei(i, dim)) for i in range(dim)])
    return set(filter(isvalid, tmp))


def interaction(multiindex):
    """
    Return tuple of interaction between variables which this multiindex 
    represents.  Eg. (a) (1,1,1) -> (), (b) (1,2,1) -> (1,), (c) (2,3,1) -> (0,1)
    Tuple is always ordered smallest dim-index to largest.
    """
    return tuple(np.where(np.array(multiindex, dtype=np.int8) > 1)[0])


# ------------------------------------------------------------------------------
class IndexSet:
    """
    Index-set describing a general sparse grid.  Each "level" (array (dim,),
    dtype=int8) in the set corresponds to one difference rule in the Smolyak
    sparse-grid sum (i.e. one tensor-product of 1d rules (\Delta_i(f) = U_i(f) -
    U_{i-1}(f)).  Standard Smolyak sparse-grid can be described by setting-up
    the level-set s.t. it contains all levels k with |k| < l+dim-1.  This is
    achieved with init_simplex(l).  Level-sets are "admissible" (i.e. result in
    correct quadrature rules) iff they are 

    Adaptivity following: Gerstner + Griebel, Dimension-Adaptive Tensor-Product
    Quadrature, Computing 71(1), 2003.  By adapting the index-set the sparse-grid
    rule is adapted.

    Definitions:
      - Multi-index        - tuple() length dim, describing quadrature level in 
                             each dimension.  Minimum level == 1. Each
                             multi-index corresponds to a tensor-product 
                             quadrature rule.
      - Index-set          - a set of multi-indices describing a particular 
                             sparse-grid quadrature rule.
    """

    def __init__(self, I=None, dim=-1, level=1, fill='simplex'):
        """
        Initialize level-set according to fill-pattern and level
        """
        if I:
            self.I = copy.deepcopy(I)
            self.dim = len(next(iter(self.I)))  # Length of random element of I
        else:
            assert dim > 0, 'Dimension must be > 0'
            self.dim = dim
            # Initialize the index-set
            if fill == 'simplex':
                self.init_simplex(level)
            elif fill == 'full_factor':
                self.init_full_factor(level)
            elif fill == 'one_factor':
                self.init_one_factor(level)
            elif fill == 'none':
                pass
            else:
                raise ValueError('Unknown fill pattern: %s' % fill)

    def is_admissible(self):
        """
        Check whether I is an admissible index-set (ie. all members are valid
        multiindices, and all backward neighbours of all members of I are in I).
        """
        for mi in self.I:
            if not isvalid(mi):
                return False
            if not backward_neighbours(mi) <= self.I:
                return False
        return True

    def activeset(self):
        """
        Return the active set of I, ie. those multiindices that can be added to
        I individually, and still result in an admissible indexset. These 
        multiindices are potential adaptation directions.
        """
        A = set([])  # Forward neighbours of all members of I
        for mi in self.I:
            A |= forward_neighbours(mi)
        A = A - self.I  # Forward neighbours of I
        # Forward neighbours with all backward
        # neighbours in I
        return set(filter(lambda a: backward_neighbours(a) <= self.I, A))

    def global_refine(self):
        """Global refinement, add entire active set"""
        self.I |= self.activeset()

    def max_level(self):
        """Highest level in the indexset I in any dimension"""
        return np.max(np.array(list(self.I), dtype=np.int8))

    def interactions(self):
        """
        Set of all interactions between variables represented by the index-set.
        Eg. (a) for one-at-a-time index-set return {(),(0,),(1,),...,(dim-1,)}, 
        and (b) for simplex level=3 in dim=3, return {(),(0,),(1,),(2,),
        (0,1),(0,2),(1,2)}.  Tuples are always sorted, ie. never (1,0).
        """
        return set([interaction(mi) for mi in self.I])

    def max_interaction(self):
        """
        Highest order of interaction between variables, eg. for a one-factor-
        at-a-time indexset = 1, for level L simplex indexset = L-1
        """
        orders = [np.sum(np.array(mi, dtype=np.int8) > 1) for mi in self.I]
        return max(orders)

    # --- Init index-set routines ----------------------------------------------
    def init_simplex(self, level):
        """
        Initialize with the "standard" sparse-grid index set of level L>0: 
        |k|_1 <= L+dim-1.  This choice has some nice theoretical error properties.
        """
        assert level > 0, 'Level must be >0: l = %d' % level
        self.I = {tuple(np.ones(self.dim, dtype=np.int8))}
        for l in range(level - 1):
            self.I |= self.activeset()

    def init_one_factor(self, level):
        """Initialize with one-factor analysis on each variable"""
        assert level > 0, 'Level must be >0: l = %d' % level
        self.I = {tuple(np.ones(self.dim, dtype=np.int8))}
        for i in range(self.dim):
            for j in range(2, level + 1):
                tmp = np.ones(self.dim, dtype=np.int8)
                tmp[i] = j
                self.I |= set(tuple(tmp))

    def init_full_factor(self, level):
        """
        Initialize with the full tensor-product (non-sparse) of level level>0.  
        This is just for reference purposes.
        """
        self.I = set([])
        for mi in mylib.meshgrid_flatten(*(range(1, level + 1),) * self.dim):
            self.I |= set([tuple(mi)])

    # --------------------------------------------------------------------------
    def reduce_axes(self, axes):
        """
        Create new IndexSet by reducing this one along specified dimensions.
        Corresponds to equivalent, lower- dimensional sparse-grid rule.  This
        IndexSet remains untouched.
          axes - dimensions to remove from the indexset
        Return:
          New IndexSet object
        """
        # Axes to *keep*
        axes_r = mylib.complement(axes, self.dim)
        I_r = set([])  # New, reduced I
        for mi in self.I:
            I_r |= {tuple(np.array(mi, dtype=np.int8)[axes_r])}
        return IndexSet(I=I_r)

    # --------------------------------------------------------------------------
    def plot(self, ax, highlight=None):
        """
        Plot IndexSet in 2d, each block is a multiindex, with: black->active set, 
        gray->current.
        """
        assert plt_loaded, 'Matplotlib not installed'
        assert self.dim == 2, 'Plotting only implemented in 2d'
        max_level = self.max_level() + 2
        tlev = np.zeros((max_level,) * self.dim, dtype=np.int8)
        for mi in self.I:
            tlev[mi] = 1
        for mi in self.activeset():  # Active multiindices
            tlev[mi] = 2
        tlev = tlev[1:, 1:]
        if highlight is not None:
            tlev[tuple(np.array(highlight) - 1)] = -1.99
        ax.imshow(
            tlev,
            cmap=cm.RdGy,
            vmin=-2.,
            vmax=2.,
            aspect='equal',
            origin='lower',
            interpolation='nearest',
            extent=[.5, max_level + .5, .5, max_level + .5],
        )
        ax.get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
        ax.get_yaxis().set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel('Level (dim 0)')
        ax.set_ylabel('Level (dim 1)')
        if highlight is not None:
            ax.set_title(r'$\boldsymbol{k} = (%d,%d)$' % highlight)

    def plot3d(self, ax, labels=None, with_active=False):
        """
        Plot IndexSet in 3d, each block is a level, with: blue->current,
        red->active-set.
        """
        assert plt_loaded, 'Matplotlib not installed'
        assert self.dim == 3, 'Plotting only implemented in 3d'

        def plot_cube(mi, color='b'):
            r = np.array([-0.5, 0.5])
            offx, offy, offz = mi[0], mi[1], mi[2]
            X, Y = np.meshgrid(r, r)
            Z0, Z1 = np.ones((2, 2)) * r[0], np.ones((2, 2)) * r[1]
            ax.plot_surface(X + offx, Y + offy, Z0 + offz, color=color, alpha=0.5)
            ax.plot_surface(X + offx, Y + offy, Z1 + offz, color=color, alpha=0.5)
            ax.plot_surface(X + offx, Z0 + offy, Y + offz, color=color, alpha=0.5)
            ax.plot_surface(X + offx, Z1 + offy, Y + offz, color=color, alpha=0.5)
            ax.plot_surface(Z0 + offx, Y + offy, X + offz, color=color, alpha=0.5)
            ax.plot_surface(Z1 + offx, Y + offy, X + offz, color=color, alpha=0.5)

        M = self.max_level()
        for mi in self.I:
            plot_cube(mi)
        if with_active:
            for mi in self.activeset():
                plot_cube(mi, color='r')
            M += 1
        ax.set_xticks(range(1, M + 1))
        ax.set_yticks(range(1, M + 1))
        ax.set_zticks(range(1, M + 1))
        ax.set_xticklabels(['$%d$' % i for i in range(1, M + 1)])
        ax.set_yticklabels(['$%d$' % i for i in range(1, M + 1)])
        ax.set_zticklabels(['$%d$' % i for i in range(1, M + 1)])
        ax.set_xlim(0.45, M + 0.55)
        ax.set_ylim(0.45, M + 0.55)
        ax.set_zlim(0.45, M + 0.55)
        if not labels:
            ax.set_xlabel('Level (dim 0)')
            ax.set_ylabel('Level (dim 1)')
            ax.set_zlabel('Level (dim 2)')
        else:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    I = IndexSet(dim=3, level=0, fill='none')
    I.init_simplex(3)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    # ax.set_aspect("equal")
    I.plot3d(ax, with_active=True)

    plt.savefig('tmp/1.pdf')
