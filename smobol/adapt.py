#   smobol - Sparse grid and Sobol approximations
#   This file is part of smobol.
#   Copyright 2013,2015 Richard Dwight <richard.dwight@gmail.com>
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
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
except ImportError:
    plt_loaded = False

from quad_cc import QuadratureCC
from indexset import IndexSet, interaction
from sparse import SparseGrid

#===============================================================================
def sobol_adaptive(f, dim, S_cutoff=0.95, max_samples=200, max_iters=10,
                   max_level=10, max_k=10,
                   fval=None, plotting=False, labels=None):
    """
    Sobol-based dimension-adaptive sparse grids.
      f           - function to sample
      dim         - number of input variables/dimensions
      S_cutoff    - Sobol index cutoff, adapt dimensions with Sobol indices
                    adding up to the cutoff
      max_samples - termination criteria, maximum allowed samples of f
      max_iters   - termination criteria, maximum adaptation iterations
      max_level   - maximum level allowed in any single variable, ie. don't allow
                    very-high resolution in any direction. Enforce
                    max(multiindex) <= max_level
      max_k       - enforce |multiindex|_1 <= dim + max_k - 1, ie. constrain
                    to simplex-rule of level max_k
      fval        - if samples of f already exist at sparse-grid nodes, pass
                    dictionary containing values - these will be used first
    The iteration will also terminate if the grid is unchanged after an 
    iteration.
    """
                                        # Initialize with simplex level 2, ie.
                                        # one-factor, 3 points in each direction
    K = IndexSet(dim=dim, level=2, fill='simplex')
    quad = QuadratureCC()
    sp = SparseGrid(dim, quad, indexset=K)
    iter = 1
    fval = {} if fval is None else fval # Dictionary of function values
                                        # Main adaptation loop 
    while sp.n_nodes() <= max_samples and iter <= max_iters:
                                        # Sampling call, don't recompute already
                                        # known values
        sp.sample_fn(f, fargs=(), fval=fval)
        print('Iter', iter, '='*100)
        print('sp.n_nodes() =', sp.n_nodes())
        if plotting:
            sp.plot(outfile='tmp/sobol_adapt_iter%d.pdf'%iter, labels=labels)
                                        # 2. Compute Sobol indices, up to maximum
                                        # interaction defined by the current K
        D, mu, var = sp.compute_sobol_variances(fval, 
                                                cardinality=K.max_interaction(), 
                                                levelrefine=2)
        del D[()]                       # Remove variance (==var) 

        print('# %6d %12.6e %12.6e' % (sp.n_nodes(), mu, var))
        ### ------------------------------------------- RESULT <==
                                        # 3. Interaction selection
                                        # Sort according to variance large->small
        print('D =', D)
        Dsort = sorted(D.iteritems(), key=lambda (k,v): -v)
        print('Dsort =', Dsort)
        print('var =', var)
        sobol_total,i,U = 0.,0,set([])  # Select most important interactions
        while sobol_total < S_cutoff and i < len(Dsort):
            sobol_total += Dsort[i][1] / var
            U |= set([Dsort[i][0]])
            i += 1
        print('U =', U)
                                        # 4. Interaction augmentation
                                        # Find set of potential *new*
                                        # interactions present in active set
        A = K.activeset()
        potential_interactions = set([interaction(a) for a in A]) - \
                                 K.interactions()
        print('A = ', A)
        print('potential_interactions =', potential_interactions)
                                        # Select potential new interactions 
                                        # satisfying 
        Uplus = set([])
        for interac in potential_interactions:
            all_subsets = set([])
            for r in range(1, len(interac)):
                all_subsets |= set(itertools.combinations(interac, r))
            if all_subsets <= U:
                Uplus |= set([interac])
        U |= Uplus
        print('Uplus =', Uplus)
        print('new U =', U)
                                        # 5. Indexset extension - new sparse grid
        unchanged = True
        for a in A:
            if np.sum(a) > max_k+dim-1: # Enforce simplex-constraint on indexset
                continue
            if np.max(a) > max_level:   # Enforce maximum level constraint
                continue
            if interaction(a) in U:
                unchanged = False
                K.I |= set([a])
        if unchanged:
            print('No new multi-indices added to index-set, terminate adapation')
            break
        print('K.I =', K.I)
        sp.set_indexset(K)
        iter += 1


#===============================================================================
def gerstnerandgriebel_adaptive(f, dim, max_samples=200, max_iters=10,
                                min_error=1.e-16, max_level=10, max_k=10,
                                fval=None, plotting=False, labels=None):
    """
    Gerstner+Griebel style dimension-adaptive sparse grids.
      f           - function to sample
      dim         - number of input variables/dimensions
      max_samples - termination criteria, maximum allowed samples of f
      max_iters   - termination criteria, maximum adaptation iterations
      max_level   - maximum level allowed in any single variable, ie. don't allow
                    very-high resolution in any direction. Enforce
                    max(multiindex) <= max_level
      max_k       - enforce |multiindex|_1 <= dim + max_k - 1, ie. constrain
                    to simplex-rule of level max_k
      fval        - if samples of f already exist at sparse-grid nodes, pass
                    dictionary containing values - these will be used first
    """
                                        # Initialize with simplex level 1
    K = IndexSet(dim=dim, level=1, fill='simplex')
    quad = QuadratureCC()
    sp = SparseGrid(dim, quad, indexset=K)
    iter, eta = 1, 1e100
    fval = {} if fval is None else fval # Dictionary of function values
                                        # Main adaptation loop 
    while sp.n_nodes() <= max_samples and iter <= max_iters and eta > min_error:
                                        # Sampling call, don't recompute already
                                        # known values
        sp.sample_fn(f, fargs=(), fval=fval)
        print('Iter', iter, '='*100)
        print('sp.n_nodes() =', sp.n_nodes())
        if plotting:
            sp.plot(outfile='tmp/GandG_adapt_iter%d.png'%iter, labels=labels)

        r = sp.integrate(fval)
        print('r =', r)
                                        # For each member of the active set,
                                        # compute the difference between the 
                                        # objective, with and without that member
        A = K.activeset()
        g = {}
        for a in A:
            if np.sum(a) > max_k+dim-1: # Enforce simplex-constraint on indexset
                continue
            if np.max(a) > max_level:   # Enforce maximum level constraint
                continue
            Kmod = copy.deepcopy(K)
            Kmod.I |= set([a])
            sp.set_indexset(Kmod)
            sp.sample_fn(f, fargs=(), fval=fval)
            rmod = sp.integrate(fval)
            g[a] = abs(rmod - r)            
        if len(g) == 0:
            print('No new multi-indices added to index-set, terminate adapation')
            break
        a_adapt = max(g, key=g.get)
        eta = sum(g.values())
        print('eta =',eta)
        K.I |= set([a_adapt])
        sp.set_indexset(K)
        iter += 1

    return r
