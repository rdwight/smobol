.. smobol documentation master file, created by
   sphinx-quickstart on Fri Dec  4 12:39:04 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sparse grid and Sobol Index surrogate modelling
===============================================

Sparse grid (Smolyak) and Sobol index code written for uncertainty
quantification in the course of my research. It is available here under the LGPL
in the hope that it will be useful.

Features
--------

- Arbetrary dimension, tested up to 40-dim.
- High-order polynomial integration and interpolation on sparse grids.
- Generalized sparse grid fill patterns (following the "index-set" construction
  of [2]).
- Sobol main-effect indices of arbetrary degree. NB: These are based on the
  sparse-grid surrogate; they will only be as accurate as that is.
- Gauss-Patterson and Clenshaw-Curtis rules (easily extensible for other
  hierarchical rules by adding a new int_*.py file with the same interface).
- The "index-set" construction allows definition of arbetrary sparse-grid
  fill-patterns by hand. Automatic adaptivity should be added in due course.

.. warning::

   CODENAME is not yet at a stage of development where it can be used in anger.
   If you're looking for a working flow-solver see :ref:`label_alternatives`.

Contents
--------

.. toctree::
   :maxdepth: 2


.. _label_alternatives:

Alternatives
------------
Other open-source sparse grid implementations:

- **Smolpack**, Knut Petras (C): [http://people.sc.fsu.edu/~jburkardt/c_src/smolpack/smolpack.html]
- **Sparse_grid**, Jochen Garcke (Python): [http://people.sc.fsu.edu/~jburkardt/py_src/sparse_grid/]
- **SPinterp**, Andreas Klimke (MatLAB): [http://www.ians.uni-stuttgart.de/spinterp/]


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

