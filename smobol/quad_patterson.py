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
import numpy as np
from quad_nested import QuadratureNested


class QuadraturePatterson(QuadratureNested):
    """
    Gauss-Patterson quadrature rules in 1d.  Reference example of nested rules,
    only those orders are defined which use *all* points from the lower-order rule.
    Defined on [0,1], does not include interval end-points (though gets very
    close).  All levels are initialized and maintained by this class.

    Levels are indexed from 0 with, 0 the lowest-order rule (nnodes_0=1).  The
    number of nodes in subsequent rules follows nnodes_{n+1} = 2*nnodes_n+1, as
    each interval in the previous level recieves a new node.

    Nodes of the rules are indexes w.r.t. a "global grid" of the highest order
    rule, using a "global index".  E.g. the global-indices of the points in the
    2nd-highest rule are [1, 3, 5, ..., 29], indicating the nodes of this rule
    are every-other node of the finest rule.  This info is stored in self.idx.

    Members:
      x     - nodes on [0,1] per level, list of ndarrays
      w     - weights per level, list of ndarrays
      dw    - weights of rule \Delta used in Smolyak construction
      idx   - indexes of nodes wrt the highest-order rule available,
              handy when reusing data, list of array(int).  Hence 
              x[i] == x[-1][idx[i]]
      fdata - stored evals from most recent call of integrate()
    """

    def init_idx(self):
        nlevel = self.nlevel()
        maxidx = self.maxnnode()
        self.idx = []
        for level in range(nlevel):
            defc = nlevel - level - 1  # = 0 for top level, decreasing
            self.idx.append(np.arange(2 ** defc - 1, maxidx, 2 ** defc))

    def init_rule(self):
        """Gauss-Patterson nodes and weights on [0,1] (rescaling at end of fn)"""
        self.x = [np.array([0.])]  # 1-node rule (level 0)
        self.w = [np.array([2.])]

        self.x.append(
            np.array(  # 3-node rule (level 1)
                [-0.77459666924148337704, 0.0, 0.77459666924148337704]
            )
        )
        self.w.append(
            np.array(
                [
                    0.555555555555555555556,
                    0.888888888888888888889,
                    0.555555555555555555556,
                ]
            )
        )

        self.x.append(
            np.array(  # 7-node rule (level 2)
                [
                    -0.96049126870802028342,
                    -0.77459666924148337704,
                    -0.43424374934680255800,
                    0.0,
                    0.43424374934680255800,
                    0.77459666924148337704,
                    0.96049126870802028342,
                ]
            )
        )
        self.w.append(
            np.array(
                [
                    0.104656226026467265194,
                    0.268488089868333440729,
                    0.401397414775962222905,
                    0.450916538658474142345,
                    0.401397414775962222905,
                    0.268488089868333440729,
                    0.104656226026467265194,
                ]
            )
        )

        self.x.append(
            np.array(  # 15-node rule (level 3)
                [
                    -0.99383196321275502221,
                    -0.96049126870802028342,
                    -0.88845923287225699889,
                    -0.77459666924148337704,
                    -0.62110294673722640294,
                    -0.43424374934680255800,
                    -0.22338668642896688163,
                    0.0,
                    0.22338668642896688163,
                    0.43424374934680255800,
                    0.62110294673722640294,
                    0.77459666924148337704,
                    0.88845923287225699889,
                    0.96049126870802028342,
                    0.99383196321275502221,
                ]
            )
        )
        self.w.append(
            np.array(
                [
                    0.0170017196299402603390,
                    0.0516032829970797396969,
                    0.0929271953151245376859,
                    0.134415255243784220360,
                    0.171511909136391380787,
                    0.200628529376989021034,
                    0.219156858401587496404,
                    0.225510499798206687386,
                    0.219156858401587496404,
                    0.200628529376989021034,
                    0.171511909136391380787,
                    0.134415255243784220360,
                    0.0929271953151245376859,
                    0.0516032829970797396969,
                    0.0170017196299402603390,
                ]
            )
        )

        self.x.append(
            np.array(  # 31-node rule (level 4)
                [
                    -0.99909812496766759766,
                    -0.99383196321275502221,
                    -0.98153114955374010687,
                    -0.96049126870802028342,
                    -0.92965485742974005667,
                    -0.88845923287225699889,
                    -0.83672593816886873550,
                    -0.77459666924148337704,
                    -0.70249620649152707861,
                    -0.62110294673722640294,
                    -0.53131974364437562397,
                    -0.43424374934680255800,
                    -0.33113539325797683309,
                    -0.22338668642896688163,
                    -0.11248894313318662575,
                    0.0,
                    0.11248894313318662575,
                    0.22338668642896688163,
                    0.33113539325797683309,
                    0.43424374934680255800,
                    0.53131974364437562397,
                    0.62110294673722640294,
                    0.70249620649152707861,
                    0.77459666924148337704,
                    0.83672593816886873550,
                    0.88845923287225699889,
                    0.92965485742974005667,
                    0.96049126870802028342,
                    0.98153114955374010687,
                    0.99383196321275502221,
                    0.99909812496766759766,
                ]
            )
        )
        self.w.append(
            np.array(
                [
                    0.00254478079156187441540,
                    0.00843456573932110624631,
                    0.0164460498543878109338,
                    0.0258075980961766535646,
                    0.0359571033071293220968,
                    0.0464628932617579865414,
                    0.0569795094941233574122,
                    0.0672077542959907035404,
                    0.0768796204990035310427,
                    0.0857559200499903511542,
                    0.0936271099812644736167,
                    0.100314278611795578771,
                    0.105669893580234809744,
                    0.109578421055924638237,
                    0.111956873020953456880,
                    0.112755256720768691607,
                    0.111956873020953456880,
                    0.109578421055924638237,
                    0.105669893580234809744,
                    0.100314278611795578771,
                    0.0936271099812644736167,
                    0.0857559200499903511542,
                    0.0768796204990035310427,
                    0.0672077542959907035404,
                    0.0569795094941233574122,
                    0.0464628932617579865414,
                    0.0359571033071293220968,
                    0.0258075980961766535646,
                    0.0164460498543878109338,
                    0.00843456573932110624631,
                    0.00254478079156187441540,
                ]
            )
        )

        self.x.append(
            np.array(  # 63-node rule (level 5)
                [
                    -0.99987288812035761194,
                    -0.99909812496766759766,
                    -0.99720625937222195908,
                    -0.99383196321275502221,
                    -0.98868475754742947994,
                    -0.98153114955374010687,
                    -0.97218287474858179658,
                    -0.96049126870802028342,
                    -0.94634285837340290515,
                    -0.92965485742974005667,
                    -0.91037115695700429250,
                    -0.88845923287225699889,
                    -0.86390793819369047715,
                    -0.83672593816886873550,
                    -0.80694053195021761186,
                    -0.77459666924148337704,
                    -0.73975604435269475868,
                    -0.70249620649152707861,
                    -0.66290966002478059546,
                    -0.62110294673722640294,
                    -0.57719571005204581484,
                    -0.53131974364437562397,
                    -0.48361802694584102756,
                    -0.43424374934680255800,
                    -0.38335932419873034692,
                    -0.33113539325797683309,
                    -0.27774982202182431507,
                    -0.22338668642896688163,
                    -0.16823525155220746498,
                    -0.11248894313318662575,
                    -0.056344313046592789972,
                    0.0,
                    0.056344313046592789972,
                    0.11248894313318662575,
                    0.16823525155220746498,
                    0.22338668642896688163,
                    0.27774982202182431507,
                    0.33113539325797683309,
                    0.38335932419873034692,
                    0.43424374934680255800,
                    0.48361802694584102756,
                    0.53131974364437562397,
                    0.57719571005204581484,
                    0.62110294673722640294,
                    0.66290966002478059546,
                    0.70249620649152707861,
                    0.73975604435269475868,
                    0.77459666924148337704,
                    0.80694053195021761186,
                    0.83672593816886873550,
                    0.86390793819369047715,
                    0.88845923287225699889,
                    0.91037115695700429250,
                    0.92965485742974005667,
                    0.94634285837340290515,
                    0.96049126870802028342,
                    0.97218287474858179658,
                    0.98153114955374010687,
                    0.98868475754742947994,
                    0.99383196321275502221,
                    0.99720625937222195908,
                    0.99909812496766759766,
                    0.99987288812035761194,
                ]
            )
        )
        self.w.append(
            np.array(
                [
                    0.000363221481845530659694,
                    0.00126515655623006801137,
                    0.00257904979468568827243,
                    0.00421763044155885483908,
                    0.00611550682211724633968,
                    0.00822300795723592966926,
                    0.0104982469096213218983,
                    0.0129038001003512656260,
                    0.0154067504665594978021,
                    0.0179785515681282703329,
                    0.0205942339159127111492,
                    0.0232314466399102694433,
                    0.0258696793272147469108,
                    0.0284897547458335486125,
                    0.0310735511116879648799,
                    0.0336038771482077305417,
                    0.0360644327807825726401,
                    0.0384398102494555320386,
                    0.0407155101169443189339,
                    0.0428779600250077344929,
                    0.0449145316536321974143,
                    0.0468135549906280124026,
                    0.0485643304066731987159,
                    0.0501571393058995374137,
                    0.0515832539520484587768,
                    0.0528349467901165198621,
                    0.0539054993352660639269,
                    0.0547892105279628650322,
                    0.0554814043565593639878,
                    0.0559784365104763194076,
                    0.0562776998312543012726,
                    0.0563776283603847173877,
                    0.0562776998312543012726,
                    0.0559784365104763194076,
                    0.0554814043565593639878,
                    0.0547892105279628650322,
                    0.0539054993352660639269,
                    0.0528349467901165198621,
                    0.0515832539520484587768,
                    0.0501571393058995374137,
                    0.0485643304066731987159,
                    0.0468135549906280124026,
                    0.0449145316536321974143,
                    0.0428779600250077344929,
                    0.0407155101169443189339,
                    0.0384398102494555320386,
                    0.0360644327807825726401,
                    0.0336038771482077305417,
                    0.0310735511116879648799,
                    0.0284897547458335486125,
                    0.0258696793272147469108,
                    0.0232314466399102694433,
                    0.0205942339159127111492,
                    0.0179785515681282703329,
                    0.0154067504665594978021,
                    0.0129038001003512656260,
                    0.0104982469096213218983,
                    0.00822300795723592966926,
                    0.00611550682211724633968,
                    0.00421763044155885483908,
                    0.00257904979468568827243,
                    0.00126515655623006801137,
                    0.000363221481845530659694,
                ]
            )
        )
        # Rescale from [-1,1] to [0,1]
        for xi in self.x:
            xi += 1.
            xi *= 0.5
        for wi in self.w:
            wi *= 0.5

        # Unit tests


if __name__ == '__main__':
    # Integrals of various orders
    qp = QuadraturePatterson()
    print('%10s %10s %20s' % ('Level', '#nodes', 'Error'))
    for level in range(qp.nlevel()):
        print(
            '%10d %10d %20.10g'
            % (
                level,
                qp.nnode(level),
                qp.integrate(level, lambda x: np.cos(4 * np.pi * x)),
            )
        )
    print('%10s %10s %20.10g' % ('Exact', '', 0.))

    for i, l in enumerate(qp.idx):
        print(l)  # Indices wrt finest rule
        # Should be all 0.
        print(qp.x[i] - qp.global_index_to_x(qp.idx[i]))

    if True:  # Test interpolation

        def f(x):
            return np.cos(4 * np.pi * x)

        l = 5
        xx = np.linspace(-0.1, 1.1, 101)
        fq = np.array([qp.interpolate(x, l, f) for x in xx])
        ff = np.array([f(x) for x in xx])
        fp = np.array([f(x) for x in qp.x[l]])

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xx, ff, '-k')
        ax.plot(xx, fq, '-r')
        ax.plot(qp.x[l], fp, 'sk')
        fig.savefig('int_patterson.pdf')