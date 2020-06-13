# Copyright (C) 2015-2019 by the RBniCS authors
# Copyright (C) 2016-2019 by the multiphenics authors
#
# This file is part of the RBniCS interface to multiphenics.
#
# RBniCS and multiphenics are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and multiphenics are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

#From FB adapted by TK

from numpy import isclose
import numpy as np
from dolfin import Identity, Mesh, Constant, DOLFIN_EPS, dx, Expression, FunctionSpace, grad, inner, IntervalMesh, PETScOptions, pi, plot, project, parameters, UnitSquareMesh, ds, dS, SubDomain, MeshFunction, near, Measure, VectorFunctionSpace, TensorFunctionSpace, FacetNormal, CellVolume, CellDiameter, FacetArea, FacetNormal, avg, jump, Function, interpolate, dot, sqrt, XDMFFile
from multiphenics import block_solve, assign, block_assemble, block_derivative, BlockDirichletBC, BlockFunction, BlockFunctionSpace, block_split, BlockTestFunction, BlockTrialFunction, DirichletBC
#import matplotlib
#import matplotlib.pyplot as plt
#from time_stepping import TimeStepping
from utils_function import *

# ~~~ Block case ~~~ #
def m_linear(integrator_type, mesh, subdomains, boundaries, t_start, dt, T, solution0, \
                 alpha_0, K_0, mu_l_0, lmbda_l_0, Ks_0, \
                 alpha_1, K_1, mu_l_1, lmbda_l_1, Ks_1, \
                 alpha, K, mu_l, lmbda_l, Ks, \
                 cf_0, phi_0, rho_0, mu_0, k_0,\
                 cf_1, phi_1, rho_1, mu_1, k_1,\
                 cf, phi, rho, mu, k, \
                 pressure_freeze):
    # Create mesh and define function space
    parameters["ghost_mode"] = "shared_facet" # required by dS

    dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    dS = Measure('dS', domain=mesh, subdomain_data=boundaries)

    C = VectorFunctionSpace(mesh, "CG", 2)
    C = BlockFunctionSpace([C])
    TM = TensorFunctionSpace(mesh, 'DG', 0)
    PM = FunctionSpace(mesh, 'DG', 0)
    n = FacetNormal(mesh)
    vc = CellVolume(mesh)
    fc = FacetArea(mesh)

    h = vc/fc
    h_avg = (vc('+') + vc('-'))/(2*avg(fc))

    monitor_dt = dt

    f_stress_x = Constant(-1.e3)
    f_stress_y = Constant(-20.0e6)

    f = Constant((0.0, 0.0)) #sink/source for displacement

    I = Identity(mesh.topology().dim())

    # Define variational problem
    psiu, = BlockTestFunction(C)
    block_u = BlockTrialFunction(C)
    u, = block_split(block_u)
    w = BlockFunction(C)

    theta = -1.0

    a_time = inner(-alpha*pressure_freeze*I,sym(grad(psiu)))*dx #quasi static

    a = inner(2*mu_l*strain(u)+lmbda_l*div(u)*I, sym(grad(psiu)))*dx

    rhs_a = inner(f,psiu)*dx \
        + dot(f_stress_y*n,psiu)*ds(2)


    r_u = [a]

    #DirichletBC
    bcd1 = DirichletBC(C.sub(0).sub(0), 0.0, boundaries, 1) # No normal displacement for solid on left side
    bcd3 = DirichletBC(C.sub(0).sub(0), 0.0, boundaries, 3) # No normal displacement for solid on right side
    bcd4 = DirichletBC(C.sub(0).sub(1), 0.0, boundaries, 4) # No normal displacement for solid on bottom side
    bcs = BlockDirichletBC([bcd1,bcd3,bcd4])

    AA = block_assemble([r_u])
    FF = block_assemble([rhs_a - a_time])
    bcs.apply(AA)
    bcs.apply(FF)

    block_solve(AA, w.block_vector(), FF, "mumps")

    export_solution = w

    return export_solution, T
