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
from dolfin import as_vector, inv, assemble, Mesh, Constant, DOLFIN_EPS, dx, Expression, FunctionSpace, grad, inner, IntervalMesh, PETScOptions, pi, plot, project, parameters, UnitSquareMesh, ds, dS, SubDomain, MeshFunction, near, Measure, VectorFunctionSpace, TensorFunctionSpace, FacetNormal, CellVolume, CellDiameter, FacetArea, FacetNormal, avg, jump, Function, interpolate, dot, sqrt, XDMFFile, VectorElement, FiniteElement, Identity, split, sym, div
from multiphenics import block_assign, as_backend_type, assign, block_assemble, block_derivative, BlockDirichletBC, BlockFunction, BlockFunctionSpace, block_split, BlockTestFunction, BlockTrialFunction, DirichletBC
#import matplotlib
#import matplotlib.pyplot as plt
from time_stepping import TimeStepping
from utils_function import *

# ~~~ Block case ~~~ #
def h_linear(integrator_type, mesh, subdomains, boundaries, t_start, dt, T, solution0, \
                 alpha_0, K_0, mu_l_0, lmbda_l_0, Ks_0, \
                 alpha_1, K_1, mu_l_1, lmbda_l_1, Ks_1, \
                 alpha, K, mu_l, lmbda_l, Ks, \
                 cf_0, phi_0, rho_0, mu_0, k_0,\
                 cf_1, phi_1, rho_1, mu_1, k_1,\
                 cf, phi, rho, mu, k, \
                 sigma_v_freeze, dphi_c_dt):
    # Create mesh and define function space
    parameters["ghost_mode"] = "shared_facet" # required by dS


    dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    dS = Measure('dS', domain=mesh, subdomain_data=boundaries)

    BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
    PDG = FiniteElement("DG", mesh.ufl_cell(), 0)

    BDM_F = FunctionSpace(mesh, BDM)
    PDG_F = FunctionSpace(mesh, PDG)

    W = BlockFunctionSpace([BDM_F, PDG_F],
                            restrict=[None, None])

    TM = TensorFunctionSpace(mesh, 'DG', 0)
    PM = FunctionSpace(mesh, 'DG', 0)
    n = FacetNormal(mesh)
    vc = CellVolume(mesh)
    fc = FacetArea(mesh)

    h = vc/fc
    h_avg = (vc('+') + vc('-'))/(2*avg(fc))

    I = Identity(mesh.topology().dim())

    monitor_dt = dt

    p_outlet = 0.1e6
    p_inlet = 1000.0

    M_inv = phi_0*cf + (alpha-phi_0)/Ks

    # Define variational problem
    trial = BlockTrialFunction(W)
    dv, dp = block_split(trial)

    trial_dot = BlockTrialFunction(W)
    dv_dot, dp_dot = block_split(trial_dot)

    test = BlockTestFunction(W)
    psiv, psip = block_split(test)

    block_w = BlockFunction(W)
    v, p = block_split(block_w)

    block_w_dot = BlockFunction(W)
    v_dot, p_dot = block_split(block_w_dot)

    a_time = Constant(0.0)*inner(v_dot,psiv)*dx #quasi static

    # k is a function of phi
    #k = perm_update_rutqvist_newton(p,p0,phi0,phi,coeff)
    lhs_a = inner(dot(v,mu*inv(k)), psiv)*dx - p*div(psiv)*dx #+ 6.0*inner(psiv,n)*ds(2)  # - inner(gravity*(rho-rho0), psiv)*dx

    b_time = (M_inv + pow(alpha,2.)/K)*p_dot*psip*dx

    lhs_b = div(v)*psip*dx #div(rho*v)*psip*dx #TODO rho

    rhs_v = - p_outlet*inner(psiv,n)*ds(3)

    rhs_p = - alpha/K*sigma_v_freeze*psip*dx - dphi_c_dt*psip*dx

    r_u = [lhs_a, lhs_b]

    j_u = block_derivative(r_u, block_w, trial)

    r_u_dot = [a_time, b_time]

    j_u_dot = block_derivative(r_u_dot, block_w_dot, trial_dot)

    r = [r_u_dot[0] + r_u[0] - rhs_v, \
         r_u_dot[1] + r_u[1] - rhs_p]

    def bc(t):
        #bc_v = [DirichletBC(W.sub(0), (.0, .0), boundaries, 4)]
        v1 = DirichletBC(W.sub(0), (1.e-4*2.0, 0.0), boundaries, 1)
        v2 = DirichletBC(W.sub(0), (0.0, 0.0), boundaries, 2)
        v4 = DirichletBC(W.sub(0), (0.0, 0.0), boundaries, 4)
        bc_v = [v1, v2, v4]

        return BlockDirichletBC([bc_v, None])

    # Define problem wrapper
    class ProblemWrapper(object):
        def set_time(self, t):
            pass
            #g.t = t

        # Residual and jacobian functions
        def residual_eval(self, t, solution, solution_dot):
            #print(as_backend_type(assemble(p_time - p_time_error)).vec().norm())
            #print("gravity effect", as_backend_type(assemble(inner(gravity*(rho-rho0), psiv)*dx)).vec().norm())

            return r
        def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
            return [[Constant(solution_dot_coefficient)*j_u_dot[0, 0] + j_u[0, 0], \
                     Constant(solution_dot_coefficient)*j_u_dot[0, 1] + j_u[0, 1]], \
                    [Constant(solution_dot_coefficient)*j_u_dot[1, 0] + j_u[1, 0], \
                     Constant(solution_dot_coefficient)*j_u_dot[1, 1] + j_u[1, 1]]]

        # Define boundary condition
        def bc_eval(self, t):
            return bc(t)

        # Define initial condition
        def ic_eval(self):
            return solution0

        # Define custom monitor to plot the solution
        def monitor(self, t, solution, solution_dot):
            pass


    # Solve the time dependent problem
    problem_wrapper = ProblemWrapper()
    (solution, solution_dot) = (block_w, block_w_dot)
    solver = TimeStepping(problem_wrapper, solution, solution_dot)
    solver.set_parameters({
        "initial_time": t_start,
        "time_step_size": dt,
        "monitor": {
            "time_step_size": monitor_dt,
        },
        "final_time": T,
        "exact_final_time": "stepover",
        "integrator_type": integrator_type,
        "problem_type": "linear",
        "linear_solver": "mumps",
        "report": True
    })
    export_solution = solver.solve()

    return export_solution, T
