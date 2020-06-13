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
from dolfin import assemble, Identity, Mesh, Constant, DOLFIN_EPS, dx, Expression, FunctionSpace, grad, inner, IntervalMesh, PETScOptions, pi, plot, project, parameters, UnitSquareMesh, ds, dS, SubDomain, MeshFunction, near, Measure, VectorFunctionSpace, TensorFunctionSpace, FacetNormal, CellVolume, CellDiameter, FacetArea, FacetNormal, avg, jump, Function, interpolate, dot, sqrt, XDMFFile
from multiphenics import as_backend_type,assign, block_assemble, block_derivative, BlockDirichletBC, BlockFunction, BlockFunctionSpace, block_split, BlockTestFunction, BlockTrialFunction, DirichletBC
#import matplotlib
#import matplotlib.pyplot as plt
from time_stepping import TimeStepping
from utils_function import *
from post_processing import *


# ~~~ Block case ~~~ #
def transport_linear(integrator_type, mesh, subdomains, boundaries, t_start, dt, T, solution0, \
                 alpha_0, K_0, mu_l_0, lmbda_l_0, Ks_0, \
                 alpha_1, K_1, mu_l_1, lmbda_l_1, Ks_1, \
                 alpha, K, mu_l, lmbda_l, Ks, \
                 cf_0, phi_0, rho_0, mu_0, k_0,\
                 cf_1, phi_1, rho_1, mu_1, k_1,\
                 cf, phi, rho, mu, k, \
                 d_0, d_1, d_t,
                 vel_c, p_con, A_0, Temp, c_extrapolate):
    # Create mesh and define function space
    parameters["ghost_mode"] = "shared_facet" # required by dS

    dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    dS = Measure('dS', domain=mesh, subdomain_data=boundaries)

    C_cg = FiniteElement("CG", mesh.ufl_cell(), 1)
    C_dg = FiniteElement("DG", mesh.ufl_cell(), 0)
    mini = C_cg+C_dg
    C = FunctionSpace(mesh, mini)
    C = BlockFunctionSpace([C])
    TM = TensorFunctionSpace(mesh, 'DG', 0)
    PM = FunctionSpace(mesh, 'DG', 0)
    n = FacetNormal(mesh)
    vc = CellVolume(mesh)
    fc = FacetArea(mesh)

    h = vc/fc
    h_avg = (vc('+') + vc('-'))/(2*avg(fc))

    penalty1 = Constant(1.0)

    tau = Function(PM)
    tau = tau_cal(tau,phi,-0.5)

    tuning_para = 0.25

    vel_norm = (dot(vel_c, n) + abs(dot(vel_c, n)))/2.0

    cell_size = CellDiameter(mesh)
    vnorm = sqrt(dot(vel_c, vel_c))

    I = Identity(mesh.topology().dim())
    d_eff = Function(TM)
    d_eff = diff_coeff_cal_rev(d_eff,d_0,tau,phi) + tuning_para*cell_size*vnorm*I

    monitor_dt = dt

    # Define variational problem
    dc, = BlockTrialFunction(C)
    dc_dot, = BlockTrialFunction(C)
    psic, = BlockTestFunction(C)
    block_c = BlockFunction(C)
    c, = block_split(block_c)
    block_c_dot = BlockFunction(C)
    c_dot, = block_split(block_c_dot)

    theta = -1.0

    a_time = phi*rho*inner(c_dot,psic)*dx

    a_dif = dot(rho*d_eff*grad(c),grad(psic))*dx \
        - dot(avg_w(rho*d_eff*grad(c),weight_e(rho*d_eff,n)), jump(psic, n))*dS \
        + theta*dot(avg_w(rho*d_eff*grad(psic),weight_e(rho*d_eff,n)), jump(c, n))*dS \
        + penalty1/h_avg*k_e(rho*d_eff,n)*dot(jump(c, n), jump(psic, n))*dS

    a_adv = -dot(rho*vel_c*c,grad(psic))*dx \
        + dot(jump(psic), rho('+')*vel_norm('+')*c('+') - rho('-')*vel_norm('-')*c('-') )*dS \
        + dot(psic, rho*vel_norm*c)*ds(3)

    R_c = R_c_cal(c_extrapolate, p_con, Temp)
    c_D1 = Constant(0.5)
    rhs_c = R_c*A_s_cal(phi, phi_0, A_0)*psic*dx - dot(rho*phi*vel_c,n)*c_D1*psic*ds(1)

    r_u = [a_dif + a_adv]
    j_u = block_derivative(r_u, [c], [dc])

    r_u_dot = [a_time]
    j_u_dot = block_derivative(r_u_dot, [c_dot], [dc_dot])
    r = [r_u_dot[0] + r_u[0] - rhs_c]

    # this part is not applied.
    exact_solution_expression1 = Expression("1.0", t=0, element=C[0].ufl_element())
    def bc(t):
        p5 = DirichletBC(C.sub(0), exact_solution_expression1, boundaries, 1, method = "geometric")
        return BlockDirichletBC([p5])

    # Define problem wrapper
    class ProblemWrapper(object):
        def set_time(self, t):
            pass

        # Residual and jacobian functions
        def residual_eval(self, t, solution, solution_dot):
            return r
        def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
            return [[Constant(solution_dot_coefficient)*j_u_dot[0, 0] + j_u[0, 0]]]

        # Define boundary condition
        def bc_eval(self, t):
            pass

        # Define initial condition
        def ic_eval(self):
            return solution0

        # Define custom monitor to plot the solution
        def monitor(self, t, solution, solution_dot):
            pass

    problem_wrapper = ProblemWrapper()
    (solution, solution_dot) = (block_c, block_c_dot)
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
