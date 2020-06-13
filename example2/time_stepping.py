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

from petsc4py import PETSc
from multiphenics.fem import block_assemble, BlockDirichletBC
from multiphenics.function import block_assign
from multiphenics.la import BlockPETScMatrix, BlockPETScVector
from petsc_ts_integrator import PETScTSIntegrator

class TimeStepping(object):
    def __init__(self, problem_wrapper, solution, solution_dot):
        ic = problem_wrapper.ic_eval()
        if ic is not None:
            block_assign(solution, ic)
        self.problem = _TimeDependentProblem(problem_wrapper.residual_eval, solution, solution_dot, problem_wrapper.bc_eval, problem_wrapper.jacobian_eval, problem_wrapper.set_time)
        self.solver = PETScTSIntegrator(self.problem, solution, solution_dot)
        self.solver.monitor.monitor_callback = _BlockMonitor(problem_wrapper.monitor)
        # Set default linear solver
        self.set_parameters({
            "linear_solver": "default"
        })

    def set_parameters(self, parameters):
        self.solver.set_parameters(parameters)

    def solve(self):
        self.solver.solve()
        #Meen
        return self.solver.solution

class _TimeDependentProblem(object):
    def __init__(self, residual_eval, solution, solution_dot, bc_eval, jacobian_eval, set_time):
        # Store input arguments
        self.residual_eval = residual_eval
        self.solution = solution
        self.solution_dot = solution_dot
        self.bc_eval = bc_eval
        self.jacobian_eval = jacobian_eval
        self.set_time = set_time
        # Make sure that block residual vector and block jacobian matrix are properly initialized
        self.residual_vector = self._residual_vector_assemble(self.residual_eval(0., self.solution, self.solution_dot))
        self.jacobian_matrix = self._jacobian_matrix_assemble(self.jacobian_eval(0., self.solution, self.solution_dot, 0.))

    def residual_vector_eval(self, ts, t, petsc_solution, petsc_solution_dot, petsc_residual):
        """
           TSSetIFunction - Set the function to compute F(t,U,U_t) where F() = 0 is the DAE to be solved.

           Logically Collective on TS

           Input Parameters:
                +  ts  - the TS context obtained from TSCreate()
                .  r   - vector to hold the residual (or NULL to have it created internally)
                .  f   - the function evaluation routine
                -  ctx - user-defined context for private data for the function evaluation routine (may be NULL)

           Calling sequence of f:
                $  f(TS ts,PetscReal t,Vec u,Vec u_t,Vec F,ctx);

                +  t   - time at step/stage being solved
                .  u   - state vector
                .  u_t - time derivative of state vector
                .  F   - function vector
                -  ctx - [optional] user-defined context for matrix evaluation routine

           (from PETSc/src/ts/interface/ts.c)
        """
        # 1. Store solution and solution_dot in multiphenics data structures, as well as current time
        self.set_time(t)
        self.update_solution(petsc_solution)
        self.update_solution_dot(petsc_solution_dot)
        # 2. Assemble the block residual
        self._residual_vector_assemble(self.residual_eval(t, self.solution, self.solution_dot), petsc_residual)
        # 3. Apply boundary conditions
        bcs = self.bc_eval(t)
        self._residual_bcs_apply(bcs)

    def _residual_vector_assemble(self, residual_block_form, petsc_residual=None):
        if petsc_residual is None:
            return block_assemble(residual_block_form)
        else:
            self.residual_vector = BlockPETScVector(petsc_residual)
            block_assemble(residual_block_form, block_tensor=self.residual_vector)

    def _residual_bcs_apply(self, bcs):
        if bcs is not None:
            bcs.apply(self.residual_vector, self.solution.block_vector())

    def jacobian_matrix_eval(self, ts, t, petsc_solution, petsc_solution_dot, solution_dot_coefficient, petsc_jacobian, petsc_preconditioner):
        """
           TSSetIJacobian - Set the function to compute the matrix dF/dU + a*dF/dU_t where F(t,U,U_t) is the function
                            provided with TSSetIFunction().

           Logically Collective on TS

           Input Parameters:
                +  ts  - the TS context obtained from TSCreate()
                .  Amat - (approximate) jacobian matrix
                .  Pmat - matrix used to compute preconditioner (usually the same as Amat)
                .  f   - the jacobian evaluation routine
                -  ctx - user-defined context for private data for the bjacobian evaluation routine (may be NULL)

           Calling sequence of f:
                $  f(TS ts,PetscReal t,Vec U,Vec U_t,PetscReal a,Mat Amat,Mat Pmat,void *ctx);

                +  t    - time at step/stage being solved
                .  U    - state vector
                .  U_t  - time derivative of state vector
                .  a    - shift
                .  Amat - (approximate) jacobian of F(t,U,W+a*U), equivalent to dF/dU + a*dF/dU_t
                .  Pmat - matrix used for constructing preconditioner, usually the same as Amat
                -  ctx  - [optional] user-defined context for matrix evaluation routine

           Notes:
           The matrices Amat and Pmat are exactly the matrices that are used by SNES for the nonlinear solve.

           If you know the operator Amat has a null space you can use MatSetNullSpace() and MatSetTransposeNullSpace()
           to supply the null space to Amat and the KSP solvers will automatically use that null space
           as needed during the solution process.

           The matrix dF/dU + a*dF/dU_t you provide turns out to be
           the jacobian of F(t,U,W+a*U) where F(t,U,U_t) = 0 is the DAE to be solved.
           The time integrator internally approximates U_t by W+a*U where the positive "shift"
           a and vector W depend on the integration method, step size, and past states. For example with
           the backward Euler method a = 1/dt and W = -a*U(previous timestep) so
           W + a*U = a*(U - U(previous timestep)) = (U - U(previous timestep))/dt

           (from PETSc/src/ts/interface/ts.c)
        """
        # 1. There is no need to store solution and solution_dot in multiphenics data structures, nor current time,
        #    since this has already been done by the residual
        # 2. Assemble the block jacobian
        assert petsc_jacobian == petsc_preconditioner
        self._jacobian_matrix_assemble(self.jacobian_eval(t, self.solution, self.solution_dot, solution_dot_coefficient), petsc_jacobian)
        # 3. Apply boundary conditions
        bcs = self.bc_eval(t)
        self._jacobian_bcs_apply(bcs)

    def _jacobian_matrix_assemble(self, jacobian_block_form, petsc_jacobian=None):
        if petsc_jacobian is None:
            return block_assemble(jacobian_block_form)
        else:
            self.jacobian_matrix = BlockPETScMatrix(petsc_jacobian)
            block_assemble(jacobian_block_form, block_tensor=self.jacobian_matrix)

    def _jacobian_bcs_apply(self, bcs):
        if bcs is not None:
            bcs.apply(self.jacobian_matrix)

    def update_solution(self, petsc_solution):
        petsc_solution.ghostUpdate()
        self.solution.block_vector().zero()
        self.solution.block_vector().add_local(petsc_solution.getArray())
        self.solution.block_vector().apply("add")
        self.solution.apply("to subfunctions")

    def update_solution_dot(self, petsc_solution_dot):
        petsc_solution_dot.ghostUpdate()
        self.solution_dot.block_vector().zero()
        self.solution_dot.block_vector().add_local(petsc_solution_dot.getArray())
        self.solution_dot.block_vector().apply("add")
        self.solution_dot.apply("to subfunctions")

class _BlockMonitor(object):
    def __init__(self, monitor_callback):
        self.monitor_callback = monitor_callback

    def __call__(self, t, solution, solution_dot):
        # Apply solution and solution_dot to subfunctions before calling monitor
        solution.apply("to subfunctions")
        solution_dot.apply("to subfunctions")
        self.monitor_callback(t, solution, solution_dot)
