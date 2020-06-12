from dolfin import Identity, assemble, Mesh, Constant, DOLFIN_EPS, dx, Expression, FunctionSpace, grad, inner, IntervalMesh, PETScOptions, pi, plot, project, parameters, UnitSquareMesh, ds, dS, SubDomain, MeshFunction, near, Measure, VectorFunctionSpace, TensorFunctionSpace, FacetNormal, CellVolume, CellDiameter, FacetArea, FacetNormal, avg, jump, Function, interpolate, dot, sqrt, XDMFFile, VectorElement, FiniteElement, Identity, split, sym, div
from multiphenics import block_assign, as_backend_type, assign, block_assemble, block_derivative, BlockDirichletBC, BlockFunction, BlockFunctionSpace, block_split, BlockTestFunction, BlockTrialFunction, DirichletBC
from utils_function import *
from post_processing import *
import csv

def initialization(mesh, subdomains, boundaries):

    TM = TensorFunctionSpace(mesh, 'DG', 0)
    PM = FunctionSpace(mesh, 'DG', 0)

    UCG = VectorElement("CG", mesh.ufl_cell(), 2)
    BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
    PDG = FiniteElement("DG", mesh.ufl_cell(), 0)

    UCG_F = FunctionSpace(mesh, UCG)
    BDM_F = FunctionSpace(mesh, BDM)
    PDG_F = FunctionSpace(mesh, PDG)

    W = BlockFunctionSpace([BDM_F, PDG_F],
                            restrict=[None, None])


    U = BlockFunctionSpace([UCG_F])

    I = Identity(mesh.topology().dim())

    C = FunctionSpace(mesh, "DG", 1)
    C = BlockFunctionSpace([C])

    #TODO
    solution0_h = BlockFunction(W)
    solution0_m = BlockFunction(U)
    solution0_c = BlockFunction(C)

    solution1_h = BlockFunction(W)
    solution1_m = BlockFunction(U)
    solution1_c = BlockFunction(C)

    solution2_h = BlockFunction(W)
    solution2_m = BlockFunction(U)
    solution2_c = BlockFunction(C)

    solution_h = BlockFunction(W)
    solution_m = BlockFunction(U)
    solution_c = BlockFunction(C)

    ## mechanics
    # 0 properties
    alpha1 = 0.74
    K1 = 8.4*1000.e6
    nu1 = 0.18

    alpha2 = 0.74
    K2 = 8.4*1000.e6
    nu2 = 0.18

    alpha_values = [alpha1, alpha2]
    K_values = [K1, K2]
    nu_values = [nu1, nu2]

    alpha_0 = Function(PM)
    K_0 = Function(PM)
    nu_0 = Function(PM)

    alpha_0 = init_scalar_parameter(alpha_0,alpha_values[0],500,subdomains)
    K_0 = init_scalar_parameter(K_0,K_values[0],500,subdomains)
    nu_0 = init_scalar_parameter(nu_0,nu_values[0],500,subdomains)

    alpha_0 = init_scalar_parameter(alpha_0,alpha_values[1],501,subdomains)
    K_0 = init_scalar_parameter(K_0,K_values[1],501,subdomains)
    nu_0 = init_scalar_parameter(nu_0,nu_values[1],501,subdomains)

    K_mult_min = 1.0
    K_mult_max = 1.0

    mu_l_0, lmbda_l_0, Ks_0, K_0 = \
    bulk_modulus_update(mesh,solution0_c[0],K_mult_min,K_mult_max,K_0,nu_0,alpha_0,K_0)

    # n-1 properties
    alpha1 = 0.74
    K1 = 8.4*1000.e6
    nu1 = 0.18

    alpha2 = 0.74
    K2 = 8.4*1000.e6
    nu2 = 0.18

    alpha_values = [alpha1, alpha2]
    K_values = [K1, K2]
    nu_values = [nu1, nu2]

    alpha_1 = Function(PM)
    K_1 = Function(PM)
    nu_1 = Function(PM)

    alpha_1 = init_scalar_parameter(alpha_1,alpha_values[0],500,subdomains)
    K_1 = init_scalar_parameter(K_1,K_values[0],500,subdomains)
    nu_1 = init_scalar_parameter(nu_1,nu_values[0],500,subdomains)

    alpha_1 = init_scalar_parameter(alpha_1,alpha_values[1],501,subdomains)
    K_1 = init_scalar_parameter(K_1,K_values[1],501,subdomains)
    nu_1 = init_scalar_parameter(nu_1,nu_values[1],501,subdomains)

    K_mult_min = 1.0
    K_mult_max = 1.0

    mu_l_1, lmbda_l_1, Ks_1, K_1 = \
    bulk_modulus_update(mesh,solution0_c[0],K_mult_min,K_mult_max,K_1,nu_1,alpha_1,K_0)


    # n properties
    alpha1 = 0.74
    K2 = 8.4*1000.e6
    nu1 = 0.18

    alpha2 = 0.74
    K2 = 8.4*1000.e6
    nu2 = 0.18

    alpha_values = [alpha1, alpha2]
    K_values = [K1, K2]
    nu_values = [nu1, nu2]

    alpha = Function(PM)
    K = Function(PM)
    nu = Function(PM)

    alpha = init_scalar_parameter(alpha,alpha_values[0],500,subdomains)
    K = init_scalar_parameter(K,K_values[0],500,subdomains)
    nu = init_scalar_parameter(nu,nu_values[0],500,subdomains)

    alpha = init_scalar_parameter(alpha,alpha_values[1],501,subdomains)
    K = init_scalar_parameter(K,K_values[1],501,subdomains)
    nu = init_scalar_parameter(nu,nu_values[1],501,subdomains)

    K_mult_min = 1.0
    K_mult_max = 1.0

    mu_l, lmbda_l, Ks, K = \
    bulk_modulus_update(mesh,solution0_c[0],K_mult_min,K_mult_max,K,nu,alpha,K_0)


    ## flow
    # 0 properties
    cf1 = 1e-10
    phi1 = 0.2
    rho1 = 1000.0
    mu1 = 1.

    kx = 1.e-6
    ky = 1.e-6
    k1 = np.array([kx, 0.,0., ky])

    cf2= 1e-10
    phi2 = 0.2
    rho2 = 1000.0
    mu2 = 1.

    kx = 1.e-5
    ky = 1.e-5
    k2 = np.array([kx, 0.,0., ky])

    cf_values = [cf1, cf2]
    phi_values = [phi1, phi2]
    rho_values = [rho1, rho2]
    mu_values = [mu1, mu2]

    k_values = [k1, k2]

    cf_0 = Function(PM)
    phi_0 = Function(PM)
    rho_0 = Function(PM)
    mu_0 = Function(PM)

    k_0 = Function(TM)

    cf_0 = init_scalar_parameter(cf_0,cf_values[0],500,subdomains)
    phi_0 = init_scalar_parameter(phi_0,phi_values[0],500,subdomains)
    rho_0 = init_scalar_parameter(rho_0,rho_values[0],500,subdomains)
    mu_0 = init_scalar_parameter(mu_0,mu_values[0],500,subdomains)

    #k_0 = init_tensor_parameter(k_0,k_values[0],500,subdomains,mesh.topology().dim())

    cf_0 = init_scalar_parameter(cf_0,cf_values[1],501,subdomains)
    phi_0 = init_scalar_parameter(phi_0,phi_values[1],501,subdomains)
    rho_0 = init_scalar_parameter(rho_0,rho_values[1],501,subdomains)
    mu_0 = init_scalar_parameter(mu_0,mu_values[1],501,subdomains)

    #k_0 = init_tensor_parameter(k_0,k_values[1],501,subdomains,mesh.topology().dim())
    filename = "perm4.csv"
    k_0 = init_from_file_parameter(k_0,0.,0.,filename)

    # n-1 properties
    cf1 = 1e-10
    phi1 = 0.2
    rho1 = 1000.0
    mu1 = 1.

    kx = 1.e-6
    ky = 1.e-6
    k1 = np.array([kx, 0.,0., ky])

    cf2= 1e-10
    phi2 = 0.2
    rho2 = 1000.0
    mu2 = 1.

    kx = 1.e-5
    ky = 1.e-5
    k2 = np.array([kx, 0.,0., ky])

    cf_values = [cf1, cf2]
    phi_values = [phi1, phi2]
    rho_values = [rho1, rho2]
    mu_values = [mu1, mu2]

    k_values = [k1, k2]

    cf_1 = Function(PM)
    phi_1 = Function(PM)
    rho_1 = Function(PM)
    mu_1 = Function(PM)

    k_1 = Function(TM)

    cf_1 = init_scalar_parameter(cf_1,cf_values[0],500,subdomains)
    phi_1 = init_scalar_parameter(phi_1,phi_values[0],500,subdomains)
    rho_1 = init_scalar_parameter(rho_1,rho_values[0],500,subdomains)
    mu_1 = init_scalar_parameter(mu_1,mu_values[0],500,subdomains)

    #k_1 = init_tensor_parameter(k_1,k_values[0],500,subdomains,mesh.topology().dim())

    cf_1 = init_scalar_parameter(cf_1,cf_values[1],501,subdomains)
    phi_1 = init_scalar_parameter(phi_1,phi_values[1],501,subdomains)
    rho_1 = init_scalar_parameter(rho_1,rho_values[1],501,subdomains)
    mu_1 = init_scalar_parameter(mu_1,mu_values[1],501,subdomains)

    #k_1 = init_tensor_parameter(k_1,k_values[1],501,subdomains,mesh.topology().dim())
    filename = "perm4.csv"
    k_1 = init_from_file_parameter(k_1,0.,0.,filename)

    # n properties
    cf1 = 1e-10
    phi1 = 0.2
    rho1 = 1000.0
    mu1 = 1.

    kx = 1.e-6
    ky = 1.e-6
    k1 = np.array([kx, 0.,0., ky])

    cf2= 1e-10
    phi2 = 0.2
    rho2 = 1000.0
    mu2 = 1.

    kx = 1.e-5
    ky = 1.e-5
    k2 = np.array([kx, 0.,0., ky])

    cf_values = [cf1, cf2]
    phi_values = [phi1, phi2]
    rho_values = [rho1, rho2]
    mu_values = [mu1, mu2]

    k_values = [k1, k2]

    cf = Function(PM)
    phi = Function(PM)
    rho = Function(PM)
    mu = Function(PM)

    k = Function(TM)

    cf = init_scalar_parameter(cf,cf_values[0],500,subdomains)
    phi = init_scalar_parameter(phi,phi_values[0],500,subdomains)
    rho = init_scalar_parameter(rho,rho_values[0],500,subdomains)
    mu = init_scalar_parameter(mu,mu_values[0],500,subdomains)

    #k = init_tensor_parameter(k,k_values[0],500,subdomains,mesh.topology().dim())

    cf = init_scalar_parameter(cf,cf_values[1],501,subdomains)
    phi = init_scalar_parameter(phi,phi_values[1],501,subdomains)
    rho = init_scalar_parameter(rho,rho_values[1],501,subdomains)
    mu = init_scalar_parameter(mu,mu_values[1],501,subdomains)

    #k = init_tensor_parameter(k,k_values[1],501,subdomains,mesh.topology().dim())
    filename = "perm4.csv"
    k = init_from_file_parameter(k,0.,0.,filename)

    ### transport
    # 0
    dx1 = 1e-12
    dy1 = 1e-12
    d1 = np.array([dx1, 0.,0., dy1])
    dx2 = 1e-12
    dy2 = 1e-12
    d2 = np.array([dx2, 0.,0., dy2])
    d_values = [d1, d2]

    d_0 = Function(TM)
    d_0 = init_tensor_parameter(d_0,d_values[0],500,subdomains,mesh.topology().dim())
    d_0 = init_tensor_parameter(d_0,d_values[1],501,subdomains,mesh.topology().dim())

    # n-1
    dx1 = 1e-12
    dy1 = 1e-12
    d1 = np.array([dx1, 0.,0., dy1])
    dx2 = 1e-12
    dy2 = 1e-12
    d2 = np.array([dx2, 0.,0., dy2])
    d_values = [d1, d2]

    d_1 = Function(TM)
    d_1 = init_tensor_parameter(d_1,d_values[0],500,subdomains,mesh.topology().dim())
    d_1 = init_tensor_parameter(d_1,d_values[1],501,subdomains,mesh.topology().dim())

    # n
    dx1 = 1e-12
    dy1 = 1e-12
    d1 = np.array([dx1, 0.,0., dy1])
    dx2 = 1e-12
    dy2 = 1e-12
    d2 = np.array([dx2, 0.,0., dy2])
    d_values = [d1, d2]

    d = Function(TM)
    d = init_tensor_parameter(d,d_values[0],500,subdomains,mesh.topology().dim())
    d = init_tensor_parameter(d,d_values[1],501,subdomains,mesh.topology().dim())

    ####initialization
    # initial
    u_0 = Constant((0.0, 0.0))
    u_0_project = project(u_0, U[0])
    assign(solution0_m.sub(0), u_0_project)

    p_0 = Constant(1.e6)
    p_0_project = project(p_0, W[1])
    assign(solution0_h.sub(1), p_0_project)

    # v_0 = Constant((0.0, 0.0))
    # v_0_project = project(v_0, W[0])
    # assign(solution0_h.sub(0), v_0_project)

    c0 = c_sat_cal(1.e6, 20.)
    c0_project = project(c0, C[0])
    assign(solution0_c.sub(0), c0_project)

    # n - 1
    u_0 = Constant((0.0, 0.0))
    u_0_project = project(u_0, U[0])
    assign(solution1_m.sub(0), u_0_project)

    p_0 = Constant(1.e6)
    p_0_project = project(p_0, W[1])
    assign(solution1_h.sub(1), p_0_project)

    # v_0 = Constant((0.0, 0.0))
    # v_0_project = project(v_0, W[0])
    # assign(solution1_h.sub(0), v_0_project)

    c0 = c_sat_cal(1.e6, 20.)
    c0_project = project(c0, C[0])
    assign(solution1_c.sub(0), c0_project)

    # n - 2
    u_0 = Constant((0.0, 0.0))
    u_0_project = project(u_0, U[0])
    assign(solution2_m.sub(0), u_0_project)

    p_0 = Constant(1.e6)
    p_0_project = project(p_0, W[1])
    assign(solution2_h.sub(1), p_0_project)

    # v_0 = Constant((0.0, 0.0))
    # v_0_project = project(v_0, W[0])
    # assign(solution2_h.sub(0), v_0_project)

    c0 = c_sat_cal(1.e6, 20.)
    c0_project = project(c0, C[0])
    assign(solution2_c.sub(0), c0_project)

    # n
    u_0 = Constant((0.0, 0.0))
    u_0_project = project(u_0, U[0])
    assign(solution_m.sub(0), u_0_project)

    p_0 = Constant(1.e6)
    p_0_project = project(p_0, W[1])
    assign(solution_h.sub(1), p_0_project)

    # v_0 = Constant((0.0, 0.0))
    # v_0_project = project(v_0, W[0])
    # assign(solution_h.sub(0), v_0_project)

    c0 = c_sat_cal(1.e6, 20.)
    c0_project = project(c0, C[0])
    assign(solution_c.sub(0), c0_project)

    ###iterative parameters
    phi_it = Function(PM)
    assign(phi_it, phi_0)

    print('c_sat',c_sat_cal(1.0e8, 20.))

    c_sat = c_sat_cal(1.0e8, 20.)
    c_sat = project(c_sat, PM)
    c_inject = Constant(0.0)
    c_inject = project(c_inject, PM)

    mu_c1_1 = 1.e-4
    mu_c2_1 = 5.e-0
    mu_c1_2 = 1.e-4
    mu_c2_2 = 5.e-0
    mu_c1_values = [mu_c1_1,mu_c1_2]
    mu_c2_values = [mu_c2_1,mu_c2_2]

    mu_c1 = Function(PM)
    mu_c2 = Function(PM)

    mu_c1 = init_scalar_parameter(mu_c1,mu_c1_values[0],500,subdomains)
    mu_c2 = init_scalar_parameter(mu_c2,mu_c2_values[0],500,subdomains)

    mu_c1 = init_scalar_parameter(mu_c1,mu_c1_values[1],501,subdomains)
    mu_c2 = init_scalar_parameter(mu_c2,mu_c2_values[1],501,subdomains)

    coeff_for_perm_1 = 22.2
    coeff_for_perm_2 = 22.2

    coeff_for_perm_values = [coeff_for_perm_1, coeff_for_perm_2]

    coeff_for_perm = Function(PM)

    coeff_for_perm = init_scalar_parameter(coeff_for_perm,coeff_for_perm_values[0],500,subdomains)
    coeff_for_perm = init_scalar_parameter(coeff_for_perm,coeff_for_perm_values[1],501,subdomains)

    solutionIt_h = BlockFunction(W)

    return solution0_m, solution0_h, solution0_c \
    ,solution1_m, solution1_h, solution1_c \
    ,solution2_m, solution2_h, solution2_c \
    ,solution_m, solution_h, solution_c  \
    ,alpha_0, K_0, mu_l_0, lmbda_l_0, Ks_0 \
    ,alpha_1, K_1, mu_l_1, lmbda_l_1, Ks_1 \
    ,alpha, K, mu_l, lmbda_l, Ks \
    ,cf_0, phi_0, rho_0, mu_0, k_0 \
    ,cf_1, phi_1, rho_1, mu_1, k_1 \
    ,cf, phi, rho, mu, k \
    ,d_0, d_1, d, I \
    ,phi_it, solutionIt_h, mu_c1, mu_c2 \
    ,nu_0, nu_1, nu, coeff_for_perm \
    ,c_sat, c_inject
