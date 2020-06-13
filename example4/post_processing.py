import numpy as np
from dolfin import ln, exp, Mesh, Constant, DOLFIN_EPS, dx, Expression, FunctionSpace, grad, inner, IntervalMesh, PETScOptions, pi, plot, project, parameters, UnitSquareMesh, ds, dS, SubDomain, MeshFunction, near, Measure, VectorFunctionSpace, TensorFunctionSpace, FacetNormal, CellVolume, CellDiameter, FacetArea, FacetNormal, avg, jump, Function, interpolate, dot, sqrt, XDMFFile
from multiphenics import assign, block_assemble, block_derivative, BlockDirichletBC, BlockFunction, BlockFunctionSpace, block_split, BlockTestFunction, BlockTrialFunction, DirichletBC
#import matplotlib
from utils_function import *

def v_post_processing(mesh, subdomains, boundaries, p, k, mu):
    V_u = VectorFunctionSpace(mesh, "CG", 2)

    return project(-k/mu*grad(p),V_u)

def to_dg0(mesh, subdomains, boundaries, p):
    PM = FunctionSpace(mesh, 'DG', 0)
    return project(p, PM)

def to_dg0(mesh, subdomains, boundaries, p):
    PM = FunctionSpace(mesh, 'DG', 0)
    return project(p, PM)

def porosity_update_from_flow(mesh, phi_1, alpha, K, p, p_1):
    DG0 = FunctionSpace(mesh, "DG", 0)
    porosity_from_flow_expr = phi_1 + (alpha - phi_1)/K * (p - p_1)
    porosity_from_flow = project(porosity_from_flow_expr, DG0)
    return porosity_from_flow

def porosity_update_from_mechanics(mesh, phi_0, alpha, K, u, u_0, p, p_0):
    DG0 = FunctionSpace(mesh, "DG", 0)
    porosity_from_mechanics_expr = phi_0 + (alpha - phi_0)*(vol_strain(u) - vol_strain(u_0)) \
    + (alpha-phi_0)*(1.-alpha)/K*(p - p_0)
    porosity_from_mechanics = project(porosity_from_mechanics_expr, DG0)
    return porosity_from_mechanics

def stress_v_mean_calculation(mesh, u, p, alpha, I, mu_l, lmbda_l):
    stress_v_mean = vol_sigma_total(u,p,alpha,I,mu_l,lmbda_l)/mesh.topology().dim()
    return stress_v_mean

def linear_extrapolation(mesh,p,p_n1,p_n2,p_max,p_min,dt,dt_1,p2_max,p2_min):
    DG0 = FunctionSpace(mesh, "DG", 0)
    p = (1.0+dt/dt_1)*mu_newton_linear_adapt(p_n1,p_min,p_max,p2_min,p2_max) \
    - (dt/dt_1)*mu_newton_linear_adapt(p_n2,p_min,p_max,p2_min,p2_max)
    return project(p, DG0)

def linear_n1(mesh,p,p_n1,p_max,p_min,p2_max,p2_min):
    DG0 = FunctionSpace(mesh, "DG", 0)
    p = mu_newton_linear_adapt(p_n1,p_min,p_max,p2_min,p2_max)
    return project(p, DG0)

def linear_extrapolation_for_c(mesh,p,p_n1,p_n2,dt,dt_1):
    DG1 = FunctionSpace(mesh, "DG", 1)
    p = (1.0+dt/dt_1)*p_n1 \
    - (dt/dt_1)*p_n2
    return project(p, DG1)

def linear_n1_for_c(mesh,p,p_n1):
    DG1 = FunctionSpace(mesh, "DG", 1)
    p = p_n1
    return project(p, DG1)


def project_mu(p):
    DG0 = FunctionSpace(mesh, "DG", 0)
    return project(p,DG0)

def perm_update_rutqvist_newton(mesh,p,p0,phi0,phi,coeff):
    DG0 = TensorFunctionSpace(mesh, 'DG', 0)
    mult_min = 1e-10
    expr = exp(coeff*(phi/phi0-1.0))
    mult = conditional(ge(expr,0.0),expr,mult_min)
    p = p0*mult
    return project(p, DG0)

def perm_update_kk_newton(mesh,p,p0,phi0,phi,coeff):
    DG0 = TensorFunctionSpace(mesh, 'DG', 0)
    mult = pow(phi/phi0,3.0)*pow(((1.0-phi0)/(1.0-phi)),2.0)
    expr = p0*mult
    return project(expr, DG0)

def bulk_modulus_update(mesh,c,K_mult_min,K_mult_max,K,nu,alpha,K_0):
    DG0 = FunctionSpace(mesh, "DG", 0)
    k_mult = bulk_modulus_mult_newton_linear(c, K_mult_min, K_mult_max)
    K = K_0*k_mult

    E = K_nu_to_E(K, nu)
    (mu_l, lmbda_l) = E_nu_to_mu_lmbda(E, nu)
    Ks = Ks_cal_no_isclose(alpha,K)

    mu_l = project(mu_l,DG0)
    lmbda_l = project(lmbda_l,DG0)
    Ks = project(Ks,DG0)
    K = project(K,DG0)

    return mu_l,lmbda_l,Ks, K

def c_sat_cal(p, Temp):

    p = p/1e6
    #p = p*9.8692e-6

    c_1 = 1.417e-3
    c_2 = 3.823e-6
    c_3 = -4.313e-7
    c_4 = -2.148e-8
    c_5 = 4.304e-8
    c_6 = -7.117e-8

    c_sat = c_1 + c_2*p + c_3*Temp \
    + c_4*pow(p,2.0) + c_5*p*Temp + c_6*pow(Temp,2.0)

    return c_sat*1000.


def c_tilda_cal(c, p, Temp):
    c_sat = c_sat_cal(p, Temp)

    return (c_sat - c)/c_sat


def R_c_cal(c, p, Temp):

    c_tilda = c_tilda_cal(c, p, Temp)

    r_temp_c_tilda = r_temp_c_tilda_cal(c_tilda, Temp)

    tol_R_c_cal = 1e-13

    return conditional(ge(c_tilda,tol_R_c_cal),pow(10., r_temp_c_tilda),
                       conditional(ge(c_tilda,-tol_R_c_cal), 0.0, -pow(10., r_temp_c_tilda)))


def r_temp_c_tilda_cal(c_tilda, Temp):

    a_0 = -5.73
    a_1 = 1.25e-2
    a_2 = 1.38
    a_3 = 2.61e-5
    a_4 = -4.01e-3
    a_5 = 3.26e-1

    #con_ln_log10 = 1.
    #con_ln_log10 = 2.3025850929940456840179914546843642076011
    con_ln_log10 = ln(10.)

    r_condition_1 = a_0 + a_1*Temp + a_2*ln(abs(c_tilda))/con_ln_log10\
    + a_3*pow(Temp, 2.0) + a_4*Temp*ln(abs(c_tilda))/con_ln_log10 + a_5*pow(ln(abs(c_tilda))/con_ln_log10, 2.0)

    b_0 = -6.45
    b_1 = 2.09e-2
    b_2 = -4.65e-2
    b_3 = 3.06e-5
    b_4 = 9.25e-3
    b_5 = -4.59e-1

    r_condition_2 = b_0 + b_1*Temp + b_2*ln(abs(c_tilda))/con_ln_log10\
    + b_3*pow(Temp, 2.0) + b_4*Temp*ln(abs(c_tilda))/con_ln_log10 + b_5*pow(ln(abs(c_tilda))/con_ln_log10, 2.0)

    c_0 = -5.80
    c_1 = 1.35e-2
    c_2 = 9.97e-1
    c_3 = 3.80e-5
    c_4 = 1.51e-5
    c_5 = -4.87e-4

    r_condition_3 = c_0 + c_1*Temp + c_2*ln(abs(c_tilda))/con_ln_log10\
    + c_3*pow(Temp, 2.0) + c_4*Temp*ln(abs(c_tilda))/con_ln_log10 + c_5*pow(ln(abs(c_tilda))/con_ln_log10, 2.0)
    return conditional(ge(c_tilda,0.01),r_condition_1,conditional(ge(c_tilda,-0.01),r_condition_3,r_condition_2))



def A_s_cal(phi, phi0, A_0):
    return A_0*phi/phi0*ln(phi)/ln(phi0)


def dphi_dt(phi, phi0, A_0, c, p, Temp, Omega, rho_solid):
    A_s = A_s_cal(phi, phi0, A_0)
    R_c = R_c_cal(c, p, Temp)
    return R_c*A_s/Omega/rho_solid


def dphi_dt_print(mesh, phi, phi0, A_0, c, p, Temp, Omega, rho_solid):
    DG0 = FunctionSpace(mesh, "DG", 0)
    A_s = A_s_cal(phi, phi0, A_0)
    R_c = R_c_cal(c, p, Temp)
    porosity_from_chemical_expr = R_c*A_s/Omega/rho_solid
    return project(porosity_from_chemical_expr, DG0)

def porosity_update_from_chemical(mesh, phi, phi1, phi0, A_0, c, p, Temp, Omega, rho_solid, dt):
    DG0 = FunctionSpace(mesh, "DG", 0)
    #porosity_from_chemical_expr = phi1 + dt*dphi_dt(phi, phi0, A_0, c, p, Temp, Omega, rho_solid)
    porosity_from_chemical_expr = phi1 + rungeKutta(phi, phi0, A_0, c, p, Temp, Omega, rho_solid, dt)
    porosity_from_chemical = project(porosity_from_chemical_expr, DG0)
    return porosity_from_chemical

def rungeKutta(phi, phi0, A_0, c, p, Temp, Omega, rho_solid, h):
    # Count number of iterations using step size or

    #Apply Runge Kutta Formulas to find next value of y
    k1 = h * dphi_dt(phi, phi0, A_0, c, p, Temp, Omega, rho_solid)
    k2 = h * dphi_dt(phi + 0.5 * k1, phi0, A_0, c, p, Temp, Omega, rho_solid)
    k3 = h * dphi_dt(phi + 0.5 * k2, phi0, A_0, c, p, Temp, Omega, rho_solid)
    k4 = h * dphi_dt(phi + k3, phi0, A_0, c, p, Temp, Omega, rho_solid)

    return (1.0 / 6.0)*(k1 + 2.0 * k2 + 2.0 * k3 + k4)

def cal_delta_pm(mesh, p, p1):
    DG0 = FunctionSpace(mesh, "DG", 0)
    return_p = Function(DG0)
    return_p.vector()[:] = p.vector()[:] - p1.vector()[:]
    return return_p

def cal_delta_tm(mesh, p, p1):
    DG0 = TensorFunctionSpace(mesh, "DG", 0)
    return_p = Function(DG0)
    return_p.vector()[:] = p.vector()[:] - p1.vector()[:]
    return return_p


def mass_conservation_cal(mesh, phi_0, cf, alpha, Ks, K, p, p1, dt,
    sigma_v_freeze, dphi_c_dt, v):
    #mass conservation
    PM = FunctionSpace(mesh, 'DG', 0)
    #con = TestFunction(PM)
    M_inv = phi_0*cf + (alpha-phi_0)/Ks
    mass_con_3 = (M_inv + pow(alpha,2.)/K)*(p-p1)/dt \
        + alpha/K*sigma_v_freeze \
        + dphi_c_dt \
        + div(v)

    mass_3 = project(mass_con_3,PM)
    return mass_3
