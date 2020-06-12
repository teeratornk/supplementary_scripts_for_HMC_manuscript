#from hm_time_stepping_nl import *
#from hm_time_stepping_l import *
from h_time_stepping_l import *
from m_time_stepping_l import *
from post_processing import *
from transport_time_stepping_l import *
from init_of_main import *

PETScOptions.set("ts_bdf_order", "4")
PETScOptions.set("ts_bdf_adapt", "false")
PETScOptions.set("ts_adapt_type", "none")
#PETScOptions.set("ts_theta_theta", "0.5")
#PETScOptions.set("ts_theta_endpoint", "")

mesh = Mesh("test.xml")

subdomains = MeshFunction("size_t",mesh,"test_physical_region.xml")
boundaries = MeshFunction("size_t",mesh,"test_facet_region.xml")

T = 120000
dt = 0.001
t = 0.
t1 = t + dt
n_print = 50
n_count = 0


VEL = VectorFunctionSpace(mesh, "CG", 2)

xdmc = XDMFFile("hm_result/c.xdmf")
xdmp = XDMFFile("hm_result/p.xdmf")
xdmv = XDMFFile("hm_result/v.xdmf")
xdmu = XDMFFile("hm_result/u.xdmf")
xdmphi = XDMFFile("hm_result/phi.xdmf")
xdmmu = XDMFFile("hm_result/mu.xdmf")
xdmperm = XDMFFile("hm_result/perm.xdmf")
xdmK = XDMFFile("hm_result/K.xdmf")
xdmmass = XDMFFile("hm_result/mass.xdmf")

xdmdphi_mech = XDMFFile("hm_result/dphi_mech.xdmf")
xdmdphi_mech_0 = XDMFFile("hm_result/dphi_mech_0.xdmf")
xdmdphi_chem = XDMFFile("hm_result/dphi_chem.xdmf")
xdmdphi_chem_after_mech = XDMFFile("hm_result/dphi_chem_after_mech.xdmf")
xdmdperm_mech = XDMFFile("hm_result/dperm_mech.xdmf")
xdmdperm_mech_0 = XDMFFile("hm_result/dperm_mech_0.xdmf")
xdmdperm_chem = XDMFFile("hm_result/dperm_chem.xdmf")

solution0_m, solution0_h, solution0_c \
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
,phi_it, solutionIt_h, mu_min, mu_max \
,nu_0, nu_1, nu, coeff_for_perm \
,c_sat, c_inject = initialization(mesh, subdomains, boundaries)

solution_m_temp, t_temp = m_linear("beuler", mesh=mesh, subdomains=subdomains, boundaries=boundaries, \
                 t_start=0.0, dt=dt, T=t1, solution0=solution0_m, \
                 alpha_0=alpha_0, K_0=K_0, mu_l_0=mu_l_0, lmbda_l_0=lmbda_l_0, Ks_0=Ks_0, \
                 alpha_1=alpha_1, K_1=K_1, mu_l_1=mu_l_1, lmbda_l_1=lmbda_l_1, Ks_1=Ks_1, \
                 alpha=alpha, K=K, mu_l=mu_l, lmbda_l=lmbda_l, Ks=Ks, \
                 cf_0=cf_0, phi_0=phi_0, rho_0=rho_0, mu_0=mu_0, k_0=k_0,\
                 cf_1=cf_1, phi_1=phi_1, rho_1=rho_1, mu_1=mu_1, k_1=k_1,\
                 cf=cf, phi=phi, rho=rho, mu=mu, k=k, \
                 pressure_freeze=solution0_h[1])

#TODO assign and block_assign are the same?
block_assign(solution0_m, solution_m_temp)
block_assign(solution1_m, solution_m_temp)
block_assign(solution_m, solution_m_temp)

print('v now',np.average(solution_h[0].vector()[:]))
print('p now',np.average(solution_h[1].vector()[:]))
print('u now',np.average(solution_m[0].vector()[:]))


porosity_from_mechanics = porosity_update_from_mechanics(mesh, phi_0, alpha, K, \
                                                         solution_m[0], solution0_m[0], \
                                                         solution_h[1], solution0_h[1])
assign(phi, porosity_from_mechanics)
assign(phi_1, porosity_from_mechanics)
assign(phi_it, porosity_from_mechanics)
tol = 1.e-5
#mu = mu0
err = 10.
first_time = True

# for calcite
A_0 = 5000. #m-1
Temp = 20.
Omega = 10.
rho_solid = 2500.
mixing_flag = True
del_phi_chem = Constant(0.)
while t < T:
    t0 = t
    t = t + dt

    ### fixed-stress loop
    stress_v_mean_n = stress_v_mean_calculation(mesh, solution_m[0],solution_h[1], \
                                              alpha, I, mu_l, lmbda_l)
    stress_v_mean_n1 = stress_v_mean_calculation(mesh, solution1_m[0],solution1_h[1], \
                                              alpha, I, mu_l, lmbda_l)
    stress_v_mean_dot = stress_v_mean_n - stress_v_mean_n1

    k_from_mech = perm_update_rutqvist_newton(mesh,k,k_0,phi_0,phi,coeff=coeff_for_perm)
    assign(k, k_from_mech)

    print('HM - with fixed-stress')
    it_num = 0
    while err > tol:

        solution_h_temp, t \
        = h_linear("beuler", mesh=mesh, subdomains=subdomains, boundaries=boundaries, \
                         t_start=t0, dt=dt, T=t, solution0=solution1_h, \
                         alpha_0=alpha_0, K_0=K_0, mu_l_0=mu_l_0, lmbda_l_0=lmbda_l_0, Ks_0=Ks_0, \
                         alpha_1=alpha_1, K_1=K_1, mu_l_1=mu_l_1, lmbda_l_1=lmbda_l_1, Ks_1=Ks_1, \
                         alpha=alpha, K=K, mu_l=mu_l, lmbda_l=lmbda_l, Ks=Ks, \
                         cf_0=cf_0, phi_0=phi_0, rho_0=rho_0, mu_0=mu_0, k_0=k_0,\
                         cf_1=cf_1, phi_1=phi_1, rho_1=rho_1, mu_1=mu_1, k_1=k_1,\
                         cf=cf, phi=phi, rho=rho, mu=mu, k=k, \
                         sigma_v_freeze=stress_v_mean_dot, dphi_c_dt = del_phi_chem)

        block_assign(solution_h, solution_h_temp)

        print('v now',np.average(solution_h[0].vector()[:]))
        print('p now',np.average(solution_h[1].vector()[:]))

        porosity_from_flow = porosity_update_from_flow(mesh, phi_it, alpha, K, \
                                                       solution_h[1], solutionIt_h[1])

        assign(phi, porosity_from_flow)

        solution_m_temp, t = m_linear("beuler", mesh=mesh, subdomains=subdomains, boundaries=boundaries, \
                         t_start=t0, dt=dt, T=t, solution0=solution1_m, \
                         alpha_0=alpha_0, K_0=K_0, mu_l_0=mu_l_0, lmbda_l_0=lmbda_l_0, Ks_0=Ks_0, \
                         alpha_1=alpha_1, K_1=K_1, mu_l_1=mu_l_1, lmbda_l_1=lmbda_l_1, Ks_1=Ks_1, \
                         alpha=alpha, K=K, mu_l=mu_l, lmbda_l=lmbda_l, Ks=Ks, \
                         cf_0=cf_0, phi_0=phi_0, rho_0=rho_0, mu_0=mu_0, k_0=k_0,\
                         cf_1=cf_1, phi_1=phi_1, rho_1=rho_1, mu_1=mu_1, k_1=k_1,\
                         cf=cf, phi=phi, rho=rho, mu=mu, k=k, \
                         pressure_freeze=solution_h[1])

        block_assign(solution_m, solution_m_temp)



        porosity_from_mechanics = porosity_update_from_mechanics(mesh, phi_0, alpha, K, \
                                                                 solution_m[0], solution0_m[0], \
                                                                 solution_h[1], solution0_h[1])

        assign(phi, porosity_from_mechanics)
        assign(phi_it, porosity_from_mechanics)

        err = np.linalg.norm((porosity_from_mechanics.vector()[:] - porosity_from_flow.vector()[:]) \
                             /porosity_from_mechanics.vector()[:])

        it_num = it_num +1
        print('err', err, 'it_num', it_num)


        print('phi now',np.average(phi.vector()[:]))
        print('phi 0',np.average(phi_0.vector()[:]))
        print('porosity_from_mechanics',np.average(porosity_from_mechanics.vector()[:]))
        print('porosity_from_flow',np.average(porosity_from_flow.vector()[:]))

        stress_v_mean_n = stress_v_mean_calculation(mesh, solution_m[0],solution_h[1], \
                                                  alpha, I, mu_l, lmbda_l)

        ### update Iterative parameters
        stress_v_mean_dot = stress_v_mean_n - stress_v_mean_n1
        k_from_mech = perm_update_rutqvist_newton(mesh,k,k_0,phi_0,phi,coeff=coeff_for_perm)
        assign(k, k_from_mech)
        block_assign(solutionIt_h, solution_h_temp)


    del_phi_mech = cal_delta_pm(mesh, porosity_from_mechanics, phi_1)
    del_perm_mech = cal_delta_tm(mesh, k, k_1)
    del_phi_mech_0 = cal_delta_pm(mesh, porosity_from_mechanics, phi_0)
    del_perm_mech_0 = cal_delta_tm(mesh, k, k_0)


    mass_con = mass_conservation_cal(mesh, phi_0, cf, alpha, Ks, K, solution_h[1], solution1_h[1], dt,
        stress_v_mean_dot, del_phi_chem, solution_h[0])
    print('mass_con',np.max(mass_con.vector()[:]))
    print('mass_con',np.average(mass_con.vector()[:]))
    # linear_extrapolation for c - rhs
    if first_time:
        c_extrapolate = linear_n1_for_c(mesh,solution_c[0],solution1_c[0])
    else:
        pass

    print('C calculation')
    # solution0 = linear_extrapolation
    solution_c_temp, t \
    = transport_linear("bdf", mesh=mesh, subdomains=subdomains, boundaries=boundaries, \
                     t_start=t0, dt=dt, T=t, solution0=solution1_c, \
                     alpha_0=alpha_0, K_0=K_0, mu_l_0=mu_l_0, lmbda_l_0=lmbda_l_0, Ks_0=Ks_0, \
                     alpha_1=alpha_1, K_1=K_1, mu_l_1=mu_l_1, lmbda_l_1=lmbda_l_1, Ks_1=Ks_1, \
                     alpha=alpha, K=K, mu_l=mu_l, lmbda_l=lmbda_l, Ks=Ks, \
                     cf_0=cf_0, phi_0=phi_0, rho_0=rho_0, mu_0=mu_0, k_0=k_0,\
                     cf_1=cf_1, phi_1=phi_1, rho_1=rho_1, mu_1=mu_1, k_1=k_1,\
                     cf=cf, phi=phi, rho=rho, mu=mu, k=k, \
                     d_0=d_0, d_1=d_1, d_t=d,
                     vel_c=solution_h[0], p_con = solution_h[1], A_0 = A_0, Temp = Temp, c_extrapolate = c_extrapolate)

    block_assign(solution_c, solution_c_temp)

    # mu is here
    if first_time and mixing_flag:
        mu_temp = linear_n1(mesh,solution_c[0],solution1_c[0],mu_max,mu_min,c_sat,c_inject)
        assign(mu, mu_temp)
    elif mixing_flag:
        mu_temp = linear_extrapolation(mesh,solution_c[0],solution1_c[0]\
                                       ,solution2_c[0],mu_max,mu_min,dt,dt_1,c_sat,c_inject)
        assign(mu, mu_temp)


    # linear_extrapolation for c - rhs
    if first_time:
        pass
    else:
        c_extrapolate = linear_extrapolation_for_c(mesh,solution_c[0],solution1_c[0]\
                                       ,solution2_c[0],dt,dt_1)
    # # for mechanics properties as a function of concentration
    # K_mult_min = 0.01
    # K_mult_max = 1.0
    # mu_l, lmbda_l, Ks, K = \
    # bulk_modulus_update(mesh,solution_c[0],K_mult_min,K_mult_max,K,nu,alpha,K_0)

    #mu_before_flow = project_mu(mu)

    # calcite!
    porosity_chemical \
    = porosity_update_from_chemical(mesh, phi, phi, phi_0, A_0 \
                                    , solution_c[0], solution_h[1], Temp, Omega, rho_solid, dt)
    assign(phi, porosity_chemical)
    #rhs_calcite_test = dphi_dt_test(mesh, phi, phi_0, A_0, solution_c[0], solution_h[1], Temp, Omega, rho_solid)
    print('phi_after_chem',np.average(porosity_chemical.vector()[:]))
    #quit()
    k_from_chem = perm_update_rutqvist_newton(mesh,k,k_0,phi_0,phi,coeff=coeff_for_perm)


    del_phi_chem_after_mech = cal_delta_pm(mesh, porosity_chemical, porosity_from_mechanics)
    del_phi_chem = dphi_dt_print(mesh, phi, phi_0, A_0, solution_c[0], solution_h[1], Temp, Omega, rho_solid)
    del_perm_chem = cal_delta_tm(mesh, k_from_chem, k_from_mech)
    assign(k, k_from_chem)

    # next time
    err = 10.

    print('mu now',np.average(mu.vector()[:]))

    block_assign(solution2_h, solution1_h)
    block_assign(solution2_m, solution1_m)
    block_assign(solution2_c, solution1_c)

    assign(phi_1, phi)
    assign(k_1, k)
    block_assign(solution1_h, solution_h)
    block_assign(solution1_m, solution_m)
    block_assign(solution1_c, solution_c)

    velocity_print = project(solution_h[0], VEL)

    # update time step for linear_extrapolation
    dt_1 = dt

    # adjust time step length
    max_velocity = np.max(np.abs(velocity_print.vector()[:]))
    hmin = mesh.hmin()
    cfl = 0.1
    dt_cfl = cfl*hmin/max_velocity
    dt_max = 4.0
    if dt_cfl < dt:
        print('reduced dt from ',dt,' to ', dt_cfl)
        dt = dt_cfl
    elif dt_cfl < dt_max:
        print('increased dt from ',dt,' to ', dt_cfl)
        dt = dt_cfl
    else:
        print('increased dt from ',dt,' to ', dt_max)
        dt = dt_max
    first_time = False
    # print when time = time step print
    n_count += 1
    print('time until print:', (n_print-n_count))

    if np.isclose(t, t1):
        xdmu.write_checkpoint(solution_m[0],'u',t,append = False)
        xdmv.write_checkpoint(velocity_print,'v',t,append = False)
        xdmp.write_checkpoint(solution_h[1],'p',t,append = False)
        xdmc.write_checkpoint(solution_c[0],'c',t,append = False)
        xdmphi.write_checkpoint(phi,'phi',t,append = False)
        xdmmu.write_checkpoint(mu,'mu',t,append = False)
        xdmperm.write_checkpoint(k,'perm',t,append = False)
        xdmK.write_checkpoint(K,'K',t,append = False)

        xdmdphi_mech.write_checkpoint(del_phi_mech,'del_phi_mech',t,append = False)
        xdmdphi_mech_0.write_checkpoint(del_phi_mech_0,'del_phi_mech_0',t,append = False)
        xdmdphi_chem.write_checkpoint(del_phi_chem,'del_phi_chem',t,append = False)
        xdmdphi_chem_after_mech.write_checkpoint(del_phi_chem_after_mech,'del_phi_chem_after_mech',t,append = False)
        xdmdperm_mech.write_checkpoint(del_perm_mech,'del_perm_mech',t,append = False)
        xdmdperm_mech_0.write_checkpoint(del_perm_mech_0,'del_perm_mech_0',t,append = False)
        xdmdperm_chem.write_checkpoint(del_perm_chem,'del_perm_chem',t,append = False)
        n_count = 0
        mass_con.rename("mass", "mass")
        xdmmass.write(mass_con,t)
    elif(n_count == n_print):
        xdmu.write_checkpoint(solution_m[0],'u',t,append = True)
        xdmv.write_checkpoint(velocity_print,'v',t,append = True)
        xdmp.write_checkpoint(solution_h[1],'p',t,append = True)
        xdmc.write_checkpoint(solution_c[0],'c',t,append = True)
        xdmphi.write_checkpoint(phi,'phi',t,append = True)
        xdmmu.write_checkpoint(mu,'mu',t,append = True)
        xdmperm.write_checkpoint(k,'perm',t,append = True)
        xdmK.write_checkpoint(K,'K',t,append = True)

        xdmdphi_mech.write_checkpoint(del_phi_mech,'del_phi_mech',t,append = True)
        xdmdphi_mech_0.write_checkpoint(del_phi_mech_0,'del_phi_mech_0',t,append = True)
        xdmdphi_chem.write_checkpoint(del_phi_chem,'del_phi_chem',t,append = True)
        xdmdphi_chem_after_mech.write_checkpoint(del_phi_chem_after_mech,'del_phi_chem_after_mech',t,append = True)
        xdmdperm_mech.write_checkpoint(del_perm_mech,'del_perm_mech',t,append = True)
        xdmdperm_mech_0.write_checkpoint(del_perm_mech_0,'del_perm_mech_0',t,append = True)
        xdmdperm_chem.write_checkpoint(del_perm_chem,'del_perm_chem',t,append = True)
        n_count = 0
        mass_con.rename("mass", "mass")
        xdmmass.write(mass_con,t)
