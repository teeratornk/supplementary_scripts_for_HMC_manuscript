import numpy as np
from dolfin import div, dot, sym, grad, tr, conditional, ge, exp, ln
import csv

def E_nu_to_mu_lmbda(E, nu):
    mu = E/(2*(1.0+nu))
    lmbda = (nu*E)/((1-2*nu)*(1+nu))
    return (mu, lmbda)

def K_nu_to_E(K, nu):
    return 3*K*(1-2*nu)

def Ks_cal(alpha,K):
    if np.isclose(alpha, 1.0):
    #if alpha == 1.0:
        Ks = 1e35
    else:
        Ks = K/(1.0-alpha)
    return Ks

def Ks_cal_no_isclose(alpha,K):
    Ks = K/(1.0-alpha)
    return Ks

def sigma(u,I,mu_l,lmbda_l):
    return 2*mu_l*sym(grad(u)) + lmbda_l*div(u)*I

def sigma_total(u,p,alpha,I,mu_l,lmbda_l):
    return sigma(u,I,mu_l,lmbda_l) - alpha*p*I

#if you look at mean, do not forget to divide by the dimension
def vol_sigma_total(u,p,alpha,I,mu_l,lmbda_l):
    return tr(sigma_total(u,p,alpha,I,mu_l,lmbda_l))

def strain(u):
    return sym(grad(u))

def vol_strain(u):
    return tr(strain(u))

def coeff_of_consolidation(K,v,k_i,vis):
   return 3.0*K*((1.0-v)/(1.0+v))*(k_i/vis)

def init_scalar_const_parameter(p,p_value):
    p.vector()[:]=p_value
    return p

def init_scalar_parameter(p,p_value,index,sub):
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            p.vector()[cell_no] = p_value
    return p

def init_het_scalar_parameter(p,p_mu,p_var,p_min,p_max,index,sub):
    X = get_truncated_normal(mean=p_mu, sd=np.sqrt(p_var), low=p_min, up=p_max)
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            p.vector()[cell_no] = X.rvs()
    return p

def init_het_LN_scalar_parameter(p,p_mu,p_var,p_min,p_max,index,sub):
    mu_normal_cal = 2*np.log(p_mu)-1.0/2.0*np.log(p_var+np.power(p_mu,2))
    var_normal_cal = -2*np.log(p_mu) + np.log(p_var+np.power(p_mu,2))
    min_cal = np.log(p_min)
    max_cal = np.log(p_max)
    print(mu_normal_cal, var_normal_cal, min_cal, max_cal)
    X = get_truncated_normal(mean=mu_normal_cal, sd=np.sqrt(var_normal_cal), low=min_cal, up=max_cal)
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            p.vector()[cell_no] = np.exp(X.rvs())
    return p

def init_tensor_parameter(p,p_value,index,sub,dim):
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            k_j = 0
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                p.vector()[k_i] = p_value[k_j]
                k_j = k_j + 1
    return p

def init_het_tensor_parameter(p,p_mu,p_var,p_min,p_max,index,sub,dim):
    X = get_truncated_normal(mean=p_mu, sd=np.sqrt(p_var), low=p_min, up=p_max)
    for cell_no in range(len(sub.array())):
        p_x = X.rvs()
        p_y = p_x
        p_mat = np.array([p_x, 0.,0., p_y])
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            k_j = 0
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                p.vector()[k_i] = p_mat[k_j]
                k_j = k_j + 1
    return p

def init_het_LN_tensor_parameter(p,p_mu,p_var,p_min,p_max,index,sub,dim):
    mu_normal_cal = 2*np.log(p_mu)-1.0/2.0*np.log(p_var+np.power(p_mu,2))
    var_normal_cal = -2*np.log(p_mu) + np.log(p_var+np.power(p_mu,2))
    min_cal = np.log(p_min)
    max_cal = np.log(p_max)
    print(mu_normal_cal, var_normal_cal, min_cal, max_cal)
    X = get_truncated_normal(mean=mu_normal_cal, \
                             sd=np.sqrt(var_normal_cal), \
                             low=min_cal, up=max_cal)
    for cell_no in range(len(sub.array())):
        p_x = np.exp(X.rvs())
        p_y = p_x
        p_mat = np.array([p_x, 0.,0., p_y])
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            k_j = 0
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                p.vector()[k_i] = p_mat[k_j]
                k_j = k_j + 1
    return p

def adjusted_het_tensor_parameter(p,p_mu,index,sub,dim):
    ad = avg_diagonal_tensor_parameter(p,index,sub,dim)/p_mu
    p.vector()[:] =  p.vector()[:]/ad
    return p

def adjusted_het_scalar_parameter(p,p_mu):
    ad = np.average(p.vector()[:])/p_mu
    p.vector()[:] =  p.vector()[:]/ad
    return p

def avg_w(x,w):
    return (w*x('+')+(1-w)*x('-'))

def k_normal(k,n):
    return dot(dot(np.transpose(n),k),n)

def k_plus(k,n):
    return dot(dot(n('+'),k('+')),n('+'))

def k_minus(k,n):
    return dot(dot(n('-'),k('-')),n('-'))

def weight_e(k,n):
    return (k_minus(k,n))/(k_plus(k,n)+k_minus(k,n))

def k_e(k,n):
    return (2*k_plus(k,n)*k_minus(k,n)/(k_plus(k,n)+k_minus(k,n)))

def k_har(k):
    return (2*k*k/(k+k))

def weight_k_homo(k):
    return (k)/(k+k)

def phi_update(phi,phi0,vs):
    phi_min = 0.0001
    for i in range(len(vs.vector())):
        phi.vector()[i] = 1.0-(1.0-phi0.vector()[i])/np.exp(vs.vector()[i])
        if phi.vector()[i]<phi_min:
            phi.vector()[i] = phi_min
    #phi.vector()[:] = 1.0-(1.0-phi0.vector()[:])/np.exp(vs.vector()[:])
    return phi

def perm_update(phi,phi0,k,k0,m,sub,dim):
    perm_min = 1e-20
    for cell_no in range(len(sub.array())):
        for k_i in range(cell_no*np.power(dim,2), \
            cell_no*np.power(dim,2)+np.power(dim,2)):
            perm_cal = k0.vector()[k_i] \
            *np.power(phi.vector()[cell_no]/phi0.vector()[cell_no],m)
            if k.vector()[k_i]>0:
                if perm_cal < perm_min:
                    k.vector()[k_i] = perm_min
                else:
                    k.vector()[k_i] = perm_cal
    return k

def perm_update_wong(vs,phi0,k,k0,sub,dim):
    perm_min = 1e-20
    for cell_no in range(len(sub.array())):
        for k_i in range(cell_no*np.power(dim,2), \
            cell_no*np.power(dim,2)+np.power(dim,2)):
            perm_cal = k0.vector()[k_i] \
            * np.power(1+vs.vector()[cell_no]/phi0.vector()[cell_no],3.0) \
            / (1.0+vs.vector()[cell_no])
            if k.vector()[k_i]>0:
                if perm_cal < perm_min:
                    k.vector()[k_i] = perm_min
                else:
                    k.vector()[k_i] = perm_cal
    return k

def get_truncated_normal(mean=0, sd=1, low=0, up=10):
    np.random.seed(seed=3)
    return truncnorm(\
        (low - mean) / sd, (up - mean) / sd, loc=mean, scale=sd)


def avg_scalar_parameter(p,index,sub):
    p_cum = 0.0
    n_cum = 0.0
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            p_cum = p_cum + p.vector()[cell_no]
            n_cum = n_cum + 1.0
    return (p_cum/n_cum)

def avg_diagonal_tensor_parameter(p,index,sub,dim):
    p_cum = 0.0
    n_cum = 0.0
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                p_cum = p_cum + p.vector()[k_i]
            n_cum = n_cum + 2
    return (p_cum/n_cum)

def min_diagonal_tensor_parameter(p,index,sub,dim):
    p_min = 10.0
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                if p.vector()[k_i]>0:
                    if p.vector()[k_i]<p_min:
                        p_min = p.vector()[k_i]
    return p_min

def log_diagonal_tensor_parameter(p,log_p,index,sub,dim):
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                if p.vector()[k_i] == 0:
                    log_p.vector()[k_i] = p.vector()[k_i]
                else:
                    log_p.vector()[k_i] = np.log(p.vector()[k_i])
    return log_p

def init_from_file_parameter(p,index,sub,filename):
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
        i = 0
        for row in readCSV:
            p.vector()[i] = row[0]
            i +=1
    return p

def tau_cal(p,phi,con):
    p.vector()[:] = np.power(phi.vector()[:],con)
    return p

def diff_coeff_cal(p,p0,tau,phi,sub,dim):
    d_eff_min = 1e-20
    for cell_no in range(len(sub.array())):
        for k_i in range(cell_no*np.power(dim,2), \
            cell_no*np.power(dim,2)+np.power(dim,2)):
            d_eff_cal = p0.vector()[k_i] * phi.vector()[cell_no] \
            / tau.vector()[cell_no]
            if p0.vector()[k_i]>0:
                if d_eff_cal < d_eff_min:
                    p.vector()[k_i] = d_eff_min
                else:
                    p.vector()[k_i] = d_eff_cal
    return p

def diff_coeff_cal_rev(p,p0,tau,phi):
    mult_min = 1e-15
    mult = conditional(ge(phi/tau,0.0),phi/tau,mult_min)
    p = p0*mult
    return p



def perm_update_wong_newton(vs,phi0,k0):
    mult_min = 1e-15
    mult = conditional(ge(pow(1.0+vs/phi0,3.0)/(1.0+vs),0.0) \
        ,pow(1.0+vs/phi0,3.0)/(1.0+vs),mult_min)
    k = k0*mult
    return k

def rho_cal_linear(p,c,rho_1,rho_2):
    p.vector()[:] = rho_1.vector()[:] \
    + np.multiply(c.vector()[:],(rho_2.vector()[:]-rho_1.vector()[:]))
    return p

def mu_cal_linear(p,c,mu_1,mu_2):
    p.vector()[:] = np.multiply(c.vector()[:],mu_1.vector()[:]) \
    + np.multiply(1.0-c.vector()[:],mu_2.vector()[:])
    return p


def rho_newton_linear(c, rho_min, rho_max):
    return (rho_max*c + rho_min*(1.0-c))


def mu_newton_linear(c, mu_min, mu_max):
    return (mu_min*c + mu_max*(1.0-c))

def mu_newton_linear_adapt(c, mu_light, mu_heavy, c_light, c_heavy):
    mu_light = ln(mu_light)
    mu_heavy = ln(mu_heavy)
    mu = mu_light + (c-c_light)/(c_heavy-c_light)*(mu_heavy-mu_light)
    return exp(mu)

def bulk_modulus_mult_newton_linear(c, K_min, K_max):
    return (K_min*c + K_max*(1.0-c))

def mu_newton_qm(c, mu_min, mu_max):
    return 1.0/(pow((pow(c*mu_min,-0.25) + pow((1.0-c)*mu_max,-0.25)),4.0))
