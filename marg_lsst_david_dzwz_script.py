"""
This is specifically for LSST likelihood where we run only gc and wl
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
import sacc
from getdist import loadMCSamples
from cobaya.model import get_model
import yaml

# decide on delta_z (fraction of uncertainty)
delta_z_frac = np.float(sys.argv[1]) #0.1, 0.05, 0.2

# get bestfit parameters
#chain_name = "/mnt/extraspace/gravityls_3/chains/desgc_deswl_dz_s8zpaper/desgc_deswl_dz_s8zpaper"
#chain_name = "chains/desgc_deswl_dzwz_s8zpaper"
chain_name = "chains/lsst_3x2pt_dzwz_david"

chain = loadMCSamples(chain_name,
                      settings={'ignore_rows':0.3})

params = chain.paramNames.list()
param_dict = chain.getParamBestFitDict('sigma8')

#for i, param in enumerate(params):
#    print(f"      {param:s}:", param_dict[param])


# Read original data and remove everything we don't need
#s = sacc.Sacc.load_fits("/mnt/extraspace/gravityls_3/S8z/Cls_new_pipeline/4096_DES_eBOSS_CMB/cls_covG_new.fits")
s = sacc.Sacc.load_fits("/users/boryanah/repos/xCell-likelihoods/simple_marginalization/data/cls_LSST_linear.fits")
s.remove_selection(data_type='cl_eb')
s.remove_selection(data_type='cl_be')
s.remove_selection(data_type='cl_bb')
s.remove_selection(data_type='cl_0b')
"""
for i in range(2):
    s.remove_selection(data_type='cl_00', tracers=(f'eBOSS__{i}', 'PLAcv'))
    s.remove_selection(data_type='cl_00', tracers=(f'eBOSS__{i}', f'eBOSS__{i}'))
for i in range(4):
    s.remove_selection(data_type='cl_0e', tracers=(f'DESwl__{i}', 'PLAcv'))
for i in range(5):
    s.remove_selection(data_type='cl_00', tracers=(f'DESgc__{i}', 'PLAcv'))
"""

# Generate cobaya model (note that we remove all scale cuts by hand)

def get_cobaya_model(config_fn, include_all_scales=True):
    with open(config_fn, "r") as fin:
        info = yaml.load(fin, Loader=yaml.FullLoader)
    info['debug'] = False
    if include_all_scales:
        info['likelihood']['cl_like.ClLike']['defaults']['kmax'] = 100000
        info['likelihood']['cl_like.ClLike']['defaults']['lmin'] = 0
        info['likelihood']['cl_like.ClLike']['defaults']['lmax'] = 100000
        #for i in range(4):
        #    info['likelihood']['cl_like.ClLike']['defaults'][f'DESwl__{i}']['lmin'] = 0

    # Get the mean proposed in the yaml file for each parameter
    p0 = {}
    s0 = {}
    for p in info['params']:
        print(p)
        if isinstance(info['params'][p], dict):
            if 'ref' in info['params'][p]:
                p0[p] = info['params'][p]['ref']['loc']
                s0[p] = info['params'][p]['ref']['scale']
            elif 'prior' in info['params'][p]:
                p0[p] = info['params'][p]['prior']['loc']
                s0[p] = info['params'][p]['prior']['scale']
    #p0['Omega_m'] = p0['Omega_c'] + p0['Omega_b']
    model = get_model(info)
    return model, p0, s0, info

model, p0, s0, info = get_cobaya_model("config/lsst_3x2pt_dzwz_david.yaml")
#model, p0, s0, info = get_cobaya_model("config/desgc_deswl_dzwz_s8zpaper.yaml")
#model, p0, s0, info = get_cobaya_model("/mnt/extraspace/gravityls_3/chains/desgc_deswl_dz_s8zpaper/desgc_deswl_dz_s8zpaper.input.yaml")
print("p0, s0 = ", p0.items(), s0.items())

# Generate a prediction for it
loglikes, derived = model.loglikes(p0)
l = model.likelihood['cl_like.ClLike']
p = l.current_state['params'].copy()
print("p without cosmo = ", p.items())
s_pred = l.get_sacc_file(**p)

assert s.mean.shape == s_pred.mean.shape

tr_pairs = s_pred.get_tracer_combinations()

indices = []
dtypes = []
for t1, t2 in tr_pairs:
    pol1 = 'e' if 'wl' in t1 else '0'
    pol2 = 'e' if 'wl' in t2 else '0'
    dtype = f'cl_{pol1}{pol2}'
    if dtype == 'cl_e0':
        dtype == 'cl_0e'
    dtypes.append(dtype)
    _, _, ind_original = s.get_ell_cl(dtype, t1, t2, return_ind=True)
    indices.append(ind_original)


# Finally, we write a function to generate the predicted data vector for a given set of parameters in the right order!
def get_data_vector(par):
    # Evaluate likelihood at the right params
    loglikes, derived = model.loglikes(par)
    # Generate prediction
    l = model.likelihood['cl_like.ClLike']
    p = l.current_state['params'].copy()
    sp = l.get_sacc_file(**p)
    assert sp.mean.shape == s.mean.shape
    # Reorder
    data = np.zeros(len(s.mean))
    for (t1, t2), ind, dtype in zip(tr_pairs, indices, dtypes):
        _, cl = sp.get_ell_cl(dtype, t1, t2)
        data[ind] = cl
    return data

d0 = get_data_vector(p0)
print(d0.shape)
print(s.mean.shape)

# get new parameter dictionary with bestfit params
p_bf = p0.copy()
for par in p0.keys():
    p_bf[par] = param_dict[par]
print("bestfit params = ", p_bf.items())

# count number of parameters
sum = 0
for par in p0.keys():
    if ("dz" in par) or ("wz" in par):
        print(par)
        sum += 1

# compute derivatives
t = np.zeros((s.mean.shape[0], sum))
P = np.zeros(sum)
print("number of params = ", sum)
sum = 0
for par in p0.keys():
    if ("dz" in par) or ("wz" in par):

        p_p = p_bf.copy()
        p_m = p_bf.copy()
        h = s0[par]*delta_z_frac
        print("h of param", h, par)

        p_p[par] = p_bf[par] + h
        p_m[par] = p_bf[par] - h
        print("bf p m ", p_bf[par], p_p[par], p_m[par])
        
        d_p = get_data_vector(p_p)
        d_m = get_data_vector(p_m)
        print("d_p-d_m", np.sum(d_p-d_m))
        
        t[:, sum] = (d_p - d_m)/(2.*h)
        P[sum] = s0[par]**2
        sum += 1
P = np.diag(P)

# add to covariance
cov_extra = np.dot(np.dot(t, P), t.T)
cov = s.covariance.covmat.copy()
cov += cov_extra
s.add_covariance(cov)
print(cov_extra)
print(P)
print(t)
np.save("t.npy", t)
quit()
#np.save(f"data/t_derfrac{delta_z_frac:.2f}_dzwz.npy", t)
#np.save(f"data/cov_extra_derfrac{delta_z_frac:.2f}_dzwz.npy", cov_extra)
fn = f'data/cls_LSST_linear_marg_derfrac{delta_z_frac:.2f}_dzwz.fits'
s.save_fits(fn, overwrite=True)
