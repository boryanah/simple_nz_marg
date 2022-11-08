"""
This is for grand conjuration study where we have DES, DECALS, KiDS, CMBk (eBOSS removed for now)
Might need to make sure that the best fit parameters agree with Carlos' paper
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
cn1 = "../chains/DELS_KiDS_CMBk_placv_marg_heft_b1b2bsbn_kmax0.40_learn"
cn2 = "../chains/DES_CMBk_placv_marg_heft_b1b2bsbn_kmax0.40_learn"

param_dict = {}
def get_param_dict(chain_name):
    chain = loadMCSamples(chain_name,
                          settings={'ignore_rows':0.3})
    
    params = chain.paramNames.list()
    param_dict = chain.getParamBestFitDict('sigma8')

    return params, param_dict

p1, dict1 = get_param_dict(cn1)
p2, dict2 = get_param_dict(cn2)
params = p1+p2
for p in dict1.keys():
    param_dict[p] = dict1[p]
for p in dict2.keys():
    param_dict[p] = dict2[p]

    
for i, param in enumerate(params):
    print(f"      {param:s}:", param_dict[param])
    if param not in param_dict.keys():
        print("missing", param)
print(len(params), len(param_dict.keys())) # different because params has duplicates

# Read original data and remove everything we don't need
s = sacc.Sacc.load_fits("/users/boryanah/repos/GrandConjuration/data/cls_FD_covG_mMarg.fits")
print(s.tracers)
s.remove_selection(data_type='cl_eb')
s.remove_selection(data_type='cl_be')
s.remove_selection(data_type='cl_bb')
s.remove_selection(data_type='cl_0b')
for i in range(2):
    s.remove_selection(data_type='cl_00', tracers=(f'eBOSS__{i}', 'PLAcv'))
    s.remove_selection(data_type='cl_00', tracers=(f'eBOSS__{i}', f'eBOSS__{i}'))
"""
for i in range(4):
    s.remove_selection(data_type='cl_0e', tracers=(f'DESwl__{i}', 'PLAcv'))
for i in range(5):
    s.remove_selection(data_type='cl_00', tracers=(f'DESgc__{i}', 'PLAcv'))
"""
# not sure if this exists
s.remove_selection(data_type='cl_ee', tracers=(f'PLAcv', 'PLAcv'))

# Generate cobaya model (note that we remove all scale cuts by hand)

def get_cobaya_model(config_fn, include_all_scales=True):
    with open(config_fn, "r") as fin:
        info = yaml.load(fin, Loader=yaml.FullLoader)
    info['debug'] = False
    if include_all_scales:
        info['likelihood']['cl_like.ClLike']['defaults']['kmax'] = 100000
        info['likelihood']['cl_like.ClLike']['defaults']['lmin'] = 0
        info['likelihood']['cl_like.ClLike']['defaults']['lmax'] = 100000
        # TESTING!!!!!!!!!!!!!!!!!!!!!!!!
        #for i in range(4):
        #    info['likelihood']['cl_like.ClLike']['defaults'][f'DESwl__{i}']['lmin'] = 0

    # Get the mean proposed in the yaml file for each parameter
    p0 = {}
    s0 = {}
    for p in info['params']:
        if isinstance(info['params'][p], dict):
            if 'ref' in info['params'][p]:
                p0[p] = info['params'][p]['ref']['loc']
                s0[p] = info['params'][p]['ref']['scale']
            elif 'prior' in info['params'][p]:
                p0[p] = info['params'][p]['prior']['loc']
                s0[p] = info['params'][p]['prior']['scale']
    #p0['Omega_m'] = p0['Omega_c'] + p0['Omega_b']
    print(info.items())
    model = get_model(info)
    return model, p0, s0, info


model, p0, s0, info = get_cobaya_model("config/DES_DELS_KiDS_CMBk_placv_marg_heft_b1b2bsbn_kmax0.40_learn.yml")
print("p0, s0 = ", p0.items(), s0.items())

# Generate a prediction for it
loglikes, derived = model.loglikes(p0)
l = model.likelihood['cl_like.ClLike']
p = l.current_state['params'].copy()
print("p without cosmo = ", p.items())
s_pred = l.get_sacc_file(**p)

assert s.mean.shape == s_pred.mean.shape, print(s.mean.shape,s_pred.mean.shape)

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
    if "dz" in par:
        print(par)
        sum += 1
print(p_bf)

# compute derivatives
t = np.zeros((s.mean.shape[0], sum))
P = np.zeros(sum)
print("number of params = ", sum)
sum = 0
for par in p0.keys():
    if "dz" in par:

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

np.save(f"data/grand_t_derfrac{delta_z_frac:.2f}.npy", t)
np.save(f"data/grand_cov_extra_derfrac{delta_z_frac:.2f}.npy", cov_extra)
fn = f'data/grand_cls_covG_new_derfrac{delta_z_frac:.2f}.fits'
s.save_fits(fn, overwrite=True)
