import sys

import numpy as np
import sacc
import pyccl as ccl
from cobaya.model import get_model
import yaml

# what fraction of error on dz's to use for derivatives
delta_z_frac = np.float(sys.argv[1])

# const
srad_to_arcmin2 = 11818080
n_bar_gc = 0.5 # gals/arcmin^2
n_bar_wl = 3.5 # gals/arcmin^2
n_bar_gc *= srad_to_arcmin2
n_bar_wl *= srad_to_arcmin2
sigma_gamma = 0.28

# bandwidths
bpw = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073, 2354, 2673, 3035, 3446, 3914, 4444, 5047, 5731, 6508, 7390, 8392, 9529, 10821, 12288])
dell = np.diff(bpw)
nl = len(dell)

# Planck cosmology
h = 0.6736
param_dict = {}
param_dict['Omega_b'] = 0.02237/h**2
param_dict['Omega_c'] = 0.1200/h**2
param_dict['h'] = h
param_dict['A_sE9'] = 2.0830
param_dict['n_s'] = 0.9649
cosmo_dict = param_dict.copy()
cosmo_dict['A_s'] = cosmo_dict['A_sE9']*1.e-9
del cosmo_dict['A_sE9']
cosmo = ccl.Cosmology(**cosmo_dict)

# definitions
massdef = ccl.halos.MassDef(200., 'critical')
hb_name = "Tinker10"
cm_name = "Duffy08"
mf_name = "Tinker08"

# Mass function
mfc = ccl.halos.mass_function_from_name(mf_name)
# Halo bias
hbc = ccl.halos.halo_bias_from_name(hb_name)
# Concentration
cmc = ccl.halos.concentration_from_name(cm_name)
cm = cmc(mdef=massdef)

# Default profiles for different quantities
profs = {'galaxy_density': None,
         'galaxy_shear': ccl.halos.HaloProfileNFW(cm)}
prof = profs['galaxy_shear']

# init objects
mf = mfc(cosmo, mass_def=massdef)
hb = hbc(cosmo, mass_def=massdef)
hmc = ccl.halos.HMCalculator(cosmo, mf, hb, massdef)

# Read original data and remove everything we don't need
s = sacc.Sacc.load_fits("/mnt/extraspace/gravityls_3/S8z/Cls_new_pipeline/4096_DES_eBOSS_CMB/cls_covG_new.fits")
s.remove_selection(data_type='cl_eb')
s.remove_selection(data_type='cl_be')
s.remove_selection(data_type='cl_bb')
s.remove_selection(data_type='cl_0b')
s.remove_tracers(['eBOSS__0', 'eBOSS__1', 'PLAcv'])

# Generate cobaya model (note that we remove all scale cuts by hand)
def get_cobaya_model(config_fn, include_all_scales=True):
    with open(config_fn, "r") as fin:
        info = yaml.load(fin, Loader=yaml.FullLoader)
    info['debug'] = False
    if include_all_scales:
        info['likelihood']['cl_like.ClLike']['defaults']['kmax'] = 100000
        info['likelihood']['cl_like.ClLike']['defaults']['lmin'] = 0
        info['likelihood']['cl_like.ClLike']['defaults']['lmax'] = 100000
        for i in range(4):
            info['likelihood']['cl_like.ClLike']['defaults'][f'DESwl__{i}']['lmin'] = 0

    # Get the mean proposed in the yaml file for each parameter
    p0 = {}
    s0 = {}
    for p in info['params']:
        if isinstance(info['params'][p], dict):
            if p in param_dict.keys():
                p0[p] = param_dict[p]
            elif 'ref' in info['params'][p]:
                p0[p] = info['params'][p]['ref']['loc']
                s0[p] = info['params'][p]['ref']['scale']
            elif 'prior' in info['params'][p]:
                p0[p] = info['params'][p]['prior']['loc']
                s0[p] = info['params'][p]['prior']['scale']
    model = get_model(info)
    return model, p0, s0, info

# get model that Carlos uses
model, p0, s0, info = get_cobaya_model("/mnt/extraspace/gravityls_3/chains/desgc_deswl_dz_s8zpaper/desgc_deswl_dz_s8zpaper.input.yaml")
print("p0 = ", p0.items())

# Generate a prediction for it (there are 3 orderings: s, s_new and snew and we don't assume they're the same)
loglikes, derived = model.loglikes(p0)
l = model.likelihood['cl_like.ClLike']
p_new = l.current_state['params'].copy()
s_new = l.get_sacc_file(**p_new)
assert s.mean.shape == s_new.mean.shape

# get all tracers
ts = ['DESgc__0', 'DESgc__1', 'DESgc__2', 'DESgc__3', 'DESgc__4', 'DESwl__0', 'DESwl__1', 'DESwl__2', 'DESwl__3']
qs = ['galaxy_density'] * 5 + ['galaxy_shear'] * 4
ps = ['0'] * 5 + ['e']  * 4
nt = len(ts)

# initialize dictionary with bias and ccl tracers
bias_dict = {}
ccl_ts = {}

# create empty sacc object and stuff it with things
snew = sacc.Sacc()
for t, q in zip(ts, qs):
    if f'clk_{t}_b1' in info['params'].keys():
        bias_dict[t] = info['params'][f'clk_{t}_b1']['ref']['loc']
    else:
        bias_dict[t] = 0. # 0 for weak lensing
    snew.add_tracer('NZ', t, quantity=q, spin=0 if q == 'galaxy_density' else 2,
                    z=s.tracers[t].z, nz=s.tracers[t].nz)

    if q == 'galaxy_density':
        nz = (s.tracers[t].z, s.tracers[t].nz)
        bz = (s.tracers[t].z, np.ones_like(s.tracers[t].z)) # set to 1 since we change later
        ccl_ts[t] = ccl.NumberCountsTracer(cosmo, dndz=nz,
                                       bias=bz, has_rsd=False)
    elif q == 'galaxy_shear':
        nz = (s.tracers[t].z, s.tracers[t].nz)
        ia = None
        ccl_ts[t] = ccl.WeakLensingTracer(cosmo, nz, ia_bias=ia)

def get_dtype(t1, t2):
    pol1 = 'e' if 'wl' in t1 else '0'
    pol2 = 'e' if 'wl' in t2 else '0'
    dtype = f'cl_{pol1}{pol2}'
    if dtype == 'cl_e0':
        dtype = 'cl_0e'
    return dtype

def tracer_iterator():
    i_d = 0
    for i1 in range(nt):
        for i2 in range(i1, nt):
            t1 = ts[i1]
            t2 = ts[i2]
            dtype = get_dtype(t1, t2)
            # no cross between different galaxy samples, so don't yield anything
            if ('gc' in t1) and ('gc' in t2) and (t1 != t2):
                continue
            yield i1, i2, t1, t2, i_d, dtype
            i_d += 1


for i1, i2, t1, t2, ii, dtype in tracer_iterator():
    # take cl from predicted model
    l, cl = s_new.get_ell_cl(dtype, t1, t2)
    # get window from original file for these tracers
    _, _, ind = s.get_ell_cl(dtype, t1, t2, return_ind=True)
    bpws = s.get_bandpower_windows(ind)
    # add to final sacc object
    snew.add_ell_cl(dtype, t1, t2, l, cl, window=bpws)

def get_cl(t1, t2):
    # given two tracers, get dtype
    dtype = get_dtype(t1, t2)

    if ('gc' in t1) and ('gc' in t2) and (t1 != t2):
        # assume no cross-correlation between different gc samples
        cl = np.zeros(nl)
    elif ('wl' in t1) and ('gc' in t2):
        # always galaxy tracer is first
        _, cl = snew.get_ell_cl(dtype, t2, t1)
    elif ('wl' in t1) and ('wl' in t2) and (t1 > t2):
        # always smaller bin number is first
        _, cl = snew.get_ell_cl(dtype, t2, t1)
    else:
        # otherwise normal
        _, cl = snew.get_ell_cl(dtype, t1, t2)

    if t1 == t2:
        if 'gc' in t1:
            cl += 1./n_bar_gc
        elif 'wl' in t1:
            cl += sigma_gamma**2/n_bar_wl
    assert len(cl) == nl
    return cl

def check_nr_cts(tr):
    # silly function for deciding whether tracer is number counts or not
    is_number_counts = True if 'gc' in tr else False
    return is_number_counts

fsky = 0.4
ncl = (nt*(nt+1)) // 2 - 5*(5-1)//2 # getting rid of the cross gc correlations
cov = np.zeros([ncl*nl, ncl*nl])
cov_extra = np.zeros([ncl*nl, ncl*nl])
nmodes = fsky*(2*l+1)*dell
sum = 0
for i1, i2, ti1, ti2, ii, idtype in tracer_iterator():
    for j1, j2, tj1, tj2, jj, jdtype in tracer_iterator():
        print("all tracers = ", ti1, ti2, tj1, tj2, sum)

        iix = snew.indices(data_type=idtype, tracers=(ti1, ti2))
        jix = snew.indices(data_type=jdtype, tracers=(tj1, tj2))

        # get all necessary cl's
        cli1j1 = get_cl(ti1, tj1)
        cli1j2 = get_cl(ti1, tj2)
        cli2j1 = get_cl(ti2, tj1)
        cli2j2 = get_cl(ti2, tj2)

        # Cov_ab,cd = (C_ell^ac*C_ell'^bd + C_ell^ad*C_ell'^bc)/(2ell+1) Delta ell fsky delta_ellell'
        cov_this = np.diag((cli1j1*cli2j2 + cli1j2*cli2j1)/nmodes)
        cov[np.ix_(iix, jix)] = cov_this

        # generate trispectrum
        tkk = ccl.halos.halo_model.halomod_Tk3D_SSC_linear_bias(cosmo, hmc, prof, bias1=bias_dict[ti1], bias2=bias_dict[ti2], bias3=bias_dict[tj1],  bias4=bias_dict[tj2], is_number_counts1=check_nr_cts(ti1), is_number_counts2=check_nr_cts(ti2), is_number_counts3=check_nr_cts(tj1), is_number_counts4=check_nr_cts(tj2))

        # get supersample covariance given tkk
        cov_ssc = ccl.covariances.angular_cl_cov_SSC(cosmo, ccl_ts[ti1], ccl_ts[ti2], l, tkka=tkk, fsky=fsky, cltracer3=ccl_ts[tj1], cltracer4=ccl_ts[tj2])
        cov_extra[np.ix_(iix, jix)] = cov_ssc

        sum += 1

# combine covariances and reshape
np.save(f"data/lsst_cov_extra.npy", cov_extra)
np.save(f"data/lsst_cov_G.npy", cov)
cov += cov_extra

# save the covariance matrix without marginalization
snew.add_covariance(cov)

# write out
fn = f'data/cls_covG_lsst.fits'
snew.save_fits(fn, overwrite=True)

# this for loop is just to get what index each predicted tracer pair has in the final sacc file
indices = []
dtypes = []
tr_pairs = s_new.get_tracer_combinations() # prediction ordering
for t1, t2 in tr_pairs:
    dtype = get_dtype(t1, t2)
    dtypes.append(dtype)
    _, _, ind_original = snew.get_ell_cl(dtype, t1, t2, return_ind=True) # indexing of the final sacc file
    indices.append(ind_original)

def get_data_vector(params):
    # generate model given parameter values
    loglikes, derived = model.loglikes(params)
    l = model.likelihood['cl_like.ClLike']
    p_new = l.current_state['params'].copy()
    s_new = l.get_sacc_file(**p_new)
    assert snew.mean.shape == s_new.mean.shape

    # Reorder into final sacc file ordering
    data = np.zeros(len(snew.mean)) # data vector
    for (t1, t2), ind, dtype in zip(tr_pairs, indices, dtypes):
        _, cl = s_new.get_ell_cl(dtype, t1, t2)
        data[ind] = cl
    return data


# compute derivatives
t = np.zeros((snew.mean.shape[0], nt)) # derivative matrix
P = np.zeros(nt) # in this case, covariance between dz's
print("number of params = ", nt)
sum = 0
for par in p0.keys(): # p0 is fiducial cosmology
    if "dz" in par:
        p_p = p0.copy()
        p_m = p0.copy()
        h = s0[par]*delta_z_frac
        print("h of param", h, par)

        p_p[par] = p0[par] + h
        p_m[par] = p0[par] - h
        print("bf p m ", p0[par], p_p[par], p_m[par])

        d_p = get_data_vector(p_p)
        d_m = get_data_vector(p_m)
        print("d_p-d_m", np.sum(d_p-d_m))

        t[:, sum] = (d_p - d_m)/(2.*h)
        P[sum] = s0[par]**2
        sum += 1
P = np.diag(P)

# add to covariance
cov_extra = np.dot(np.dot(t, P), t.T)
cov = snew.covariance.covmat.copy()
cov += cov_extra
snew.add_covariance(cov)

# save updated sacc object
np.save(f"data/lsst_t_derfrac{delta_z_frac:.2f}.npy", t)
np.save(f"data/lsst_cov_extra_derfrac{delta_z_frac:.2f}.npy", cov_extra)
fn = f'data/cls_covG_lsst_derfrac{delta_z_frac:.2f}.fits'
snew.save_fits(fn, overwrite=True)
