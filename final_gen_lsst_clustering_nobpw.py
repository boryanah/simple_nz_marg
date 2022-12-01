"""
This is for futuristic LSST with almost no noise and same bins as DES (can modulate N(z) uncertainties)
First generates fake data with Gaussian + SSC covariance;
"""
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import sacc

# bandwidths
bpw_edges = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073, 2354, 2673, 3035, 3446, 3914, 4444, 5047, 5731, 6508, 7390, 8392, 9529, 10821, 12288])

# Data generator
class Generator():
    def __init__(self, reference_sacc, nbar=3.5, bpw_edges=bpw_edges, fsky=0.4,
                 **kwargs):
        self.cosmo = ccl.CosmologyVanillaLCDM(**kwargs)
        # This could depend on the tracer
        srad_to_arcmin2 = 11818102.86
        self.s_ref = reference_sacc.copy()
        self.s_new = None
        self.bpw_edges = bpw_edges
        self.dell = np.diff(bpw_edges)
        self.nl = len(self.dell)
        self.ell = None
        self.fsky = fsky
        self.ccl_tracers = {}

        # nbar in gals/arcmin2
        if isinstance(nbar, float):
            self.nbar = {}
            for trn in self.s_ref.tracers:
                self.nbar[trn] = nbar * srad_to_arcmin2
        elif isinstance(nbar, dict):
            self.nbar = {trn: tr_nbar * srad_to_arcmin2 for trn, tr_nbar in
                         nbar.items()}

    def get_ell_eff(self):
        # This assumes the same binning for all Cells
        if self.ell is None:
            trs = self.s_ref.get_tracer_combinations()[0]
            dt = self.s_ref.get_data_types(tracers=trs)[0]
            self.ell, _ = self.s_ref.get_ell_cl(dt, trs[0], trs[1])
        return self.ell

    def get_tracer_ccl(self, trn):
        if trn not in self.ccl_tracers:
            tr = self.s_ref.tracers[trn]
            if tr.quantity == 'galaxy_density':
                z, nz = tr.z, tr.nz
                dndz = (z, nz)
                bias = (z, np.ones_like(z))
                ccl_tracer = ccl.NumberCountsTracer(self.cosmo, has_rsd=False,
                                                    dndz=dndz, bias=bias,
                                                    mag_bias=None)
            else:
                raise NotImplementedError("To be implemented when needed")

            self.ccl_tracers[trn] = ccl_tracer
        return self.ccl_tracers[trn]

    def get_ell_cl(self, data_type, trn1, trn2):
        ell, _ = self.s_ref.get_ell_cl(data_type, trn1, trn2)
        if 'b' in data_type:
            cl = np.zeros_like(ell)
        else:
            ccl_tracer1 = self.get_tracer_ccl(trn1)
            ccl_tracer2 = self.get_tracer_ccl(trn2)
            # This does not have into account the bpw. This is what we want for
            # now
            cl = ccl.angular_cl(self.cosmo, ccl_tracer1, ccl_tracer2, ell)

        return ell, cl

    def get_ell_nl(self, trn1, trn2):
        ell = self.get_ell_eff()

        nl = np.zeros_like(ell)
        if trn1 == trn2:
            nl += 1.0/self.nbar[trn1]

        return ell, nl

    def get_indep_dtype_trs_in_comb(self, data_type, trn1, trn2):
        # Normally they will be in order but it could be the case that not. Eg.
        # for cl_0e
        dt1 = data_type[-2]
        dt2 = data_type[-1]

        if data_type in ['cl_0e', 'cl_0b']:
            q1 = self.s_ref.tracer[trn1].quantity
            if q1 != 'galaxy_density':
                # Wrong order
                return dt2, dt2
        return dt1, dt2

    def get_covariance(self, s_new=None):
        ncl = self.s_ref.mean.size
        cov = np.zeros([ncl, ncl])
        ell = self.get_ell_eff()
        nmodes = self.fsky*(2*ell+1)*self.dell

        trs_comb = self.s_ref.get_tracer_combinations()

        # To avoid recomputing
        if s_new is not None:
            def get_ell_cl(dt, tr1, tr2):
                if (tr1, tr2) in trs_comb:
                    return s_new.get_ell_cl(dt, tr1, tr2)
                elif (tr2, tr1) in trs_comb:
                    return s_new.get_ell_cl(dt, tr2, tr1)
                else:
                    return self.get_ell_cl
        else:
            get_ell_cl = self.get_ell_cl

        for i, trs1 in enumerate(trs_comb):
            for trs2 in trs_comb[i:]:
                for dt1 in self.s_ref.get_data_types(tracers=trs1):
                    ix1 = s_new.indices(data_type=dt1, tracers=trs1)
                    for dt2 in self.s_ref.get_data_types(tracers=trs1):
                        ix2 = s_new.indices(data_type=dt2, tracers=trs2)
                        # Get the data types for the cells
                        dti1, dti2 = self.get_indep_dtype_trs_in_comb(
                            dt1, *trs1
                        )
                        dtj1, dtj2 = self.get_indep_dtype_trs_in_comb(
                            dt2, *trs2
                        )
                        dti1j1 = f"cl_{dti1}{dtj1}"
                        dti1j2 = f"cl_{dti1}{dtj2}"
                        dti2j1 = f"cl_{dti2}{dtj1}"
                        dti2j2 = f"cl_{dti2}{dtj2}"

                        # get all necessary cl's
                        cli1j1 = get_ell_cl(dti1j1, trs1[0], trs2[0])[1]
                        cli1j2 = get_ell_cl(dti1j2, trs1[0], trs2[1])[1]
                        cli2j1 = get_ell_cl(dti2j1, trs1[1], trs2[0])[1]
                        cli2j2 = get_ell_cl(dti2j2, trs1[1], trs2[1])[1]

                        # add noise
                        cli1j1 += self.get_ell_nl(trs1[0], trs2[0])[1]
                        cli1j2 += self.get_ell_nl(trs1[0], trs2[1])[1]
                        cli2j1 += self.get_ell_nl(trs1[1], trs2[0])[1]
                        cli2j2 += self.get_ell_nl(trs1[1], trs2[1])[1]

                        # Cov_ab,cd = (C_ell^ac*C_ell'^bd + C_ell^ad*C_ell'^bc)/(2ell+1) Delta ell fsky delta_ellell'
                        cov_this = np.diag((cli1j1*cli2j2 + cli1j2*cli2j1)/nmodes)
                        cov[np.ix_(ix1, ix2)] = cov_this
                        cov[np.ix_(ix2, ix1)] = cov_this.T
        return cov

    def generate_fake_data(self, write_to=None):
        if self.s_new is None:
            s_new = sacc.Sacc()
            for tr in self.s_ref.tracers.values():
                s_new.add_tracer_object(tr)

            for trs in self.s_ref.get_tracer_combinations():
                for dt in self.s_ref.get_data_types(tracers=trs):
                    ell, cl = self.get_ell_cl(dt, *trs)
                    s_new.add_ell_cl(dt, trs[0], trs[1], ell, cl)

            s_new.add_covariance(self.get_covariance(s_new))

            self.s_new = s_new

        if write_to is not None:
            self.s_new.save_fits(write_to, overwrite=True)

        return self.s_new


if __name__ == "__main__":
    s = sacc.Sacc.load_fits("data/DES_dummy_clustering.fits")
    generator = Generator(s, transfer_function="eisenstein_hu")
    snew = generator.generate_fake_data(
        write_to='data/cls_covG_lsst_clustering_eisenstein_hu_no_noise.fits'
    )

    for trs in snew.get_tracer_combinations():
        for dt in snew.get_data_types(tracers=trs):
            ell, cl, cov = snew.get_ell_cl(dt, *trs, return_cov=True)
            err = np.sqrt(np.diag(cov))
            plt.errorbar(ell, cl, yerr=err)
            if np.abs(cl[3] / cl[-1]) > 10:
                plt.loglog()
            else:
                plt.semilogx()
            plt.ylabel(r"$C_\ell$")
            plt.xlabel(r"$\ell$")
            plt.title(trs)
            plt.show()


