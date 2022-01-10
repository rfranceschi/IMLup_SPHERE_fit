import getpass
import logging
import pickle
import shutil
from multiprocessing import Pool
from pathlib import Path

import astropy.constants as c
import astropy.units as u
import dsharp_helper as dh
import dsharp_opac as do
import emcee
import numpy as np
from astropy.io import fits

from helper_functions import chop_forward_scattering
from helper_functions import get_normalized_profiles
from helper_functions import get_profile_from_fits
from helper_functions import make_opacs
from log_prob import log_prob

# np.seterr(all='raise')
logging.basicConfig(filename='run_model.log', filemode='w', level=logging.DEBUG)

# define constants
au = c.au.cgs.value
M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
R_sun = c.R_sun.cgs.value

# select radmc3d executable path
if getpass.getuser() == 'birnstiel':
    radmc3d_exec = Path('~/.bin/radmc3d').expanduser()
else:
    radmc3d_exec = Path('~/bin/radmc3d').expanduser()

disk = 'IMLup'

# disklab grid
disklab_grid = {
    "nr": 100,
    "rin": 0.1 * au,
    "rout": 400 * au,  # 400au from avenhaus paper  #DSHARP Huang 2018 says 290 au
}

# physical parameters
disk_params = {
    "r_c": 100 * au,  # from Zhang+ 2021, was 300, not sure where this value came from
    "z0": 0.2,
    "psi": 1.27,

    "mstar": 10. ** dh.sources.loc[disk]['log M_star/M_sun'] * M_sun,
    "lstar": 10. ** dh.sources.loc[disk]['log L_star/L_sun'] * L_sun,
    "tstar": 10. ** dh.sources.loc[disk]['log T_eff/ K'],
    "alpha": 1e-3,

    "PA": dh.sources.loc[disk]['PA'],
    "inc": dh.sources.loc[disk]['inc'],
    "dpc": dh.sources.loc[disk]['distance [pc]'],
}
disk_params["rstar"] = np.sqrt(disk_params["lstar"] / (4 * np.pi * c.sigma_sb.cgs.value * disk_params["tstar"] ** 4))

# ALMA data
fname_mm_obs = dh.get_datafile(disk)
clip = 2 * disklab_grid["rout"] / au / disk_params["dpc"]
lam_mm = 0.125
RMS_jyb = 14e-6

# ALMA radial profile
x_mm_obs, y_mm_obs, dy_mm_obs = get_profile_from_fits(
    fname_mm_obs,
    clip=clip,
    inc=disk_params["inc"], PA=disk_params["PA"],
    z0=0.0,
    psi=0.0,
    dist=disk_params['dpc']
)

# Sphere data
lam_sca = 1.65e-4
fname_sca_obs_orig = 'observations/IM_Lup_reducedRob_median_Hband_12.25mas.fits'

# pixel size of the sphere image, converted to degree
pixelsize = (12.5 * u.mas).to('deg').value
# the "beam" assumed in the radial profile calculation
beam_sca = (0.040, 0.040, 0.0)

# RMS of the observations
RMS_sca = ...

# The image does not contain all the required info, so we make a copy of the fits file and modify that one
fname_sca_obs = fname_sca_obs_orig.replace('.fits', '_mod.fits')

shutil.copy(fname_sca_obs_orig, fname_sca_obs)

fits.setval(fname_sca_obs, 'cdelt1', value=-pixelsize)
fits.setval(fname_sca_obs, 'cdelt2', value=pixelsize)
fits.setval(fname_sca_obs, 'crpix1', value=fits.getval(fname_sca_obs_orig, 'naxis1') // 2 + 0.5)
fits.setval(fname_sca_obs, 'crpix2', value=fits.getval(fname_sca_obs_orig, 'naxis2') // 2 + 0.5)
fits.setval(fname_sca_obs, 'crval1', value=0.0)
fits.setval(fname_sca_obs, 'crval2', value=0.0)
fits.setval(fname_sca_obs, 'crval3', value=1.65e-4)
fits.setval(fname_sca_obs, 'BUNIT', value='JY/PIXEL')

# read it with imagecube and derive profiles
profiles_sca_obs = get_normalized_profiles(
    fname_sca_obs,
    clip=clip,
    inc=disk_params['inc'],
    PA=disk_params['PA'],
    z0=disk_params['z0'],
    psi=disk_params['psi'],
    beam=beam_sca,
)

# Opacities
# TODOtry new and old a1, and n_a = 50, mostly small grains (constant size). Hopefully nothing changes as 50 sizes are
#  a lot
# TODOdo the same with n_a = 15, with new a_1, should still look the same. The old a_1 should look different as we
#  under-sample the grain distribution

# Define the wavelength, size, and angle grids then calculate opacities and store them in a local file,
# if it doesn't exist yet. Careful, that takes of the order of >2h
n_lam = 200  # number of wavelength points
n_a = 15  # number of particle sizes
n_theta = 181  # number of angles in the scattering phase function
porosity = 0.3

# wavelength and particle sizes grids
lam_opac = np.logspace(-5, 1, n_lam)
a_opac = np.logspace(-5, 1, n_a)

# make opacities if necessary
opac_dict = make_opacs(a_opac, lam_opac, fname='opacities/dustkappa_IMLUP', porosity=porosity, n_theta=n_theta)
fname_opac = opac_dict['filename']

# This part chops the very-forward scattering part of the phase function.
# This part is basically the same as no scattering, but are treated by the code as a scattering event.
# By cutting this part out of the phase function, we avoid those non-scattering scattering events.

# fname_opac_chopped = fname_opac.replace('.', '_chopped.')
fname_opac_chopped = '_chopped.'.join(fname_opac.rsplit('.', 1))

k_sca_nochop = opac_dict['k_sca']
g_nochop = opac_dict['g']

zscat, zscat_nochop, k_sca, g = chop_forward_scattering(opac_dict)

opac_dict['k_sca'] = k_sca
opac_dict['zscat'] = zscat
opac_dict['g'] = g

rho_s = opac_dict['rho_s']
m = 4 * np.pi / 3 * rho_s * a_opac ** 3

do.write_disklab_opacity(fname_opac_chopped, opac_dict)

# Put all options in a dictionary

options = {'disk': disk, 'PA': disk_params['PA'], 'inc': disk_params['inc'], 'distance': disk_params['dpc'],
           'clip': clip, 'lam_mm': lam_mm, 'RMS_jyb': RMS_jyb, 'mstar': disk_params['mstar'],
           'lstar': disk_params['lstar'], 'tstar': disk_params['tstar'], 'rstar': disk_params['rstar'],
           'x_mm_obs': x_mm_obs, 'y_mm_obs': y_mm_obs, 'dy_mm_obs': dy_mm_obs, 'fname_mm_obs': fname_mm_obs,
           'z0': disk_params['z0'], 'psi': disk_params['psi'], 'alpha': disk_params['alpha'], 'lam_sca': lam_sca,
           'fname_sca_obs': fname_sca_obs, 'beam_sca': beam_sca, 'RMS_sca': RMS_sca,
           'profiles_sca_obs': profiles_sca_obs, 'fname_opac': fname_opac_chopped, 'nr': disklab_grid['nr'],
           'rin': disklab_grid['rin'], 'r_c': disk_params['r_c'], 'rout': disklab_grid['rout']}

pickle.dump(options, open("options.pickle", "wb"))

# Emcee
# Here we define some inputs and initial parameter sets for the optimization

# defining number of walkers
nwalkers = 30  # it  does not work with fewer  walkers than the number  of dimensions
ndim = 7

# Setting the priors for some parameters instead of letting them be uniform randoms between (0.1)
sigma_coeff_0 = 10 ** ((np.random.rand(nwalkers) - 0.5) * 4)
others_0 = np.random.rand(ndim - 3, nwalkers)
d2g_coeff_0 = (np.random.rand(nwalkers) + 0.5) / 100
d2g_exp_0 = (np.random.rand(nwalkers) - 0.5)

# Input matrix of priors
p0 = np.vstack((sigma_coeff_0, others_0, d2g_coeff_0, d2g_exp_0)).T

# hpt save file
filename = 'chain.hdf5'

backend = emcee.backends.HDFBackend(filename)
# backend.reset(nwalkers, ndim)

procs = 4  # 30
steps = 8  # 30

# if procs > 1:
#     # Parallelize the simulation
#     with Pool(processes=procs) as pool:
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[options, False], pool=pool, backend=backend)
#         res = sampler.run_mcmc(p0, steps, progress=True, store=True)
# else:
#     sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[options, False], backend=backend)
#     res = sampler.run_mcmc(p0, steps, progress=True, store=True)
