#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import shutil
import tempfile
import warnings
from pathlib import Path
import random

import astropy.constants as c
import disklab
import dsharp_opac as opacity
import numpy as np
from dipsy.utils import Capturing
from disklab.radmc3d import write
from gofish import imagecube
from radmc3dPy import *
from pathlib import Path

from helper_functions import get_normalized_profiles
from helper_functions import get_profile_from_fits
from helper_functions import make_disklab2d_model
from helper_functions import read_opacs
from helper_functions import write_radmc3d

au = c.au.cgs.value
M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
R_sun = c.R_sun.cgs.value


def calculate_chisquared(sim_data, obs_data, error):
    """

    Args:
        sim_data:
        obs_data:
        error:

    Returns:

    """

    error = error + 1e-100
    return np.sum((obs_data - sim_data) ** 2 / (error ** 2))


def log_prob(parameters, options, debugging=False):
    params = {
        "sigma_coeff": parameters[0],
        "sigma_exp": parameters[1],
        "size_exp": parameters[2],
        "amax_coeff": parameters[3],
        "amax_exp": parameters[4],
        "d2g_coeff": parameters[5],
        "d2g_exp": parameters[6],
    }

    output_dict = {}

    output = Capturing()

    if not (0 < params['sigma_coeff'] < 1e4 or
            -5 < params['sigma_exp'] < 5 or
            -5 < params['size_exp'] < 5 or
            1e-4 < params['amax_coeff'] < 100 or
            -5 < params['amax_exp'] < 5 or
            1e-6 < params['d2g_coeff'] < 1e2 or
            -5 < params['d2g_exp'] < 5):
        print("Parameters out of prior")
        return -np.Inf, "Parameters out of prior"

    radmc3d_exec = Path('~/bin/radmc3d').expanduser()

    # temp_directory = tempfile.TemporaryDirectory(dir='.')
    # temp_path = temp_directory.name
    ...
    temp_number = random.getrandbits(32)
    output_dir = Path('runs')
    temp_path = output_dir / f'test_{temp_number}'
    temp_path.mkdir(parents=True, exist_ok=True)

    # make the disklab 2D model
    disk2d = make_disklab2d_model(
        parameters,
        options['mstar'],
        options['lstar'],
        options['tstar'],
        options['nr'],
        options['alpha'],
        options['rin'],
        options['rout'],
        options['r_c'],
        options['fname_opac'],
        show_plots=False
    )

    print(f'disk to star mass ratio = {disk2d.disk.mass / disk2d.disk.mstar:.2g}')

    # read the wavelength grid from the opacity file and write out radmc setup
    opac_dict = read_opacs(options['fname_opac'])
    lam_opac = opac_dict['lam']
    n_a = len(opac_dict['a'])

    try:
        write_radmc3d(disk2d, lam_opac, temp_path, show_plots=False)
    except Exception:
        warnings.warn(f"Index error in write_radmc3d")
        return -np.Inf, "Index error in write_radmc3d"

    # calculate the mm continuum image
    fname_mm_sim = temp_path / 'image_mm.fits'

    with output:
        disklab.radmc3d.radmc3d(
            f"image incl {options['inc']} posang {options['PA'] - 90} npix 500 lambda {options['lam_mm'] * 1e4} sizeau {2 * options['rout'] / au} secondorder  setthreads 1",
            path=temp_path,
            executable=str(radmc3d_exec)
        )

    radmc_image_path = temp_path / 'image.out'

    if radmc_image_path.is_file():
        im_mm_sim = image.readImage(str(radmc_image_path))
        radmc_image_path.replace(temp_path / 'image_mm.out')
        im_mm_sim.writeFits(str(fname_mm_sim), dpc=options['distance'], coord='15h56m09.17658s -37d56m06.1193s')

    iq_mm_obs = imagecube(str(options['fname_mm_obs']), FOV=options['clip'])
    iq_mm_sim = imagecube(str(fname_mm_sim))
    iq_mm_sim.bmaj, iq_mm_sim.bmin, iq_mm_sim.bpa = iq_mm_obs.beam
    iq_mm_sim.beamarea_arcsec = iq_mm_sim._calculate_beam_area_arcsec()
    iq_mm_sim.beamarea_str = iq_mm_sim._calculate_beam_area_str()

    x_mm_sim, y_mm_sim, dy_mm_sim = get_profile_from_fits(
        str(fname_mm_sim),
        rvals=options['x_mm_obs'],
        inc=options['inc'],
        PA=options['PA'],
        z0=0.0,
        psi=0.0,
        beam=iq_mm_obs.beam,
        show_plots=False)

    if not np.allclose(x_mm_sim, options['x_mm_obs']):
        raise ValueError('observed and simulated millimeter radial profile grids are not equal')

    print(f"y_mm_sim: {y_mm_sim}")
    print(f"y_mm_obs: {options['y_mm_obs']}")
    print(f"dy_mm_obs: {options['dy_mm_obs']}")

    chi_squared = calculate_chisquared(y_mm_sim, options['y_mm_obs'], options['dy_mm_obs'])

    print(f"Chi quadro {temp_number}: {chi_squared}")

    for i_grain in range(n_a):
        opacity.write_radmc3d_scatmat_file(i_grain, opac_dict, f'{i_grain}', path=temp_path)

    with open(Path(temp_path) / 'dustopac.inp', 'w') as f:
        write(f, '2               Format number of this file')
        write(f, '{}              Nr of dust species'.format(n_a))

        for i_grain in range(n_a):
            write(f, '============================================================================')
            write(f, '10               Way in which this dust species is read')
            write(f, '0               0=Thermal grain')
            write(f, '{}              Extension of name of dustscatmat_***.inp file'.format(i_grain))

        write(f, '----------------------------------------------------------------------------')

    iq_sca_obs = imagecube(str(options['fname_sca_obs']), FOV=options['clip'])

    disklab.radmc3d.write_radmc3d_input(
        {
            'scattering_mode': 5,
            'scattering_mode_max': 5,  # was 5 (most realistic scattering), 1 is isotropic
            'nphot': 10000000,
            'dust_2daniso_nphi': '60',
            'mc_scat_maxtauabs': '5.d0',
        },
        path=temp_path)

    with output:
        disklab.radmc3d.radmc3d(
            f"image incl {options['inc']} posang {options['PA'] - 90} npix {iq_sca_obs.data.shape[0]} lambda {options['lam_sca'] / 1e-4} sizeau {2 * options['rout'] / au} setthreads 4",
            path=temp_path,
            executable=str(radmc3d_exec))

    fname_sca_sim = temp_path / 'image_sca.fits'
    if (temp_path / 'image.out').is_file():
        (temp_path / 'image.out').replace(fname_sca_sim.with_suffix('.out'))

    output_dict["radmc_call"] = f"image incl {options['inc']} posang {options['PA'] - 90} npix {iq_sca_obs.data.shape[0]} lambda {options['lam_sca'] / 1e-4} sizeau {2 * options['rout'] / au} setthreads 4"
    output_dict["folder_path"] = str(fname_sca_sim)

    try:
        im = image.readImage(str(fname_sca_sim.with_suffix('.out')))
        # output_dict["output"] = output
    except FileNotFoundError:
        warnings.warn(f"File was not found {str(fname_sca_sim.with_suffix('.out'))}")

        shutil.copytree(temp_path, temp_path + "_error")

        print("Image file was not found")
        return -np.inf, "Image file was not found"

    im.writeFits(str(fname_sca_sim), dpc=options['distance'], coord='15h56m09.17658s -37d56m06.1193s')

    iq_sca_sim = imagecube(str(fname_sca_sim), clip=options['clip'])

    for iq in [iq_sca_obs, iq_sca_sim]:
        iq.bmaj, iq.bmin, iq.bpa = options['beam_sca']
        iq.beamarea_arcsec = iq._calculate_beam_area_arcsec()
        iq.beamarea_str = iq._calculate_beam_area_str()

    profiles_sca_sim = get_normalized_profiles(
        str(fname_sca_sim),
        rvals=options['profiles_sca_obs']['B']['x'],
        # clip=options['clip'],
        inc=options['inc'],
        PA=options['PA'],
        z0=options['z0'],
        psi=options['psi'],
        beam=options['beam_sca'])

    try:
        assert np.allclose(options['profiles_sca_obs']['B']['x'], profiles_sca_sim['B']['x'])
    except Exception:
        warnings.warn(f"observed and simulated scattered radial profile grids are not equal")

    for i, key in enumerate(options['profiles_sca_obs'].keys()):
        profile_obs = options['profiles_sca_obs'][key]
        profile_sim = profiles_sca_sim[key]

        # we ignore the inner 1 arcsec as we are interested in the outer disk
        i_obs_0 = profile_obs['x'].searchsorted(1.0)
        i_sim_0 = profile_sim['x'].searchsorted(1.0)
        max_len = min(len(profile_obs['x']), len(profile_sim['x'])) - 1

        assert np.allclose(profile_sim['x'][i_sim_0:max_len],
                           profile_obs['x'][i_obs_0:max_len]), 'x arrays do not agree'

        chi_squared += calculate_chisquared(profile_sim['y'][i_sim_0:max_len],
                                            profile_obs['y'][i_obs_0:max_len],
                                            profile_obs['dy'][i_obs_0:max_len])

        print(f"y_sca_sim: {profile_sim['y'][i_sim_0:max_len]}")
        print(f"y_sca_obs: {profile_obs['y'][i_obs_0:max_len]}")
        print(f"dy_sca_obs: {profile_obs['dy'][i_obs_0:max_len]}")

        print(f"Chi quadro {temp_number}: {chi_squared}")


    logp = -np.log(chi_squared)

    if debugging:

        # output_directory = 'out_dir'
        # Path(output_directory).mkdir(parents=True, exist_ok=True)
        # filename = Path(output_directory) / f'run_{fnr}.pickle'

        filename = output_dir / f'run_{temp_number}.pickle'
        print(str(filename))

        with filename.open('wb') as fn:
            pickle.dump(output_dict, fn)
        return logp, temp_number
    else:
        return logp, ''
