#!/usr/bin/env python
# coding: utf-8
# import os
# import tempfile
import logging
import pickle
import random
import shutil
import warnings
from pathlib import Path

import astropy.constants as c
import astropy.units as u
import disklab
import dsharp_opac as opacity
import numpy as np
from astropy.io import fits
from dipsy.utils import Capturing
from disklab.radmc3d import write
from gofish import imagecube
from radmc3dPy import image

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


def log_prob(parameters, options, debugging=False, run_id=None):
    params = {
        "size_exp": parameters[0],
        "amax_exp": parameters[1],
        "d2g_coeff": parameters[2],
        "d2g_exp": parameters[3],
        "cutoff_exp_d2g": parameters[4],
        "amax_coeff": parameters[5],
        "cutoff_r": parameters[6],
        # "cutoff_exp_amax": parameters[7],
    }

    temp_number = random.getrandbits(32)
    output_dir = options['output_dir']
    if run_id is None:
        temp_path = output_dir / f'run_{temp_number}'
    else:
        temp_path = output_dir / f'run_{run_id}'
    temp_path.mkdir(parents=True, exist_ok=True)

    output_dict = {'params': params}

    output = Capturing()

    if not (
            (0 < params['size_exp'] < 4)
            and (0 < params['amax_exp'] < 10)
            and (1e-6 < params['d2g_coeff'] < 1e-1)
            and (0 < params['d2g_exp'] < 3)
            # and (280 < params['cutoff_r'] < 320)
            and (params['cutoff_exp_d2g'] >= 0)
            and (1e-5 < params['amax_coeff'] < 1e1)
            and (250 < params['cutoff_r'] < 350)
            # and (params['cutoff_exp_d2g'] >= 0)
    ):
        print("Parameters out of prior")
        return -np.Inf, -1

    radmc3d_exec = options['radmc3d_exec']

    # temp_directory = tempfile.TemporaryDirectory(dir='.')
    # temp_path = temp_directory.name

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

    output_dict['disk2d'] = disk2d

    # read the wavelength grid from the opacity file and write out radmc setup
    opac_dict = read_opacs(options['fname_opac'])
    lam_opac = opac_dict['lam']
    n_a = len(opac_dict['a'])

    # try:
    write_radmc3d(disk2d, lam_opac, temp_path, show_plots=False)
    # except Exception:
    #    warnings.warn("Index error in write_radmc3d")
    #    return -np.Inf, "Index error in write_radmc3d"

    # write the detailed scattering matrix files
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

    # read the real disk images

    with output:
        iq_mm_obs = imagecube(str(options['fname_mm_obs']), FOV=options['clip'])
        iq_sca_obs = imagecube(str(options['fname_sca_obs']), FOV=options['clip'])

        # calculate the mm continuum image
        fname_mm_sim = temp_path / 'image_mm.fits'

        radmc_call_mm = f"image incl {options['inc']} posang {options['PA'] - 90} npix 500 lambda {options['lam_mm'] * 1e4} sizeau {2 * options['rout'] / au} setthreads 4"
        disklab.radmc3d.radmc3d(
            radmc_call_mm,
            path=temp_path,
            executable=str(radmc3d_exec)
        )

    # convert to fits file

    radmc_image_path = temp_path / 'image.out'
    if radmc_image_path.is_file():
        im_mm_sim = image.readImage(str(radmc_image_path))
        radmc_image_path.replace(temp_path / 'image_mm.out')
        im_mm_sim.writeFits(str(fname_mm_sim), dpc=options['distance'], coord='15h56m09.17658s -37d56m06.1193s')
    else:
        shutil.move(temp_path, str(temp_path) + "_mm_error")
        warnings.warn(
            f"continuum image failed to run, folder copied to {str(temp_path) + '_mm_error'}, radmc3d call was {radmc_call_mm}")
        output_dict['error'] = "continuum image failed to run"
        output_dict['radmc_call_mm'] = radmc_call_mm
        print(output_dict['error'])

        filename = output_dir / f'run_{temp_number}.pickle'
        with filename.open('wb') as fn:
            pickle.dump(output_dict, fn)

        return -np.inf, temp_number

    # read as image cube and copy beam properties from observations
    with output:
        iq_mm_sim = imagecube(str(fname_mm_sim))
        iq_mm_sim.bmaj, iq_mm_sim.bmin, iq_mm_sim.bpa = iq_mm_obs.beam
        iq_mm_sim.beamarea_arcsec = iq_mm_sim._calculate_beam_area_arcsec()
        iq_mm_sim.beamarea_str = iq_mm_sim._calculate_beam_area_str()

        # derive the radial profile
        x_mm_sim, y_mm_sim, dy_mm_sim = get_profile_from_fits(
            str(fname_mm_sim),
            clip=options['clip'],
            inc=options['inc'],
            PA=options['PA'],
            z0=0.0,
            psi=0.0,
            beam=iq_mm_obs.beam,
            show_plots=False,
            dist=options['distance'])

    # clip the profiles if they are not of the same length

    x_mm_obs = options['x_mm_obs']
    y_mm_obs = options['y_mm_obs']
    dy_mm_obs = options['dy_mm_obs']

    if not (len(options['x_mm_obs']) == len(x_mm_sim)):
        i_max = min(len(x_mm_obs), len(x_mm_sim)) - 1
        i_min_obs = x_mm_obs.searchsorted(1.0)
        i_min_sim = x_mm_sim.searchsorted(1.0)
        x_mm_sim = x_mm_sim[i_min_sim:i_max]
        y_mm_sim = y_mm_sim[i_min_sim:i_max]
        dy_mm_sim = dy_mm_sim[i_min_sim:i_max]
        x_mm_obs = x_mm_obs[i_min_obs:i_max]
        y_mm_obs = y_mm_obs[i_min_obs:i_max]
        dy_mm_obs = dy_mm_obs[i_min_obs:i_max]

    if not np.allclose(x_mm_sim, x_mm_obs):
        raise ValueError(f'observed and simulated millimeter radial profile grids are not equal (run {temp_number})')

    # calculate the chi-squared value from it
    x_beam_as = np.sqrt(iq_mm_obs.beamarea_arcsec * 4 * np.log(2) / np.pi)
    rms = options['RMS_jyb'] / (iq_mm_obs.beamarea_arcsec * (u.arcsec ** 2).to('sr')) * (1 * u.Jy).cgs.value
    rms_weighted = rms / np.sqrt(x_mm_obs / (2 * np.pi * x_beam_as))

    output_dict['y_mm_sim'] = y_mm_sim
    output_dict['y_mm_obs'] = y_mm_obs
    output_dict['x_mm_sim'] = x_mm_sim
    output_dict['x_mm_obs'] = x_mm_obs
    output_dict['dy_mm_obs'] = dy_mm_obs
    output_dict['dy_mm_sim'] = dy_mm_sim
    output_dict['error'] = np.maximum(rms_weighted, dy_mm_obs)

    chi_squared = calculate_chisquared(y_mm_sim, y_mm_obs, np.maximum(rms_weighted, dy_mm_obs))

    # call RADMC-3D to calculate sphere image

    with output:
        # a bit complicated probably due to difference in pixel center / interface
        sizeau = np.diff(iq_sca_obs.xaxis[[-1, 0]])[0] * options['distance'] * iq_sca_obs.nxpix / (
                iq_sca_obs.nxpix - 1) * 1.0000000000000286
        radmc_call_sca = f"image incl {options['inc']} posang {options['PA'] - 90} npix {iq_sca_obs.data.shape[0]} lambda {options['lam_sca'] * 1e4} sizeau {sizeau} setthreads 4 stokes"
        disklab.radmc3d.radmc3d(
            radmc_call_sca,
            path=temp_path,
            executable=str(radmc3d_exec))

    if not (temp_path / 'image.out').is_file():
        shutil.move(str(temp_path), str(temp_path) + "_sca_error")
        warnings.warn(
            f"scattered light image failed to run, folder copied to {str(temp_path) + '_sca_error'}, radmc3d call was {radmc_call_sca}")
        output_dict['error'] = "scattered image failed to run"
        output_dict['output'] = output.copy()
        output_dict['radmc_call_sca'] = radmc_call_sca

        filename = output_dir / f'run_{run_id}.pickle'
        with filename.open('wb') as fn:
            pickle.dump(output_dict, fn)

        raise ValueError('breakpoint')
        return -np.inf, temp_number

    radmc_out = temp_path / 'image_sca.out'
    (temp_path / 'image.out').replace(radmc_out)
    for i, _stokes in enumerate(['Q', 'U']):
        # if image was created: rename and read it in and write it as fits file
        fname_sca_sim = temp_path / f'image_{_stokes}.fits'
        im = image.readImage(str(radmc_out))
        # here we need to shift the pixels to match the scattered light image
        im.writeFits(str(fname_sca_sim), dpc=options['distance'],
                     fitsheadkeys={'CRPIX1': iq_sca_obs.nxpix / 2 + 1, 'CRPIX2': iq_sca_obs.nxpix / 2 + 1},
                     stokes=_stokes)

    # compute Qphi image
    fname_sca_q_sim = temp_path / 'image_Q.fits'
    fname_sca_u_sim = temp_path / 'image_U.fits'

    for fname in (fname_sca_q_sim, fname_sca_u_sim):
        fits.setval(fname, 'CUNIT3', value='Hz')
        fits.setval(fname, 'BUNIT', value='Jy/pixel')

    with fits.open(fname_sca_q_sim, mode='update') as hdul:
        # RADMC convention is different than the standard
        hdul[0].data[0] = -hdul[0].data[0]
        data_sca_q_sim = hdul[0].data[0]
        hdul.flush()

    with fits.open(fname_sca_u_sim, mode='update') as hdul:
        # RADMC convention is different than the standard
        hdul[0].data[0] = -hdul[0].data[0]
        data_sca_u_sim = hdul[0].data[0]
        hdul.flush()

    if fits.getval(fname_sca_q_sim, 'CRPIX1') != fits.getval(fname_sca_u_sim, 'CRPIX1') or \
            fits.getval(fname_sca_q_sim, 'CRPIX2') != fits.getval(fname_sca_u_sim, 'CRPIX2'):
        raise ValueError("The simulated Q and U images have a different central pixel")

    x0 = fits.getval(fname_sca_q_sim, 'CRPIX1')
    y0 = fits.getval(fname_sca_q_sim, 'CRPIX2')

    data_sca_qphi_sim = np.zeros_like(data_sca_q_sim)
    for i in range(len(data_sca_q_sim)):
        i = len(data_sca_qphi_sim) - i - 1
        for j in range(len(data_sca_q_sim[i])):
            if j == y0:
                continue
            phi = np.arctan((i - x0) / (j - y0))
            data_sca_qphi_sim[i, j] = data_sca_q_sim[i, j] * np.cos(2 * phi) + data_sca_u_sim[i, j] * np.sin(2 * phi)

    with fits.open(fname_sca_q_sim) as hdul:
        header = hdul[0].header

    fname_qphi_sim = temp_path / 'image_Qphi.fits'
    hdu = fits.PrimaryHDU(data_sca_qphi_sim, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(fname_qphi_sim)

    with output:
        # read as image cube and copy beam properties from observations
        iq_qphi_sim = imagecube(str(fname_qphi_sim), FOV=options['clip'])

        for iq in [iq_sca_obs, iq_qphi_sim]:
            iq.bmaj, iq.bmin, iq.bpa = options['beam_sca']
            iq.beamarea_arcsec = iq._calculate_beam_area_arcsec()
            iq.beamarea_str = iq._calculate_beam_area_str()

        profiles_sca_sim = get_normalized_profiles(
            str(fname_qphi_sim),
            clip=options['clip'],
            inc=options['inc'],
            PA=options['PA'],
            z0=options['z0'],
            psi=options['psi'],
            beam=options['beam_sca'])

    try:
        assert np.allclose(options['profiles_sca_obs']['B']['x'], profiles_sca_sim['B']['x'])
    except Exception:
        warnings.warn(f"observed and simulated scattered radial profile grids are not equal [run {temp_number}]")

    for i, key in enumerate(options['profiles_sca_obs'].keys()):
        profile_obs = options['profiles_sca_obs'][key]
        profile_sim = profiles_sca_sim[key]

        # we ignore the inner 1 arcsec as we are interested in the outer disk
        i_obs_0 = profile_obs['x'].searchsorted(1.0)
        i_sim_0 = profile_sim['x'].searchsorted(1.0)
        max_len = min(len(profile_obs['x']), len(profile_sim['x'])) - 1

        assert np.allclose(profile_sim['x'][i_sim_0:max_len],
                           profile_obs['x'][i_obs_0:max_len]), 'x arrays do not agree'

        x_beam_sca_as = np.sqrt(iq_sca_obs.beamarea_arcsec * 4 * np.log(2) / np.pi)
        rms_sca = profile_obs['dy'][i_obs_0:max_len] / (iq_sca_obs.beamarea_arcsec * (u.arcsec ** 2).to('sr')) * (
                1 * u.Jy).cgs.value
        # in the next line 10 deg is the aperture of the cones from which we extracted the profiles
        rms_sca_weighted = rms_sca / np.sqrt(
            profile_obs['x'][i_obs_0:max_len] / (2 * np.pi * x_beam_sca_as / (10 * u.deg).to(u.rad).value))

        # divide by 4 because we have 4 sca and one mm profiles
        chi_squared += 0.25 * calculate_chisquared(profile_sim['y'][i_sim_0:max_len],
                                                   profile_obs['y'][i_obs_0:max_len],
                                                   np.maximum(rms_sca_weighted, profile_obs['dy'][i_obs_0:max_len]))
    # Jeffreys’ prior
    logp = -chi_squared + np.log(parameters[1] * parameters[3])

    # we keep some results stored in a dictionary

    output_dict["radmc_call_mm"] = radmc_call_mm
    output_dict["radmc_call_sca"] = radmc_call_sca
    output_dict["folder_path"] = str(temp_path)
    output_dict['iq_mm_obs'] = iq_mm_obs
    output_dict['iq_mm_sim'] = iq_mm_sim
    output_dict['iq_sca_obs'] = iq_sca_obs
    output_dict['iq_sca_sim'] = iq_qphi_sim
    output_dict['profiles_sca_sim'] = profiles_sca_sim
    output_dict['profiles_sca_obs'] = options['profiles_sca_obs']
    output_dict['profiles_sca_obs'] = options['profiles_sca_obs']
    output_dict['x_mm_sim'] = x_mm_sim
    output_dict['y_mm_sim'] = y_mm_sim
    output_dict['dy_mm_sim'] = dy_mm_sim
    output_dict['logp'] = logp

    # store the pickled dictionary
    if run_id is None:
        filename = output_dir / f'run_{temp_number}.pickle'
    else:
        filename = output_dir / f'run_{run_id}.pickle'

    with filename.open('wb') as fn:
        pickle.dump(output_dict, fn)

    if not debugging:
        shutil.rmtree(temp_path, ignore_errors=True)

    if np.isnan(logp):
        logging.warning(f"Probability function returned NaN for run ID {temp_number}")
        return -np.inf, temp_number

    return logp, temp_number


# if __name__ == '__main__':
def main():
    """
    "sigma_coeff": parameters[0],
    "sigma_exp": parameters[1],
    "size_exp": parameters[2], a**(4 - size_exp) grain size distribution
    "amax_coeff": parameters[3],
    "amax_exp": parameters[4],
    "d2g_coeff": parameters[5],
    "d2g_exp": parameters[6],
    """
    # import warnings
    # warnings.simplefilter('error', category=RuntimeWarning)

    fname = "options.pickle"

    with open(fname, "rb") as fb:
        options = pickle.load(fb)

    # a_max_300 = options['lam_mm'] / (2 * np.pi)
    options['rin'] = 0.2 * au

    # original
    p0 = [
        0.657,
        3.592,
        0.007,
        0.352,
        0,
        0.020,
        300,
    ]

    #  - dust density at 1 au ~ 200 g / cm3

    # i_param = 4
    # param_array = np.linspace(0.1, 1, 4, endpoint=True)
    #
    # for _i, _param in enumerate(param_array):
    #     params = p0
    #     params[i_param] = _param
    #     prob, blob = log_prob(params, options, debugging=True, run_id=f'p{i_param}_{_param:.1f}')

    prob, blob = log_prob(p0, options, debugging=True, run_id='test_cutoff_amax_lower_cutoff')
    print(prob, blob)

    # with open('run_results.txt', 'a') as fff:
    #     for i, _par in enumerate(param_change):
    #         pars = p0
    #         pars[param_index] = _par
    #         prob, blob = log_prob(pars, options, debugging=True, run_id=f'p{param_index}_{_par:.2f}')
    #         fff.write(f'p{param_index}={_par}, logp={prob}, blob={blob}, pars: {pars}\n')


if __name__ == '__main__':
    import time

    start_time = time.time()
    main()
    print(f"---- execution time {time.time() - start_time} ----")
