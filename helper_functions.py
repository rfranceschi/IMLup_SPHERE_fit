import warnings
from pathlib import Path

import astropy.constants as c
import disklab
import dsharp_opac as opacity
import matplotlib.pyplot as plt
import numpy as np
from dipsy import get_powerlaw_dust_distribution
from dipsy.utils import get_interfaces_from_log_cell_centers
from gofish import imagecube

au = c.au.cgs.value
M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
R_sun = c.R_sun.cgs.value


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def make_disklab2d_model(
        parameters,
        mstar,
        lstar,
        tstar,
        nr,
        alpha,
        rin,
        rout,
        r_c,
        opac_fname,
        show_plots=False
):
    # The different indices in the parameters list correspond to different physical paramters

    sigma_coeff = parameters[0]
    sigma_exp = parameters[1]
    size_exp = parameters[2]
    amax_coeff = parameters[3]
    amax_exp = parameters[4]
    d2g_coeff = parameters[5]
    d2g_exp = parameters[6]

    # read some values from the parameters file

    with np.load(opac_fname) as fid:
        a_opac = fid['a']
        rho_s = fid['rho_s']
        n_a = len(a_opac)

    # start with the 1D model

    d = disklab.DiskRadialModel(mstar=mstar, lstar=lstar, tstar=tstar, nr=nr, alpha=alpha, rin=rin, rout=rout)
    d.make_disk_from_simplified_lbp(sigma_coeff, r_c, sigma_exp)

    if d.mass / mstar > 0.2:
        warnings.warn(f'Disk mass is unreasonably high: M_disk / Mstar = {d.mass / mstar:.2g}')

    # add the dust, based on the dust-to-gas parameters

    d2g = d2g_coeff * ((d.r / au) ** d2g_exp)
    a_max = amax_coeff * (d.r / au) ** (-amax_exp)

    a_i = get_interfaces_from_log_cell_centers(a_opac)
    a, a_i, sig_da = get_powerlaw_dust_distribution(d.sigma * d2g, np.minimum(a_opac[-1], a_max), q=4 - size_exp,
                                                    na=n_a, a0=a_i[0], a1=a_i[-1])

    for _sig, _a in zip(np.transpose(sig_da), a_opac):
        d.add_dust(agrain=_a, xigrain=rho_s, dtg=_sig / d.sigma)

    if show_plots:
        f, ax = plt.subplots()

        ax.contourf(d.r / au, a_opac, np.log10(sig_da.T))

        ax.loglog(d.r / au, a_max, label='a_max')
        ax.loglog(d.r / au, d2g, label='d2g')

        ax.set_xlabel('radius [au]')
        ax.set_ylabel('grain size [cm]')
        ax.set_ylim(1e-5, 1e0)
        ax.legend()

    # load the opacity from the previously calculated opacity table
    for dust in d.dust:
        dust.grain.read_opacity(str(opac_fname))

    # compute the mean opacities
    d.meanopacitymodel = ['dustcomponents', {'method': 'simplemixing'}]
    d.compute_mean_opacity()

    if show_plots:
        f, ax = plt.subplots()
        ax.loglog(d.r / au, d.mean_opacity_planck, label='mean plack')
        ax.loglog(d.r / au, d.mean_opacity_rosseland, label='mean rosseland')

        ax.set_xlabel('radius [au]')
        ax.set_ylabel('mean opacity')
        ax.legend()

    # smooth the mean opacities

    d.mean_opacity_planck[7:-7] = movingaverage(d.mean_opacity_planck, 10)[7:-7]
    d.mean_opacity_rosseland[7:-7] = movingaverage(d.mean_opacity_rosseland, 10)[7:-7]

    if show_plots:
        ax.loglog(d.r / au, d.mean_opacity_planck, 'C0--')
        ax.loglog(d.r / au, d.mean_opacity_rosseland, 'C1--')

        f, ax = plt.subplots()
        ax.loglog(d.r / au, d.tmid)

        ax.set_xlabel('radius [au]')
        ax.set_ylabel(r'T$_{mid}$')

    # iterate the temperature
    for iter in range(100):
        d.compute_hsurf()
        d.compute_flareindex()
        d.compute_flareangle_from_flareindex(inclrstar=True)
        d.compute_disktmid(keeptvisc=False)
        d.compute_cs_and_hp()

    # ---- Make a 2D model out of it ----

    disk2d = disklab.Disk2D(
        disk=d,
        meanopacitymodel=d.meanopacitymodel,
        nz=100)

    # taken from snippet vertstruc 2d_1
    for vert in disk2d.verts:
        vert.iterate_vertical_structure()
    disk2d.radial_raytrace()
    for vert in disk2d.verts:
        vert.solve_vert_rad_diffusion()
        vert.tgas = (vert.tgas ** 4 + 15 ** 4) ** (1 / 4)
        for dust in vert.dust:
            dust.compute_settling_mixing_equilibrium()

    # --- done setting up the radmc3d model ---
    return disk2d


def get_normalized_profiles(fname, **kwargs):
    """calculates the profiles in all major and minor axes

    Arguments
    ---------

    fname : str
        file name

    kwargs : are passed to `get_profile`
    """

    # determine the norm as average at 1 arcsec
    if 'r_norm' in kwargs:
        raise ValueError('do not pass r_norm to scale different profiles')

    x, y, dy = get_profile_from_fits(fname, **kwargs)
    norm = np.interp(1.0, x, y)
    kwargs['norm'] = norm

    # set the names and angles of the profiles

    masks = {}
    masks['B'] = dict(PA_min=85, PA_max=95)
    masks['T'] = dict(PA_min=-95, PA_max=-85)
    masks['L'] = dict(PA_min=-10, PA_max=0)
    masks['R'] = dict(PA_min=-180, PA_max=-170)

    profiles = dict()

    for key in masks.keys():
        x, y, dy = get_profile_from_fits(fname, **kwargs, **masks[key])
        profiles[key] = dict(x=x, y=y, dy=dy, mask=masks[key], norm=norm)

    return profiles


def get_profile_from_fits(fname, clip=2.5, show_plots=False, inc=0, PA=0, z0=0.0, psi=0.0, beam=None, r_norm=None,
                          norm=None, **kwargs):
    """Get radial profile from fits file.

    Reads a fits file and determines a radial profile with `imagecube`

    fname : str | path
        path to fits file

    clip : float
        clip the image at that many image units (usually arcsec)

    show_plots : bool
        if true: produce some plots for sanity checking

    inc, PA : float
        inclination and position angle used in the radial profile

    z0, psi : float
        the scale height at 1 arcse and the radial exponent used in the deprojection

    beam : None | tuple
        if None: will be determined by imgcube
        if 3-element tuple: assume this beam a, b, PA.

    r_norm : None | float
        if not None: normalize at this radius

    norm : None | float
        divide by this norm

    kwargs are passed to radial_profile

    Returns:
    x, y, dy: arrays
        radial grid, intensity (cgs), error (cgs)
    """

    if norm is not None and r_norm is not None:
        raise ValueError('only norm or r_norm can be set, not both!')

    data = imagecube(fname, FOV=clip)

    if beam is not None:
        data.bmaj, data.bmin, data.bpa = beam
        data.beamarea_arcsec = data._calculate_beam_area_arcsec()
        data.beamarea_str = data._calculate_beam_area_str()

    x, y, dy = data.radial_profile(inc=inc, PA=PA, z0=z0, psi=psi, **kwargs)

    if data.bunit.lower() == 'jy/beam':
        y *= 1e-23 / data.beamarea_str
        dy *= 1e-23 / data.beamarea_str
    elif data.bunit.lower() == 'jy/pixel':
        y *= 1e-23 * data.pix_per_beam / data.beamarea_str
        dy *= 1e-23 * data.pix_per_beam / data.beamarea_str
    else:
        raise ValueError('unknown unit, please implement conversion to CGS here')

    if r_norm is not None:
        norm = np.interp(r_norm, x, y)
        y /= norm
        dy /= norm

    if norm is not None:
        y /= norm
        dy /= norm

    if show_plots:
        f, ax = plt.subplots()
        ax.semilogy(x, y)
        ax.fill_between(x, y - dy, y + dy, alpha=0.5)
        ax.set_ylim(bottom=1e-16)

    return x, y, dy


def azimuthal_profile(cube, n_theta=30, **kwargs):
    """derive an azimuthal profile

    Arguments:
    ----------

    r : float
        radius around which to take the azimuthal bin

    dr : float
        radial width of the annulus

    cube : imgcube instance
        the image cube from `gofish.imgcube`

    n_theta : int
        number of bins in azimuth

    kwargs are passed to `cube.disk_coords`  and `get_mask` but can contain PA_min and PA_max
    to constrain the azimuthal extent
    """

    if kwargs.get('r_min') is None or kwargs.get('r_max') is None:
        raise ValueError('need to set at least r_min and r_max')

    r_min = kwargs.pop('r_min')
    r_max = kwargs.pop('r_max')
    PA_min = kwargs.pop('PA_min', -180)
    PA_max = kwargs.pop('PA_max', 180)

    mask = cube.get_mask(r_min=r_min, r_max=r_max, PA_min=PA_min, PA_max=PA_max, **kwargs)
    rvals, tvals, _ = cube.disk_coords(**kwargs)

    # rvals_annulus = rvals[mask]
    tvals_annulus = tvals[mask]
    dvals_annulus = cube.data[mask]

    tbins = np.linspace(np.radians(PA_min), np.radians(PA_max), n_theta + 1)
    bin_centers = 0.5 * (tbins[1:] + tbins[:-1])
    assert bin_centers.size == n_theta
    tidx = np.digitize(tvals_annulus, tbins)

    return bin_centers, \
           np.array([np.mean(dvals_annulus[tidx == t]) for t in range(1, n_theta + 1)]), \
           np.array([np.std(dvals_annulus[tidx == t]) for t in range(1, n_theta + 1)])


def make_opacs(a, lam, fname='dustkappa_IMLUP', porosity=None, constants=None, n_theta=101):
    """make optical constants file"""

    if n_theta // 2 == n_theta / 2:
        n_theta += 1
        print(f'n_theta needs to be odd, will set it to {n_theta}')

    n_a = len(a)
    n_lam = len(lam)

    if constants is None:
        if porosity is None:
            porosity = 0.0

        if porosity < 0.0 or porosity >= 1.0:
            raise ValueError('porosity has to be >=0 and <1!')

        if porosity > 0.0:
            fname = fname + f'_p{100 * porosity:.0f}'

        constants = opacity.get_dsharp_mix(porosity=porosity)
    else:
        if porosity is not None:
            raise ValueError('if constants are given, porosity keyword cannot be used')

    opac_fname = Path(fname).with_suffix('.npz')

    diel_const, rho_s = constants

    run_opac = True

    if opac_fname.is_file():

        opac_dict = read_opacs(opac_fname)

        if (
                (len(opac_dict['a']) == n_a) and
                np.allclose(opac_dict['a'], a) and
                (len(opac_dict['lam']) == n_lam) and
                np.allclose(opac_dict['lam'], lam) and
                (len(opac_dict['theta']) == n_theta) and
                (opac_dict['rho_s'] == rho_s)
        ):
            print(f'reading from file {opac_fname}')
            run_opac = False

    if run_opac:
        # call the Mie calculation & store the opacity in a npz file
        opac_dict = opacity.get_smooth_opacities(
            a,
            lam,
            rho_s=rho_s,
            diel_const=diel_const,
            extrapolate_large_grains=False,
            n_angle=(n_theta + 1) // 2)

        print(f'writing opacity to {opac_fname} ... ', end='', flush=True)
        opacity.write_disklab_opacity(opac_fname, opac_dict)
        print('Done!')

    opac_dict['filename'] = str(opac_fname)

    return opac_dict


def read_opacs(fname):
    with np.load(fname) as fid:
        opac_dict = {k: v for k, v in fid.items()}
    return opac_dict


def chop_forward_scattering(opac_dict, chopforward=5):
    """Chop the forward scattering.

    This part chops the very-forward scattering part of the phase function.
    This very-forward scattering part is basically the same as no scattering,
    but is treated by the code as a scattering event. By cutting this part out
    of the phase function, we avoid those non-scattering scattering events.
    This needs to recalculate the scattering opacity kappa_sca and asymmetry
    factor g.

    Parameters
    ----------
    opac_dict : dict
        opacity dictionary as produced by dsharp_opac

    chopforward : float
        up to which angle to we truncate the forward scattering
    """

    k_sca = opac_dict['k_sca']
    S1 = opac_dict['S1']
    S2 = opac_dict['S2']
    theta = opac_dict['theta']
    g = opac_dict['g']
    rho_s = opac_dict['rho_s']
    lam = opac_dict['lam']
    a = opac_dict['a']
    m = 4 * np.pi / 3 * rho_s * a ** 3

    n_a = len(a)
    n_lam = len(lam)

    zscat = opacity.calculate_mueller_matrix(lam, m, S1, S2)['zscat']
    zscat_nochop = zscat.copy()

    mu = np.cos(theta * np.pi / 180.)
    # dmu = np.diff(mu)

    for grain in range(n_a):
        for i in range(n_lam):
            if chopforward > 0:
                iang = np.where(theta < chopforward)
                if theta[0] == 0.0:
                    iiang = np.max(iang) + 1
                else:
                    iiang = np.min(iang) - 1
                zscat[grain, i, iang, :] = zscat[grain, i, iiang, :]

                # zav = 0.5 * (zscat[grain, i, 1:, 0] + zscat[grain, i, :-1, 0])
                # dum = -0.5 * zav * dmu
                # integral = dum.sum() * 4 * np.pi
                # k_sca[grain, i] = integral

                # g = <mu> = 2 pi / kappa * int(Z11(mu) mu dmu)
                # mu_av = 0.5 * (mu[1:] + mu[:-1])
                # P_mu = 2 * np.pi / k_sca[grain, i] * 0.5 * (zscat[grain, i, 1:, 0] + zscat[grain, i, :-1, 0])
                # g[grain, i] = np.sum(P_mu * mu_av * dmu)

                k_sca[grain, i] = -2 * np.pi * np.trapz(zscat[grain, i, :, 0], x=mu)
                g[grain, i] = -2 * np.pi * np.trapz(zscat[grain, i, :, 0] * mu, x=mu) / k_sca[grain, i]

    return zscat, zscat_nochop, k_sca, g


def write_radmc3d(disk2d, lam, path, show_plots=False, nphot=10000000):
    """
    convert the disk2d object to radmc3d format and write the radmc3d input files.

    disk2d : disklab.Disk2D instance
        the disk

    lam : array
        wavelength grid [cm]

    path : str | path
        the path into which to write the output

    show_plots : bool
        if true: produce some plots for checking

    nphot : int
        number of photons to send
    """

    rmcd = disklab.radmc3d.get_radmc3d_arrays(disk2d, showplots=show_plots)

    # Assign the radmc3d data

    ri = rmcd['ri']
    thetai = rmcd['thetai']
    phii = rmcd['phii']
    rho = rmcd['rho']
    n_a = rho.shape[-1]

    # we need to tile this for each species

    rmcd_temp = rmcd['temp'][:, :, None] * np.ones(n_a)[None, None, :]

    # Define the wavelength grid for the radiative transfer

    lam_mic = lam * 1e4

    # Write the `RADMC3D` input

    disklab.radmc3d.write_stars_input(disk2d.disk, lam_mic, path=path)
    disklab.radmc3d.write_grid(ri, thetai, phii, mirror=False, path=path)
    disklab.radmc3d.write_dust_density(rmcd_temp, fname='dust_temperature.dat', path=path, mirror=False)
    disklab.radmc3d.write_dust_density(rho, mirror=False, path=path)
    disklab.radmc3d.write_wavelength_micron(lam_mic, path=path)
    disklab.radmc3d.write_opacity(disk2d, path=path)
    disklab.radmc3d.write_radmc3d_input(
        {
            'scattering_mode': 5,
            'scattering_mode_max': 1,  # was 5 (most realistic scattering), 1 is isotropic
            'nphot': nphot,
            'dust_2daniso_nphi': '60',
            'mc_scat_maxtauabs': '5.d0',
        },
        path=path)
