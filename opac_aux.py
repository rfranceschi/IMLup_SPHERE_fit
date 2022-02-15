from pathlib import Path
import subprocess
import tqdm
import numpy as np


def get_line(filehandle, comments=('=', '#')):
    "helper function: reads next line from file but skips comments and empty lines"
    line = filehandle.readline()
    while line.startswith(comments) or line.strip() == '':
        line = filehandle.readline()
    return line


def read_radmc3d_opacities(path, scattermatrix=True):
    # reading from RADMC3D input files to get the names of the species and number of wavelength
    lines = open(path / 'dustopac.inp', 'r')
    _ = int(get_line(lines).split()[0])
    n_a = int(get_line(lines).split()[0])

    n_f = int(np.fromfile(path / 'wavelength_micron.inp', count=1, dtype=int, sep=' '))

    names = []
    for _ in range(n_a):
        itype = int(get_line(lines).split()[0])
        if itype not in [1, 10]:
            print('ERROR!')

        gtype = int(get_line(lines).split()[0])
        if gtype != 0:
            print('strange grains detected')

        names += [get_line(lines).split()[0].strip()]

    # check which file format is present

    if scattermatrix and (path / f'dustkapscatmat_{names[0]}.inp').is_file():
        stem = 'dustkapscatmat_{}.inp'
        scattermatrix = True
    else:
        scattermatrix = False
        stem = 'dustkappa_{}.inp'

    # now for each name: read the file

    opacs = {}
    for name in names:
        file = path / stem.format(name)

        # store in dict
        opacs[name] = read_radmc_opacityfile(file)

    # the order should be according to particle size, so we reformat the dict to an array
    k_abs = np.zeros([len(names), n_f])
    k_sca = np.zeros_like(k_abs)
    g = np.zeros_like(k_abs)
    zscat = None

    for i, name in enumerate(names):
        k_abs[i, :] = opacs[name]['k_abs']
        k_sca[i, :] = opacs[name]['k_sca']
        g[i, :] = opacs[name]['g']

        # put data of this species into the big arrays
        lam = opacs[name]['lam']

        if scattermatrix:
            if zscat is None:
                theta = opacs[name]['theta']
                zscat = np.zeros([len(names), len(lam), len(theta), 6])
            zscat[i, ...] = opacs[name]['zscat']

    output = {
        'k_abs': k_abs,
        'k_sca': k_sca,
        'g': g,
        'lam': lam,
    }

    if scattermatrix:
        output['zscat'] = zscat
        output['theta'] = theta
        output['n_th'] = len(theta)

    return output


def read_radmc_opacityfile(file):
    """reads RADMC-3D opacity files, returns dictionary with its contents."""
    file = Path(file)

    if 'dustkapscatmat' in file.name:
        scatter = True

    name = '_'.join(file.stem.split('_')[1:])

    if not file.is_file():
        raise FileNotFoundError(f'file not found: {file}')

    with open(file, 'r') as f:
        iformat = int(get_line(f))
        if iformat == 2:
            ncol = 3
        elif iformat in [1, 3]:
            ncol = 4
        else:
            raise ValueError('Format of opacity file unknown')
        n_f = int(get_line(f))

        # read also number of angles for scattering matrix
        if scatter:
            n_th = int(get_line(f))

        # read wavelength, k_abs, k_sca, g
        data = np.fromfile(f, dtype=np.float64, count=n_f * ncol, sep=' ')

        # read angles and zscat
        if scatter:
            theta = np.fromfile(f, dtype=np.float64, count=n_th, sep=' ')
            zscat = np.fromfile(f, dtype=np.float64, count=n_th * n_f * 6, sep=' ').reshape([n_th, n_f, 6])
            zscat = np.moveaxis(zscat, 0, 1)

    data = data.reshape(n_f, ncol)
    lam = 1e-4 * data[:, 0]
    k_abs = data[:, 1]
    k_sca = data[:, 2]

    if iformat in [1, 3]:
        opac_gsca = 1.0 * data[:, 3]

    # define the output

    output = {
        'lam': lam,
        'k_abs': k_abs,
        'k_sca': k_sca,
        'name': name,
    }

    if iformat in [1, 3]:
        output['g'] = opac_gsca
        if scatter:
            output['theta'] = theta
            output['n_th'] = n_th
            output['zscat'] = zscat

    return output


def optool_wrapper(a, lam, chop=5, porosity=0.3):
    """
    Wrapper for optool to calculate DSHARP opacities in RADMC-3D format.

    Parameters
    ----------
    a : array
        particle size array
    lam : array | str
        either a string pointing to a RADMC-3d wavelength file or 3-elements: min & max & number of wavelengths
    chop : float, optional
        below how many degrees to chop forward scattering peak, by default 5
    porosity : float, optional
        grain porosity, by default 0.3

    Returns
    -------
    dict

    """
    if isinstance(lam, str):
        lam_str = lam
        nlam = int(np.fromfile(lam_str, count=1, dtype=int, sep=' '))
    elif len(lam) == 3:
        nlam = lam[3]
        lam_str = '%e %e %d' % tuple(lam)
    else:
        raise ValueError('lam needs to be a file or of length 3 (lmin, lmax, nl)')

    # initialize arrays

    k_abs = np.zeros([len(a), nlam])
    k_sca = np.zeros_like(k_abs)
    g = np.zeros_like(k_abs)
    zscat = None

    # start reading

    for ia, _a in tqdm.tqdm(enumerate(a), total=len(a)):
        cmd = f'optool -mie -chop {chop} -s 180 -p {porosity} -c h2o-w 0.2 -c astrosil 0.3291 -c fes 0.0743 -c c-org 0.3966 -a {_a * 0.9e4} {_a * 1.1e4} 3.5 10 -l {lam_str} -radmc'
        result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
        output = result.stdout.decode()

        if output.split()[-1] == 'dustkapscatmat.inp':
            scatter = True
            fname = 'dustkapscatmat.inp'
        elif output.split()[-1] == 'dustkappa.inp':
            scatter = False
            fname = 'dustkappa.inp'

        # read data, remove file
        optool_data = read_radmc_opacityfile(fname)
        Path(fname).unlink()

        # put data of this particle into the big arrays
        k_abs[ia, :] = optool_data['k_abs']
        k_sca[ia, :] = optool_data['k_sca']
        g[ia, :] = optool_data['g']
        lam = optool_data['lam']

        if scatter:
            theta = optool_data['theta']
            if zscat is None:
                zscat = np.zeros([len(a), len(lam), len(theta), 6])
            zscat[ia, ...] = optool_data['zscat']

    output = {
        'lam': lam,
        'k_abs': k_abs,
        'k_sca': k_sca,
        'g': g,
        'output': output,
    }

    if scatter:
        output['zscat'] = zscat
        output['theta'] = theta
        output['n_th'] = len(theta)

    return output
