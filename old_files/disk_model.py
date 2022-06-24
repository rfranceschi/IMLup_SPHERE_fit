import logging
import pickle
import random
import shutil
import warnings
from dataclasses import dataclass, field
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

@dataclass
class DiskModel:
    params: dict = field(default_factory=dict)

    default_params = {...:...,}

    def __post_init__(self):
        self.set_default_param()

    def set_default_param(self):
        """
        Set parameters in params dict to default values when not defined.
        """
        for key, value in self.default_params:
            if key not in self.params:
                self.params[key] = value

DiskModel.params