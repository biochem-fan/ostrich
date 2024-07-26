# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

import math
import numpy as np

from cctbx import factor_ev_angstrom
from cctbx.eltbx import attenuation_coefficient
from dials.array_family import flex
from dxtbx.model.beam import BeamFactory
from dxtbx.model.detector import Detector
from dxtbx.model import ParallaxCorrectedPxMmStrategy
from dxtbx.format.FormatStill import FormatStill
from scitbx import matrix

class FormatSACLAInMemory(FormatStill):
    def __init__(self, buffers, geometry, energy, adu_per_photon, mask=None, distance=50.0):
        assert len(buffers) == len(geometry.panels)
        if mask is not None:
            assert len(mask) == len(geometry.panels)

        self._mask = mask
        self._image = tuple(flex.float(buf) for buf in buffers)
        self._beam = BeamFactory.simple(factor_ev_angstrom / energy) 
        self.setup_detector(geometry, distance, adu_per_photon)

    def setup_detector(self, geometry, distance, adu_per_photon):
        wavelength = self.get_beam().get_wavelength()

        table = attenuation_coefficient.get_table("Si")
        # According to dxtbx/model/parallax_correction.h,
        # the unit of mu is 1/mm according, thickness t0 is mm, while geometry.thickness is um.
        # According to cctbx/eltbx/attenuation_coefficient.h,
        # mu_at_angstrom returns 1/cm
        mu = table.mu_at_angstrom(wavelength) / 10.0
        px_mm = ParallaxCorrectedPxMmStrategy(mu, geometry.thickness / 1000.0)

        detector = Detector()
        root = detector.hierarchy()
        root.set_frame((-1, 0, 0),
                       ( 0, 1, 0),
                       ( 0, 0, -distance))

        for i, panel in enumerate(geometry.panels):
            angle = panel.rotation
            # TODO: confirm the signs!
            fast = matrix.col((math.cos(angle), math.sin(angle), 0))
            slow = matrix.col((-math.sin(angle), math.cos(angle), 0))
            normal = fast.cross(slow)

            origin = matrix.col((-panel.pos_x,
                                  panel.pos_y,
                                  panel.pos_z)) / 1000.0
            p = root.add_panel()
            p.set_type("SENSOR_PAD")
            p.set_name('Panel%d' % i)
            p.set_image_size((geometry.width, geometry.height))
            # we don't really apply saturation cutoff
            p.set_trusted_range((-10 * adu_per_photon, 65535 * adu_per_photon))
            p.set_pixel_size((geometry.pixel_size, geometry.pixel_size))
            p.set_thickness(geometry.thickness)
            p.set_local_frame(fast.elems, slow.elems, origin.elems)
            p.set_px_mm_strategy(px_mm)
            p.set_gain(adu_per_photon)

        self._detector = detector

    def get_num_images(self):
        return 1

    def get_raw_data(self, index=None):
        return self._image

    def get_detector(self, index=None):
        return self._detector

    def get_beam(self, index=None):
        return self._beam

    def get_goniometer(self, index=None):
        return None

    def get_static_mask(self):
        # TODO: I don't understand why Format.get_static_mask is never called.
        print("DEBUG: get_static_mask")
        return self._mask
