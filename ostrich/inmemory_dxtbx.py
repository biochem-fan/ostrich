# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

import math
import numpy as np

from cctbx import factor_ev_angstrom
from cctbx.eltbx import attenuation_coefficient
from dials.array_family import flex
from dxtbx.format.FormatStill import FormatStill
from dxtbx.model.beam import BeamFactory
from dxtbx.model.detector import Detector
from dxtbx.model import ParallaxCorrectedPxMmStrategy
from scitbx import matrix

class FormatSACLAInMemory(FormatStill):
    def __init__(self, buffers, geometry, energy, adu_per_photon, distance=50.0, binning=1):
        assert len(buffers) == len(geometry.panels)

        self.invalid_panels = [buf is None for buf in buffers]
        self._mask = None
        self._image = tuple(flex.float(buf) for buf in buffers if buf is not None)
        self._beam = BeamFactory.simple(factor_ev_angstrom / energy)
        self.setup_detector(geometry, distance, adu_per_photon, binning)

    def setup_detector(self, geometry, distance, adu_per_photon, binning):
        assert geometry.width % binning == 0
        assert geometry.height % binning == 0

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
        # This makes panel's coordinate system the same as CrystFEL (and NeXus/McStats).
        root.set_frame((-1, 0, 0),
                       ( 0, 1, 0),
                       ( 0, 0, -distance))

        for i, panel in enumerate(geometry.panels):
            if self.invalid_panels[i]:
                continue

            angle = panel.rotation
            # The sign before cosines is the opposite of my old Cheetah & dxtbx class
            # but I believe this is correct.
            fast = matrix.col((-math.cos(angle), math.sin(angle), 0))
            slow = matrix.col((-math.sin(angle), -math.cos(angle), 0))
            normal = fast.cross(slow)

            origin = matrix.col((-panel.pos_x,
                                  panel.pos_y,
                                  panel.pos_z)) / 1000.0
            p = root.add_panel()
            p.set_type("SENSOR_PAD")
            p.set_name('Panel%d' % i)
            p.set_image_size((geometry.width // binning, geometry.height // binning))
            # we don't really apply saturation cutoff
            p.set_trusted_range((-10 * adu_per_photon, 65535 * adu_per_photon))
            p.set_pixel_size((geometry.pixel_size * binning, geometry.pixel_size * binning))
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
        # WARNING: This is not called if ImageSet is manually created!
        return self._mask
