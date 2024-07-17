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

class FormatMPCCDInMemory(FormatStill):
    def __init__(self, buffers, geometry, energy, distance=50.0):
        assert len(buffers) == len(geometry.panels)

        self._image = tuple(flex.int(buf.astype(np.int32)) for buf in buffers)
        self._beam = BeamFactory.simple(factor_ev_angstrom / energy) 
        self.setup_detector(geometry, distance)

    def setup_detector(self, geometry, distance):
        wavelength = self.get_beam().get_wavelength()

        table = attenuation_coefficient.get_table("Si")
        mu = table.mu_at_angstrom(wavelength) / 10.0
        px_mm = ParallaxCorrectedPxMmStrategy(mu, geometry.thickness)

        detector = Detector()
        root = detector.hierarchy()
        root.set_frame((-1, 0, 0),
                       ( 0, 1, 0),
                       ( 0, 0, distance))

        for i, panel in enumerate(geometry.panels):
            angle = panel['rotation'] * math.pi / 180.0
            fast = matrix.col((math.cos(angle), math.sin(angle), 0))
            slow = matrix.col((-math.sin(angle), math.cos(angle), 0))
            normal = fast.cross(slow)

            origin = matrix.col((-panel['pos_x'],
                                  panel['pos_y'],
                                  panel['pos_z'])) / 1000.0
            p = root.add_panel()
            p.set_type("SENSOR_PAD")
            p.set_name('Panel%d' % i)
            p.set_image_size((geometry.width, geometry.height))
            p.set_trusted_range((-1, 65535))
            p.set_pixel_size((geometry.pixel_size, geometry.pixel_size))
            p.set_thickness(geometry.thickness)
            p.set_local_frame(fast.elems, slow.elems, origin.elems)
            p.set_px_mm_strategy(px_mm)
            p.set_gain(10)

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
