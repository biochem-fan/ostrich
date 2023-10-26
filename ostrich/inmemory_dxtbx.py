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
    THICKNESS = 0.050 # mm

    def __init__(self, buffers, det_infos, energy, distance=50.0):
        assert len(buffers) == 8
        assert len(det_infos) == 8

        self._image = tuple(flex.int(buf.astype(np.int32)) for buf in buffers)
        self._beam = BeamFactory.simple(factor_ev_angstrom / energy) 
        self.setup_detector(det_infos, distance)

    def setup_detector(self, det_infos, distance):
        wavelength = self.get_beam().get_wavelength()

        table = attenuation_coefficient.get_table("Si")
        mu = table.mu_at_angstrom(wavelength) / 10.0
        px_mm = ParallaxCorrectedPxMmStrategy(mu, self.THICKNESS)

        detector = Detector()
        root = detector.hierarchy()
        root.set_frame((-1, 0, 0),
                       ( 0, 1, 0),
                       ( 0, 0, distance))

        for i, det_info in enumerate(det_infos):
            angle = det_info['mp_rotationangle'] * math.pi / 180.0
            fast = matrix.col((math.cos(angle), math.sin(angle), 0))
            slow = matrix.col((-math.sin(angle), math.cos(angle), 0))
            normal = fast.cross(slow)

            origin = matrix.col((-det_info['mp_posx'],
                                  det_info['mp_posy'],
                                  det_info['mp_posz'])) / 1000.0
            p = root.add_panel()
            p.set_type("SENSOR_PAD")
            p.set_name('Panel%d' % i)
            p.set_image_size((det_info['xsize'], det_info['ysize']))
            p.set_trusted_range((-1, 65535))
            p.set_pixel_size((det_info['mp_pixelsizex'], det_info['mp_pixelsizey']))
            p.set_thickness(self.THICKNESS)
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
