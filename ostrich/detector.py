# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

from numpy import pi
import re
import stpy

# ctdapy_xfel is not available in anapc
try:
    import ctdapy_xfel
except:
    pass

SI_eV_per_ELECTRON = 3.65

def bin_image(img, binning):
    assert img.ndim == 2
    assert img.shape[0] % binning == 0
    assert img.shape[1] % binning == 0

    # cf. https://stackoverflow.com/a/36102436/23975867
    return img.reshape(img.shape[0] // binning, binning, img.shape[1] // binning, binning).sum(axis=3).sum(axis=1)

class DetectorPanel:
    def __init__(self):
        self.name = ""
        self.long_name = ""
        self.index = 0
        self.pos_x = 0.0 # um
        self.pos_y = 0.0 # um
        self.pos_z = 0.0 # um
        self.rotation = 0.0 # rad
        self.gain = 1.0 # e/ADU

class DetectorGeometry:
    def __init__(self):
        self.name = ""
        self.pixel_size = 0 # um / px
        self.width = 0 # fast scan
        self.height = 0 # slow scan
        self.thickness = 0 # um
        self.panels = []
        self.groups = [] # array of (group_name, [panel_name])

class Detector:
    def __init__(self, det_ids, bl, runid, first_tag):
        self.bl = bl
        self.runid = runid
        self.first_tag = first_tag
        self.det_ids = det_ids
        self.geometry = None
        self.readers = None
        self.buffers = None

        self.allocate_readers()

    def allocate_readers(self):
        raise NotImplementedError

    @staticmethod
    def validate_and_set_geometry(det_infos):
        raise NotImplementedError

    def read_detinfos(self):
        raise NotImplementedError

class CITIUSDetector(Detector):
    def __init__(self, det_ids, det_longname, bl, runid, first_tag):
        self.det_longname = det_longname
        super().__init__(det_ids, bl, runid, first_tag)

    @staticmethod
    def is_20M(detector_name):
        return detector_name.startswith("CITIUS-20.2M-")

    @staticmethod
    def filter_prbs_by_roi(det_ids, roi="all"):
        # This function assumes roi="all" for non-20.2M detectors
        if roi == "all" or roi == "72":
            return det_ids
        elif roi == "24":
            within_roi = set([   19, 20, 21, 22,
                                 25, 26, 27, 28,
                                 31, 32, 33, 34,
                                 37, 38, 39, 40,
                                 43, 44, 45, 46,
                                 49, 50, 51, 52])
        elif roi == "40":
            within_roi = set([   13, 14, 15, 16,
                                 19, 20, 21, 22,
                             24, 25, 26, 27, 28, 29,
                             30, 31, 32, 33, 34, 35,
                             36, 37, 38, 39, 40, 41,
                             42, 43, 44, 45, 46, 47,
                                 49, 50, 51, 52,
                                 55, 56, 57, 58])
        elif roi == "48":
            within_roi = set([        8,  9,
                                 13, 14, 15, 16,
                             18, 19, 20, 21, 22, 23,
                             24, 25, 26, 27, 28, 29,
                             30, 31, 32, 33, 34, 35,
                             36, 37, 38, 39, 40, 41,
                             42, 43, 44, 45, 46, 47,
                             48, 49, 50, 51, 52, 53,
                                 55, 56, 57, 58,
                                     62, 63])
        else:
             raise ValueError("CITIUS ROI type must be one of all, 24, 40, 48")

        return [x for x in det_ids if x in within_roi]

    def allocate_readers(self):
        # CITIUS API does not have readers
        if self.buffers is not None:
            return

        # CITIUS API uses only one buffer for alll sensors
        try:
            self.buffers = ctdapy_xfel.CtrlBuffer(self.bl, self.runid)
        except:
            raise RuntimeError("FailedOn_CtrlBuffer")

        if self.geometry is None:
            self.read_detinfos()

    @staticmethod
    def validate_and_set_geometry(det_infos, longname="CITIUS-20.2M-#UNK"):
        geometry = DetectorGeometry()

        # WARNING: this function does not set up width and height, because
        # they are different between ctdapy_xfel and ctolpy_xfel.
        geometry.name = longname
        geometry.pixel_size = det_infos[0]['pixel_size_x']
        geometry.thickness = 650

        for det_info in det_infos:
            # All panels must be the same shape
            assert geometry.pixel_size == det_info['pixel_size_x']
            assert geometry.pixel_size == det_info['pixel_size_y']

            panel = DetectorPanel()
            panel.name = "prb%02d" % det_info['id']
            # TODO: use ctdapy_xfel.CtrlBuffer.read_detidlist
            panel.long_name = "%s-%03d" % (longname, det_info['id'])
            panel.index = det_info['id']
            panel.pos_x = det_info['position_x']
            panel.pos_y = det_info['position_y']
            panel.pos_z = det_info['position_z']
            panel.rotation = det_info['position_theta'] * pi / 180.0
            panel.gain = 40 # 1 ADU = 40 electrons
            geometry.panels.append(panel)

        # Analyze 2 x 4 SSS (Sensor Sub System) blocks, each containing 3x3 panels
        geometry.groups = []
        if CITIUSDetector.is_20M(longname):
            det_ids = [p['id'] for p in det_infos]
            for y in [0, 3, 6, 9]:
                for x in [0, 3]:
                    prb_in_sss = []
                    for dy in range(3):
                        for dx in range(3):
                            prb = x + dx + (y + dy) * 6
                            if prb in det_ids:
                                prb_in_sss.append("prb%02d" % prb)
    
                    if len(prb_in_sss) > 0:
                        geometry.groups.append(("sss%d%d" % (x, y), prb_in_sss))
        else: # CITIUS 2.2M does not have the SSS hierarchy
             for panel in geometry.panels:
                 geometry.groups.append((panel.name, [panel.name]))

        return geometry

    def read_detinfos(self):
        det_infos = [self.buffers.read_reconstinfo(prb_id, self.first_tag) for prb_id in self.det_ids]
        for det_info, prb_id in zip(det_infos, self.det_ids):
            det_info['id'] = prb_id

        self.geometry = CITIUSDetector.validate_and_set_geometry(det_infos, self.det_longname)
        if self.bl == 2 and self.runid < 228000:
            for panel in self.geometry.panels:
                panel.gain = 1.0 # Old data

        self.geometry.width = ctdapy_xfel.CITIUS_IMAGE_WIDTH
        self.geometry.height = ctdapy_xfel.CITIUS_IMAGE_HEIGHT

    def deallocate_readers(self):
        self.readers = None
        self.buffers = None

class MPCCDDetector(Detector):
    def __init__(self, det_ids, bl, runid, first_tag):
        super().__init__(det_ids, bl, runid, first_tag)

    def filter_mpccd_octal(det_ids):
        mpccds = sorted([x for x in det_ids if re.match("^MPCCD-8.*-[1-8]$", x)])
        if len(mpccds) != 8:
            raise RuntimeError("NoSupportedDetectorFound")

        return mpccds

    def allocate_readers(self):
        if self.readers is not None:
            return

        try:
            self.readers = [stpy.StorageReader(det_id, self.bl, (self.runid,)) for det_id in self.det_ids]
        except:
            raise RuntimeError("FailedOn_create_streader")
        try:
            self.buffers = [stpy.StorageBuffer(reader) for reader in self.readers]
        except:
            raise RuntimeError("FailedOn_create_stbuf")

        if self.geometry is None:
            self.read_detinfos()

    @staticmethod
    def validate_and_set_geometry(det_infos):
        geometry = DetectorGeometry()

        geometry.name = det_infos[0]['id'][:-2] # Remove panel ID "-N"
        geometry.width = det_infos[0]["xsize"]
        geometry.height = det_infos[0]["ysize"]
        geometry.pixel_size = det_infos[0]["mp_pixelsizex"]

        if geometry.name.startswith("MPCCD-8N"):
            geometry.thickness = 50 # um for Phase I
        else:
            geometry.thickness = 300 # um for Phase III
        geometry.groups = []

        for i, det_info in enumerate(det_infos):
            # All panels must be the same shape
            assert geometry.width == det_info["xsize"]
            assert geometry.height == det_info["ysize"]
            assert geometry.pixel_size == det_info["mp_pixelsizex"]
            assert geometry.pixel_size == det_info["mp_pixelsizey"]

            # The direct beam aperture must be closed
            assert det_info["mp_manipulator_shift"] == 0

            panel = DetectorPanel()
            panel.name = "q%d" % (i + 1)
            panel.long_name = det_info['id']
            panel.index = i
            panel.pos_x = det_info['mp_posx']
            panel.pos_y = det_info['mp_posy']
            panel.pos_z = det_info['mp_posz']
            panel.rotation = det_info['mp_rotationangle'] * pi / 180.0
            panel.gain = det_info['mp_absgain']
            geometry.panels.append(panel)
            geometry.groups.append(("group%d" % (i + 1), [panel.name]))

        return geometry

    def read_detinfos(self):
        for reader, buf in zip(self.readers, self.buffers):
            try:
                reader.collect(buf, self.first_tag)
            except:
                raise RuntiemError("FailedOn_collect_data")

        # Store the detector name
        det_infos = [buf.read_det_info(0) for buf in self.buffers]
        for i, det_info in enumerate(det_infos):
            det_info['id'] = self.det_ids[i]
        self.geometry = MPCCDDetector.validate_and_set_geometry(det_infos)

    def deallocate_readers(self):
        self.readers = None
        self.buffers = None
