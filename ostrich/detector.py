# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

import stpy

class DetectorGeometry:
    def __init__(self):
        self.pixel_size = 0
        self.width = 0 # fast scan
        self.height = 0 # slow scan
        self.panels = []

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

    def validate_and_set_geometry(self, det_infos):
        self.geometry = DetectorGeometry()

        self.geometry.width = det_infos[0]["xsize"]
        self.geometry.height = det_infos[0]["ysize"]
        self.geometry.pixel_size = det_infos[0]["mp_pixelsizex"]

        self.geometry.thickness = 300 # um for Phase III (was 50 um for Phase I)
        # CITIUS had 650 um

        for det_info in det_infos:
            # All panels must be the same shape
            assert self.geometry.width == det_info["xsize"]
            assert self.geometry.height == det_info["ysize"]
            assert self.geometry.pixel_size == det_info["mp_pixelsizex"]
            assert self.geometry.pixel_size == det_info["mp_pixelsizey"]

            # The direct beam aperture must be closed
            assert det_info["mp_manipulator_shift"] == 0

            self.geometry.panels.append({\
              "id": det_info['id'],
              "pos_x": det_info['mp_posx'],
              "pos_y": det_info['mp_posy'],
              "pos_z": det_info['mp_posz'],
              "rotation": det_info['mp_rotationangle'],
              "gain": det_info['mp_absgain']})

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
        self.validate_and_set_geometry(det_infos)

    def deallocate_readers(self):
        self.readers = None
        self.buffers = None
