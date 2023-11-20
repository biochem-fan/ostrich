# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

import stpy

class Detector:
    def __init__(self, det_ids, bl, runid, first_tag):
        self.bl = bl
        self.runid = runid
        self.first_tag = first_tag
        self.det_ids = det_ids
        self.det_infos = None
        self.readers = None

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

        if self.det_infos is None:
            self.read_detinfos()

    def read_detinfos(self):  
        for reader, buf in zip(self.readers, self.buffers):
            try:
                reader.collect(buf, self.first_tag)
            except:
                raise RuntiemError("FailedOn_collect_data")

        # Store the detector name
        self.det_infos = [buf.read_det_info(0) for buf in self.buffers]
        for i, det_info in enumerate(self.det_infos):
            det_info['id'] = self.det_ids[i]

    def deallocate_readers(self):
        self.readers = None
        self.buffers = None
