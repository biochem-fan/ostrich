# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

from numpy import pi
import ctolpy_xfel
from ostrich.detector import CITIUSDetector
import re

class CITIUSOnlineDetector():
    def __init__(self, det_ids, det_longname, image_buffer, ctrl_buffer=None, conn_id=0):
        self.det_longname = det_longname
        self.ctrl_buffer = ctrl_buffer
        self.image_buffer = image_buffer
        self.det_ids = det_ids
        self.geometry = None

        self.allocate_ctrl_buffer(conn_id)

    def allocate_ctrl_buffer(self, conn_id=0):
        if self.ctrl_buffer is not None:
            return

        try:
            self.ctrl_buffer = ctolpy_xfel.CtrlBuffer(conn_id)
        except:
            raise RuntimeError("FailedOn_CtrlBuffer")

        if self.geometry is None:
            self.read_detinfos()

    def read_detinfos(self):
        det_infos = self.ctrl_buffer.collect_data(self.image_buffer, self.det_ids, ctolpy_xfel.NEWEST)['reconst_info_list']
 
        for det_info, prb_id in zip(det_infos, self.det_ids):
            det_info['id'] = prb_id

        self.geometry = CITIUSDetector.validate_and_set_geometry(det_infos, self.det_longname)
        # The above function uses constants in the offline API, so we have to apply binning.
        _, _, self.geometry.width, self.geometry.height = self.ctrl_buffer.read_binninginfo()

    def deallocate_readers(self):
        self.ctrl_buffer = None
        self.image_buffer = None
