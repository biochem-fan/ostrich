#!/usr/bin/env dials.python

# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

import datetime
import h5py
import libtbx
from multiprocessing import set_start_method, freeze_support, shared_memory
import numpy as np
import os.path
import re
import sys
import traceback

# SACLA APIs
import ctolpy_xfel
import dbpy

from ostrich import VERSION, OSTRICH_ONLINE_SHM_NAME
from ostrich.detector import CITIUSDetector, MPCCDDetector
from ostrich.online_detector import CITIUSOnlineDetector
from ostrich.geometry import *
from ostrich.online_hitfinder import find_hits

def run(params):
    photon_energy = params.photon_energy
    bl = params.bl
    clen = params.clen
    framebuffer_size = params.framebuffer_size
    citius_roi = params.hitfinding_roi

    print("ctdapy_xfel version:", ctolpy_xfel.get_api_version())

    # Get Run info
    runid = dbpy.read_runnumber_newest(bl)
    #runid = 216241 # debug using simulator (20.2 M)
    #runid = 264585 # debug using simulator (2.2M)
    if photon_energy == libtbx.Auto:
        comment = dbpy.read_comment(bl, runid)
        photon_energy = 1000.0 * dbpy.read_config_photonenergy(bl, runid)
        print("Read the configured photon energy as %.1f eV from run %d (%s).\n" % (photon_energy, runid, comment))
    else:
        print("Using the specified photon energy %.1f eV." % photon_energy)

    # Find detectors
    try:
        ctrl_buf = ctolpy_xfel.CtrlBuffer(0) # TODO: conn_id
        longnames = ctrl_buf.read_detidlist()
        print("CITIUS detector available sensor full names:", longnames)
        det_ids_all = sorted(ctrl_buf.read_sensoridlist())
        print("CITIUS detector available sensor IDs:", det_ids_all)
        det_ids = CITIUSDetector.filter_prbs_by_roi(det_ids_all, citius_roi)
        print("CITIUS detector sensors within the ROI:", det_ids)
        is_citius = True
    except:
        raise NotImplementedError("MPCCD is not supported.")
    print()

    # Setup buffers
    if is_citius:
        binsize_x, binsize_y, binned_width, binned_height = ctrl_buf.read_binninginfo()
        assert binsize_x == binsize_y
        binning = binsize_x
        print("CITIUS detector binning is %d, thus (width, height) = (%d, %d) px." % (binning, binned_width, binned_height))

        # TODO: ensure unlinking; what happens if existing SHM is owned by others?
        # Python 3.13 has a resource tracker (track=True)
        buffer_shape = (framebuffer_size, len(det_ids), binned_height, binned_width)
        shm = shared_memory.SharedMemory(name=OSTRICH_ONLINE_SHM_NAME, create=True, size=np.prod(buffer_shape) * np.dtype(np.float32).itemsize)
        shared_buffer = np.ndarray(buffer_shape, dtype=np.float32, buffer=shm.buf)

        detector_name = longnames[0][:-4]
        detector = CITIUSOnlineDetector(det_ids, detector_name, shared_buffer[0, :, :, :])
    else:
        raise NotImplementedError("MPCCD is not supported.")

    # For old data in simulator:
    # Note that this beam center affects only the test geometry files, NOT hit finding.
    # Take care when using the radial profile and/or resolution filters.
    # This test data is from Run 216157 tag 628776479 - 628776581.
    if False:
        beam_center = (165.6, 206.9) # x, y in mm, NeXus-McStats system
        for panel in detector.geometry.panels:
            panel.gain = 1.0

    # Prepare pixel mask (with bad pixel masks from API for CITIUS)
    # Unlike the offline API, the online API requires a 3D output array because it works
    # on multiple sensors in one go.
    bad_mask = np.zeros((len(det_ids), detector.geometry.height, detector.geometry.width), dtype=np.uint8)
    try:
        ctrl_buf.read_badpixel_mask(bad_mask, det_ids)
        for i, det_id in enumerate(det_ids):
            print("The number of bad pixels on sensor %d from API: %d" % (det_id, np.sum(bad_mask[i, :, :])))
    except:
        print("This is CITIUS but bad pixel masks are unavailable from API")
        bad_mask = None

    bad_mask = bad_mask.reshape((-1, detector.geometry.width))
    pixel_mask = make_pixelmask(detector.geometry, bl, runid, binning=1)

    # Write metadata for debugging
    if False:
        binned_pixel_mask = make_pixelmask(detector.geometry, bl, runid, binning=binning)
        adu_per_photon = 10 # dummy value for geometry output (debug use only)

        output_filename = "online-debug.h5"
        # TODO: binning is set to 1 because detector.geometry is already binned! This is irrelevant for NeXuS
        write_crystfel_geom("online.geom", True, detector.geometry, photon_energy, adu_per_photon, clen, bl, runid, beam_center, 1)
        write_nexus(output_filename, detector.geometry, bl, runid, "Online Debug", start_time, end_time, clen, adu_per_photon, binned_pixel_mask, beam_center, binning)
    print()

    # Make a boolean mask for DIALS hit finder.
    pixel_mask = np.split(pixel_mask == 0, len(detector.geometry.panels), axis=0)

    find_hits(detector, shared_buffer, photon_energy, pixel_mask, params)

phil_str = '''
bl = 2
 .help = Beam line (2 or 3)
 .type = int(value_min = 2, value_max = 3)

clen = 50.0
 .help = Detector distance in millimeter
 .type = float(value_min = 0)

photon_energy = Auto
 .help = Photon energy in eV. Leave this Auto to retrieve from the latest run.
 .type = float(value_min = 1000, value_max = 20000)

framebuffer_size = 100
 .help = Number of frames in the buffer
 .type = int(value_min = 1, value_max = 200)

nproc_hitfinder = 24
 .help = Number of hitfinder processes
 .type = int(value_min = 1, value_max = 24)

nproc_reader = 3
 .help = Number of image retriever processes
 .type = int(value_min = 1, value_max = 4)

hitfinding_roi = *all 24 40 48
 .help = ROI for CITIUS detectors
 .type = choice

output {
    shoeboxes = False
        .type = bool
}

include scope dials.algorithms.spot_finding.factory.phil_scope
'''

default_override_phil = '''
spotfinder {
    filter {
#        border= 4
    }
}
'''

if __name__ == "__main__":
    # Very annoyingly, stpy is not compatible with fork.
    # Internally, it never releases sockets to MySQL DAQ DB.
    # BufferReaders shares file handles.
    freeze_support()
    set_start_method("spawn")

    # Importing these at the top makes spawning much slower
    from dials.util.options import ArgumentParser
    from libtbx.phil import parse

    phil_scope = parse(phil_str, process_includes=True).fetch(parse(default_override_phil))
    params, options = ArgumentParser(phil=phil_scope).parse_args(show_diff_phil=True, return_unhandled=False)

    print("Ostrich: SACLA Data Preprocessing System version %d" % VERSION)
    print(" by Takanori Nakane at Institute of Protein Research, Osaka University")
    print()
    print("Option: bl                = %d" % params.bl)
    if params.photon_energy == libtbx.Auto:    
        print("Option: photon_energy     = Auto (read from the latest run)")
    else:
        print("Option: photon_energy     = %.1f eV" % params.photon_energy)
    print("Option: clen              = %.1f mm" % params.clen)
    print("Option: framebuffer_size  = %d" % params.framebuffer_size)
    print("Option: nproc_hitfinder  = %d" % params.nproc_hitfinder)
    print("Option: nproc_reader      = %d" % params.nproc_reader)
    print("Option: hitfinding_roi    = %s" % params.hitfinding_roi)
    print()

    try:
        run(params)
    except:
        print(traceback.format_exc())
