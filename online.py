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

from ostrich import VERSION, update_status, OSTRICH_ONLINE_SHM_NAME
from ostrich.detector import CITIUSDetector, MPCCDDetector
from ostrich.online_detector import CITIUSOnlineDetector
from ostrich.geometry import *
from ostrich.online_hitfinder import find_hits

def run(params):
    runid = params.runid
    bl = params.bl
    clen = params.clen
    framebuffer_size = params.framebuffer_size
    status = params.status
    citius_roi = params.citius_roi

    # Get Run info
    # TODO: use the latest when not specified
    try:
        run_info = dbpy.read_runinfo(bl, runid)
    except:
        update_status(status, "Status=Error-BadRunID")
        raise RuntimeError("Failed to get the run information. Probably the requested run does not exist.")
    high_tag = dbpy.read_hightagnumber(bl, runid)
    tags = dbpy.read_taglist_byrun(bl, runid)
    start_time = datetime.datetime.fromtimestamp(dbpy.read_starttime(bl, runid), tz=datetime.timezone.utc).isoformat()
    end_time = datetime.datetime.fromtimestamp(dbpy.read_stoptime(bl, runid), tz=datetime.timezone.utc).isoformat()
    print("Collecting metadata from the run %d" % runid)
    print("Run %d: HighTag %d, Tags %d - %d (inclusive) with %d images" % (runid, high_tag, tags[0], tags[-1], len(tags)))
    print("Run time: %s to %s" % (start_time, end_time))

    # Collect pulse energies
    config_photon_energy = 1000.0 * dbpy.read_config_photonenergy(bl, runid)
    print("Configured photon energy: %f eV\n" % config_photon_energy)

    # Find detectors
    try:
        ctrl_buf = ctolpy_xfel.CtrlBuffer(0) # TODO: conn_id
        longnames = ctrl_buf.read_detidlist()
        print("CITIUS detector available sensor full names:", longnames)
        det_ids_all = sorted(ctrl_buf.read_sensoridlist())
        print("CITIUS detector available sensor IDs:", det_ids_all)
        det_ids = CITIUSDetector.filter_prbs_by_roi(det_ids_all, citius_roi)
        print("CITIUS detector sensos within the ROI:", det_ids)
        is_citius = True
    except:
        update_status(status, "Status=Error-CITIUSFailedToGetDetectors")
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

    # Create geometry files

    # The origin of MPCCD panels is roughy the beam center but that of CITIUS
    # is at the top left corner as of 2024A. This might change in the future.
    if is_citius:
        beam_center = (165.6, 206.9) # x, y in mm, NeXus-McStats system
    else:
        beam_center = (0.0, 0,0)

    adu_per_photon = 10 # TODO: fixme
    # Write metadata
    pixel_mask = make_pixelmask(detector.geometry, runid)
    binned_pixel_mask = make_pixelmask(detector.geometry, runid, binning)

    output_filename = "online-debug.h5"
    # TODO: binning is set to 1 because detector.geometry is already binned! This is irrelevant for NeXuS
    write_crystfel_geom("online.geom", True, detector.geometry, config_photon_energy, adu_per_photon, clen, runid, beam_center, 1)
    write_nexus(output_filename, detector.geometry, bl, runid, "Online Debug", start_time, end_time, clen, adu_per_photon, binned_pixel_mask, beam_center, binning)
    print()

    # Make a boolean mask for DIALS hit finder. Note that hit finder uses non-binned images.
    pixel_mask = np.split(pixel_mask == 0, len(detector.geometry.panels), axis=0)

    find_hits(detector, shared_buffer, config_photon_energy, pixel_mask, params)

phil_str = '''
bl = 2
 .help = Beam line (2 or 3)
 .type = int(value_min = 2, value_max = 3)

clen = 50.0
 .help = Detector distance in millimeter
 .type = float(value_min = 0)

hit_threshold = 20
 .help = Minimum number of spots to consider an image to be hit (inclusive)
 .type = int(value_min = 0)

runid = 216241
 .help = Run ID to process
 .type = int(value_min = 0)

status = ""
 .help = File name for status log (for integration with Ostrich Dispatcher GUI)
 .type = str

framebuffer_size = 10
 .help = Number of frames in the buffer
 .type = int(value_min = 1, value_max = 100)

nproc_hitfinder = 8
 .help = Number of hitfinder processes
 .type = int(value_min = 1, value_max = 48)

nproc_reader = 1
 .help = Number of image retriever processes
 .type = int(value_min = 1, value_max = 3)

citius_roi = *all 24 40 48
 .help = ROI for CITIUS detectors
 .type = choice

hitfinding_roi = *all 24 40 48
 .help = ROI used for hit finding (only for CITIUS detectors)
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
    print("Option: runid             = %d" % params.runid)
    print("Option: clen              = %.1f mm" % params.clen)
    print("Option: hit_threshold     = %d" % params.hit_threshold)
    print("Option: framebuffer_size  = %d" % params.framebuffer_size)
    print("Option: nproc_hihtfinder  = %d" % params.nproc_hitfinder)
    print("Option: nproc_reader      = %d" % params.nproc_reader)
    print("Option: status            = %s" % params.status)
    print("Option: citius_roi        = %s" % params.citius_roi)
    print("Option: hitfinding_roi    = %s" % params.hitfinding_roi)
    print()

    try:
        run(params)
    except:
        # Do not overwrite an existing detailed error message by a general message.
        already_reported = False
        if params.status != "":
            if os.path.exists(params.status):
                with open(params.status) as f:
                    if "Error" in f.read():
                        already_reported = True
        if not already_reported:
            update_status(params.status, "Status=Error-PleaseCheckLog")

        print(traceback.format_exc())
