#!/usr/bin/env dials.python

# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

# TODO:
# - move this to command_line.py or something
# - help message
# - test NeXus output
# - GUI integration
# - study shared memory approach
#    e.g. https://stackoverflow.com/questions/37705974/why-are-multiprocessing-sharedctypes-assignments-so-slow
#         https://qiita.com/kakinaguru_zo/items/f53e2485f15dd0d71f82

import h5py
import sys
from multiprocessing import set_start_method, freeze_support
import numpy as np
import re

# SACLA APIs
import dbpy

from ostrich import VERSION
from ostrich.dark_average import average_images
from ostrich.detector import Detector
from ostrich.geometry import *
from ostrich.hitfinder import find_hits
from ostrich.metadata import filter_mpccd_octal, is_exposed, get_photon_energies, syncdata2float

def classify_frames(params, high_tag, tags):
    # TODO: test!!
    ret = np.zeros(len(tags))

    if params.runtype.startswith("dark") or params.runtype == "light":
        if params.pd1_threshold != 0:
            pd1_values = syncdata2float(params.pd1_name, high_tag, tags)
        if params.pd2_threshold != 0:
            pd2_values = syncdata2float(params.pd2_name, high_tag, tags)
        if params.pd3_threshold != 0:
            pd3_values = syncdata2float(params.pd2_name, high_tag, tags)

        nframe_after_light = 0
        for i, tag in enumerate(tags):
            if (params.pd1_threshold != 0 and
                (not (params.pd1_threshold > 0 and  params.pd1_threshold <= pd1_value[i])) and
                (not (params.pd1_threshold < 0 and -params.pd1_threshold >  pd1_value[i]))) \
               or \
               (params.pd2_threshold != 0 and
                (not (params.pd2_threshold > 0 and  params.pd2_threshold <= pd2_value[i])) and
                (not (params.pd2_threshold < 0 and -params.pd2_threshold >  pd2_value[i]))) \
               or \
               (params.pd3_threshold != 0 and
                (not (params.pd3_threshold > 0 and  params.pd3_threshold <= pd3_value[i])) and
                (not (params.pd3_threshold < 0 and -params.pd3_threshold >  pd3_value[i]))):
                nframe_after_light += 1
            else:
                nframe_after_light = 0

            ret[i] = nframe_after_light
    else:
        # This strange logic is for compatibility with Cheetah
        width = len(tags) // params.nblock
        for i in range(params.nblock):
            istart = i * width
            iend = (i + 1) * width
            if (i == params.nblock):
                iend = len(tags)
            ret[istart:iend] = i

    return ret

def runtype_to_num(runtype):
    try:
        if runtype == "light":
            return 0
        elif runtype.startswith("dark-"):
            return int(runtype[5:])
        else:
            return int(runtype)
    except e:
        raise ValueError("runtype must be 0, 1, 2, ... or light or dark-1, dark-2, ...")

def run(params):
    runid = params.runid
    bl = params.bl
    clen = params.clen
    nproc = params.nproc

    # Get Run info
    try:
        run_info = dbpy.read_runinfo(bl, runid)
    except:
        raise RuntimeError("BadRunID")
    high_tag = dbpy.read_hightagnumber(bl, runid)
    comment = dbpy.read_comment(bl, runid)
    tags = dbpy.read_taglist_byrun(bl, runid)
    print("Run %d: HighTag %d, Tags %d - %d (inclusive) with %d images" % (runid, high_tag, tags[0], tags[-1], len(tags)))
    print("Comment: %s\n" % comment)

    # Find detectors
    det_ids_all = dbpy.read_detidlist(bl, runid)
    print("Detector IDs: " + " ".join(det_ids_all))
    det_ids = filter_mpccd_octal(det_ids_all)
    print("MPCCD Octal IDs to use: " + " ".join(det_ids))
    print()

    # Find images for dark average
    exposed = is_exposed(high_tag, tags, bl, runid)
    calib_images = [tag for tag, exposed in zip(tags, exposed) if not exposed]

    if np.sum(calib_images) == 0:
        RuntimeError("NoDarkImage")

    # Setup buffer readers
    detector = Detector(det_ids, bl, runid, calib_images[0])

    # Collect pulse energies
    pulse_energies, config_photon_energy = get_photon_energies(bl, runid, high_tag, tags)
    mean_energy = np.mean(pulse_energies)
    print("Mean photon energy: %f eV" % mean_energy)
    print("Configured photon energy: %f eV\n" % config_photon_energy)

    # Create geometry files
    write_crystfel_geom("%d.geom" % runid, detector.det_infos, mean_energy, clen, runid)
    # write_cheetah_geom("%d-geom.h5" % runid, detector.det_infos)

    # Write metadata
    output_filename = "run%d-%s.h5" % (runid, params.runtype)
    write_metadata(output_filename, detector.det_infos, clen, comment, runid)

    # Create dark average
    print("\nCalculating a dark average over %d images:\n" % len(calib_images))
    photon_energies_calib = pulse_energies[np.logical_not(exposed)]
    averaged = average_images(detector, calib_images, photon_energies_calib, nproc)

    f = h5py.File("%d-dark.h5" % runid, "w")
    f.create_dataset("/data/data", data=averaged, compression="gzip", shuffle=True)
    f.close()
    print("Dark average was written to %s" % ("%d-dark.h5" % runid))
    print()

    exposed_images = [tag for tag, exposed in zip(tags, exposed) if exposed]
    right_type = (classify_frames(params, high_tag, exposed_images) == runtype_to_num(params.runtype))
    target_images = [tag for tag, flag in zip(tags, right_type) if flag]
    print("%d images will be processed in this job out of %d exposed images" % (len(target_images), len(exposed_images)))
    photon_energies_target = pulse_energies[exposed][right_type]
    find_hits(detector, target_images, photon_energies_target, output_filename, params)

phil_str = '''
bl = 2
 .help = Beam line (2 or 3)
 .type = int(value_min = 2, value_max = 3)

runid = 180693
 .help = Run ID to process
 .type = int(value_min = 0)

clen = 50.0
 .help = Detector distance in millimeter
 .type = float(value_min = 0)

pd1_thresh = 0.1
 .help = Threshold for photodiode 1 (pd1_name) for light. Set 0 to ignore this photodide.
 .type = float

pd2_thresh = 0
 .help = Threshold for photodiode 2 (pd1_name2) for light. Set 0 to ignore this photodide.
 .type = float

pd3_thresh = 0
 .help = Threshold for photodiode 3 (pd1_name3) for light. Set 0 to ignore this photodide.
 .type = float

pd1_name = "xfel_bl_2_st_3_pd_user_10_fitting_peak/voltage"
 .help = SyncDAQ DB signal name for photodiode 1
 .type = str

pd2_name = "xfel_bl_2_st_3_pd_user_9_fitting_peak/voltage"
 .help = SyncDAQ DB signal name for photodiode 2
 .type = str

pd3_name = "xfel_bl_2_st_3_laser_fitting_peak/voltage"
 .help = SyncDAQ DB signal name for photodiode 3
 .type = str

hit_threshold = 20
 .help = Minimum number of spots to consider an image to be hit (inclusive)
 .type = int(value_min = 0)

runtype = 0
 .help = Parallelization mode (0, 1, ..., nblocks-1 for non-TR SFX and light, dark-1, dark-2, ... for TR-SFX)
 .type = str

nblock = 3
 .help = Number of parallelization blocks (ignored in TR-SFX)
 .type = int(value_min = 1, value_max = 5)

nproc = 8
 .help = Number of processes
 .type = int(value_min = 1, value_max = 48)

compression_level = 6
 .help = GZIP compression level for output frames
 .type = int(value_min = 0, value_max = 9)

output {
    shoeboxes = False
        .type = bool
}

include scope dials.algorithms.spot_finding.factory.phil_scope
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

    phil_scope = parse(phil_str, process_includes=True)
    params, options = ArgumentParser(phil=phil_scope).parse_args(show_diff_phil=True, return_unhandled=False)

    print("Ostrich: SACLA Data Preprocessing System version %d" % VERSION)
    print(" by Takanori Nakane at Institute of Protein Research, Osaka University")
    print()
    print("Option: bl                = %d" % params.bl)
    print("Option: runid             = %d" % params.runid)
    print("Option: clen              = %.1f mm" % params.clen)
    print("Option: pd1_name          = %s" % params.pd1_name)
    print("Option: pd1_threshold     = %.2f" % params.pd1_thresh)
    print("Option: pd2_name          = %s" % params.pd2_name)
    print("Option: pd2_threshold     = %.2f" % params.pd2_thresh)
    print("Option: pd3_name          = %s" % params.pd3_name)
    print("Option: pd3_threshold     = %.2f" % params.pd3_thresh)
    print("Option: hit_threshold     = %d" % params.hit_threshold)
    print("Option: runtype           = %s" % params.runtype)
    print("Option: nproc             = %d" % params.nproc)
    print("Option: compression_level = %d" % params.compression_level)
    print()

    run(params)
