#!/usr/bin/env dials.python

# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

# TODO: GUI integration
# TODO: Binning
#       Where/Whether should we modify the geometry?
#       pixel_size is doubled, posx/posy are halved and TRUNCATED,
#       border width is halved and then CEILED.

import datetime
import h5py
import sys
from multiprocessing import set_start_method, freeze_support
import numpy as np
import re

# SACLA APIs
import ctdapy_xfel
import dbpy

from ostrich import VERSION
from ostrich.dark_average import average_images
from ostrich.detector import CITIUSDetector, MPCCDDetector
from ostrich.geometry import *
from ostrich.hitfinder import find_hits
from ostrich.metadata import is_exposed, get_photon_energies, syncdata2float

def classify_frames(params, high_tag, tags):
    # TODO: test time-resolved mode
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
            if (i == params.nblock - 1):
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
    adu_per_photon = params.adu_per_photon
    citius_roi = params.citius_roi
    use_nexus = params.nexus

    if not params.runtype.startswith("dark") and not params.runtype == "light":
        if int(params.runtype) >= params.nblock:
            raise ValueError("runtype must be 0, 1, 2, ..., (nblock - 1)")

    # Get Run info
    try:
        run_info = dbpy.read_runinfo(bl, runid)
    except:
        raise RuntimeError("BadRunID")
    high_tag = dbpy.read_hightagnumber(bl, runid)
    comment = dbpy.read_comment(bl, runid)
    tags = dbpy.read_taglist_byrun(bl, runid)
    start_time = datetime.datetime.fromtimestamp(dbpy.read_starttime(bl, runid), tz=datetime.timezone.utc).isoformat()
    end_time = datetime.datetime.fromtimestamp(dbpy.read_stoptime(bl, runid), tz=datetime.timezone.utc).isoformat()
    print("Run %d: HighTag %d, Tags %d - %d (inclusive) with %d images" % (runid, high_tag, tags[0], tags[-1], len(tags)))
    print("Run time: %s to %s" % (start_time, end_time))
    print("Comment: %s\n" % comment)

    # Find detectors
    try:
        det_ids_all = dbpy.read_detidlist(bl, runid)
        print("Detector IDs: " + " ".join(det_ids_all))
        det_ids = MPCCDDetector.filter_mpccd_octal(det_ids_all)
        print("MPCCD Octal IDs to use: " + " ".join(det_ids))
        is_citius = False
    except:
        ctrl_buf = ctdapy_xfel.CtrlBuffer(bl, runid)
        det_ids_all = sorted(ctrl_buf.read_prbidlist())
        print("CITIUS detector available PRB IDs:", det_ids_all)
        det_ids = CITIUSDetector.filter_prbs_by_roi(det_ids_all, citius_roi)
        print("CITIUS detector PRBs within the ROI:", det_ids)
        is_citius = True
    print()

    # Find images for dark average
    exposed = is_exposed(high_tag, tags, bl, runid)
    calib_images = [tag for tag, exposed in zip(tags, exposed) if not exposed]
    exposed_images = [tag for tag, exposed in zip(tags, exposed) if exposed]
    assert len(calib_images) + len(exposed_images) == len(tags)

    if not is_citius and np.sum(calib_images) == 0:
        RuntimeError("NoDarkImage")

    # Setup buffer readers
    if is_citius:
        detector = CITIUSDetector(det_ids, bl, runid, exposed_images[0])
    else:
        detector = MPCCDDetector(det_ids, bl, runid, exposed_images[0])

    # Collect pulse energies
    pulse_energies, config_photon_energy = get_photon_energies(bl, runid, high_tag, tags)
    mean_energy = np.mean(pulse_energies)
    print("Mean photon energy: %f eV" % mean_energy)
    print("Configured photon energy: %f eV\n" % config_photon_energy)

    # Create geometry files

    # The origin of MPCCD panels is roughy the beam center but that of CITIUS
    # is at the top left corner as of 2024A. This might change in the future.
    if is_citius:
        beam_center = (165.6, 206.9) # x, y in mm, NeXus-McStats system
    else:
        beam_center = (0.0, 0,0)

    write_crystfel_geom("%d.geom" % runid, use_nexus, detector.geometry, mean_energy, adu_per_photon, clen, runid, beam_center)
    # write_cheetah_geom("%d-geom.h5" % runid, detector.geometry)

    # Write metadata
    pixel_mask = make_pixelmask(detector.geometry, runid)
    output_filename = "run%d-%s.h5" % (runid, params.runtype)

    if use_nexus:
        write_nexus(output_filename, detector.geometry, bl, runid, comment, start_time, end_time, clen, pixel_mask, beam_center)
    else:
        write_metadata(output_filename, detector.geometry, clen, comment, runid, adu_per_photon, pixel_mask)
    print()

    # Make a boolean mask for DIALS hit finder
    pixel_mask = np.split(pixel_mask == 0, len(detector.geometry.panels), axis=0)

    # Create dark average
    dark_average = None
    if not is_citius:
        print("Calculating a dark average over %d images:\n" % len(calib_images))
        photon_energies_calib = pulse_energies[np.logical_not(exposed)]
        dark_average = average_images(detector, calib_images, photon_energies_calib, adu_per_photon, nproc)

        if False:
            f = h5py.File("%d-dark.h5" % runid, "w")
            f.create_dataset("/data/data", data=dark_average, compression="gzip", shuffle=True)
            f.close()
            print("Dark average was written to %s" % ("%d-dark.h5" % runid))

        dark_average = np.split(dark_average, len(detector.geometry.panels), axis=0)
        print()

    # Enumerate target tags for this job
    right_type = (classify_frames(params, high_tag, exposed_images) == runtype_to_num(params.runtype))
    assert len(exposed_images) == len(right_type)

    if is_citius:
        citius_valid_tags = set(ctrl_buf.read_taglist())
        are_valid = np.array([tag in citius_valid_tags for tag in exposed_images])
        if not np.all(are_valid):
            invalid_tags = [tag for tag, flag in zip(tags, are_valid) if not flag]
            print("Warning: CITIUS images are unavailable for %d tag(s):" % len(invalid_tags), invalid_tags)
            right_type = np.logical_and(right_type, are_valid)
            print()

    target_images = [tag for tag, flag in zip(exposed_images, right_type) if flag]
    #print("exposed", exposed_images)
    #print("target", target_images)
    print("%d images will be processed in this job out of %d exposed images." % (len(target_images), len(exposed_images)))
    print()

    photon_energies_target = pulse_energies[exposed][right_type]
    find_hits(detector, target_images, photon_energies_target, output_filename, dark_average, pixel_mask, params)

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

pd1_thresh = 0
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

adu_per_photon = 10
 .help = Output value per photon
 .type = float(value_min=0.1, value_max = 100)

citius_roi = *all 24 40 48
 .help = ROI for CITIUS detectors
 .type = choice

hitfinding_roi = *all 24 40 48
 .help = ROI used for hit finding (only for CITIUS detectors)
 .type = choice

nexus = True
 .help = Output in the NeXus NXmx format (CITIUS images must be written in NXmx to be processed in DIALS)
 .type = bool

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
    print("Option: adu_per_photon    = %.1f / photon" % params.adu_per_photon)
    print("Option: citius_roi        = %s" % params.citius_roi)
    print("Option: hitfinding_roi    = %s" % params.hitfinding_roi)
    print("Option: nexus             = %s" % params.nexus)
    print()

    run(params)
