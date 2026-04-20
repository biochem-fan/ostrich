#!/usr/bin/env dials.python

# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

import datetime
import h5py
import libtbx
from multiprocessing import set_start_method, freeze_support, set_forkserver_preload
import numpy as np
import os.path
import re
import sys
import traceback

# SACLA APIs
import ctdapy_xfel
import dbpy

from ostrich import VERSION, update_status
from ostrich.dark_average import average_images
from ostrich.detector import CITIUSDetector, MPCCDDetector
from ostrich.geometry import *
from ostrich.hitfinder import find_hits
from ostrich.metadata import is_exposed, get_photon_energies, syncdata2float

def classify_frames(params, high_tag, tags):
    ret = np.zeros(len(tags))

    if params.runtype.startswith("dark") or params.runtype == "light":
        if params.pd1_threshold != 0:
            pd1_values = syncdata2float(params.pd1_name, high_tag, tags)
        if params.pd2_threshold != 0:
            pd2_values = syncdata2float(params.pd2_name, high_tag, tags)
        if params.pd3_threshold != 0:
            pd3_values = syncdata2float(params.pd3_name, high_tag, tags)

        nframe_after_light = 0
        for i, tag in enumerate(tags):
            if (params.pd1_threshold != 0 and
                (not (params.pd1_threshold > 0 and  params.pd1_threshold <= pd1_values[i])) and
                (not (params.pd1_threshold < 0 and -params.pd1_threshold >  pd1_values[i]))) \
               or \
               (params.pd2_threshold != 0 and
                (not (params.pd2_threshold > 0 and  params.pd2_threshold <= pd2_values[i])) and
                (not (params.pd2_threshold < 0 and -params.pd2_threshold >  pd2_values[i]))) \
               or \
               (params.pd3_threshold != 0 and
                (not (params.pd3_threshold > 0 and  params.pd3_threshold <= pd3_values[i])) and
                (not (params.pd3_threshold < 0 and -params.pd3_threshold >  pd3_values[i]))):
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
        elif runtype.startswith("dark"):
            return int(runtype[4:])
        else:
            return int(runtype)
    except ValueError:
        raise ValueError("runtype must be 0, 1, 2, ... or light or dark1, dark2, ...")

def run(params):
    runid = params.runid
    bl = params.bl
    clen = params.clen
    nproc = params.nproc
    status = params.status
    citius_roi = params.citius_roi
    hitfinding_roi = params.hitfinding_roi
    binning = params.binning
    use_nexus = params.nexus

    if not params.runtype.startswith("dark") and not params.runtype == "light":
        if int(params.runtype) >= params.nblock:
            update_status(status, "Status=Error-BadRunType")
            raise ValueError("runtype must be 0, 1, 2, ..., (nblock - 1).")

    if binning != 1 and not use_nexus:
        update_status(status, "Status=Error-InvalidBinning")
        raise ValueError("binning is available only for NXmx output.")

    print("ctdapy_xfel version:", ctdapy_xfel.get_api_version())
    print("dbpy version:", dbpy.get_api_version())
    print()

    # Get Run info
    try:
        run_info = dbpy.read_runinfo(bl, runid)
    except:
        update_status(status, "Status=Error-BadRunID")
        raise RuntimeError("Failed to get the run information. Probably the requested run does not exist.")
    high_tag = dbpy.read_hightagnumber(bl, runid)
    comment = dbpy.read_comment(bl, runid)
    tags = dbpy.read_taglist_byrun(bl, runid)
    start_time = datetime.datetime.fromtimestamp(dbpy.read_starttime(bl, runid), tz=datetime.timezone.utc).isoformat()
    end_time = datetime.datetime.fromtimestamp(dbpy.read_stoptime(bl, runid), tz=datetime.timezone.utc).isoformat()
    print("Run %d: HighTag %d, Tags %d - %d (inclusive) with %d images" % (runid, high_tag, tags[0], tags[-1], len(tags)))
    print("Run time: %s to %s" % (start_time, end_time))
    print("Comment: %s\n" % comment)

    # Find detectors
    citius_status = ctdapy_xfel.get_runstatus(bl ,runid)
    if citius_status == ctdapy_xfel.CTDA_RUN_STATUS_CAN_READ:
        try:
            ctrl_buf = ctdapy_xfel.CtrlBuffer(bl, runid)
            longnames = ctrl_buf.read_detidlist()
            detector_name = longnames[0][:-4]
            is_20M = CITIUSDetector.is_20M(detector_name)
            if is_20M and (citius_roi != "all" or hitfinding_roi != "all"):
                update_status(status, "Status=Error-ROINotSupportedForNon20.2MDetectors")
                raise RuntimeError("ROI is supported only for CITIUS 20.2M")
            print("CITIUS detector available sensor full names:", longnames)
            det_ids_all = sorted(ctrl_buf.read_sensoridlist())
            print("CITIUS detector available sensor IDs:", det_ids_all)
            det_ids = CITIUSDetector.filter_prbs_by_roi(det_ids_all, citius_roi)
            print("CITIUS detector sensors within the ROI:", det_ids)
            is_citius = True
        except:
            update_status(status, "Status=Error-CITIUSFailedToGetDetectors")
            raise RuntimeError("Found a CITIUS detector but failed to get sensor IDs.")
    elif citius_status == ctdapy_xfel.CTDA_RUN_STATUS_WAIT_CALIB:
        update_status(status, "Status=Error-CITIUSNotYetReady")
        raise RuntimeError("Found a CITIUS detector but calibration is still underway. Please try again later.")
    else:
        try:
            det_ids_all = dbpy.read_detidlist(bl, runid)
            print("Detector IDs: " + " ".join(det_ids_all))
            det_ids = MPCCDDetector.filter_mpccd_octal(det_ids_all)
            print("MPCCD Octal IDs to use: " + " ".join(det_ids))
            is_citius = False
        except:
            update_status(status, "Status=Error-NoSupportedDetectorFound")
            raise RuntimeError("Neither MPCCD or CITIUS was found for this run.")
    print()

    if params.adu_per_photon == libtbx.Auto:
        if is_citius:
            params.adu_per_photon = 8
        else:
            params.adu_per_photon = 10
    adu_per_photon = params.adu_per_photon

    if params.output_dtype == libtbx.Auto:
        if is_citius:
            params.output_dtype = "int32"
        else:
            params.output_dtype = "uint16"
    output_dtype = np.dtype(getattr(np, params.output_dtype))

    # Find images for dark average
    try:
        exposed = is_exposed(high_tag, tags, bl, runid)
    except Exception as e:
        update_status(status, "Status=Error-NoShutterStatus")
        raise e
    # exposed = [True] * len(tags) # for debugging
    calib_images = [tag for tag, exposed in zip(tags, exposed) if not exposed]
    exposed_images = [tag for tag, exposed in zip(tags, exposed) if exposed]
    assert len(calib_images) + len(exposed_images) == len(tags)

    if not is_citius and len(calib_images) == 0:
        update_status(status, "Status=Error-NoDarkImage")
        raise RuntimeError("No dark image was available for MPCCD dark current subtraction.")

    if len(exposed_images) == 0:
        update_status(status, "Status=Error-NoImage")
        raise RuntimeError("No exposed image was available. Nothing to do.")

    # Setup buffer readers
    if is_citius:
        detector = CITIUSDetector(det_ids, detector_name, bl, runid, exposed_images[0])
    else:
        detector = MPCCDDetector(det_ids, bl, runid, exposed_images[0])

    # Collect pulse energies
    pulse_energies, config_photon_energy = get_photon_energies(bl, runid, high_tag, tags)
    mean_energy = np.mean(pulse_energies)
    print("Mean photon energy: %f eV" % mean_energy)
    print("Configured photon energy: %f eV\n" % config_photon_energy)

    # Create geometry files

    # The origin of CITIUS panels was the top left corner until 2024/11/18.
    # This has been changed to the beam center (same as MPCCD) since then.
    if is_citius and bl == 2 and runid < 223251:
        # NOTE that this fudge affects geometry files but not hit finding.
        # Take care when using the radial profile and/or resolution filters.
        beam_center = (165.6, 206.9) # x, y in mm, NeXus-McStats system
    else:
        beam_center = (0.0, 0.0)

    write_crystfel_geom("%d.geom" % runid, use_nexus, detector.geometry, mean_energy, adu_per_photon, clen, bl, runid, beam_center, binning)

    # Prepare pixel mask (with bad pixel masks from API for CITIUS)
    if is_citius:
        ysize = detector.geometry.height
        bad_mask = np.zeros((ysize * len(det_ids), detector.geometry.width),
                            dtype=np.uint8)
        try:
            for i, det_id in enumerate(det_ids):
                ctrl_buf.read_badpixel_mask(bad_mask[(ysize * i):(ysize * (i + 1)),:], det_id)
            print("The total number of bad pixels from API: %d" % np.sum(bad_mask))
        except:
            print("This is CITIUS but bad pixel masks are unavailable from API")
            bad_mask = None
    else:
        bad_mask = None

    pixel_mask = make_pixelmask(detector.geometry, bl, runid, bad_mask, binning=1)
    if binning != 1:
        binned_pixel_mask = make_pixelmask(detector.geometry, bl, runid, bad_mask, binning)
    else:
        binned_pixel_mask = pixel_mask

    # Write metadata
    output_filename = "run%d-%s.h5" % (runid, params.runtype)
    if use_nexus:
        write_nexus(output_filename, detector.geometry, bl, runid, comment, start_time, end_time, clen, adu_per_photon, binned_pixel_mask, beam_center, binning)
    else:
        write_metadata(output_filename, detector.geometry, clen, comment, runid, adu_per_photon, binned_pixel_mask)
    print()

    # Make a boolean mask for DIALS hit finder. Note that hit finder uses non-binned images.
    pixel_mask = np.split(pixel_mask == 0, len(detector.geometry.panels), axis=0)

    # Create dark average
    dark_average = None
    if not is_citius:
        print("Calculating a dark average over %d images:\n" % len(calib_images))
        photon_energies_calib = pulse_energies[np.logical_not(exposed)]
        dark_average = average_images(detector, calib_images, photon_energies_calib, status, nproc)

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
    # TODO: Allow non-zero beam center. This is necessary for radial profiles on detector shifted experiments
    find_hits(detector, target_images, photon_energies_target, output_filename, dark_average, pixel_mask, output_dtype, params)

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

pd1_threshold = 0
 .help = Threshold for photodiode 1 (pd1_name) for light. Set 0 to ignore this photodiode.
 .type = float

pd2_threshold = 0
 .help = Threshold for photodiode 2 (pd2_name) for light. Set 0 to ignore this photodiode.
 .type = float

pd3_threshold = 0
 .help = Threshold for photodiode 3 (pd3_name) for light. Set 0 to ignore this photodiode.
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
 .help = Parallelization mode (0, 1, ..., nblocks-1 for non-TR SFX and light, dark1, dark2, ... for TR-SFX)
 .type = str

nblock = 3
 .help = Number of parallelization blocks (ignored in TR-SFX)
 .type = int(value_min = 1, value_max = 5)

status = ""
 .help = File name for status log (for integration with Ostrich Dispatcher GUI)
 .type = str

nproc = 8
 .help = Number of processes
 .type = int(value_min = 1, value_max = 48)

compression_level = 6
 .help = GZIP compression level for output frames
 .type = int(value_min = 0, value_max = 9)

adu_per_photon = Auto
 .help = Output value per photon. Auto means 10 for MPCCD, 8 for CITIUS.
 .type = float(value_min=0.1, value_max = 100)

output_dtype = Auto
 .help = Output data type. Values beyond the valid range are truncated. \\
         For floats, adu_per_photon will be set to 1. \\
         Auto means uint16 for MPCCD, int32 for CITIUS.
 .type = str

citius_roi = *all 24 40 48
 .help = ROI for CITIUS detectors
 .type = choice

hitfinding_roi = *all 24 40 48
 .help = ROI used for hit finding (only for CITIUS detectors)
 .type = choice

binning = 1
 .help = Binning (1 = no binning, 2 = half size)
 .type = int(value_min = 1, value_max = 2)

nexus = True
 .help = Output in the NeXus NXmx format (CITIUS images must be written in NXmx to be processed in DIALS)
 .type = bool

output {
    shoeboxes = False
        .type = bool
}

include scope dials.algorithms.spot_finding.factory.phil_scope
'''

# Users can override these defaults by adding more options
default_override_phil = '''
spotfinder {
    filter {
#        border = 4
    }
    background {
        dispersion {
# global_threshold is after gain correction, i.e., number of photons
            global_threshold = 10
        }
    }
}
'''

if __name__ == "__main__":
    # Very annoyingly, stpy is not compatible with fork.
    # Internally, it never releases sockets to MySQL DAQ DB.
    # BufferReaders shares file handles.
    # forkserver starts a vanilla parent that imports
    # only those set at `set_forkserver_preload`.
    # Thus, it is faster than `spawn` and safer than `fork`.

    # Python 3.13 (in DIALS 3.27.1) requires this order!
    # https://github.com/python/cpython/issues/140814
    set_start_method("forkserver")
    freeze_support()
    set_forkserver_preload(['dials.array_family.flex'])

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
    print("Option: pd1_threshold     = %.2f" % params.pd1_threshold)
    print("Option: pd2_name          = %s" % params.pd2_name)
    print("Option: pd2_threshold     = %.2f" % params.pd2_threshold)
    print("Option: pd3_name          = %s" % params.pd3_name)
    print("Option: pd3_threshold     = %.2f" % params.pd3_threshold)
    print("Option: hit_threshold     = %d" % params.hit_threshold)
    print("Option: runtype           = %s" % params.runtype)
    print("Option: nproc             = %d" % params.nproc)
    print("Option: status            = %s" % params.status)
    print("Option: compression_level = %d" % params.compression_level)
    if params.adu_per_photon == libtbx.Auto:
        print("Option: adu_per_photon    = Auto (10 for MPCCD, 8 for CITIUS)")
    else:
        print("Option: adu_per_photon    = %.1f / photon" % params.adu_per_photon)
    if params.output_dtype == libtbx.Auto:
        print("Option: output_dtype      = Auto (uint16 for MPCCD, int32 for CITIUS)")
    else:
        print("Option: output_dtype      = %s" % params.output_dtype)
    print("Option: citius_roi        = %s" % params.citius_roi)
    print("Option: hitfinding_roi    = %s" % params.hitfinding_roi)
    print("Option: binning           = %d" % params.binning)
    print("Option: nexus             = %s" % params.nexus)
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
