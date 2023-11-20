#!/usr/bin/env dials.python

# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

# TODO:
# - move this to command_line.py or something
# - output file name
# - target selection
# - test NeXus output
# - GUI integration

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
from ostrich.metadata import filter_mpccd_octal, is_exposed, get_photon_energies

def run(runid, bl, clen, nproc):
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
    target_images = [tag for tag, exposed in zip(tags, exposed) if exposed]

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
    write_metadata("%d.h5" % runid, detector.det_infos, clen, comment, runid)

    # Create dark average
    print("\nCalculating a dark average over %d images:\n" % len(calib_images))
    photon_energies_calib = pulse_energies[np.logical_not(exposed)]
    averaged = average_images(detector, calib_images, photon_energies_calib, nproc)

    f = h5py.File("%d-dark.h5" % runid, "w")
    f.create_dataset("/data/data", data=averaged, compression="gzip", shuffle=True)
    f.close()
    print("Dark average was written to %s" % ("%d-dark.h5" % runid))

    # DEBUG; move this below!
    photon_energies_target = pulse_energies[exposed]
    find_hits(detector, target_images, photon_energies_target, nproc, params)
    return

phil_str = '''
runid = 180693
 .type = int(value_min = 0)

bl = 2
 .type = int(value_min = 2, value_max = 3)

clen = 50.0
 .type = float(value_min = 0)

nproc = 8
 .type = int(value_min = 1, value_max = 48)

hit_threshold = 20
 .type = int(value_min = 0)

compression_level = 6
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
    print("Option: bl               = %d" % params.bl)
    print("Option: clen             = %.1f mm" % params.clen)
    print()

    run(runid=params.runid, bl=params.bl, clen=params.clen, nproc=params.nproc)
