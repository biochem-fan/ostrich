#!/usr/bin/env dials.python

# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

import dbpy
import stpy

import optparse
import sys
import h5py
from multiprocessing import set_start_method, freeze_support
import numpy as np
import re

from ostrich import VERSION
from ostrich.dark_average import average_images
from ostrich.geometry import *
from ostrich.metadata import filter_mpccd_octal, is_exposed, get_photon_energies

def run(runid, bl=3, clen=50.0):
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
    try:
        readers = [stpy.StorageReader(det_id, bl, (runid,)) for det_id in det_ids]
    except:
        raise RuntimeError("FailedOn_create_streader")
    try:
        buffers = [stpy.StorageBuffer(reader) for reader in readers]
    except:
        raise RuntimeError("FailedOn_create_stbuf")

    # Read the first image to get the detector info
    # Store the detector name
    for reader, buf in zip(readers, buffers):
        try:
            reader.collect(buf, calib_images[0])
        except:
            raise RuntiemError("FailedOn_collect_data")
    det_infos = [buf.read_det_info(0) for buf in buffers]
    for i, det_info in enumerate(det_infos):
        det_info['id'] = det_ids[i]

    # Collect pulse energies
    pulse_energies, config_photon_energy = get_photon_energies(bl, runid, high_tag, tags)
    mean_energy = np.mean(pulse_energies)
    print("Mean photon energy: %f eV" % mean_energy)
    print("Configured photon energy: %f eV\n" % config_photon_energy)

    # Create geometry files
    write_crystfel_geom("%d.geom" % runid, det_infos, mean_energy, clen, runid)
    write_cheetah_geom("%d-geom.h5" % runid, det_infos)

    # Write metadata
    write_metadata("%d.h5" % runid, det_infos, clen, comment, runid)

    # Create dark average
    print("\nCalculating a dark average over %d images:\n" % len(calib_images))
    photon_energies_calib = pulse_energies[np.logical_not(exposed)]
    averaged = average_images(readers, buffers, det_ids, bl, runid, calib_images, det_infos, photon_energies_calib, nproc=8)

    f = h5py.File("%d-dark.h5" % runid, "w")
    f.create_dataset("/data/data", data=averaged, compression="gzip", shuffle=True)
    f.close()
    print("Dark average was written to %s" % ("%d-dark.h5" % runid))

if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn")

    parser = optparse.OptionParser()
    parser.add_option("--bl", dest="bl", type=int, default=3, help="Beamline")
    parser.add_option("--clen", dest="clen", type=float, default=50.0, help="Camera distance")
    opts, args = parser.parse_args()

    if (opts.bl != 2 and opts.bl !=3):
        print("--bl must be 2 or 3.")
        sys.exit(-1)

    if len(args) != 1:
        print("Usage: prepare-cheetah-sacla-api2.py runid [--bl 3] [--clen 50.0]")
        sys.exit(-1)
    runid = int(args[0])

    print("prepare-cheetah-sacla-api2.py version %d" % VERSION)
    print(" by Takanori Nakane at Institute of Protein Research, Osaka University")
    print()
    print("Option: bl               = %d" % opts.bl)
    print("Option: clen             = %.1f mm" % opts.clen)
    print()

    run(runid=runid, bl=opts.bl, clen=opts.clen)
