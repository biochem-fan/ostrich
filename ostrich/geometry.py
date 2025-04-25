# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

# This `ostrich.geometry` module deals with the detector geometry and masks.
# This also deals with various glitches specific to certain beam time.

import h5py
import math
import numpy as np
import re

from ostrich import VERSION

# NeXus McStats:
#  Z along the beam, towards the detector
#  X towards horizontally left, looking from the source to the detector
#  Y completes the right handed coordinate system (towards the ceiling)
#
# CrystFEL:
#  Z along the beam, towards the detector
#  Y towards the ceiling
#  X completes the right handed coordinate system (towards horizontally left)
#  Thus this is the same as NeXus McStats.
#  Note that in the CrystFEL GUI, +X is horizontal, towards RIGHT,
#  +Y is vertical, towards top. Thus it is looking the detector from the back
#  towards the source.
#
# CBF (DIALS):
#  Z along the beam, towards the source
#  X goniometer axis
#  Y completes the right handed coordinate system (towards ceiling, if X is horizontally right)
#  For stills, it does not matter which way X is pointing. The difference ends up as an in-plane rotation.
#  For example, we can take +X horizontally left and +Y gravity.
#  The difference is merely a 180 degree rotation around the beam.
#
# MPCCD & CITIUS API:
#  Z along the beam, towards the source (e.g. outer columns are closer to the source)
#  X towards horizontally right, looking from the source to the detector
#  Y completes the right handed coordinate system (i.e. towards the ceiling)
#  Rotation is anti-clockwise.
#  Thus, Z and X must be flipped for NeXus; X and Y for CBF/DIALS/dxtbx.

def write_crystfel_geom(filename, use_nexus, geometry, energy, adu_per_photon, clen, bl, runid, beam_center, binning=1):
    assert geometry.width % binning == 0
    assert geometry.height % binning == 0

    xsize = geometry.width // binning
    ysize = geometry.height // binning
    pixel_size = geometry.pixel_size * binning
    npanels = len(geometry.panels)

    with open(filename, "w") as out:
        out.write("; CrystFEL geometry file produced by Ostrich version %d\n" % VERSION)
        out.write(";   Takanori Nakane (tnakane.protein@osaka-u.ac.jp)\n")
        out.write("clen = %.4f    ; %.1f mm camera length. You SHOULD optimize this!\n" % (clen * 1E-3, clen))
        out.write("res = %.1f     ; = 1 m / %.4f micron (binning %d) \n" % (1E6 / pixel_size, pixel_size, binning))
        if use_nexus:
            out.write("data = /entry/data/data\n")
            out.write("dim0 = %\n")
            out.write("dim1 = ss\n")
            out.write("dim2 = fs\n")
            out.write("photon_energy = /entry/instrument/beam/incident_energy ; roughly %.1f eV\n" % energy)
        else:
            out.write("data = /%/data\n")
            out.write("photon_energy = /%%/photon_energy_ev ; roughly %.1f eV\n" % energy)
        out.write("\n")

        out.write("; === Masks =========================\n")
        out.write(";  Unfortunately, this pixel mask does not work in earlier versions of CrystFEL.\n")
        out.write(";  CrystFEL 0.5 to 0.9: the mask is ignored because they request per-shot masks.\n")
        out.write(";  CrystFEL > 0.10: OK\n")
        if use_nexus:
            out.write("mask = /entry/instrument/detector/pixel_mask\n")
        else:
            out.write("mask = /metadata/pixelmask\n")
        out.write("mask_good = 0x00\n")
        out.write("mask_bad = 0xFF\n")
        out.write("\n")

        out.write("; Group definitions for geoptimiser\n")
        for panel in geometry.panels:
            out.write("rigid_group_%s = %s\n" % (panel.name, panel.name))
        out.write("rigid_group_collection_independent = " + \
                  ",".join([panel.name for panel in geometry.panels]) + "\n")

        for (group_name, members) in geometry.groups:
            out.write("rigid_group_%s = " % group_name + ",".join([panel_name for panel_name in members]) + "\n")
        out.write("rigid_group_collection_connected = " + ",".join([group_name for (group_name, _) in geometry.groups]) + "\n")
        out.write("\n")

        out.write("; Panel definitions\n")
        for i, panel in enumerate(geometry.panels):
            name = panel.name
            gain = panel.gain
            detx = panel.pos_x
            dety = panel.pos_y
            detz = panel.pos_z
            rotation = panel.rotation
            print("panel %s gain %f pos (%f, %f, %f) rotation %f energy %f" % (name, gain, detx, dety, detz, rotation, energy))

            out.write("; sensor %s\n" % panel.long_name)
            out.write("%s/adu_per_eV = %f\n" % (name, adu_per_photon / energy))
            out.write("%s/min_fs = %d\n" % (name, 0))
            out.write("%s/min_ss = %d\n" % (name, i * ysize))
            out.write("%s/max_fs = %d\n" % (name, xsize - 1))
            out.write("%s/max_ss = %d\n" % (name, (i + 1) * ysize - 1))
            # The sign before cosines is the opposite of my old Cheetah & dxtbx class
            # but I believe this is correct.
            out.write("%s/fs = %fx %+fy\n" % (name, -math.cos(rotation), math.sin(rotation)))
            out.write("%s/ss = %fx %+fy\n" % (name, -math.sin(rotation), -math.cos(rotation)))
            out.write("%s/corner_x = %f\n" % (name, (-detx + 1000 * beam_center[0]) / pixel_size)) # px
            out.write("%s/corner_y = %f\n" % (name, (dety + 1000 * beam_center[1]) / pixel_size)) # px
            out.write("%s/coffset = %f\n\n" % (name, -detz * 1E-6)) # m

        border, outer_border = get_border(geometry.name)
        border = int(np.ceil(border / binning))
        outer_border = int(np.ceil(outer_border / binning))
        if border != 0:
            out.write("; Bad regions near edges of each sensor.\n")
            out.write(";  l: long axis, s: short axis\n")
            out.write("; NOTE: ranges are 0-indexed and inclusive in CrystFEL\n")

            for i, panel in enumerate(geometry.panels):
                name = panel.name
                out.write("bad%sl1/min_fs = %d\n"    % (name, 0))
                out.write("bad%sl1/max_fs = %d\n"    % (name, border - 1))
                out.write("bad%sl1/min_ss = %d\n"    % (name, ysize * i))
                out.write("bad%sl1/max_ss = %d\n"    % (name, ysize * (i + 1) - 1))
                out.write("bad%sl1/panel  = %s\n\n"  % (name, name))

                out.write("bad%sl2/min_fs = %d\n"    % (name, xsize - border))
                out.write("bad%sl2/max_fs = %d\n"    % (name, xsize - 1))
                out.write("bad%sl2/min_ss = %d\n"    % (name, ysize * i))
                out.write("bad%sl2/max_ss = %d\n"    % (name, ysize * (i + 1) - 1))
                out.write("bad%sl2/panel  = %s\n\n"  % (name, name))

                out.write("bad%ss1/min_fs = %d\n"    % (name, 0))
                out.write("bad%ss1/max_fs = %d\n"    % (name, xsize - 1))
                out.write("bad%ss1/min_ss = %d\n"    % (name, ysize * i))
                out.write("bad%ss1/max_ss = %d\n"    % (name, ysize * i + border - 1))
                out.write("bad%ss1/panel  = %s\n\n"  % (name, name))

        if outer_border != 0:
            out.write("; Bad regions near outer edges of each sensor due to amplifier shields;\n")
            out.write("; you might want to optimize these widths (edit min_ss).\n")

            for i, panel in enumerate(geometry.panels):
                name = panel.name
                out.write("bad%ss2/min_fs = %d\n"    % (name, 0))
                out.write("bad%ss2/max_fs = %d\n"    % (name, xsize - 1))
                out.write("bad%ss2/min_ss = %d\n"    % (name, ysize * (i + 1) - outer_border))
                out.write("bad%ss2/max_ss = %d\n"    % (name, ysize * (i + 1) - 1))
                out.write("bad%ss2/panel  = %s\n\n"  % (name, name))

        # Probably this should be moved to the detector class.
        if re.match("MPCCD-8B0-2-003", geometry.panels[0].long_name):
            assert binning == 1

            out.write("; Severly damaged Phase 3 detector\n")
            out.write("baddamage1/min_fs = 501\n")
            out.write("baddamage1/max_fs = 511\n")
            out.write("baddamage1/min_ss = 1024\n")
            out.write("baddamage1/max_ss = 2047\n")
            out.write("baddamage1/panel  = q2\n\n")

            out.write("baddamage2/min_fs = 0\n")
            out.write("baddamage2/max_fs = 12\n")
            out.write("baddamage2/min_ss = 2048\n")
            out.write("baddamage2/max_ss = 3071\n")
            out.write("baddamage2/panel  = q3\n\n")

            if runid >= 73832 and bl == 2:
                out.write("; 2019B: broken ports in Panel 2,3,6,7\n")
                out.write("badport2/min_fs = 448\n")
                out.write("badport2/max_fs = 511\n")
                out.write("badport2/min_ss = 1024\n")
                out.write("badport2/max_ss = 2047\n")
                out.write("badport2/panel  = q2\n\n")

                out.write("badport3/min_fs = 0\n")
                out.write("badport3/max_fs = 63\n")
                out.write("badport3/min_ss = 2048\n")
                out.write("badport3/max_ss = 3071\n")
                out.write("badport3/panel  = q3\n\n")

                out.write("badport6/min_fs = 0\n")
                out.write("badport6/max_fs = 63\n")
                out.write("badport6/min_ss = 5120\n")
                out.write("badport6/max_ss = 6143\n")
                out.write("badport6/panel  = q6\n\n")

                out.write("badport7/min_fs = 448\n")
                out.write("badport7/max_fs = 511\n")
                out.write("badport7/min_ss = 6144\n")
                out.write("badport7/max_ss = 7167\n")
                out.write("badport7/panel  = q7\n\n")

        elif re.match("MPCCD-8B0-2-008", geometry.panels[0].long_name):
            assert binning == 1

            if runid >= 223598 and bl == 2:
                out.write("; Damaged port 1\n")
                out.write("badq3port1/min_fs = 0\n")
                out.write("badq3port1/max_fs = 63\n")
                out.write("badq3port1/min_ss = 2048\n")
                out.write("badq3port1/max_ss = 3071\n")
                out.write("badq3port1/panel  = q3\n\n")

# Returns (border, outer_border)
# outer_border is along the fast edge at the largest slow values
def get_border(det_name):
    if re.match("MPCCD-8B0-2-008", det_name): # New Phase 3 detector
        return (5, 30) # based on 24Feb-Shimada @ 10keV
    elif re.match("MPCCD-8B0-2-007", det_name): # New Phase 3 detector
        return (5, 33) # based on 22Nov-Iwata @ 10keV
    elif re.match("MPCCD-8B0-2-006", det_name): # New Phase 3 detector
        return (5, 30) # based on 20Feb-Ueno @ 10keV
    elif re.match("MPCCD-8B0-2-005", det_name): # New Phase 3 detector
        # I wasn't aware of this detector until its retirement on 22Nov and
        # used to manually modify the geometry every beam time.
        # This line was introduced on 22Nov, just in case someone reprocesses old data.
        return (5, 35) # based on 22Oct-Ueno @ 10keV
    elif re.match("MPCCD-8B", det_name): # Other Phase 3 detector
        return (5, 23) # based on 17Jul-P3Lys @ 10 keV
    elif re.match("MPCCD-8N", det_name): # Compact detector with amp shields
        return (0, 22) # based on 17Jul-Kuma @ 7 keV
    elif re.match("CITIUS 20.2M", det_name): # based on 2024-Jul-12, 180 mm
        return (1, 14)
    else:
        return (5, 30) # default assumes New Phase 3 detector

# This returns a non-binned mask, because it is used for in-memory hitfinding.
def make_pixelmask(geometry, bl, runid, binning=1):
    assert geometry.width % binning == 0
    assert geometry.height % binning == 0

    xsize = geometry.width // binning
    ysize = geometry.height // binning
    npanels = len(geometry.panels)

    border, outer_border = get_border(geometry.name)
    border = int(np.ceil(border / binning))
    outer_border = int(np.ceil(outer_border / binning))

    # NeXus bit masks
    GAP = 1 # bit 0
    DEAD = 2 # bit 1
    COLD = 4 # bit 2
    HOT = 8 # bit 3
    NOISY = 16 # bit 4

    mask = np.zeros((ysize * npanels, xsize), dtype=np.uint32)
    mask[:, 0:border] = NOISY
    mask[:, (xsize - border):xsize] = NOISY

    for i in range(npanels):
        mask[(ysize * i):(ysize * i + border), :] = NOISY
        mask[(ysize * (i + 1) - outer_border):(ysize * (i + 1)), :] = COLD

    if re.match("MPCCD-8B0-2-003", geometry.name): # Severly damaged Phase 3 detector
        assert binning == 1
        mask[1024:2048, 501:512] = NOISY
        mask[2048:3072, 0:11] = NOISY

        if runid >= 73832 and bl == 2:
            mask[1024:2048, 448:512] = NOISY
            mask[2048:3072, 0:64] = NOISY
            mask[5120:6144, 0:64] = NOISY
            mask[6144:7168, 448:512] = NOISY

    elif re.match("MPCCD-8B0-2-008", geometry.panels[0].long_name):
        assert binning == 1

        if runid >= 223598 and bl == 2:
            mask[2048:3072, 0:64] = NOISY

    return mask

# We don't support binning here, because our MPCCD dxtbx class does not support it anyway.
def write_metadata(filename, geometry, clen, comment, runid, adu_per_photon, pixel_mask):
    f = h5py.File(filename, "w")

    pixel_size = geometry.pixel_size

    if adu_per_photon != 10.0:
        print("WARNING: DIALS assumes adu_per_photon is 10.0.")
        print("         Because you set it to be %f, you have to explicitly specify it during data processing." % adu_per_photon)
    f["/metadata/pipeline_version"] = VERSION
    f["/metadata/run_comment"] = comment
    f["/metadata/adu_per_photon"] = adu_per_photon
    f["/metadata/sensor_id"] = [panel.long_name for panel in geometry.panels]
    f["/metadata/posx_in_um"] = [panel.pos_x for panel in geometry.panels]
    f["/metadata/posy_in_um"] = [panel.pos_y for panel in geometry.panels]
    f["/metadata/posz_in_um"] = [panel.pos_z for panel in geometry.panels]
    # Although this field name is angles_in_RAD, the dxtbx class interprets it as degrees...
    # For compatibility, we keep it as is.
    f["/metadata/angle_in_rad"] = [panel.rotation * (180.0 / math.pi) for panel in geometry.panels]
    f["/metadata/pixelsizex_in_um"] = [pixel_size] * len(geometry.panels)
    f["/metadata/pixelsizey_in_um"] = [pixel_size] * len(geometry.panels)
    f["/metadata/distance_in_mm"] = clen
    f.create_dataset("/metadata/pixelmask", data=pixel_mask, compression="gzip", shuffle=True)
    f.close()

def write_nexus(filename, geometry, bl, runid, comment, start_time, end_time, clen, adu_per_photon, pixel_mask, beam_center, binning=1):
    assert geometry.width % binning == 0
    assert geometry.height % binning == 0

    xsize = geometry.width // binning
    ysize = geometry.height // binning
    pixel_size = geometry.pixel_size * binning
    npanels = len(geometry.panels)

    assert pixel_mask.shape == (ysize * npanels, xsize)

    f = h5py.File(filename, "w")

    # Entry
    entry = f.create_group("entry")
    entry.attrs['NX_class'] = "NXentry"
    entry.create_dataset("definition", data="NXmx")
    entry.create_dataset("title", data="SFX dataset collected at SACLA BL%d" % bl)
    entry.create_dataset("experiment_identifier", data="Run %d" %  runid)
    entry.create_dataset("notes", data="Only hit images are recoded in this file. Bin %d." % binning)
    program_name = entry.create_dataset("program_name", data="Ostrich")
    program_name.attrs['version'] = str(VERSION)

    entry.create_dataset("start_time", data=start_time) # required
    entry.create_dataset("end_time", data=end_time) # not required
    entry.create_dataset("end_time_estimated", data=end_time) # this IS required

    sample = entry.create_group("sample")
    sample.attrs['NX_class'] = "NXsample"
    sample.create_dataset("name", data="SFX microcrystals")
    sample.create_dataset("depends_on", data=".")

    instrument = entry.create_group("instrument")
    instrument.attrs['NX_class'] = "NXinstrument"
    instrument.create_dataset("name", data="SACLA BL%d" % bl)

    # Beam
    source = instrument.create_group("source")
    source.attrs['NX_class'] = "NXsource"
    source.create_dataset("type", data="Free-Electron Laser")
    source.create_dataset("probe", data="x-ray")
    source.create_dataset("frequency", data=30.0)

    beam = instrument.create_group("beam")
    beam.attrs['NX_class'] = "NXbeam"
    incident_wavelength = beam.create_dataset("incident_wavelength", shape=(0,), maxshape=(None, ), dtype=np.float32)
    incident_wavelength.attrs['units'] = "angstrom"
    incident_energy = beam.create_dataset("incident_energy", shape=(0,), maxshape=(None, ), dtype=np.float32)
    incident_energy.attrs['units'] = "eV"

    # Detector
    detector = instrument.create_group("detector")
    detector.attrs['NX_class'] = "NXdetector"
    detector.create_dataset("depends_on", data=".")
    detector.create_dataset("description", data=geometry.name)
    detector.create_dataset("pixel_mask_applied", data=0, dtype=np.int8) # data=True creates H5T_ENUM
    detector.create_dataset("pixel_mask", dtype=np.uint32, data=pixel_mask, compression="gzip", shuffle=True)
    detector["data"] = h5py.SoftLink('/entry/data/data')

    sensor_material = detector.create_dataset("sensor_material", data="Si")
    sensor_thickness = detector.create_dataset("sensor_thickness", data=geometry.thickness * 0.001)
    sensor_thickness.attrs['units'] = 'mm'
    # distance, beam_center_x/y are not mandatory because they will be derived from the transformation chain

    transformations = detector.create_group("transformations")
    transformations.attrs['NX__class'] = "NXtransformations"
    whole = transformations.create_dataset("whole", data=clen)
    whole.attrs['units'] = 'mm'
    whole.attrs['transformation_type'] = "translation"
    whole.attrs['vector'] = (0.0, 0.0, 1.0)
    whole.attrs['offset'] = (beam_center[0], beam_center[1], 0.0)
    whole.attrs['depends_on'] = '.'

    # MPCCD does not have a panel hierarchy
    if not geometry.name.startswith("MPCCD"):
        for (group_name, members) in geometry.groups:
            sensor_group= transformations.create_dataset(group_name, data=0.0)
            sensor_group.attrs['units'] = 'degrees'
            sensor_group.attrs['transformation_type'] = "rotation"
            sensor_group.attrs['vector'] = (0.0, 0.0, -1.0)
            sensor_group.attrs['offset'] = (0.0, 0.0, 0.0)
            sensor_group.attrs['depends_on'] = '/entry/instrument/detector/transformations/whole'

    for i, panel in enumerate(geometry.panels):
        detector_module = detector.create_group(panel.name)
        detector_module.attrs["NX_class"] = "NXdetector_module"
        origin = (-panel.pos_x * 1E-3, panel.pos_y * 1E-3, -panel.pos_z * 1E-3)

        parent_transform = "/entry/instrument/detector/transformations/whole"
        if not geometry.name.startswith("MPCCD"):
            for group_name, members in geometry.groups:
                if panel.name in members:
                    parent_transform = "/entry/instrument/detector/transformations/" + group_name
                    break

        fast_pixel_direction = detector_module.create_dataset("fast_pixel_direction", data=pixel_size * 1E-3)
        fast_pixel_direction.attrs['units'] = 'mm'
        fast_pixel_direction.attrs['vector'] = (-math.cos(panel.rotation), math.sin(panel.rotation), 0)
        fast_pixel_direction.attrs['offset'] = origin
        fast_pixel_direction.attrs['depends_on'] = parent_transform
        slow_pixel_direction = detector_module.create_dataset("slow_pixel_direction", data=pixel_size * 1E-3)
        slow_pixel_direction.attrs['units'] = 'mm'
        slow_pixel_direction.attrs['vector'] = (-math.sin(panel.rotation), -math.cos(panel.rotation), 0)
        slow_pixel_direction.attrs['offset'] = origin
        slow_pixel_direction.attrs['depends_on'] = parent_transform
        detector_module.create_dataset("data_size", data=(ysize, xsize))
        detector_module.create_dataset("data_origin", data=(ysize * i, 0))

    # Data
    data_group = entry.create_group("data")
    data_group.attrs['NX_class'] = "NXdata"
    data_group.attrs['signal'] = "data"
    data_group.create_dataset("data_scale_factor", data=1.0 / adu_per_photon)

    f.close()
