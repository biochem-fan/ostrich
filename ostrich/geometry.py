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

def write_crystfel_geom(filename, geometry, energy, adu_per_photon, clen, runid):
    xsize = geometry.width
    ysize = geometry.height
    npanels = len(geometry.panels)

    with open(filename, "w") as out:
        out.write("; CrystFEL geometry file produced by Ostrich version %d\n" % VERSION)
        out.write(";   Takanori Nakane (tnakane.protein@osaka-u.ac.jp)\n")
        # out.write("; for tiled but NOT reassembled images (512x8192 pixels)\n\n")
        out.write("clen = %.4f    ; %.1f mm camera length. You SHOULD optimize this!\n" % (clen * 1E-3, clen))
        out.write("res = %.1f     ; = 1 m / %.4f micron\n" % (1E6 / geometry.pixel_size, geometry.pixel_size))
        out.write("data = /%/data\n")
        out.write("; === Masks =========================\n")
        out.write(";  Unfortunately, this pixel mask does not work in earlier versions of CrystFEL.\n")
        out.write(";  CrystFEL 0.5 to 0.9: the mask is ignored because they request per-shot masks.\n")
        out.write(";  CrystFEL > 0.10: OK\n")
        out.write("mask = /metadata/pixelmask\n")
        out.write("mask_good = 0x00\n")
        out.write("mask_bad = 0xFF\n")
        out.write("photon_energy = /%%/photon_energy_ev ; roughly %.1f eV\n" % energy)
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
            pixel_size = geometry.pixel_size
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
            out.write("%s/corner_x = %f\n" % (name, -detx / pixel_size)) # px
            out.write("%s/corner_y = %f\n" % (name, dety / pixel_size)) # px
            out.write("%s/coffset = %f\n\n" % (name, detz * 1E-6)) # m

        border, outer_border = get_border(geometry.panels[0].long_name)
        if border != 0:
            out.write("; Bad regions near edges of each sensor.\n")
            out.write(";  l: long axis, s: short axis\n")
            out.write("; NOTE: ranges are 0-indexed and inclusive in CrystFEL\n")

            for i in range(npanels):
                out.write("bad%sl1/min_fs = %d\n"    % (name, 0))
                out.write("bad%sl1/max_fs = %d\n"    % (name, border - 1))
                out.write("bad%sl1/min_ss = %d\n"    % (name, ysize * i))
                out.write("bad%sl1/max_ss = %d\n"    % (name, ysize * (i + 1) - 1))
                out.write("bad%sl1/panel  = %s\n\n" % (name, name))

                out.write("bad%sl2/min_fs = %d\n"    % (name, xsize - border))
                out.write("bad%sl2/max_fs = %d\n"    % (name, xsize - 1))
                out.write("bad%sl2/min_ss = %d\n"    % (name, ysize * i))
                out.write("bad%sl2/max_ss = %d\n"    % (name, ysize * (i + 1) - 1))
                out.write("bad%sl2/panel  = %s\n\n" % (name, name))

                out.write("bad%ss1/min_fs = %d\n"    % (name, 0))
                out.write("bad%ss1/max_fs = %d\n"    % (name, xsize - 1))
                out.write("bad%ss1/min_ss = %d\n"    % (name, ysize * i))
                out.write("bad%ss1/max_ss = %d\n"    % (name, ysize * i + border - 1))
                out.write("bad%ss1/panel  = %s\n\n" % (name, name))

        if outer_border != 0:
            out.write("; Bad regions near outer edges of each sensor due to amplifier shields;\n")
            out.write("; you might want to optimize these widths (edit min_ss).\n")

            for i in range(npanels):
                out.write("bad%ss2/min_fs = %d\n"    % (name, 0))
                out.write("bad%ss2/max_fs = %d\n"    % (name, xsize - 1))
                out.write("bad%ss2/min_ss = %d\n"    % (name, ysize * (i + 1) - outer_border))
                out.write("bad%ss2/max_ss = %d\n"    % (name, ysize * (i + 1) - 1))
                out.write("bad%ss2/panel  = q%d\n\n" % (name, i + 1))

        if re.match("MPCCD-8B0-2-003", geometry.panels[0].long_name):
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

            if runid >= 73832:
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

def write_cheetah_geom(filename, geometry):
    xsize = geometry.width
    ysize = geometry.height

    npanels = len(geometry.panels)
    posx = np.zeros((ysize * npanels, xsize), dtype=np.float32)
    posy = posx.copy()
    posz = posx.copy()

    for i, panel in enumerate(geometry.panels):
        gain = panel.gain
        detx = panel.pos_x * 1E-6 # m
        dety = panel.pos_y * 1E-6
        detz = panel.pos_z * 1E-6
        rotation = panel.rotation
        pixel_size = geometry.pixel_size * 1E-6 # m
        
        fast_x = math.cos(rotation) * pixel_size
        fast_y = math.sin(rotation) * pixel_size;
        slow_x = -math.sin(rotation) * pixel_size
        slow_y = math.cos(rotation) * pixel_size;

        y = np.arange(ysize)
        for x in range(xsize):
            posx[i * ysize + y, x] = fast_x * x + slow_x * y - detx
            posy[i * ysize + y, x] = -(fast_y * x + slow_y * y + dety)
            posz[i * ysize + y, x] = 0

    f = h5py.File(filename, "w")
    f.create_dataset("x", data=posx, compression="gzip", shuffle=True)
    f.create_dataset("y", data=posy, compression="gzip", shuffle=True)
    f.create_dataset("z", data=posz, compression="gzip", shuffle=True)
    f.close()

def get_border(det_name):
    if re.match("MPCCD-8B0-2-007", det_name): # New Phase 3 detector
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
    else:
        return (0, 0)

def make_pixelmask(geometry, runid):
    #TODO: In NeXus, the mask is uint32 with bit 1: dead, 2: cold, 3: hot, 4: noisy.
    xsize = geometry.width
    ysize = geometry.height
    npanels = len(geometry.panels)

    det_name = geometry.panels[0].long_name
    border, outer_border = get_border(det_name)

    mask = np.zeros((ysize * npanels, xsize), dtype=np.uint16)
    mask[:, 0:border] = 1
    mask[:, (xsize - border):xsize] = 1

    for i in range(npanels):
        mask[(ysize * i):(ysize * i + border), :] = 1
        mask[(ysize * (i + 1) - outer_border):(ysize * (i + 1)), :] = 1

    if re.match("MPCCD-8B0-2-003", det_name): # Severly damaged Phase 3 detector
        mask[1024:2048, 501:512] = 1
        mask[2048:3072, 0:11] = 1

        if runid >= 73832:
            mask[1024:2048, 448:512] = 1
            mask[2048:3072, 0:64] = 1
            mask[5120:6144, 0:64] = 1
            mask[6144:7168, 448:512] = 1

    return mask

def write_metadata(filename, geometry, clen, comment, runid, adu_per_photon, pixel_mask):
    f = h5py.File(filename, "w")
   
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
    f["/metadata/pixelsizex_in_um"] = [geometry.pixel_size] * len(geometry.panels)
    f["/metadata/pixelsizey_in_um"] = [geometry.pixel_size] * len(geometry.panels)
    f["/metadata/distance_in_mm"] = clen
    f.create_dataset("/metadata/pixelmask", data=pixel_mask, compression="gzip", shuffle=True)
    f.close()
