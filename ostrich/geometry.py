# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

# This `ostrich.geometry` module deals with the detector geometry and masks.
# This also deals with various glitches specific to certain beam time.

import h5py
import math
import numpy as np
import re

from ostrich import VERSION

def validate_mpccd_geometry(det_infos):
    xsize = det_infos[0]["xsize"]
    ysize = det_infos[0]["ysize"]

    # At the moment we assume all panels have the same shape
    for det_info in det_infos:
        assert xsize == det_info["xsize"]
        assert ysize == det_info["ysize"]

def write_crystfel_geom(filename, det_infos, energy, clen, runid):
    validate_mpccd_geometry(det_infos)
    xsize = det_infos[0]["xsize"]
    ysize = det_infos[0]["ysize"]
    npanels = len(det_infos)

    with open(filename, "w") as out:
        out.write("; CrystFEL geometry file produced by Ostrich version %d\n" % VERSION)
        out.write(";   Takanori Nakane (tnakane.protein@osaka-u.ac.jp)\n")
        out.write("; Detector ID: %s\n" % det_infos[0]['id'])
        out.write("; for tiled but NOT reassembled images (512x8192 pixels)\n\n")
        out.write("clen = %.4f               ; %.1f mm camera length. You SHOULD optimize this!\n" % (clen * 1E-3, clen))
        out.write("res = 20000                 ; = 1 m /50 micron\n")
        out.write(";badrow_direction = x\n")
        out.write(";max_adu = 250000           ; should NOT be used. see best practice on CrystFEL's Web site\n")
        out.write("data = /%/data\n")
        out.write(";mask = /metadata/pixelmask ; this does not work in CrystFEL 0.6.2 (reported bug)\n")
        out.write("mask_good = 0x00            ; instead, we can specify bad regions below if necessary\n")
        out.write("mask_bad = 0xFF\n")
        out.write("photon_energy = /%%/photon_energy_ev ; roughly %.1f eV\n\n" % energy)
        out.write("; Definitions for geoptimiser\n")
        out.write("rigid_group_q1 = q1\n")
        out.write("rigid_group_q2 = q2\n")
        out.write("rigid_group_q3 = q3\n")
        out.write("rigid_group_q4 = q4\n")
        out.write("rigid_group_q5 = q5\n")
        out.write("rigid_group_q6 = q6\n")
        out.write("rigid_group_q7 = q7\n")
        out.write("rigid_group_q8 = q8\n\n")
        out.write("rigid_group_collection_connected = q1,q2,q3,q4,q5,q6,q7,q8\n")
        out.write("rigid_group_collection_independent = q1,q2,q3,q4,q5,q6,q7,q8\n\n")

        out.write("; Panel definitions\n")
        for i, det_info in enumerate(det_infos):
            name = det_info['id']
            gain = det_info['mp_absgain']
            detx = det_info['mp_posx']
            dety = det_info['mp_posy']
            detz = det_info['mp_posz']
            rotation = det_info['mp_rotationangle'] * (math.pi / 180.0) # rad
            pixel_size = det_info['mp_pixelsizex']
            print("panel %s gain %f pos (%f, %f, %f) rotation %f energy %f" % (name, gain, detx, dety, detz, rotation, energy))

            detx /= pixel_size; dety /= pixel_size;
            det_id = i + 1

            # Nphotons = S [ADU] * G [e-/ADU] / (E [eV] * 3.65 [eV/e-]) according to the manual.
            # Thus, ADU/eV = 1/(3.65*G)
            out.write("; sensor %s\n" % name)
            out.write("q%d/adu_per_eV = %f\n" % (det_id, 1.0 / (0.1 * energy))) # Keitaro's 0.1 photon
            out.write("q%d/min_fs = %d\n" % (det_id, 0))
            out.write("q%d/min_ss = %d\n" % (det_id, i * ysize))
            out.write("q%d/max_fs = %d\n" % (det_id, xsize - 1))
            out.write("q%d/max_ss = %d\n" % (det_id, (i + 1) * ysize - 1))
            out.write("q%d/fs = %fx %+fy\n" % (det_id, math.cos(rotation), math.sin(rotation)))
            out.write("q%d/ss = %fx %+fy\n" % (det_id, -math.sin(rotation), math.cos(rotation)))
            out.write("q%d/corner_x = %f\n" % (det_id, -detx))
            out.write("q%d/corner_y = %f\n\n" % (det_id, dety))

        border, outer_border = get_border(det_infos[0]['id'])
        if border != 0:
            out.write("; Bad regions near edges of each sensor.\n")
            out.write(";  l: long axis, s: short axis\n")
            out.write("; NOTE: ranges are 0-indexed and inclusive in CrystFEL\n")

            for i in range(npanels):
                out.write("badq%dl1/min_fs = %d\n"    % (i + 1, 0))
                out.write("badq%dl1/max_fs = %d\n"    % (i + 1, border - 1))
                out.write("badq%dl1/min_ss = %d\n"    % (i + 1, ysize * i))
                out.write("badq%dl1/max_ss = %d\n"    % (i + 1, ysize * (i + 1) - 1))
                out.write("badq%dl1/panel  = q%d\n\n" % (i + 1, i + 1))

                out.write("badq%dl2/min_fs = %d\n"    % (i + 1, xsize - border))
                out.write("badq%dl2/max_fs = %d\n"    % (i + 1, xsize - 1))
                out.write("badq%dl2/min_ss = %d\n"    % (i + 1, ysize * i))
                out.write("badq%dl2/max_ss = %d\n"    % (i + 1, ysize * (i + 1) - 1))
                out.write("badq%dl2/panel  = q%d\n\n" % (i + 1, i + 1))

                out.write("badq%ds1/min_fs = %d\n"    % (i + 1, 0))
                out.write("badq%ds1/max_fs = %d\n"    % (i + 1, xsize - 1))
                out.write("badq%ds1/min_ss = %d\n"    % (i + 1, ysize * i))
                out.write("badq%ds1/max_ss = %d\n"    % (i + 1, ysize * i + border - 1))
                out.write("badq%ds1/panel  = q%d\n\n" % (i + 1, i + 1))

        if outer_border != 0:
            out.write("; Bad regions near outer edges of each sensor due to amplifier shields;\n")
            out.write("; you might want to optimize these widths (edit min_ss).\n")

            for i in range(npanels):
                out.write("badq%ds2/min_fs = %d\n"    % (i + 1, 0))
                out.write("badq%ds2/max_fs = %d\n"    % (i + 1, xsize - 1))
                out.write("badq%ds2/min_ss = %d\n"    % (i + 1, ysize * (i + 1) - outer_border))
                out.write("badq%ds2/max_ss = %d\n"    % (i + 1, ysize * (i + 1) - 1))
                out.write("badq%ds2/panel  = q%d\n\n" % (i + 1, i + 1))

        if re.match("MPCCD-8B0-2-003", det_infos[0]['id']):
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

def write_cheetah_geom(filename, det_infos):
    validate_mpccd_geometry(det_infos)
    xsize = det_infos[0]["xsize"]
    ysize = det_infos[0]["ysize"]

    npanels = len(det_infos)
    posx = np.zeros((ysize * npanels, xsize), dtype=np.float32)
    posy = posx.copy()
    posz = posx.copy()

    for i, det_info in enumerate(det_infos):
        gain = det_info['mp_absgain']
        detx = det_info['mp_posx'] * 1E-6 # m
        dety = det_info['mp_posy'] * 1E-6
        detz = det_info['mp_posz'] * 1E-6
        rotation = det_info['mp_rotationangle'] * (math.pi / 180.0) # rad
        pixel_size = det_info['mp_pixelsizex'] * 1E-6 # m
        
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

def make_pixelmask(det_infos, runid):
    validate_mpccd_geometry(det_infos)
    xsize = det_infos[0]["xsize"]
    ysize = det_infos[0]["ysize"]
    npanels = len(det_infos)

    det_name = det_infos[0]["id"]
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

def write_metadata(filename, det_infos, clen, comment, runid):
    f = h5py.File(filename, "w")
    
    f["/metadata/pipeline_version"] = VERSION
    f["/metadata/run_comment"] = comment
    f["/metadata/sensor_id"] = [det_info['id'] for det_info in det_infos]
    f["/metadata/posx_in_um"] = [det_info['mp_posx'] for det_info in det_infos]
    f["/metadata/posy_in_um"] = [det_info['mp_posy'] for det_info in det_infos]
    f["/metadata/posz_in_um"] = [det_info['mp_posz'] for det_info in det_infos]
    f["/metadata/angle_in_rad"] = [det_info['mp_rotationangle'] for det_info in det_infos]
    f["/metadata/pixelsizex_in_um"] = [det_info['mp_pixelsizex'] for det_info in det_infos]
    f["/metadata/pixelsizey_in_um"] = [det_info['mp_pixelsizey'] for det_info in det_infos]
    f["/metadata/distance_in_mm"] = clen
    pixel_mask = make_pixelmask(det_infos, runid)
    f.create_dataset("/metadata/pixelmask", data=pixel_mask, compression="gzip", shuffle=True)
    f.close()
