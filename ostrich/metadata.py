# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

import dbpy
import numpy as np
import re

def str2float(str):
    if str == "not-converged": return float("nan")
    if str == "saturated": return float("inf")

    m = re.match("-?\d+(.\d+)?(e[+-]?\d+)?", str)
    if m is not None:
        return float(m.group(0))
    else:
        return None

def syncdata2float(sensor, high_tag, tags):
    return [str2float(s) for s in dbpy.read_syncdatalist(sensor, high_tag, tags)]

def filter_mpccd_octal(det_ids):
    mpccds = sorted([x for x in det_ids if re.match("^MPCCD-8.*-[1-8]$", x)])
    if len(mpccds) != 8:
        raise RuntimeError("NoSupportedDetectorFound")

    return mpccds

def is_exposed(high_tag, tags, bl, runid):
    # Beamline specific constants
    if bl == 2:
        sensor_spec = "xfel_bl_2_tc_spec_1/energy"
        sensor_shutter_open = "xfel_bl_2_shutter_1_open_valid/status"
        sensor_shutter_close = "xfel_bl_2_shutter_1_close_valid/status"
    elif bl == 3:
        sensor_spec = "xfel_bl_3_tc_spec_1/energy"
        sensor_shutter_open = "xfel_bl_3_shutter_1_open_valid/status"
        sensor_shutter_close = "xfel_bl_3_shutter_1_close_valid/status"
    else:
        RuntimeError("BadBeamline")

    if (bl == 2 and runid >= 81550) or (bl == 3 and runid >= 909709):
        # Current strategy from 2020 Jan (run 81550-): look at both open and close status.
        #  This method was unreliable at 2018 but Dr. Tono says it should be fine now.
        #  If the detector alarm triggered shutter closure, these values can still be nonsense.
        #  This is why we look at "only tags at the beginning of a run" to avoid false positives.
        try:
            shutter_open = syncdata2float(sensor_shutter_open, high_tag, tags)
        except:
            raise RuntimeError("NoShutterOpenStatus")
        try:
            shutter_close = syncdata2float(sensor_shutter_close, high_tag, tags)
        except:
            raise RuntimeError("NoShutterClosedStatus")

        exposed = [True] * len(tags)
        for idx, (is_open, is_close) in enumerate(zip(shutter_open, shutter_close)):
             if int(is_open) == 0 and int(is_close) == 1:
                 exposed[idx] = False
             else:
                 break
    if bl == 2 and runid >= 32348 and runid < 81550:
        # Another strategy:
	#  2018 Feb (run 32348-): Unreliable shutter status. We should use BM1 PD and take darks only at the beginning of a run
        #  2020 Jan (run 81550-): X-ray PD does not necessarily show "not-converged" but can have values around 1e-11.
        #    Thus, xray_pd_thresh = 1e-10 was introduced, but it still does not work perfectly...
        print("The shutter open status has been unreliable for runs since 2018 Feb,")
        print("so we use X-ray PD values instead.")
        xray_pd = "xfel_bl_2_st_3_bm_1_pd/charge"
        xray_pd_thresh = 1e-10
        pd_values = syncdata2float(xray_pd, high_tag, tag_list)

        exposed = [True] * len(tags)
        for idx, pd in enumerate(pd_values):
             if (np.isnan(pd) or pd < xray_pd_thresh):
                 exposed[idx] = False
             else:
                 break
    else:
        # The oldest strategy: only using the open status because the close status was unreliable at the time.
        try:
            shutter_open = syncdata2float(sensor_shutter_open, high_tag, tags)
        except:
            raise RuntimeError("NoShutterOpenStatus")

        exposed = [int(is_open) != 0 for is_open in shutter_open]

    return exposed

def get_photon_energies(bl, runid, high_tag, tags):
    if bl == 2:
        sensor_spec = "xfel_bl_2_tc_spec_1/energy"
    elif bl == 3:
        sensor_spec = "xfel_bl_3_tc_spec_1/energy"
    else:
        RuntimeError("BadBeamline")

    config_photon_energy = 1000.0 * dbpy.read_config_photonenergy(bl, runid)
    config_photon_energy_sensible = True
    if config_photon_energy < 5000 or config_photon_energy > 14000:
        print("WARNING: dbpy.read_config_photonenergy returned %f eV, which is absurd!" % config_photon_energy)
        print("         Report this to SACLA DAQ team.")
        print("         This is not problematic unless the inline spectrometer is also broken.")
        config_photon_energy_sensible = False

    pulse_energies_in_keV  = syncdata2float(sensor_spec, high_tag, tuple(tags))
    pulse_energies = []
    for tag, energy in zip(tags, pulse_energies_in_keV):
        if energy is not None and energy > 0:
            pulse_energies.append(energy * 1000.0)
        else:
            print("WARNING: The wavelength from the inline spectrometer does not look sensible for tag %d." % tag)
            if config_photon_energy_sensible:
                pulse_energies.append(config_photon_energy)
                print("         Used the accelerator config value instead.")
            else:
                pulse_energies.append(7000.0)
                print("         The accelerator config value is also broken; assumed 7 keV as a last resort!")

    return np.array(pulse_energies), config_photon_energy
