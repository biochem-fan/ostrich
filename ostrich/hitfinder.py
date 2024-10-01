# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

import h5py
from multiprocessing import Process, Queue
import numpy as np
import zlib

import stpy

# DIALS functions
from dxtbx.format.image import ImageBool
from dxtbx.imageset import ImageSet, ImageSetData, MemReader
from dxtbx.model.experiment_list import ExperimentListFactory
from scitbx import matrix

from ostrich import update_status
from ostrich.detector import CITIUSDetector, MPCCDDetector, bin_image

def queue_based_worker(read_queue, result_queue, chunksize, detector, dtype, dark_average, pixel_mask, params):
    from dials.array_family import flex
    from ostrich.inmemory_dxtbx import FormatSACLAInMemory

    detector.allocate_readers()
    hit_threshold = params.hit_threshold
    adu_per_photon = params.adu_per_photon
    hitfinding_roi = params.hitfinding_roi
    binning = params.binning

    assert detector.geometry.width % binning == 0
    assert detector.geometry.height % binning == 0

    xsize = detector.geometry.width // binning
    ysize = detector.geometry.height // binning
    gains = [panel.gain for panel in detector.geometry.panels]
    npanels = len(detector.geometry.panels)
    cp, cy, cx = (npanels, ysize // chunksize[0], xsize // chunksize[1])
    compression_level = params.compression_level
    is_citius = isinstance(detector, CITIUSDetector)

    if is_citius:
        hitfinding_panels = CITIUSDetector.filter_prbs_by_roi(detector.det_ids, hitfinding_roi)
        in_hitfinding_roi = [det_id in hitfinding_panels for det_id in detector.det_ids]

    # Convert the pixel_mask to DIALS's flex array
    if pixel_mask is not None:
        if is_citius:
            pixel_mask = [mask for mask, valid in zip(pixel_mask, in_hitfinding_roi) if valid]
        pixel_mask = ImageBool(tuple([flex.bool(m) for m in pixel_mask]))

    while True:
        task = read_queue.get()
        if task is None:
            result_queue.put(None)
            break
        tag, pulse_energy = task

        if not is_citius:
            for reader, buf in zip(detector.readers, detector.buffers):
                reader.collect(buf, tag)

            # We quantize such that one photon is adu_per_photon output (i.e. DIALS's gain = 1 / adu_per_photon by definition).
            #
            # ValuesInFile = N_photon * adu_per_photon
            #  = CameraValueFromAPI [ADU] * GainSacla [e-/ADU] / (E_photon [eV] / 3.65 [eV/e-]) * adu_per_photon
            # CameraValueFromAPI = N_photon * E_photon / 3.65 / G
            # 3.65 is the energy required to make an electron-hole pair in silicon.
            # SACLA's gain (G) is the number of electron-hole pair per ADU, while DIALS's gain is photon/ADU.
            # For CITIUS, G = 1.00, since the values from API are normalized to the number of electrons.

            image_buf = [buf.read_det_data(0) * (gain * 3.65 * adu_per_photon / pulse_energy) - dark \
                         for gain, buf, dark in zip(gains, detector.buffers, dark_average)]
        else:
            image_buf = [None] * len(detector.geometry.panels)

            for i, panel in enumerate(detector.geometry.panels):
                if not in_hitfinding_roi[i]:
                    continue

                image_buf[i] = detector.buffers.read_image(panel.index, tag) * (adu_per_photon * 3.65 / pulse_energy)

        if False: # skip DIALS
            print(tag)
            continue

        # Images are NOT binned yet!
        image = FormatSACLAInMemory(image_buf, detector.geometry, pulse_energy, adu_per_photon, distance=params.clen)
        imageset = ImageSet(ImageSetData(MemReader([image,]), None))
        imageset.set_beam(image.get_beam())
        imageset.set_detector(image.get_detector())
        # This is usually populated by Format.get_imageset() but since we created imageset manually,
        # we have to fill this explicitly.
        imageset.external_lookup.mask.data = pixel_mask
        experiments = ExperimentListFactory.from_imageset_and_crystal(imageset, None)

        if False: # skip spot-finding
            print(tag)
            continue

        observed = flex.reflection_table.from_observations(experiments, params, is_stills=True)
        xyzobs = observed['xyzobs.px.value']
        if len(xyzobs) < hit_threshold:
            result_queue.put([tag, len(xyzobs), pulse_energy, None])
            continue

        # Read missing panels (only for CITIUS)
        for i, panel in enumerate(detector.geometry.panels):
            if not is_citius or in_hitfinding_roi[i]:
                continue
            image_buf[i] = detector.buffers.read_image(panel.index, tag) * (adu_per_photon * 3.65 / pulse_energy)

        if binning != 1:
            image_buf = [bin_image(img, binning) for img in image_buf]

        # shuffle and compress in workers (see my PR https://github.com/keitaroyam/cheetah/pull/1)
        chunkidx = 0
        compressed_chunks = [None] * (cp * cy * cx)
        for ip in range(cp):
            image_buf[ip].clip(np.iinfo(dtype).min, np.iinfo(dtype).max, out=image_buf[ip])
            rounded_buf = np.rint(image_buf[ip]).astype(dtype)
            byteview = rounded_buf.view(dtype=np.uint8) # ONLY the length of the fast axis changes
            itemsize = rounded_buf.dtype.itemsize # dtype.itemsize (e.g. np.int32.itemsize) doesn't work

            for iy in range(cy):
                for ix in range(cx):
                    sy = iy * chunksize[0]
                    sx = ix * chunksize[1]
                    ey = sy + chunksize[0]
                    ex = sx + chunksize[1]

                    chunk = byteview[sy:ey, (sx * itemsize):(ex * itemsize)]
                    shuffled = chunk.reshape((-1, itemsize)).transpose().reshape(-1)
                    compressed_chunks[chunkidx] = zlib.compress(shuffled.tobytes(), compression_level)
                    chunkidx += 1

        result_queue.put([tag, len(xyzobs), pulse_energy, compressed_chunks])

def find_hits(detector, tags, pulse_energies, output_filename, dark_average, pixel_mask, params):
    nproc = params.nproc
    hit_threshold = params.hit_threshold
    adu_per_photon = params.adu_per_photon
    use_nexus = params.nexus
    binning = params.binning
    status = params.status
    is_citius = isinstance(detector, CITIUSDetector)

    assert detector.geometry.width % binning == 0
    assert detector.geometry.height % binning == 0

    xsize = detector.geometry.width // binning
    ysize = detector.geometry.height // binning
    gains = [panel.gain for panel in detector.geometry.panels]
    npanels = len(detector.geometry.panels)

    if is_citius:
        dtype = np.int32
    else:
        dtype = np.uint16

    if isinstance(detector, MPCCDDetector):
        chunksize = (256, 256) # slow, fast
    else:
        chunksize = (ysize, xsize)

    assert ysize % chunksize[0] == 0
    assert xsize % chunksize[1] == 0
    cp, cy, cx = (npanels, ysize // chunksize[0], xsize // chunksize[1])
    compression_level = params.compression_level

    read_queue = Queue()
    result_queue = Queue()
    workers = []

    # Send tasks first so that workers can start processing
    # while other workers are being spawned.
    i = 0
    for tag, pulse_energy in zip(tags, pulse_energies):
        read_queue.put([tag, pulse_energy])
        i += 1
#        if i > 99: break # DEBUG
    for i in range(nproc): read_queue.put(None)

    # Create workers
    detector.deallocate_readers()
    for i in range(nproc):
        # TODO: serialization of dark_average and pixel_mask is very slow.
        # Testing showed that when flex.array_family was imported, serialization of NumPy
        # arrays became slower (https://github.com/dials/dials/issues/2708).
        # NumPy array backed by multiprocessing.shared_memory is much much faster.
        p = Process(target=queue_based_worker, args=(read_queue, result_queue, \
                    chunksize, detector, dtype, dark_average, pixel_mask, params))
        p.start()
        print("Worker process", i, "started.")
        workers.append(p)

    n_finished = 0
    n_hit = 0
    n_processed = 0
    h5out = h5py.File(output_filename, "a")

    if use_nexus:
        data_group = h5out["/entry/data"]
        # The scale factor is unofficial at the moment. See https://github.com/nexusformat/definitions/pull/1343.
        data_group.create_dataset("data_scale_factor", data=1.0 / adu_per_photon)
        d = data_group.create_dataset("data", shape=(0, 0, 0), dtype=dtype, \
                                      maxshape=(None, ysize * npanels, xsize), chunks=(1, chunksize[0], chunksize[1]), \
                                      compression="gzip", shuffle=True)
    #         compression=hdf5plugin.Blosc2(cname='blosclz', clevel=9, filters=hdf5plugin.Blosc2.BITSHUFFLE))
    #         compression=hdf5plugin.BZip2())
    #         compression=hdf5plugin.Bitshuffle(cname='zstd')) # zstd or lz4

    hit_energies = []
    hit_ids = []
    while n_finished < nproc:
        task = result_queue.get()
        if task is None:
            n_finished += 1
            continue

        tag, n_spots, pulse_energy, image = task
        n_processed += 1

        if image is not None:
            if use_nexus:
                hit_ids.append(tag)
                hit_energies.append(pulse_energy)
                d.resize((n_hit + 1, ysize * npanels, xsize))
            else:
                g = h5out.create_group("tag-%d" % tag)
                g.create_dataset("photon_energy_ev", data=pulse_energy)
                d = g.create_dataset("data", (npanels * ysize, xsize), chunks=chunksize,
                                     compression="gzip", compression_opts=compression_level, shuffle=True, dtype=dtype)
            chunkidx = 0
            for ip in range(cp):
                for iy in range(cy):
                    for ix in range(cx):
                        sy = iy * chunksize[0] + ip * ysize
                        sx = ix * chunksize[1]

                        if use_nexus:
                            d.id.write_direct_chunk(offsets=(n_hit, sy, sx), data=image[chunkidx], filter_mask=0)
                        else:
                            d.id.write_direct_chunk(offsets=(sy, sx), data=image[chunkidx], filter_mask=0)

                        chunkidx += 1
            n_hit += 1

        # The LLFpassed field is kept for backward compatibility.
        if n_processed % 10 == 0:
             update_status(status, "Total=%d,Processed=%d,LLFpassed=%d,Hits=%ld,Status=Hitfinding" % (len(pulse_energies), n_processed, n_processed, n_hit))
        print("%4d / %4d processed, %4d hits, current tag = %d with %d spot(s)" % (n_processed, len(tags), n_hit, tag, n_spots))

    if use_nexus:
        h5out["/entry/data"].create_dataset("tag_id", data=hit_ids)
        h5out["/entry/instrument/beam/incident_wavelength"].resize((n_hit,))
        h5out["/entry/instrument/beam/incident_wavelength"][:] = [12398.4193 / e for e in hit_energies]
        h5out["/entry/instrument/beam/incident_energy"].resize((n_hit,))
        h5out["/entry/instrument/beam/incident_energy"][:] = hit_energies

    h5out.close()
    [t.join() for t in workers]
    read_queue.close()
    result_queue.close()
    print("%d Hit / %d Processed." % (n_hit, n_processed))
    update_status(status, "Total=%d,Processed=%d,LLFpassed=%d,Hits=%ld,Status=Finished" % (len(pulse_energies), n_processed, n_processed, n_hit))
