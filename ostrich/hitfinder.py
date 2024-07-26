# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

import h5py
from multiprocessing import Process, Queue
import numpy as np
import zlib

import stpy

# DIALS functions
from dials.array_family import flex
from dxtbx.format.image import ImageBool
from dxtbx.imageset import ImageSet, ImageSetData, MemReader 
from dxtbx.model.experiment_list import ExperimentListFactory
from scitbx import matrix

from ostrich.detector import CITIUSDetector, MPCCDDetector
from ostrich.inmemory_dxtbx import FormatSACLAInMemory

def queue_based_worker(read_queue, result_queue, chunksize, detector, dtype, dark_average, pixel_mask, params):
    detector.allocate_readers()
    hit_threshold = params.hit_threshold
    adu_per_photon = params.adu_per_photon

    gains = [panel.gain for panel in detector.geometry.panels]
    xsize = detector.geometry.width
    ysize = detector.geometry.height
    npanels = len(detector.geometry.panels)
    cp, cy, cx = (npanels, ysize // chunksize[0], xsize // chunksize[1])
    compression_level = params.compression_level

    while True:
        task = read_queue.get()
        if task is None:
            result_queue.put(None)
            break
        tag, pulse_energy = task

        if isinstance(detector, MPCCDDetector):
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
            image_buf = [detector.buffers.read_image(panel.index, tag) * (adu_per_photon * 3.65 / pulse_energy) \
                       for panel in detector.geometry.panels]

        if False: # skip DIALS
            print(tag)
            continue

        image = FormatSACLAInMemory(image_buf, detector.geometry, pulse_energy, adu_per_photon, distance=params.clen)
        imageset = ImageSet(ImageSetData(MemReader([image,]), None))
        imageset.set_beam(image.get_beam())
        imageset.set_detector(image.get_detector())
        imageset.external_lookup.mask.data = pixel_mask
        experiments = ExperimentListFactory.from_imageset_and_crystal(imageset, None)

        if False: # skip spot-finding
            print(tag)
            continue

        observed = flex.reflection_table.from_observations(experiments, params, is_stills=True)
        xyzobs = observed['xyzobs.px.value']
        # print(tag, len(xyzobs))
        if len(xyzobs) < hit_threshold:
            result_queue.put([tag, len(xyzobs), pulse_energy, None])
            continue

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
    is_citius = isinstance(detector, CITIUSDetector)

    gains = [panel.gain for panel in detector.geometry.panels]
    xsize = detector.geometry.width
    ysize = detector.geometry.height
    npanels = len(detector.geometry.panels)

    if is_citius:
        dtype = np.int32
    else:
        dtype = np.uint16

    # Chunking parameters
    if isinstance(detector, MPCCDDetector):
        chunksize = (256, 256) # slow, fast
    else:
        chunksize = (ysize, xsize)

    assert ysize % chunksize[0] == 0
    assert xsize % chunksize[1] == 0
    cp, cy, cx = (npanels, ysize // chunksize[0], xsize // chunksize[1])
    compression_level = params.compression_level

    # Convert the pixel_mask to DIALS's flex array
    if pixel_mask is not None:
        assert len(pixel_mask) == npanels
        pixel_mask = ImageBool(tuple([flex.bool(m) for m in pixel_mask]))

    read_queue = Queue()
    result_queue = Queue()
    workers = []

    # Create workers
    detector.deallocate_readers()
    for i in range(nproc):
        p = Process(target=queue_based_worker, args=(read_queue, result_queue, \
                    chunksize, detector, dtype, dark_average, pixel_mask, params))
        p.start()
        workers.append(p)

    # Send tasks
    i = 0
    for tag, pulse_energy in zip(tags, pulse_energies):
        read_queue.put([tag, pulse_energy])
        i += 1
        if i > 200: break
    for i in range(nproc): read_queue.put(None)

    n_finished = 0
    n_hit = 0
    n_processed = 0
    h5out = h5py.File(output_filename, "a")
    while n_finished < nproc:
        task = result_queue.get()
        if task is None:
            n_finished += 1
            continue

        tag, n_spots, pulse_energy, image = task
        n_processed += 1

        if image is not None:
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

                        d.id.write_direct_chunk(offsets=(sy, sx), data= image[chunkidx], filter_mask=0)
                        chunkidx += 1
            n_hit += 1
        print(tag, n_spots)

    h5out.close()
    [t.join() for t in workers]
    read_queue.close()
    result_queue.close()
    print("%d Hit / %d Processed." % (n_hit, n_processed))
