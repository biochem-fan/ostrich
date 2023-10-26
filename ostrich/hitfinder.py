# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

import h5py
from multiprocessing import Process, Queue
import numpy as np
import zlib

import stpy

# DIALS functions
from dials.array_family import flex
from dxtbx.imageset import ImageSet, ImageSetData, MemReader 
from dxtbx.model.experiment_list import ExperimentListFactory
from scitbx import matrix

from ostrich.inmemory_dxtbx import FormatMPCCDInMemory

def queue_based_worker(read_queue, result_queue, chunksize, bl, runid, det_infos, params):
    try:
        readers = [stpy.StorageReader(det_info['id'], bl, (runid,)) for det_info in det_infos]
    except:
        raise RuntimeError("FailedOn_create_streader")
    try:
        buffers = [stpy.StorageBuffer(reader) for reader in readers]
    except:
        raise RuntimeError("FailedOn_create_stbuf")

    gains = [det_info['mp_absgain'] for det_info in det_infos]
    hit_threshold = params.hit_threshold

    xsize = det_infos[0]["xsize"]
    ysize = det_infos[0]["ysize"]
    npanels = len(det_infos)
    cp, cy, cx = (npanels, ysize // chunksize[0], xsize // chunksize[1])
    compression_level = params.compression_level

    while True:
        task = read_queue.get()
        if task is None:
            result_queue.put(None)
            break
        tag, pulse_energy = task

        for reader, buf in zip(readers, buffers):
            reader.collect(buf, tag)

        # ADU = N_photon * photon_energy / 3.65 / gain
        # 3.65 is eV to make an electron-hole pair in silicon.
        # SACLA's gain is the number of electron-hole pair per ADU, while DIALS's gain is photon/ADU.
        # Thus, N_photon = ADU * 3.65 * gain / photon_energy.
        # We use the "0.1 photon" unit (i.e. DIALS's gain = 0.1 by definition)
        image_buf = [(buf.read_det_data(0) * (gain * 3.65 * 10 / pulse_energy)).astype(np.int32) \
                     for gain, buf in zip(gains, buffers)]

        if False: # skip DIALS
            print(tag)
            continue

        image = FormatMPCCDInMemory(image_buf, det_infos, pulse_energy)
        imageset = ImageSet(ImageSetData(MemReader([image,]), None))
        imageset.set_beam(image.get_beam())
        imageset.set_detector(image.get_detector())
        experiments = ExperimentListFactory.from_imageset_and_crystal(imageset, None)

        if False: # skip spot-finding
            print(tag)
            continue

        observed = flex.reflection_table.from_observations(experiments, params, is_stills=True)
        xyzobs = observed['xyzobs.px.value']
        print(tag, len(xyzobs))
        if len(xyzobs) < hit_threshold:
            result_queue.put([tag, len(xyzobs), None])
            continue

        # shuffle and compress in workers (see my PR https://github.com/keitaroyam/cheetah/pull/1)
        chunkidx = 0
        compressed_chunks = [None] * (cp * cy * cx)
        for ip in range(cp):
            image_buf[ip].clip(0, np.iinfo(np.uint16).max, out=image_buf[ip])
            uint16buf = image_buf[ip].astype(np.uint16)
            byteview = uint16buf.view(dtype=np.uint8)
            itemsize = uint16buf.dtype.itemsize

            for iy in range(cy):
                for ix in range(cx):
                    sy = iy * chunksize[0]
                    sx = ix * chunksize[1]
                    ey = sy + chunksize[0]
                    ex = sx + chunksize[1]

                    chunk = byteview[sy:ey, (sx * itemsize):(ey * itemsize)]
                    shuffled = chunk.reshape((-1, itemsize)).transpose().reshape(-1)
                    compressed_chunks[chunkidx] = zlib.compress(shuffled.tobytes(), compression_level)
                    chunkidx += 1

        result_queue.put([tag, len(xyzobs), compressed_chunks])

def find_hits(bl, runid, det_infos, tags, pulse_energies, nproc, params):
    xsize = det_infos[0]["xsize"]
    ysize = det_infos[0]["ysize"]
    npanels = len(det_infos)
    gains = [det_info['mp_absgain'] for det_info in det_infos]

    # Chunking parameters
    chunksize = (256, 256)
    assert ysize % chunksize[0] == 0
    assert xsize % chunksize[1] == 0
    cp, cy, cx = (npanels, ysize // chunksize[0], xsize // chunksize[1])
    compression_level = params.compression_level

    read_queue = Queue()
    result_queue = Queue()
    workers = []

    # Create workers
    for i in range(nproc):
        p = Process(target=queue_based_worker, args=(read_queue, result_queue, \
                      chunksize, bl, runid, det_infos, params))
        p.start()
        workers.append(p)

    # Send tasks
    for tag, pulse_energy in zip(tags, pulse_energies):
        read_queue.put([tag, pulse_energy])
    for i in range(nproc): read_queue.put(None)

    n_finished = 0
    n_hit = 0
    n_processed = 0
    h5out = h5py.File("test.h5", "w")
    while n_finished < nproc:
        task = result_queue.get()
        if task is None:
            n_finished += 1
            continue

        tag, n_spots, image = task
        n_processed += 1

        if image is not None:
            d = h5out.create_dataset("tag-%d" % tag, (npanels * ysize, xsize), chunks=chunksize,
                                     compression="gzip", compression_opts=compression_level, shuffle=True, dtype=np.uint16)
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
