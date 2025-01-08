# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

import h5py
from multiprocessing import Process, Queue, shared_memory
import numpy as np

import ctolpy_xfel

# DIALS functions
from dxtbx.format.image import ImageBool
from dxtbx.imageset import ImageSet, ImageSetData, MemReader
from dxtbx.model.experiment_list import ExperimentListFactory
from scitbx import matrix

from ostrich import update_status, OSTRICH_ONLINE_SHM_NAME
from ostrich.detector import CITIUSDetector, MPCCDDetector, bin_image

FRAME_NS = int(1E9 / 30.0) # 30 FPS

def image_reading_worker(worker_id, start_frame, read_queue, hitfinding_queue, detector, shared_buffer_shape, dtype, params):
    from dials.array_family import flex
    from ostrich.inmemory_dxtbx import FormatSACLAInMemory

    shm = shared_memory.SharedMemory(OSTRICH_ONLINE_SHM_NAME)
    framebuffer = np.ndarray(shared_buffer_shape, dtype, buffer=shm.buf)

    detector.allocate_ctrl_buffer()
    nproc_reader = params.nproc_reader

    xsize = detector.geometry.width
    ysize = detector.geometry.height
    gains = [panel.gain for panel in detector.geometry.panels]

    frame_idx = 0
    prev_frame = start_frame
    while True:
        task = read_queue.get()
        if task is None:
            result_queue.put(None)
            break
        slot = task

        while True:
            next_frame = start_frame + int(FRAME_NS * (- 0.1 + (1 + np.round(frame_idx)) * nproc_reader))
            info = detector.ctrl_buffer.collect_data(framebuffer[slot, :, :, :,], detector.det_ids, next_frame)
            cur_frame = info['data_time']
            delta_prev = (cur_frame - prev_frame) / FRAME_NS
            delta_req = (cur_frame - next_frame) / FRAME_NS
            prev_frame = cur_frame
            frame_idx = (((cur_frame - start_frame) / (FRAME_NS * nproc_reader))) # TODO: not right!
            print("Reader %d retrieved frame %d (req %d) to slot %d, delta prev = %f, delta req = %f, frame idx = %f, %d" %
                      (worker_id, cur_frame, next_frame, slot, delta_prev, delta_req, frame_idx, np.round(frame_idx)))
            if delta_req > 0.3:
                frame_idx = np.round(frame_idx) + int(16 / nproc_reader)
                print("Reader %d is too slow! fast forwarding frame idx to %f" % (worker_id, frame_idx))
            else:
                break

def hitfinding_worker(woreker_id, hitfind_queue, result_queue, detector, dtype, pixel_mask, params):
    # Convert the pixel_mask to DIALS's flex array
    if pixel_mask is not None:
        pixel_mask = ImageBool(tuple([flex.bool(m) for m in pixel_mask]))

    while True:
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
        result_queue.put([tag, len(xyzobs)])

def find_hits(detector, shared_buffer, photon_energy, pixel_mask, params):
    nproc_reader = params.nproc_reader
    framebuffer_size = params.framebuffer_size
    adu_per_photon = 10 # TODO
    status = params.status

    xsize = detector.geometry.width
    ysize = detector.geometry.height
    gains = [panel.gain for panel in detector.geometry.panels]
    npanels = len(detector.geometry.panels)
    dtype = np.float32

    read_queue = Queue()
    hitfind_queue = Queue()
    result_queue = Queue()
    image_read_workers = []

    # Fill all empty slots
    for i in range(framebuffer_size):
        read_queue.put(i)

    # Get the start time
    info = detector.ctrl_buffer.collect_data(shared_buffer[0, :, :, :,], detector.det_ids, ctolpy_xfel.NEWEST)
    cur_frame = info['data_time']
    print("Base time %d" % cur_frame)

    # Create workers
    detector.deallocate_readers()
    for i in range(nproc_reader):
        start_frame = cur_frame + FRAME_NS * i
        p = Process(target=image_reading_worker, args=(i, start_frame, read_queue, hitfind_queue, \
                    detector, shared_buffer.shape, dtype, params))
        p.start()
        print("Image reading worker process %d started with base time %d" % (i, start_frame))
        image_read_workers.append(p)

    n_finished = 0
    n_processed = 0

    while True:
        task = result_queue.get()
        if task is None:
            n_finished += 1
            continue

        tag, n_spots = task
        n_processed += 1

        print("%4d processed, current tag = %d with %d spot(s)" % (n_processed, tag, n_spots))

    [t.join() for t in workers]
    read_queue.close()
    result_queue.close()
    print("%d Hit / %d Processed." % (n_hit, n_processed))
