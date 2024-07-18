# Ostrich for SACLA SFX data proprocessing
# written by Takanori Nakane at Osaka University

from multiprocessing import Process, Queue
import numpy as np

import stpy

def add_image_par(read_queue, result_queue, detector, adu_per_photon):
    detector.allocate_readers()

    xsize = detector.geometry.width
    ysize = detector.geometry.height
    npanels = len(detector.geometry.panels)
    gains = [panel.gain for panel in detector.geometry.panels]

    local_buffer = np.zeros((ysize * npanels, xsize), dtype=np.float32)
    local_n_added = 0

    while True:
        task = read_queue.get()
        if task is None:
            result_queue.put((local_buffer, local_n_added))
            break
        else:
            tag, energy = task
            try:
                for i in range(npanels):
                    detector.readers[i].collect(detector.buffers[i], tag)
                    data = detector.buffers[i].read_det_data(0)
                    data *= gains[i] * 3.65 * adu_per_photon / energy
                    local_buffer[(ysize * i):(ysize * (i + 1)),] += data
            except Exception as e:
                print(e)
                continue # FIXME: report error
            local_n_added += 1

def average_images(detector, tags, photon_energies, adu_per_photon, nproc=8):
    xsize = detector.geometry.width
    ysize = detector.geometry.height
    npanels = len(detector.geometry.panels)
    gains = [panel.gain for panel in detector.geometry.panels]

    n_added = 0
    sum_buffer = np.zeros((ysize * npanels, xsize), dtype=np.float32)

    if nproc == 1:
        detector.allocate_readers()
        def add_image(tag, energy):
            for i in range(npanels):
                try:
                    detector.readers[i].collect(detector.buffers[i], tag)
                except:
                    raise RuntimeError("FailedOn_collect_data")

                data = detector.buffers[i].read_det_data(0)
                data *= gains[i] * 3.65 * adu_per_photon / energy
                sum_buffer[(ysize * i):(ysize * (i + 1)),] += data

            return 1

        for idx, tag in enumerate(tags):
            print("Processing tag %d (%2.1f%% done)" % (tag, 100.0 * (idx + 1) / len(tags)))
            if (idx % 5 == 0):
                with open("status.txt", "w") as status:
                    status.write("Status: Total=%d,Processed=%d,Status=DarkAveraging\n" % (len(tags), idx + 1))
            n_added += add_image(tag, photon_energies[idx])
    else:
        read_queue = Queue()
        result_queue = Queue()
        workers = []
        detector.deallocate_readers()
        for i in range(nproc):
            p = Process(target=add_image_par, args=(read_queue, result_queue, detector, adu_per_photon))
            p.start()
            workers.append(p)
        
        for tag, energy in zip(tags, photon_energies):
            read_queue.put([tag, energy])
        for i in range(nproc):
            read_queue.put(None)

        n_finished = 0
        while n_finished < nproc:
            local_buffer, local_n_added = result_queue.get()
            print(local_n_added)
            sum_buffer += local_buffer
            n_added += local_n_added
            n_finished += 1

        [t.join() for t in workers]
        read_queue.close()
        result_queue.close()

    if (n_added < 1):
       raise RuntimeError("NoImageToAverage")

    sum_buffer /= n_added

    # In the Phase 3 detector, some pixels average to negative values.
    # Most are around -0.1 and all those below -1 are at panel edges that will be masked.
    # So we don't have to worry about them.
    ushort_max = np.iinfo(np.uint16).max
    print("\nCalibration image statistics: averaged %d images, #neg (< 0) %d, #overflow (> %d) %d" %
            (n_added, np.sum(sum_buffer < 0), ushort_max, np.sum(sum_buffer > ushort_max)))
    sum_buffer.clip(0, ushort_max, out=sum_buffer)
    averaged = sum_buffer.astype(np.uint16)

    return averaged
