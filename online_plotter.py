# Ref:
# http://stackoverflow.com/questions/21579658/embedding-a-live-updating-matplotlib-graph-in-wxpython
# http://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib

import matplotlib.figure as mfigure
import matplotlib.animation as manim
import numpy as np
import optparse
import os
import re
import sys
import time
import wx

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg

class RingBuffer():
    def __init__(self, nmax):
        self.nmax = nmax
        self.buf = np.zeros(self.nmax)
        self.n = 0

    def append(self, x):
        self.buf[self.n % self.nmax] = x
        self.n += 1

    def clear(self):
        self.n = 0
        self.buf.fill(0)

    def get(self):
        return self.buf[:min(self.n, self.nmax)]

    def __len__(self):
        return min(self.n, self.nmax)

class PlotWindow(wx.Frame):
    NUM_DISPLAY = 600

    def __init__(self, parent, opts):
        super(PlotWindow,self).__init__(None, wx.ID_ANY, size=(1024, 550), title="Hitrate Plotter")

        self.hit_window = opts.window
        self.hit_threshold = opts.threshold
        self.fixed_filename = opts.filename
        self.filename = self.fixed_filename

        self.fig = mfigure.Figure()
        self.fig.subplots_adjust(left=0.07, bottom=0.15, right=0.93, top=0.95, wspace=0.15, hspace=0.15)
        self.ax = self.fig.add_subplot(111)
        self.ax_hit = self.ax.twinx()
        self.canvas = FigureCanvasWxAgg(self, wx.ID_ANY, self.fig)

        # Data
        self.peaks = RingBuffer(self.NUM_DISPLAY)
        self.framenumber = RingBuffer(self.NUM_DISPLAY)

        # Parser internals
        self.frame_read_bytes = 0
        self.peak_read_bytes = 0
        self.peak_cur_tag = -1

        self.ax.set_ylim([0,300])
        self.ax_hit.set_ylim([0, 100])
        self.ax.set_xlabel("Seconds before the latest data")
        self.ax.set_ylabel("Number of peaks")
        self.ax_hit.set_ylabel("Hit rate (%)")
        self.ax.get_xaxis().get_major_formatter().set_useOffset(False)
        self.ax.get_xaxis().get_major_formatter().set_scientific(False) # no exponential notation

        self.peakdots, = self.ax.plot(0, 0, color="r", marker="o", linestyle="None", markersize=3, markeredgewidth=0)
        self.hitline, = self.ax_hit.plot(0, 0, 'b-')
        self.animator = manim.FuncAnimation(self.fig, self.anim, interval=3000) # 3 sec. cf. saveInterval in cheetah

        self.vsizer = wx.BoxSizer(wx.VERTICAL)
        self.vsizer.Add(self.canvas, 0, wx.EXPAND | wx.RIGHT | wx.TOP)
        self.SetSizer(self.vsizer)

        self.Show()

    def getLatest(self):
        import os
        import glob

        frames = glob.glob("spotcount-from-*.log")
        frames.sort(key=os.path.getmtime, reverse=True)
        return frames[0]

    def anim(self, i):
        if self.fixed_filename is None:
            new_filename = self.getLatest()
            if self.filename != new_filename:
                self.frame_read_bytes = 0
                self.peak_read_bytes = 0
                self.filename = new_filename
                
        self.SetTitle(self.filename)

        self.parse_frames(self.filename)
        if len(self.peaks) == 0:
            return
        peaks = self.peaks.get()
        framenumber = self.framenumber.get()
        order = framenumber.argsort()
        peaks = peaks[order]
        framenumber = framenumber[order]
        frametime = [(x - framenumber[-1]) * (1.0 / 60) for x in framenumber] # ctag is at 60 Hz

        self.ax.set_ylim([0, np.max(peaks)])
        self.ax.set_xlim([frametime[0], frametime[-1]])

        self.peakdots.set_xdata(frametime)
        self.peakdots.set_ydata(peaks)

        hitrate = np.convolve(100 * (peaks[0:len(peaks)] > self.hit_threshold), 
                              np.ones(self.hit_window) / self.hit_window, mode="valid")
        self.hitline.set_xdata(frametime[0:len(hitrate)])
        self.hitline.set_ydata(hitrate)

        self.fig.canvas.draw()
 
    def parse_frames(self, filename):
        try:
            size = os.path.getsize(filename)
        except:
            size = 0
        # This is necessary to limit latency on some file systems.

        cnt = 0
        with open(filename, "r") as f:
            f.seek(self.frame_read_bytes)

            for line in f:
                cnt += 1
                self.frame_read_bytes += len(line)
                
                if line[0] == "#":
                    continue
                columns = line.split(" ")
                if len(columns) != 2:
                    continue

                try:
                    framenumber, peaks = int(columns[0]), int(columns[1])
                    self.framenumber.append(framenumber)
                    self.peaks.append(peaks)
                except:
                    print("Parse error in %s line: %s" % (filename, line))

                if self.frame_read_bytes > size:
                    break

print()
print("Cheetah Online plotter version 2026/04/20")
print("   by Takanori Nakane")
print()

parser = optparse.OptionParser()
parser.add_option("--window", dest="window", type=int, default=30, help="window size for moving average")
parser.add_option("--threshold", dest="threshold", type=int, default=20, help="minimum number of peaks for hits")
parser.add_option("--filename", dest="filename", type=str, default=None, help="filename to display (set None to follow current directory)")
#parser.add_option("--saturation", dest="saturation", type=int, default=15000, help="saturation threshold")
opts, args = parser.parse_args()

print("Option: window = %d" % opts.window)
print("Option: threshold = %d" % opts.threshold)
print("Option: filename = %s" % opts.filename)
#print("Option: saturation = %d" % opts.saturation)

app = wx.App(False)
frame = PlotWindow(None, opts)
app.MainLoop()
