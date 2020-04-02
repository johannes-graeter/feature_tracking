#!/usr/bin/env python
# In[0]:
# Script for demonstrating feature_tracking
import numpy as np
import matcher_python_binding.tracker_libviso as t
import pykitti

basedir = '/limo_data'
sequence = '04'

# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.odometry(basedir, sequence)
dataset = pykitti.odometry(basedir, sequence, frames=range(0, 20, 2))

tracker = t.Tracker()
for image in dataset.cam0:
    t.push_back(tracker, image)
    tracklets = []
    tracker.get_tracklets(tracklets, 0)
    # plot

