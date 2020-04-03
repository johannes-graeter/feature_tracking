#!/usr/bin/env python
# In[0]:
# Script for demonstrating feature_tracking
import numpy as np
import matcher_python_binding.tracker_libviso as t
import pykitti

# In[1]:
basedir = '/limo_data/dataset'
sequence = '04'

# In[2]:
# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.odometry(basedir, sequence)
dataset = pykitti.odometry(basedir, sequence, frames=range(0, 20, 2))

# In[3]:
tracker = t.TrackerLibViso()
for image in dataset.cam0:
    t.push_back(tracker, np.expand_dims(np.asarray(image), axis=-1))

# In[4]:
tracklets = t.get_tracklets(tracker)
print("Number of tracklets={}".format(len(tracklets)))
