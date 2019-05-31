#!/usr/bin/env python
# coding: utf-8

# ## VISUALIZE THE PRE-PROCESSED DATA

# In[1]:


import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import imutils
import os
import itertools

np.random.seed(7)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


# In[10]:


# destination folder of preprocessed samples
NPY_FOLDER = os.path.join("NPY", "npy_28")


# In[6]:


def figure_plot(figures, nrows=1, ncols=1, title=""):
    """Plot a dictionary of figures."""

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    fig.suptitle(title, fontsize="x-large")

    for ind, img in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(img, cmap=plt.jet())
        axeslist.ravel()[ind].set_axis_off()

    plt.savefig("outputs/" + title + '.png', dpi=300)
    plt.show()


# In[3]:


def view_sample(sample, word="", set_type="", sample_name="", gray=True, rows=5, cols=5):
    if gray:
        figures = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in sample]
    else:
        figures = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in sample]

    if set_type == "":
        title = sample_name
    else:
        title = ", ".join([word, set_type, sample_name])
    figure_plot(figures, rows, cols, title)


# In[7]:


def view_stored(word, set_type, sample_name):

    sample = np.load(os.path.join(NPY_FOLDER, set_type, word, sample_name))
    view_sample(sample, set_type, word, sample_name)


# In[ ]:


# plot samples of numpy array

view_stored("ABOUT", "train", "55.npy")
#view_stored("ABOUT", "val", "0.npy")



# In[ ]:
