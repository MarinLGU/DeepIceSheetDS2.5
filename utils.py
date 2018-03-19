"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import h5py

import numpy as np

import tensorflow as tf
import scipy.ndimage as sci


xrange = range

FLAGS = tf.app.flags.FLAGS


def read_data(path):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))

    data, label=data[:,:,:,0:2], label[:,:,:,0]
    #data=np.expand_dims(data,axis=-1)
    label=np.expand_dims(label,axis=-1)

    return data, label

def make_label(label, step, scale_factor):
    s1=label.shape
    r=sci.zoom(label,(1,(step+1)*2/scale_factor,(step+1)*2/scale_factor,1))
    s2=r.shape
    r=sci.zoom(r,(1,s1[1]/s2[1],s1[2]/s2[2],1))
    return r

def make_topg(data, step, scale_factor):
    s1=data.shape
    r=data[:,:,:,1]
    r=sci.zoom(r,(1,(step+1)*2/scale_factor,(step+1)*2/scale_factor))
    s2 = r.shape
    r = sci.zoom(r,(1,s1[1]/s2[1],s1[2]/s2[2]))
    return r









def make_data(sess, data, label):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if FLAGS.is_train:
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)






