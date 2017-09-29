"""
runs pong_ae*.py code
"""

from __future__ import division, print_function, absolute_import

import datetime
import random
import tensorflow as tf
import numpy as np
import bitarray

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from PIL import Image

import pong_ae_batchnorm as ba
import pong_ae_dropout as dr
import pong_ae_dropout_batchnorm as dr_ba
import pong_ae as p

encodings_1 = ( ((3,[3,3]),([2,2],2)), ((3,[3,3]),([2,2],2)) )
encodings_2 = ( ((3,[5,5]),([2,2],2)), ((3,[5,5]),([2,2],2)) )
encodings_3 = ( ((6,[7,7]),([4,4],4)), )

amt_tr_data = [100, 250, 500, 1000, 2000, 4000, 8000]
num_epochs = [100, 95, 90, 85, 80, 70, 60]

"""
Format of command to run autoencoder on pong-v0 data:

ae.start(amount of training data, file number, number of epochs, 
  neural network encoding, encoding number/name, batch size, learning rate)
"""

for i in range(7):
	ae = ba.train_autoencoder()
	ae.start(amt_tr_data[i], 1, num_epochs[i], encodings_1, 1, 50, 0.001)

for i in range(7):
	ae = dr_ba.train_autoencoder()
	ae.start(amt_tr_data[i], 1, num_epochs[i], encodings_2, 2, 50, 0.001)
