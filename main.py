# import and set randomseeds
import os
os.environ['PYTHONHASHSEED'] = str(12)

import tensorflow as tf
tf.random.set_seed(324)

import random
random.seed(14)

import numpy as np
np.random.seed(15)

# import rest of relavant libraries
from dataloader import IMDBDataSet
from trainer import TrainNetworks
from wordembedding import GloVe
from compmod import PlotCurves
import dill

# Prep GPU:
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(\
#       memory_limit=3000)])
logical = tf.config.experimental.list_logical_devices('GPU')
print(logical[0])

# Various relavant parameters across the classes
# Shared by all classes Data Loader, Glove, Network Architectures
max_len = 200
max_words = 20000
embedding_dim = 100

# Specific to Convolutional archtecture of SingleConv1D
filters = 64
kernel_size = 5
strides = 2
pool_size = 4

# Parameters for training

epochs = 25
cutoff = 0.85

# Model parameters with default sizes
dense_output_size = 32
lstm_output_size = 100
lstm_output_size2 = 100
rate = 0.4

# Load and pratition data sets.
temp = IMDBDataSet()
tr, lbl, words = temp.load_data_default(name='train')
tr_dt = tr[:20000]
tr_lbl = lbl[:20000]
val_dt = tr[20000:]
val_lbl = lbl[20000:]

temp = GloVe(words)
weights = temp.load_glove(max_words=max_words)

# # Create model trainer
mod_trainer = TrainNetworks(tr_dt, tr_lbl, val_dt, val_lbl, weights)

# Train neural network architecture: Basic
# history, model = mod_trainer.train(name='basic', epochs=epochs, rate=rate,
#                                    dense_output_size=dense_output_size, cutoff=cutoff)
# glove_history, glove_model = mod_trainer.train(name='glove_basic', epochs=epochs, dense_output_size=dense_output_size,
#                                                rate=rate, cutoff=cutoff)




# Train neural network architecture: bidirectional. Rate is currently redundant as it has no dropout
history, model = mod_trainer.train(name='bidirectional', epochs=epochs, rate=rate)
glove_history, glove_model = mod_trainer.train(name='glove_bidirectional', epochs=epochs, rate=rate)
history = [history, glove_history]
dill.dump(history, open('history_10.pkd', 'wb'))


# Gather histories and save
# history = [history, glove_history, iter_history]
# dill.dump(history, open('history_2.pkd', 'wb'))

# Draw plots
# curves = PlotCurves()
# curves.draw(dill.load(open('history_1.pkd', 'rb')), sub_epochs=sub_epochs, iterates=iterates)
# curves.bokeh_draw(dill.load(open('history_2.pkd', 'rb')), sub_epochs=sub_epochs, iterates=iterates)
