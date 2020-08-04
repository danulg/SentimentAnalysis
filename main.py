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
from wordembedding import GloVe, Word2VecWeights
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

# Default word form. Changing these will require the creation of new tokens and tokenizers with new_tokens = True
max_len = 200
max_words = 20000
embedding_dim = 100

# Keyword arguments specific to various architectures. Can be changed by providing the parameters to the
# TrainNetworks.train()
filters = 32
strides = 1
pool_size = 2
dense_output_size = 128
lstm_output_size = 128
lstm_output_size2 = 128

# Parameters for training
epochs = 10
rate = 0.5

# Load and pratition data sets.
temp = IMDBDataSet()
tr, lbl, words = temp.load_data_default(name='train')
tr_dt = tr[:20000]
tr_lbl = lbl[:20000]
val_dt = tr[20000:]
val_lbl = lbl[20000:]

temp = GloVe(words)
glove_weights = temp.load_glove_weights(max_words=max_words, twitter=True)

temp = Word2VecWeights()
w2vec_weights = temp.load_word2vec_weights(max_words=max_words)

# # Create model trainer
mod_trainer = TrainNetworks(tr_dt, tr_lbl, val_dt, val_lbl, glove_weights, w2vec_weights)

# The following code trains the models with their default architectures
# Train neural network architecture: Basic
history = mod_trainer.train(name='basic')
glove_history = mod_trainer.train(name='glove_basic')
w2vec_history = mod_trainer.train(name='w2vec_basic', epochs=20)
history = [history, glove_history, w2vec_history]
dill.dump(history, open('history_basic.pkd', 'wb'))


# Train neural network architecture: bidirectional. Rate is currently redundant as it has no dropout
history = mod_trainer.train(name='bidirectional', epochs=20)
glove_history = mod_trainer.train(name='glove_bidirectional', epochs=20)
w2vec_history = mod_trainer.train(name='w2vec_bidirectional', epochs=20)
history = [history, glove_history, w2vec_history]
dill.dump(history, open('history_bidirectional.pkd', 'wb'))


# Draw plots
curves = PlotCurves()
curves.bokeh_draw(dill.load(open('history_basic.pkd', 'rb')), epochs=epochs, name='basic')
curves.bokeh_draw(dill.load(open('history_bidirectional.pkd', 'rb')), epochs=epochs, name='basic')
