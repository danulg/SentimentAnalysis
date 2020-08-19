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
from visulaizations import PlotCurves
import dill

# Prep GPU:
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
logical = tf.config.experimental.list_logical_devices('GPU')
print(logical[0])

# Default word form. Changing these will require the creation of new tokens and tokenizers with new_tokens = True.  Can
# be passed onto TrainNetworks.train()
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
epochs = 20
rate = 0.5

# Load and partition data sets.
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
# history = mod_trainer.train(name='basic', epochs=20)
# glove_history = mod_trainer.train(name='glove_basic', epochs=20)
# w2vec_history = mod_trainer.train(name='w2vec_basic', epochs=20)
# history = [history, glove_history, w2vec_history]
# dill.dump(history, open('history_basic.pkd', 'wb'))


# Train neural network architecture: bidirectional.
# history = mod_trainer.train(name='bidirectional', epochs=20)
# glove_history = mod_trainer.train(name='glove_bidirectional', epochs=20)
# w2vec_history = mod_trainer.train(name='w2vec_bidirectional', epochs=20)
# history = [history, glove_history, w2vec_history]
# dill.dump(history, open('history_bidirectional.pkd', 'wb'))

# Train neural network architecture: ConvLSTM.
# history = mod_trainer.train(name='conv_lstm', epochs=20, filters=filters, lstm_output_size=lstm_output_size)
# glove_history = mod_trainer.train(name='glove_conv_lstm', epochs=20, filters=filters, lstm_output_size=lstm_output_size)
# w2vec_history = mod_trainer.train(name='w2vec_conv_lstm', epochs=20, filters=filters, lstm_output_size=lstm_output_size)
# history = [history, glove_history, w2vec_history]
# dill.dump(history, open('history_conv_lstm.pkd', 'wb'))

# Further train neural network architecture: ConvLSTM
# glove_history = mod_trainer.train(name='glove_conv_lstm', epochs=100, filters=filters, lstm_output_size=lstm_output_size)
# w2vec_history = mod_trainer.train(name='w2vec_conv_lstm', epochs=100, filters=filters, lstm_output_size=lstm_output_size)
# history = [glove_history, w2vec_history]
# dill.dump(history, open('history_conv_lstm_res_100.pkd', 'wb'))

w2vec_history = mod_trainer.train(name='w2vec_conv_lstm', epochs=100, filters=filters, lstm_output_size=lstm_output_size,
                                  record=True)


# Train neural network architecture: Conv. Performs extremely poorly
# history = mod_trainer.train(name='conv', epochs=20, filters=filters)
# glove_history = mod_trainer.train(name='glove_conv', epochs=20, filters=filters)
# w2vec_history = mod_trainer.train(name='w2vec_conv', epochs=20, filters=filters)
# history = [history, glove_history, w2vec_history]
# dill.dump(history, open('history_conv.pkd', 'wb'))


# Draw plots
# curves = PlotCurves()
# curves.draw_comparison(dill.load(open('history_basic.pkd', 'rb')), epochs=epochs, name='basic')
# curves.draw_comparison(dill.load(open('history_bidirectional.pkd', 'rb')), epochs=epochs, name='bidirectional')
# curves.draw_comparison(dill.load(open('history_conv_lstm.pkd', 'rb')), epochs=epochs, name='conv_lstm')
# curves.draw_two(dill.load(open('history_conv_lstm_res_100.pkd', 'rb')), epochs=100, name='conv_lstm_100')
# curves.draw_comparison(dill.load(open('history_conv.pkd', 'rb')), epochs=epochs, name='conv')
