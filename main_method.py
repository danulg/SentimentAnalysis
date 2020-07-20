#import and set randomseeds
import os
os.environ['PYTHONHASHSEED']=str(12)

import tensorflow as tf
tf.random.set_seed(324)

import random
random.seed(14)

import numpy as np
import numpy as np
np.random.seed(15)


from data_loader import IMDBDataSet
from trainer import TrainNetworks
from glove import LoadGloVe
from compare_models import PlotCurves
import dill

#Various relavant parameters across the classes
#Shared by all classes Data Loader, Glove, Network Architectures
maxlen = 100
max_words = 10000
embedding_dim = 100
max_words = max_words #i.e. the parameters for glove and data loader should match

#Specific to Convolutional archtecture of SingleConv1D
filters = 64
kernel_size = 5
strides = 2
pool_size = 4
lstm_output_size = 4

#Prep GPU: Does this code need to be else
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(\
     memory_limit=6000)])
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(\
#       memory_limit=3000)])
logical = tf.config.experimental.list_logical_devices('GPU')
print(logical[0])

#Load and pratition data sets.
# temp = IMDBDataSet()
# tr, lbl, words = temp.load_data()
# tr_dt = tr[:20000]
# tr_lbl = lbl[:20000]
# val_dt = tr[20000:]
# val_lbl = lbl[20000:]
# unsup, *_ = temp.load_data(name='unsup')
#
# temp = LoadGloVe(words)
# weights = temp.load_glove()



# Parameters for training
epochs = 12
rate = 0.6
iterates = 3
sub_epochs = 4

# Create, train and save models
# mod_trainer = TrainNetworks(tr_dt, tr_lbl, val_dt, val_lbl, unsup, weights)
# assert sub_epochs*iterates == epochs
# basic_history, basic_model = mod_trainer.train(name='basic', epochs=epochs, rate=rate)
# glove_basic_history, glove_basic_model = mod_trainer.train(name='glove_basic', epochs=epochs, rate=rate)
# iter_history, basic_iter_model = mod_trainer.train_unlabled(name='basic', sub_epochs=sub_epochs, iterates=iterates, rate=rate)
# # biderectional_history, biderectional_model = mod_trainer.train(name='bidirectional', epochs=epochs)


# Gather histories and save
# history = [basic_history, glove_basic_history, iter_history]
# dill.dump(history, open('history_1.pkd', 'wb'))

#Draw plots
curves = PlotCurves()
curves.draw(dill.load(open('history_1.pkd', 'rb')), sub_epochs=sub_epochs, iterates=iterates)