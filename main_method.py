import tensorflow as tf
import numpy as np
import random
from data_loader import IMDBDataSet
from trainer import TrainNetworks
from glove import LoadGloVe


#Set randomseeds


#Various relavant parameters shared accross the classes
#Data Loader
maxlen = 100
max_words = 10000
embedding_dim = 100
max_words = max_words #i.e. the parameters for glove and data loader should match


#Prep GPU: Does this code need to be else
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(\
      memory_limit=6000)])
logical = tf.config.experimental.list_logical_devices('GPU')
print(logical[0])

#Load and pratition data sets.
_ = IMDBDataSet()
tr, lbl, words = _.load_data()
tr_dt = tr[:20000]
tr_lbl = lbl[:20000]
val_dt = tr[20000:]
val_lbl = lbl[20000:]

_ = LoadGloVe(words)
weights = _.load_glove()


# Create, train and save models
mod_trainer = TrainNetworks(tr_dt, tr_lbl, val_dt, val_lbl, weights)
#biderectional_history, biderectional_model = mod_trainer.train(name='bidirectional')
basic_history, basic_model = mod_trainer.train(name='basic')
glove_basic_history, glove_basic_model = mod_trainer.train(name='glove_basic')
