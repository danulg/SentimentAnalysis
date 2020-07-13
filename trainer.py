import tensorflow as tf
import numpy as np
import matplotlib

import bokeh
#Set randomseeds

#Prepare GPU:
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],\
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
logical = tf.config.experimental.list_logical_devices('GPU')
print(logical[0])

#Various relavant parameters
#Data Loader
maxlen = 100
max_words = 10000

#glove vectors
embedding_dim = 100
max_words = max_words #i.e. the parameters for glove and data loader should match
#Augmented data



#Unsupervised: Ideas include running through half trained and adding and taking out high threshold data!
#Model 1: On the 25000 smaples

#Model 2: Clustering with glove6B to add additional models

#Model 3: Transfer learning with Glove but no interaction with GloVe: Compare 25000 with 2000?

#Model 4: Train in cycles to add / remove examples with high probability

#Draw images using bokeh on train / test accuracies