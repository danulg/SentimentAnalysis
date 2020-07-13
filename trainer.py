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
#Note down times for training. Repeat on basic / advanced models

#Model 1: On the 25000 smaples - Basic

#Model 2: Transfer learning with Glove but no interaction with cluseteing or unsupervised set: all examples 25000  - Basic

#Model 3:  Transfer learning with Glove but no interaction with cluseteing or unsupervised set: with a fraction of the original data- Basic

#Model 4: Clustering with glove6B to add additional examples - Basic

#Model 5: Train in cycles to add / remove examples with high probability - Basic

#Model 6: On the 25000 smaples - Advanced

#Model 7: Transfer learning with Glove but no interaction with cluseteing or unsupervised set: all examples 25000  - Advanced

#Model 8:  Transfer learning with Glove but no interaction with cluseteing or unsupervised set: with a fraction - Advanced

#Model 9: Clustering with glove6B to add additional examples - Advanced

#Model 10: Train in cycles to add / remove examples with high probability - Advanced

############# IDEAL: 5, 4 > 2 > 3 > 1 AND 10, 9 > 7 > 8 > 6










#Possibly a fifth model that


