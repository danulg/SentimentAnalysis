import tensorflow as tf
import numpy as np
import matplotlib

#Set randomseeds

#Prepare GPU:
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
logical = tf.config.experimental.list_logical_devices('GPU')
print(logical[0])

#None augmented data

#Augmented data

#GANs? If time permits!

#Unsupervised: Ideas include running through half trained and adding and taking out high threshold data!
embedding_dim = 100