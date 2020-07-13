import numpy as np
from data_loader import IMDBDataSet
from glove import LoadGloVe
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten
import tensorflow as tf
import sklearn

class Cluster():
    def __init__(self):
        super().__init__()

class Encoder(Sequential):
    def __init__(self, max_words=10000, embedding_dim=100, maxlen=100):
        super().__init__()
        self.add(Embedding(max_words, embedding_dim, input_length=maxlen))
        self.add(Flatten())


if __name__=="__main__":
    data = IMDBDataSet()
    tr_dt, tr_lbl, word_index = data.load_data(name='train', is_numpy=True)
    glove = LoadGloVe(word_index)
    weights = glove.load_glove()
    print(weights.shape)

    # Prepare GPU:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    tf.config.experimental.set_virtual_device_configuration(gpus[0],\
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
    logical = tf.config.experimental.list_logical_devices('GPU')
    print(logical[0])

    encoder = Encoder()
    encoder.layers[0].set_weights([weights])

    #Data Split?
    train_data = tr_dt[:20000]
    train_label = tr_lbl[:20000]
    val_data = tr_dt[20000:]
    val_label = tr_dt[20000:]

    predictions = encoder.predict(train_data)
    print(predictions.shape)

