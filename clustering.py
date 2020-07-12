import numpy as np
from data_loader import IMDBDataSet
from glove import LoadGloVe
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten
import sklearn

class Cluster():
    def __init__(self):
        super().__init__()

class Encoder(Sequential):
    def __init__(self, embedding_dim=100, max_words=100, maxlen=10000):
        super().__init__()
        self.add(Embedding(max_words, embedding_dim, input_length=maxlen))
        self.add(Flatten())


if __name__=="__main__":
    data = IMDBDataSet()
    tr_dt, tr_lbl, word_index = data.load_data(name='train', is_numpy=True)
    glove_weights =
    train_data = tr_dt[:20000]
    train_label = tr_lbl[:20000]
    val_data = tr_dt[20000:]
    val_label = tr_dt[20000:]



