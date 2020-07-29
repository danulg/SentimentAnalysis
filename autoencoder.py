from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Reshape, Flatten, TimeDistributed
from tensorflow.keras.models import Sequential
from dataloader import IMDBDataSet
import tensorflow as tf

class AutoEncoder(Sequential):
    def __init__(self):
        super().__init__()
        self.max_words = 20000
        self.embedding_dim = 100
        self.max_len = 200
        self.lstm_output_size = 100
        self.inputs = self.add(Input(shape=(None,), dtype="int32"))
        self.embed = self.add(Embedding(self.max_words, self.embedding_dim))
        self.lstm1 = self.add(LSTM(self.lstm_output_size))
        self.lstm2 = self.add(LSTM(self.max_len, activation='relu', return_sequences=True))
        self.dense1 = self.add(TimeDistributed(Dense(1)))




if __name__ == '__main__':
    imdb = IMDBDataSet()
    text, *_ = imdb.load_data(name='unsup')

    # Prep GPU:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=6000)])
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(\
    #       memory_limit=3000)])
    logical = tf.config.experimental.list_logical_devices('GPU')
    print(logical[0])

    encoder = AutoEncoder()
    encoder.compile(optimizer='adam', loss='mse')
    encoder.summary()