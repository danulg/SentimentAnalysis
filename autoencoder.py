from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Reshape, Flatten, TimeDistributed, RepeatVector
from tensorflow.keras.models import Sequential
from dataloader import IMDBDataSet
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from wordembedding import Word2VecWeights


class Embedder(Sequential):
    def __init__(self):
        super().__init__()
        self.max_words = 20000
        self.embedding_dim = 100
        self.embed = self.add(Embedding(self.max_words, self.embedding_dim))


class AutoEncoder(Sequential):
    def __init__(self):
        super().__init__()
        self.max_words = 20000
        self.embedding_dim = 100
        self.max_len = 200
        self.lstm_output_size = 512
        self.lstm1 = self.add(
            LSTM(self.lstm_output_size, input_shape=(self.max_len, self.embedding_dim), name='lstm1'))
        # Copies output into higher dimension
        self.rpt = self.add(RepeatVector(self.max_len))
        self.lstm2 = self.add(LSTM(self.embedding_dim, return_sequences=True, name='lstm2'))
        self.dense1 = self.add(TimeDistributed(Dense(1, name='dense1')))

        # The input of the LSTM is always is a 3D array of (batch_size, time_steps, seq_len).
        # The output of the LSTM could be a 2D array or 3D array depending upon the return_sequences argument.
        # If return_sequence is False, the output is a 2D array of (batch_size, units)
        # If return_sequence is True, the output is a 3D array of (batch_size, time_steps, units)

        # TimeDistributed layer applies a specific layer such as Dense to every sample it receives as an input.Suppose
        # the input size is (13, 10, 6).Now, if I need to apply a Dense layer to every slice of shape(10, 6).Then I
        # would wrap the Dense layer in a TimeDistributed layer.
        #
        # model.add(TimeDistributed(Dense(12, input_shape=(10, 6))))
        #
        # The output shape of such a layer would be(13, 10, 12). Hence, the operation of the Dense layer was applied to
        # each temporal slice as mentioned.


def data_gen(seq, embedding):
    i = 0
    while i < len(seq):
        yield (embedding.predict(seq_text[i]), embedding.predict(seq_text[i]))
        i += 1


if __name__ == '__main__':
    # Prep GPU:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=6000)])
    logical = tf.config.experimental.list_logical_devices('GPU')
    print(logical[0])

    # Load data
    imdb = IMDBDataSet()
    seq_text, *_ = imdb.load_data_default(name='unsup')

    # Create embedding layer
    embedding_layer = Embedder()
    temp = Word2VecWeights()
    weights = temp.load_word2vec_weights(max_words=20000)
    embedding_layer.layers[0].set_weights([weights])



    # Create a pipeline for data. This is required to manage resources.

    # train_ds = data_gen(seq_text, embedding_layer)
    # for x in train_ds:
    #     print(x)




    # print(train_ds)

    # encoder = AutoEncoder()
    # encoder.compile(optimizer='adam', loss='mse')
    # encoder.summary()
    # checkpoint_filepath = './checkpoints/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5'
    # model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
    #                                             save_weights_only=True, monitor='loss',
    #                                             mode='min', save_best_only=True)
    #
    # encoder.fit(train_ds.batch(32).take(1), epochs=100, batch_size=1, verbose=1,
    #             callbacks=[model_checkpoint_callback])
    # encoder.save_weights('at_200.hd5')
