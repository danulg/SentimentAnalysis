from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, Embedding, Input, Bidirectional, Dropout, \
    MaxPooling1D, Concatenate

from tensorflow.keras.models import Sequential, Model


class AnalysisBasic(Sequential):
    def __init__(self, rate=0.5, max_words=20000, embedding_dim=100, max_len=200, dense_output_size=128):
        super().__init__()
        self.add(Embedding(max_words, embedding_dim, input_length=max_len))
        self.add(Dropout(rate))
        self.add(Flatten())
        self.add(Dense(dense_output_size, activation='relu'))
        self.add(Dropout(rate))
        self.add(Dense(1, activation='sigmoid'))


class AnalysisBidirectional(Sequential):  # Should have 1D CNN and dropout or batchnorm
    def __init__(self, max_words=20000, embedding_dim=100, rate=0.5, lstm_output_size=128, lstm_output_size2=128):
        super().__init__()
        # self.add(Input(shape=(None,), dtype="int32"))
        # Embed each integer in a 100-dimensional vector
        self.add(Embedding(max_words, embedding_dim))
        # Add 2 bidirectional LSTMs
        self.add(LSTM(lstm_output_size, return_sequences=True))
        self.add(Bidirectional(LSTM(lstm_output_size2)))
        # Add a classifier
        self.add(Flatten())
        self.add(Dense(256, activation='relu'))
        self.add(Dropout(rate))
        self.add(Dense(1, activation="sigmoid"))


class ConvolutionalLSTM(Model):
    def __init__(self, max_words=20000, embedding_dim=100, lstm_output_size=128, rate=.5, filters=32):
        super().__init__()
        self.embedding_layer = Embedding(max_words, embedding_dim)
        self.conv_by1 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='sigmoid')
        self.conv_by2 = Conv1D(filters=filters, kernel_size=2, strides=1, padding='same', activation='sigmoid')
        self.conv_by3 = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same', activation='sigmoid')
        self.conv_by4 = Conv1D(filters=filters, kernel_size=4, strides=1, padding='same', activation='sigmoid')
        self.conv_by5 = Conv1D(filters=filters, kernel_size=5, strides=1, padding='same', activation='sigmoid')
        self.max_pool = MaxPooling1D()
        self.dropout = Dropout(rate=rate)
        self.lstm_by1 = LSTM(lstm_output_size)
        self.conv_by2 = LSTM(lstm_output_size)
        self.conv_by3 = LSTM(lstm_output_size)
        self.conv_by4 = LSTM(lstm_output_size)
        self.conv_by5 = LSTM(lstm_output_size)
        self.concat = Concatenate()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)

        c1 = self.conv_by1(x)
        c2 = self.conv_by2(x)
        c3 = self.conv_by1(x)
        c4 = self.conv_by1(x)
        c5 = self.conv_by1(x)

        c1 = self.max_pool(c1)
        c2 = self.max_pool(c2)
        c3 = self.max_pool(c3)
        c4 = self.max_pool(c4)
        c5 = self.max_pool(c5)

        if training:
            c1 = self.dropout(c1)
            c2 = self.dropout(c2)
            c3 = self.dropout(c3)
            c4 = self.dropout(c4)
            c5 = self.dropout(c5)

        c1 = self.lstm_by1(c1)
        c2 = self.lstm_by1(c2)
        c3 = self.lstm_by1(c3)
        c4 = self.lstm_by1(c4)
        c5 = self.lstm_by1(c5)

        x = Concatenate([c1, c2, c3, c4, c5])

        x = self.dense1(x)
        return self.dense2(x)






