from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, Embedding, Bidirectional, Dropout, \
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
        self.max_pool1 = MaxPooling1D()
        self.max_pool2 = MaxPooling1D()
        self.max_pool3 = MaxPooling1D()
        self.max_pool4 = MaxPooling1D()
        self.max_pool5 = MaxPooling1D()
        self.dropout1 = Dropout(rate=rate)
        self.dropout2 = Dropout(rate=rate)
        self.dropout3 = Dropout(rate=rate)
        self.dropout4 = Dropout(rate=rate)
        self.dropout5 = Dropout(rate=rate)
        self.lstm_by1 = LSTM(lstm_output_size)
        self.lstm_by2 = LSTM(lstm_output_size)
        self.lstm_by3 = LSTM(lstm_output_size)
        self.lstm_by4 = LSTM(lstm_output_size)
        self.lstm_by5 = LSTM(lstm_output_size)
        self.concat = Concatenate()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        x = self.embedding_layer(inputs)

        c1 = self.conv_by1(x)
        c2 = self.conv_by2(x)
        c3 = self.conv_by3(x)
        c4 = self.conv_by4(x)
        c5 = self.conv_by5(x)

        c1 = self.max_pool1(c1)
        c2 = self.max_pool2(c2)
        c3 = self.max_pool3(c3)
        c4 = self.max_pool4(c4)
        c5 = self.max_pool5(c5)

        c1 = self.dropout1(c1, training=training)
        c2 = self.dropout2(c2, training=training)
        c3 = self.dropout3(c3, training=training)
        c4 = self.dropout4(c4, training=training)
        c5 = self.dropout5(c5, training=training)

        c1 = self.lstm_by1(c1)
        c2 = self.lstm_by2(c2)
        c3 = self.lstm_by3(c3)
        c4 = self.lstm_by4(c4)
        c5 = self.lstm_by5(c5)

        x = self.concat([c1, c2, c3, c4, c5])

        x = self.dense1(x)
        return self.dense2(x)

    def get_model(self):
        pass

class Convolutional(Model):
    def __init__(self, max_words=20000, embedding_dim=100, rate=.5, filters=32):
        super().__init__()
        self.embedding_layer = Embedding(max_words, embedding_dim)
        self.conv_by1 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='sigmoid')
        self.conv_by2 = Conv1D(filters=filters, kernel_size=2, strides=1, padding='same', activation='sigmoid')
        self.conv_by3 = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same', activation='sigmoid')
        self.conv_by4 = Conv1D(filters=filters, kernel_size=4, strides=1, padding='same', activation='sigmoid')
        self.conv_by5 = Conv1D(filters=filters, kernel_size=5, strides=1, padding='same', activation='sigmoid')
        self.max_pool1 = MaxPooling1D()
        self.max_pool2 = MaxPooling1D()
        self.max_pool3 = MaxPooling1D()
        self.max_pool4 = MaxPooling1D()
        self.max_pool5 = MaxPooling1D()
        self.dropout1 = Dropout(rate=rate)
        self.dropout2 = Dropout(rate=rate)
        self.dropout3 = Dropout(rate=rate)
        self.dropout4 = Dropout(rate=rate)
        self.dropout5 = Dropout(rate=rate)
        self.concat = Concatenate()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        x = self.embedding_layer(inputs)

        c1 = self.conv_by1(x)
        c2 = self.conv_by2(x)
        c3 = self.conv_by3(x)
        c4 = self.conv_by4(x)
        c5 = self.conv_by5(x)

        c1 = self.max_pool1(c1)
        c2 = self.max_pool2(c2)
        c3 = self.max_pool3(c3)
        c4 = self.max_pool4(c4)
        c5 = self.max_pool5(c5)

        c1 = self.dropout1(c1, training=training)
        c2 = self.dropout2(c2, training=training)
        c3 = self.dropout3(c3, training=training)
        c4 = self.dropout4(c4, training=training)
        c5 = self.dropout5(c5, training=training)

        x = self.concat([c1, c2, c3, c4, c5])

        x = self.dense1(x)
        return self.dense2(x)

    def get_model(self):
        pass





