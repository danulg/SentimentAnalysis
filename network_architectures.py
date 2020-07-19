from tensorflow.keras.layers import LSTM, Dense, Conv1D, Softmax, Flatten, Embedding, Input, Bidirectional, Dropout, MaxPooling1D,\
    GlobalMaxPooling1D
from tensorflow.keras.models import Sequential

class SentimentAnalysisBasic(Sequential):
    def __init__(self, rate=0.5, max_words=10000, embedding_dim=100, maxlen=100):
        super().__init__()
        self.add(Embedding(max_words, embedding_dim, input_length=maxlen))
        self.add(Dropout(rate))
        self.add(Flatten())
        self.add(Dense(32, activation='relu'))
        self.add(Dropout(rate))
        self.add(Dense(1, activation='sigmoid'))

class SentimentAnalysisBidirectional(Sequential): #Should have 1D CNN and dropout or batchnorm
    def __init__(self, max_words=10000, embedding_dim=100, maxlen=100):
        super().__init__()
        self.add(Input(shape=(None,), dtype="int32"))
        # Embed each integer in a 128-dimensional vector
        self.add(Embedding(max_words, embedding_dim))
        # Add 2 bidirectional LSTMs
        self.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.add(Bidirectional(LSTM(64)))
        # Add a classifier
        self.add(Dense(1, activation="sigmoid"))

class SentimentAnalysisSingleConv1D(Sequential):
    def __init__(self, max_words=10000, embedding_dim=100, maxlen=100, filters=64, kernel_size=5, strides=1, pool_size=4, lstm_output_size=4):
        super().__init__()
        self.add(Embedding(max_words, embedding_dim, input_length=maxlen))
        self.add(Dropout(0.25))
        self.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=strides))
        self.add(MaxPooling1D(pool_size=pool_size))
        self.add(LSTM(lstm_output_size))
        self.add(Dense(1, activation ='sigmoid'))

class SentimentAnalysisMultipleConv1D(Sequential):
    def __init__(self, max_words=10000, embedding_dim=100, maxlen=100, filters=128, kernel_size=5, strides=1,
                     pool_size=4, lstm_output_size=4):
        super().__init__()
        self.add(Input(shape=(None,), dtype="int64"))
        self.add(Embedding(max_words, embedding_dim))
        self.add(Conv1D(filters, kernel_size, padding='valid', activation="relu",strides=strides))
        self.add(MaxPooling1D(pool_size))
        self.add(Conv1D(filters, kernel_size, padding='valid', activation="relu",strides=strides))
        self.add(MaxPooling1D(pool_size))
        self.add(Conv1D(filters, kernel_size, padding='valid', activation="relu",strides=strides))
        self.add(GlobalMaxPooling1D())
        self.add(Dense(128, activation="relu"))
        self.add(Dropout(0.5))
        self.add(Dense(1, activation="sigmoid"))

#GANs
class DiscrimintorGan(Sequential):
    def __init__(self):
        super().__init__()
        self.layer0 = self.add(Input)
        self.layer1 = self.add(Conv1D)
        self.layer2 = self.add(LSTM)
        self.layer3 = self.add(Flatten)
        self.layer4 = self.add(Dense)
        self.layer5 = self.add(Softmax)

class GeneratorGan(Sequential):
    def __init__(self):
        super().__init__()
        self.layer0 = self.add(Input)
        self.layer1 = self.add(Conv1D)
        self.layer2 = self.add(LSTM)
        self.layer3 = self.add(Flatten)
        self.layer4 = self.add(Dense)
        self.layer5 = self.add(Softmax)

