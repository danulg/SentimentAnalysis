from tensorflow.keras.layers import LSTM, Dense, Conv1D, Softmax, Flatten, Embedding, Input, Bidirectional, Dropout, \
    MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential


class SentimentAnalysisBasic(Sequential):
    def __init__(self, rate=0.5, max_words=20000, embedding_dim=100, max_len=200, dense_output_size=128):
        super().__init__()
        self.add(Embedding(max_words, embedding_dim, input_length=max_len))
        self.add(Dropout(rate))
        self.add(Flatten())
        self.add(Dense(dense_output_size, activation='relu'))
        self.add(Dropout(rate))
        self.add(Dense(1, activation='sigmoid'))


class SentimentAnalysisBidirectional(Sequential):  # Should have 1D CNN and dropout or batchnorm
    def __init__(self, max_words=20000, embedding_dim=100, rate=0.5, lstm_output_size=100, lstm_output_size2=100,
                 max_len=200):
        super().__init__()
        self.add(Input(shape=(None,), dtype="int32"))
        # Embed each integer in a 100-dimensional vector
        self.add(Embedding(max_words, embedding_dim))
        # Add 2 bidirectional LSTMs
        self.add(LSTM(lstm_output_size, return_sequences=True))
        self.add(Bidirectional(LSTM(lstm_output_size2)))
        # Leads to input dimension error when commented out. Why?
        # Add a classifier
        self.add(Flatten())
        self.add(Dense(256, activation='relu'))
        self.add(Dropout(rate))
        self.add(Dense(1, activation="sigmoid"))


