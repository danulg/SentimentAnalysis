from tensorflow.keras.layers import LSTM, Dense, Conv1D, Softmax, Flatten, Input, Embedding, Convolution1D
from tensorflow.keras import Sequential

class SentimentAnalysisBasic(Sequential):
    def __init__(self, max_words=10000, embedding_dim=100, maxlen=100):
        super().__init__()
        self.add(Embedding(max_words, embedding_dim, input_length=maxlen))
        self.add(Flatten())
        self.add(Dense(32, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))

class SentimentAnalysisAdvanced(Sequential): #Should have 1D CNN and dropout or batchnorm
    def __init__(self, max_words=10000, embedding_dim=100, maxlen=100):
        super().__init__()
        self.add(Embedding(max_words, embedding_dim, input_length=maxlen))
        self.add(Conv1D)
        self.add(Flatten())
        self.add(Dense(32, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))

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

