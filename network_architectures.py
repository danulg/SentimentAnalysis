from tensorflow.keras.layers import LSTM, Dense, Conv1D, Softmax, Flatten, Input, Embedding
from tensorflow.keras import Sequential

class sentiment_analysis_basic(Sequential):
    def __int__(self, max_words=10000, embedding_dim=100, maxlen=100):
        self.__int__(super)
        self.add(Embedding(max_words, embedding_dim, input_length=maxlen))
        self.add(Flatten())
        self.add(Dense(32, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))

class sentiment_analysis_advanced(Sequential):
    def __int__(self, max_words=10000, embedding_dim=100, maxlen=100):
        self.__int__(super)
        self.add(Embedding(max_words, embedding_dim, input_length=maxlen))
        self.add(Flatten())
        self.add(Dense(32, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))

#GANs
class discrimintor_gan(Sequential):
    def __int__(self):
        self.__int__(super)
        self.layer0 = self.add(Input)
        self.layer1 = self.add(Conv1D)
        self.layer2 = self.add(LSTM)
        self.layer3 = self.add(Flatten)
        self.layer4 = self.add(Dense)
        self.layer5 = self.add(Softmax)

class generator_gan(Sequential):
    def __int__(self):
        self.__int__(super)
        self.layer0 = self.add(Input)
        self.layer1 = self.add(Conv1D)
        self.layer2 = self.add(LSTM)
        self.layer3 = self.add(Flatten)
        self.layer4 = self.add(Dense)
        self.layer5 = self.add(Softmax)

