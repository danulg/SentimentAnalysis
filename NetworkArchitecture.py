from tensorflow.keras.layers import LSTM, Dense, Conv1D, Softmax, Flatten, Input
from tensorflow.keras import Sequential

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

class sentiment_analysis(Sequential):
    def __int__(self):
        self.__int__(super)

class sentiment_analysis_pretrained(Sequential):
    def __int__(self):
        self.__int__(super)
