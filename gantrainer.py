from tensorflow.keras.layers import LSTM, Dense, Conv1D, Softmax, Flatten, Embedding, Input, Bidirectional, Dropout, MaxPooling1D,\
    GlobalMaxPooling1D
from tensorflow.keras.models import Sequential

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