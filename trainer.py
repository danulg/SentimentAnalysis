import numpy as np
from architectures import SentimentAnalysisBasic as Basic
from architectures import SentimentAnalysisBidirectional as Bidirectional



class TrainNetworks():
    def __init__(self, tr_dt, tr_lbl, val_dt, val_lbl, weights):
        super().__init__()
        # set class variables
        self.tr_dt = tr_dt
        self.tr_lbl = tr_lbl
        self.val_dt = val_dt
        self.val_lbl = val_lbl
        self.val_lbl = val_lbl
        self.glove_weights = weights
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'
        self.metrics = ['acc']


    def train(self, name='basic', rate=0.5, lstm_output_size=512, lstm_output_size2=512,
              dense_output_size=128, epochs=12, batch_size=32, verbose=1):

        if name == 'basic':
            model = Basic(rate=rate, dense_output_size=dense_output_size)

        elif name == 'glove_basic':
            model = Basic(rate=rate, dense_output_size=dense_output_size)
            model.layers[0].set_weights([self.glove_weights])
            model.layers[0].trainable = False

        elif name == 'bidirectional':
            model = Bidirectional(rate=rate, lstm_output_size=lstm_output_size, lstm_output_size2=lstm_output_size2)

        elif name == 'glove_bidirectional':
            model = Bidirectional(rate=rate)
            model.layers[0].set_weights([self.glove_weights])
            model.layers[0].trainable = False

        elif name == 'conv':
            model = SingleConv1D(rate=rate)

        elif name == 'glove_conv':
            model = SingleConv1D(rate=rate)
            model.layers[0].set_weights([self.glove_weights])
            model.layers[0].trainable = False

        elif name == 'conv_md':
            model = MultipleConv1D()

        elif name == 'glove_conv_md':
            model = MultipleConv1D(rate=rate)
            model.layers[0].set_weights([self.glove_weights])
            model.layers[0].trainable = False

        else:
            print("Type of model not identified")
            return 0, 0

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.summary()

        save_name = str(cutoff) + '_' + name + '_model_' + str(epochs) + '_' + str(
            dense_output_size) + '_' + str(rate) + '_' + str(lstm_output_size) + '_' + str(lstm_output_size2) + '.h5'
        history = model.fit(self.tr_dt, self.tr_lbl, epochs=epochs, batch_size=batch_size,
                            validation_data=(self.val_dt, self.val_lbl), verbose=verbose)
        model.save_weights(save_name)
        return history.history, model

