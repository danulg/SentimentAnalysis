from architectures import AnalysisBasic as Basic
from architectures import AnalysisBidirectional as Bidirectional
from architectures import ConvolutionalLSTM as ConvLSTM
from architectures import Convolutional as Conv
from datetime import datetime
import tensorflow.keras as keras

class TrainNetworks():
    def __init__(self, tr_dt, tr_lbl, val_dt, val_lbl, glove_weights, w2vec_weights, optimizer='adam',
                 loss='binary_crossentropy', metrics=['acc'], max_len=200, embedding_dim=100, max_words=20000):
        super().__init__()
        # set class variables
        self.tr_dt = tr_dt
        self.tr_lbl = tr_lbl
        self.val_dt = val_dt
        self.val_lbl = val_lbl
        self.glove_weights = glove_weights
        self.w2vec_weights = w2vec_weights
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.max_words = max_words

    def train(self, name='basic', rate=0.5, lstm_output_size=128, lstm_output_size2=128, dense_output_size=128,
              filters=32, epochs=20, batch_size=32, verbose=1, record=False):

        if name == 'basic':
            model = Basic(rate=rate, dense_output_size=dense_output_size)
            save_name = name + '_model_' + str(epochs) + '_' + str(dense_output_size) + '_' + str(rate) + '.h5'

        elif name == 'glove_basic':
            model = Basic(rate=rate, dense_output_size=dense_output_size)
            model.layers[0].set_weights([self.glove_weights])
            model.layers[0].trainable = False
            save_name = name + '_model_' + str(epochs) + '_' + str(dense_output_size) + '_' + str(rate) + '.h5'

        elif name == 'w2vec_basic':
            model = Basic(rate=rate, dense_output_size=dense_output_size)
            model.layers[0].set_weights([self.w2vec_weights])
            model.layers[0].trainable = False
            save_name = name + '_model_' + str(epochs) + '_' + str(dense_output_size) + '_' + str(rate) + '.h5'

        elif name == 'bidirectional':
            model = Bidirectional(rate=rate, lstm_output_size=lstm_output_size, lstm_output_size2=lstm_output_size2)
            save_name = name + '_model_' + str(epochs) + '_' + str(dense_output_size) + '_' + str(rate) + '_' + \
                        str(lstm_output_size) + '_' + str(lstm_output_size2) + '.h5'

        elif name == 'glove_bidirectional':
            model = Bidirectional(rate=rate, lstm_output_size=lstm_output_size, lstm_output_size2=lstm_output_size2)
            model.layers[0].set_weights([self.glove_weights])
            model.layers[0].trainable = False
            save_name = name + '_model_' + str(epochs) + '_' + str(dense_output_size) + '_' + str(rate) + '_' + \
                        str(lstm_output_size) + '_' + str(lstm_output_size2) + '.h5'

        elif name == 'w2vec_bidirectional':
            model = Bidirectional(rate=rate, lstm_output_size=lstm_output_size, lstm_output_size2=lstm_output_size2)
            model.layers[0].set_weights([self.w2vec_weights])
            model.layers[0].trainable = False
            save_name = name + '_model_' + str(epochs) + '_' + str(dense_output_size) + '_' + str(rate) + '_' + \
                        str(lstm_output_size) + '_' + str(lstm_output_size2) + '.h5'

        elif name == 'conv_lstm':
            model = ConvLSTM(rate=rate, filters=filters, lstm_output_size=lstm_output_size)
            model.build((self.max_words, self.embedding_dim))
            save_name = name + '_model_' + str(epochs) + '_' + str(filters) + str(lstm_output_size) + '_' + str(rate) \
                        + '.h5'

        elif name == 'glove_conv_lstm':
            model = ConvLSTM(rate=rate,  filters=filters, lstm_output_size=lstm_output_size)
            model.build((self.max_words, self.embedding_dim))
            model.layers[0].set_weights([self.glove_weights])
            model.layers[0].trainable = False
            save_name = name + '_model_' + str(epochs) + '_' + str(filters) + str(lstm_output_size) + '_' + str(rate) \
                        + '.h5'

        elif name == 'w2vec_conv_lstm':
            model = ConvLSTM(rate=rate,  filters=filters, lstm_output_size=lstm_output_size)
            model.build((self.max_words, self.embedding_dim))
            model.layers[0].set_weights([self.w2vec_weights])
            model.layers[0].trainable = False
            save_name = name + '_model_' + str(epochs) + '_' + str(filters) + str(lstm_output_size) + '_' + str(rate) \
                        + '.h5'

        elif name == 'conv':
            model = Conv(rate=rate, filters=filters)
            model.build((self.max_words, self.embedding_dim))
            save_name = name + '_model_' + str(epochs) + '_' + str(filters) + '_' + str(rate) + '.h5'

        elif name == 'glove_conv':
            model = Conv(rate=rate,  filters=filters)
            model.build((self.max_words, self.embedding_dim))
            model.layers[0].set_weights([self.glove_weights])
            model.layers[0].trainable = False
            save_name = name + '_model_' + str(epochs) + '_' + str(filters) + '_' + str(rate) + '.h5'

        elif name == 'w2vec_conv':
            model = Conv(rate=rate,  filters=filters)
            model.build((self.max_words, self.embedding_dim))
            model.layers[0].set_weights([self.w2vec_weights])
            model.layers[0].trainable = False
            save_name = name + '_model_' + str(epochs) + '_' + str(filters) + '_' + str(rate) + '.h5'

        else:
            print("Type of model not identified")
            return 0

        # Compile model if not already compiled
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        model.summary()

        if record:
            logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
            history = model.fit(self.tr_dt, self.tr_lbl, epochs=epochs, batch_size=batch_size,
                                validation_data=(self.val_dt, self.val_lbl), verbose=verbose,
                                callbacks=[tensorboard_callback])

        else:
            history = model.fit(self.tr_dt, self.tr_lbl, epochs=epochs, batch_size=batch_size,
                                validation_data=(self.val_dt, self.val_lbl), verbose=verbose)


        model.save_weights(save_name)
        return history.history