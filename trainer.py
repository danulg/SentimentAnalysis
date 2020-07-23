import numpy as np
from network_architectures import SentimentAnalysisBasic as Basic
from network_architectures import SentimentAnalysisBidirectional as Bidirectional
from network_architectures import SentimentAnalysisSingleConv1D as SingleConv1D
from network_architectures import SentimentAnalysisMultipleConv1D as MultipleConv1D


class TrainNetworks():
    def __init__(self, tr_dt, tr_lbl, val_dt, val_lbl, unsup, weights):
        super().__init__()
        # set class variables
        self.tr_dt = tr_dt
        self.tr_lbl = tr_lbl
        self.val_dt = val_dt
        self.val_lbl = val_lbl
        self.val_lbl = val_lbl
        self.glove_weights = weights
        self.unlabled = unsup

    def train(self, name='basic', data='labled_only', rate=0.5, lstm_output_size=128, lstm_output_size2=128,
              dense_output_size=128, sub_epochs=4, iterates=2, epochs=8, batch_size=32, optimizer='adam',\
              loss='binary_crossentropy', metrics=['acc'], verbose=1, cutoff=0.8):

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

        if data == 'labled_only':
            save_name = str(cutoff) + '_' + name + '_model_' + str(sub_epochs) + '_' + str(iterates) + '_' + str(
                dense_output_size) + '_' \
                        + str(lstm_output_size) + '_' + str(lstm_output_size2) +'.h5'
            history = model.fit(self.tr_dt, self.tr_lbl, epochs=epochs, batch_size=batch_size,
                                validation_data=(self.val_dt, \
                                                 self.val_lbl), verbose=verbose)
            model.save_weights(save_name)
            return history.history, model

        elif data == 'unlabled_considered':
            i = 0
            history = []
            save_name = str(cutoff) + '_' + name + '_itermodel_' + str(sub_epochs) + '_' + str(iterates) + '_' + str(
                dense_output_size) + \
                        '_' + str(lstm_output_size) + '_' + str(lstm_output_size2) + '.h5'
            while (i < iterates):
                print("iteration cycle:", i + 1)
                temp = model.fit(self.tr_dt, self.tr_lbl, batch_size=batch_size, epochs=sub_epochs, \
                                 validation_data=(self.val_dt, self.val_lbl), verbose=verbose)
                history.append(temp.history)
                self.__add_remove(model, cutoff)
                print('The number of training examples for iterate ' + str(i) + ' is:', len(self.tr_dt))
                print('The number of unlabled examples left:', len(self.unlabled))
                i += 1

            # Code for saving the model: Cannot load models this way: known tensorflow error
            # model.save(save_name)

            model.save_weights(save_name)
            return history, model

    def __add_remove(self, model, cutoff):
        predicitons = model.predict(self.unlabled)

        # Adds high confidence results to training.
        # Add more iterations and modify to keep track of added unlabled data
        i = 0
        np_one = np.array([1])
        np_zero = np.array([0])
        to_remove = []

        for x in np.nditer(predicitons):
            if x >= cutoff:
                self.tr_dt = np.append(self.tr_dt, np.array([self.unlabled[i]]), axis=0)
                self.tr_lbl = np.append(self.tr_lbl, np_one)
                to_remove.append(i)
                i += 1
            elif x <= 1 - cutoff:
                self.tr_dt = np.append(self.tr_dt, np.array([self.unlabled[i]]), axis=0)
                self.tr_lbl = np.append(self.tr_lbl, np_zero)
                to_remove.append(i)
                i += 1
            else:
                i += 1

        self.unlabled = np.delete(self.unlabled, to_remove, axis=0)

# Unsupervised: Ideas include running through half trained and adding and taking out high threshold data!
# Note down times for training.

# Model 1: On the 25000 smaples - Basic


# Model 2: Transfer learning with Glove but no interaction with cluseteing or unsupervised set: all examples 25000  - Basic

# Model 3:  Transfer learning with Glove but no interaction with cluseteing or unsupervised set: with a fraction of the original data- Basic

# Model 4: Clustering with glove6B to add additional examples - Basic

# Model 5: Train in cycles to add / remove examples with high probability - Basic

# Model 6: On the 25000 smaples - Advanced

# Model 7: Transfer learning with Glove but no interaction with cluseteing or unsupervised set: all examples 25000  - Advanced

# Model 8:  Transfer learning with Glove but no interaction with cluseteing or unsupervised set: with a fraction - Advanced

# Model 9: Clustering with glove6B to add additional examples - Advanced

# Model 10: Train in cycles to add / remove examples with high probability - Advanced

############# IDEAL: 5, 4 > 2 > 3 > 1 AND 10, 9 > 7 > 8 > 6

# Possibly a fifth model that interacts with clustering

# for x in self.unlabled:
#     y = model.predict(x)
#
#     if y>=0.8:
#        np.append(self.tr_dt, x, axis=0)
