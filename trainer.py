import tensorflow as tf
import numpy as np
import matplotlib
from data_loader import IMDBDataSet
from network_architectures import SentimentAnalysisBasic as SABasic
from network_architectures import SentimentAnalysisBidirectional as SABi
import bokeh

class TrainNetworks():
      def __init__(self, tr_dt, tr_lbl, val_dt, val_lbl, weights):
          super().__init__()
          #set class variables
          self.tr_dt = tr_dt
          self.tr_lbl = tr_lbl
          self.val_dt = val_dt
          self.val_lbl = val_lbl
          self.glove_weights = weights

      def train(self, name='basic', epochs=5, batch_size=32, optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'], verbose=1):
          if name == 'basic':
              model = SABasic()

          elif name == 'glove_basic':
              model = SABasic()
              model.layers[0].set_weights([self.glove_weights])
              model.layers[0].trainable = False

          elif name == 'bidirectional':
              model = SABi()

          else:
              print("Type of model not identified")
              return 0, 0

          model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
          history = model.fit(self.tr_dt, self.tr_lbl, epochs=epochs, batch_size=batch_size, validation_data=(self.val_dt, self.val_lbl),\
                       verbose=verbose)
          model.save(name+'_model_'+str(epochs))
          return history, model

      def train_unlabled(self, name='basic', epochs=10, batch_size=32, optimizer='rmsprop', loss='binary_crossentropy',
                metrics=['acc'], verbose=1):
          pass















#Unsupervised: Ideas include running through half trained and adding and taking out high threshold data!
#Note down times for training.

#Model 1: On the 25000 smaples - Basic




#Model 2: Transfer learning with Glove but no interaction with cluseteing or unsupervised set: all examples 25000  - Basic

#Model 3:  Transfer learning with Glove but no interaction with cluseteing or unsupervised set: with a fraction of the original data- Basic

#Model 4: Clustering with glove6B to add additional examples - Basic

#Model 5: Train in cycles to add / remove examples with high probability - Basic

#Model 6: On the 25000 smaples - Advanced

#Model 7: Transfer learning with Glove but no interaction with cluseteing or unsupervised set: all examples 25000  - Advanced

#Model 8:  Transfer learning with Glove but no interaction with cluseteing or unsupervised set: with a fraction - Advanced

#Model 9: Clustering with glove6B to add additional examples - Advanced

#Model 10: Train in cycles to add / remove examples with high probability - Advanced

############# IDEAL: 5, 4 > 2 > 3 > 1 AND 10, 9 > 7 > 8 > 6











#Possibly a fifth model that


