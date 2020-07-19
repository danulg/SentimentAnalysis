import numpy as np
from network_architectures import SentimentAnalysisBasic as Basic
from network_architectures import SentimentAnalysisBidirectional as Bidirectional
from network_architectures import SentimentAnalysisSingleConv1D as SingleConv1D
from network_architectures import SentimentAnalysisMultipleConv1D as MultipleConv1D


import bokeh

class TrainNetworks():
      def __init__(self, tr_dt, tr_lbl, val_dt, val_lbl, unsup, weights):
          super().__init__()
          #set class variables
          self.tr_dt = tr_dt
          self.tr_lbl = tr_lbl
          self.val_dt = val_dt
          self.val_lbl = val_lbl
          self.val_lbl = val_lbl
          self.glove_weights = weights
          self.unlabled = unsup

      def train(self, name='basic', epochs=10, batch_size=32, rate=0.5, optimizer='adam', loss='binary_crossentropy', metrics=['acc'], verbose=1):
          if name == 'basic':
              model = Basic(rate)

          elif name == 'glove_basic':
              model = Basic(rate)
              model.layers[0].set_weights([self.glove_weights])
              model.layers[0].trainable = False

          elif name == 'bidirectional':
              model = Bidirectional()

          elif name == 'conv':
              model = SingleConv1D()

          else:
              print("Type of model not identified")
              return 0, 0

          model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
          history = model.fit(self.tr_dt, self.tr_lbl, epochs=epochs, batch_size=batch_size, validation_data=(self.val_dt, self.val_lbl),\
                       verbose=verbose)
          model.save(name+'_model_'+str(epochs))
          return history, model

      def train_unlabled(self, iterates=2, rate=0.5, name='basic', epochs=10, batch_size=32, optimizer='adam', loss='binary_crossentropy',\
                metrics=['acc'], verbose=1):
          if epochs%iterates != 0:
              print('The epoch / iterate combination does not match: Aborting execution')
              return 0, 0

          if name == 'basic':
              model = Basic(rate=rate)

          elif name == 'glove_basic':
              model = Basic(rate=rate)
              model.layers[0].set_weights([self.glove_weights])
              model.layers[0].trainable = False

          elif name == 'bidirectional':
              model = Bidirectional()

          else:
              print("Type of model not identified")
              return 0, 0

          model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

          i = 0
          history = []
          sub_epoch = int(epochs / iterates)

          while(i<iterates):
              print("iteration cycle:", i)
              temp = model.fit(self.tr_dt, self.tr_lbl, batch_size=batch_size, epochs=sub_epoch,\
                              validation_data=(self.val_dt, self.val_lbl), verbose=1)
              history.append(temp)
              self.__add_remove(model)
              i+=1


          model.save(name + '_iterative_model_' + str(epochs))
          return history, model

      def __add_remove(self, model):
          predicitons = model.predict(self.unlabled)

          # Adds high confidence results to training.
          # Add more iterations and modify to keep track of added unlabled data
          i = 0
          k = np.array([1])
          j = np.array([0])
          for x in np.nditer(predicitons):
              if x >= 0.8:
                  x_train1 = np.append(self.tr_dt, np.array([self.unlabled[i]]), axis=0)
                  y_train1 = np.append(self.tr_lbl, k)
                  i += 1
              elif x <= 0.2:
                  x_train1 = np.append(self.tr_dt, np.array([self.unlabled[i]]), axis=0)
                  y_train1 = np.append(self.tr_lbl, j)
                  i += 1
              else:
                  i += 1




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

#Possibly a fifth model that interacts with clustering


