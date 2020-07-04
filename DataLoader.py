import re
import os, shutil
from tensorflow import keras as keras
import dill

class IMDBDataSet():
    def __init__(self):
        super().__init__()
        self.amazing_sub = {'amazing', 'awesome', 'stunning', 'astounding'}
        self.amazingly_sub = {'amazingly', 'stunningly','astoundlingly'}
        self.imdb_dir = '/home/danul-g/IMDB'
        self.train_dir = os.path.join(self.imdb_dir, 'train')
        self.test_dir = os.path.join(self.imdb_dir, 'test')
        self.unlabled_dir = os.path.join(self.imdb_dir, 'unsup')

    def data_augmentation(self):
        pass

    def random_review(self):
        pass

    def data_explorer(self):
        pass

    def token_analysis(self):
        pass

    def load_original(self):
        for label_type in ['pos', 'neg']

    def load_augmented(self):
        pass

if __name__ == "__main__":
    data = IMDBDataSet()
    #train, test = data.load_normalized_data()
    #print(train)