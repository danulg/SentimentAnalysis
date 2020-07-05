import re
import random
import os, shutil
from tensorflow import keras as keras
import dill

class IMDBDataSet():
    def __init__(self):
        super().__init__()
        self.amazing_sub = {'amazing', 'awesome', 'stunning', 'astounding'}
        self.amazingly_sub = {'amazingly', 'stunningly', 'astoundlingly'}
        self.love_sub = {'love', 'adore'}
        self.imdb_dir = '/home/danul-g/IMDB'
        self.unlabled_dir = os.path.join(self.imdb_dir, 'unsup')

    def data_augmentation(self):
        pass

    def random_original_review(self):
        text, labels = self.load_original_test()
        print(len(text), len(labels))
        i = 0
        while(i<10):
            j = random.randint(0,len(text)-1)
            print(text[j], labels[j])
            i+=1

    def random_augmented_review(self):
        pass

    def data_explorer(self):
        pass

    def token_analysis(self):
        pass

    def labled_to_pickle(self, val='train'):
        labels = []
        text = []
        dir = os.path.join(self.imdb_dir, val)
        for label_type in ['pos', 'neg']:
            dir_name = os.path.join(dir, label_type)
            for fname in os.listdir(dir_name):
                with open(os.path.join(dir_name, fname)) as f:
                    text.append(f.read())
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
        #Check number of examples
        print(type(text), len(text))
        print(type(labels), len(labels))
        dill.dump(text, open(val+'_text.pkd', 'wb'))
        dill.dump(labels, open(val+'_label.pkd', 'wb'))


    def load_augmented(self):

        pass

    def load_original_test(self):
        with open('train_text.pkd', 'rb') as f:
            text = dill.load(f)
        with open('train_label.pkd', 'rb') as f:
            labels = dill.load(f)
        return text, labels
        pass

    def load_train(self):
        pass


if __name__ == "__main__":
    data = IMDBDataSet()
    #data.labled_to_pickle('train')
    #data.labled_to_pickle('test')
    data.random_original_review()

