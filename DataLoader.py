import re
import random
import os, shutil
from collections import Counter
from tensorflow import keras as keras
import dill
import pprint

class IMDBDataSet():
    def __init__(self):
        super().__init__()
        self.amazing_sub = {'amazing', 'awesome', 'stunning', 'astounding'}
        self.amazingly_sub = {'amazingly', 'stunningly', 'astoundlingly'}
        self.love_sub = {'love', 'adore'}
        self.great_sub = {'great', 'really nice'}
        self.common_words = {'a', 'are', 'you', 'have', 'by', 'an', 'one', 'is', 'the', 'I', 'and', 'in','of',\
                             'to', 'that', 'it', 'this', '/><br', 'as', 'has', 'about', 'very', 'they', 'or'\
                             'with', 'was', 'for', 'The', 'but', 'his', 'her', 'on', 'film', 'movie', 'be', 'by'\
                             'an', 'at', 'who', 'from', 'all', 'had', 'up', 'story', 'will', 'would', 'my', 'if',\
                             'only', 'see', 'can', 'it\'s', 'he', 'she', 'which', 'their', 'when', 'so', 'or', 'out'\
                             'some', 'just', 'this'}
        self.imdb_dir = '/home/danulg/IMDB'

    #Data augmentation methods
    def data_augmentation(self):
        pass

    #Methods for looking at random data
    def random_original_review(self):
        text, labels = self.load_original_train()
        print(len(text), len(labels))
        i = 0
        while(i<10):
            j = random.randint(0,len(text)-1)
            print(text[j], labels[j])
            i+=1

    def random_unsup_review(self):
        text = self.load_unlabled()
        print(len(text))
        i = 0
        while (i < 10):
            j = random.randint(0, len(text) - 1)
            print(text[j])
            i += 1

    def random_augmented_review(self):
        pass
    #Rough usage statistics: used to create lexicon for data augmentation
    def word_usage_analysis(self, label=1, num='35'):
        text, labels = self.load_original_train()
        word_counter = Counter()
        for z in zip(text, labels):
            if z[1] == label or z[1]==2:
                for _ in z[0].split():
                    if _ not in self.common_words:
                        word_counter[_]+=1

        print(word_counter.most_common(num))
        #Add data for tokenizers?
        #Can we use bokeh and scipy to get a "nicer" look at the data?

    #Data to more convinient form
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

    def unlabled_to_pickle(self):
        text = []
        dir = os.path.join(self.imdb_dir, 'unsup')
        for fname in os.listdir(dir):
            with open(os.path.join(dir, fname)) as f:
                text.append(f.read())
        print(type(text), len(text))
        dill.dump(text, open('unsup_text.pkd', 'wb'))

    #Methods for loading data
    def load_test(self):
        with open('test_text.pkd', 'rb') as f:
            text = dill.load(f)
        with open('test_label.pkd', 'rb') as f:
            labels = dill.load(f)
        return text, labels

    def load_original_train(self):
        with open('train_text.pkd', 'rb') as f:
            text = dill.load(f)
        with open('train_label.pkd', 'rb') as f:
            labels = dill.load(f)
        return text, labels

    def load_unlabled(self):
        with open('unsup_text.pkd', 'rb') as f:
            text = dill.load(f)
        return text


    def load_augmented_test(self):
        pass


if __name__ == "__main__":
    data = IMDBDataSet()
    #Run once to create files
    #data.labled_to_pickle('train')
    #data.labled_to_pickle('test')
    #data.unlabled_to_pickle()

    #data.random_unsup_review()
    # data.random_original_review()
    # data.word_usage_analysis(1, 40)


