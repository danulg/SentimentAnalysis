import re
import random
import os
import numpy as np
import dill
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class IMDBDataSet():
    def __init__(self):
        super().__init__()
        self.amazing_sub = {'amazing', 'awesome', 'stunning', 'astounding'}
        self.amazingly_sub = {'amazingly', 'stunningly', 'astoundlingly'}
        self.love_sub = {'love', 'adore'}
        self.great_sub = {'great', 'really nice'}
        self.imdb_dir = '/home/danulg/IMDB'

    #Data augmentation methods
    def data_augmentation(self):
        #To be added
        pass

    #Methods for looking at random data
    def random_review(self, name='train'):
        name = name + '_text.pkd'
        text, labels = self.load_data(is_numpy=False)
        print(name, len(text))
        i = 0

        while(i<10):
            j = random.randint(0, len(text) - 1)
            if name == 'unsup_text.pkd':
                print(text[j], ': unlabled')
                i+=1
            else:
                print(text[j], ':', labels[j])
                i+=1

    #Data to more convinient form
    def labled_to_pickle(self, val='train'):
        labels = []
        text = []
        dir = os.path.join(self.imdb_dir, val)
        for label_type in ['pos', 'neg']:
            dir_name = os.path.join(dir, label_type)
            for fname in os.listdir(dir_name):
                with open(os.path.join(dir_name, fname)) as f:
                    _ = f.read()
                    _ = re.sub(r'\<br', '', _)
                    _ = re.sub(r'\/br\>', '', _)
                    _ = re.sub(r'\/\>', '', _)
                    _ = re.sub(r'[,.()?\']', '', _)
                    _.strip()
                    text.append(_)
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
                _ = f.read()
                _ = re.sub(r'\<br', '', _)
                _ = re.sub(r'\/br\>', '', _)
                _ = re.sub(r'\/\>', '', _)
                _ = re.sub(r'[,.()?\']', '', _)
                _.strip()
                text.append(_)
        print(type(text), len(text))
        dill.dump(text, open('unsup_text.pkd', 'wb'))


    def load_data(self, name='train', is_numpy=True, maxlen=100, max_words=10000):
        data = name+'_text.pkd'
        lbl = name+'_label.pkd'
        with open(data, 'rb') as f:
            text = dill.load(f)

        sequences, word_index = self.__tokenize(text, max_words)

        if name == 'train' or name == 'test' or name == 'augmented':
            with open(lbl, 'rb') as f:
                labels = dill.load(f)

            if is_numpy:
                data, labels = self.__data_to_numpy(sequences, labels, maxlen)
                return data, labels, word_index
            else:
                return text, labels, word_index

        else:
            if is_numpy:
                data, labels = self.__data_to_numpy(sequences, [],  maxlen)
                return data, labels, word_index
            else:
                return text, [], word_index

    def __data_to_numpy(self, sequences, labels, word_index, maxlen=100):
        #Convert to numpy
        data = pad_sequences(sequences, maxlen=maxlen)
        print('Shape of data tensor:', data.shape)

        if labels != []:
            labels = np.asarray(labels)
            print('Shape of label tensor:', labels.shape)
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            data = data[indices]
            labels = labels[indices]
            return data, labels

        else:
            return data, labels, word_index

    def __tokenize(self, text, max_words=10000):
        # tokenize the text data
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        return sequences, word_index

if __name__ == "__main__":
    data = IMDBDataSet()

    #Run once to create files and comment out
    # data.labled_to_pickle(val='train')
    # data.labled_to_pickle(val='test')
    # data.unlabled_to_pickle()


    #Code to check some of the methods
    d, l, w = data.load_data()
    w = list(w.items())
    i = 0
    while(i<5):
        print(d[i], 'label is', l[i])
        print('word', w[i])
        i+=1

    d, l, w = data.load_data(is_numpy=False)
    w = list(w.items())
    i = 0
    while (i < 5):
        print(d[i], 'label is', l[i])
        print('word', w[i])
        i += 1