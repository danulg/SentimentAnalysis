import re
import random
import os
import numpy as np
import dill
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy


class IMDBDataSet():
    def __init__(self):
        super().__init__()
        self.amazing_sub = {'amazing', 'awesome', 'stunning', 'astounding'}
        self.amazingly_sub = {'amazingly', 'stunningly', 'astoundlingly'}
        self.love_sub = {'love', 'adore'}
        self.great_sub = {'great', 'really nice'}
        self.imdb_dir = '/home/danulg/IMDB'
        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords = self.nlp.Defaults.stop_words

    #Data augmentation methods
    def data_augmentation(self):
        #To be added
        pass

    def pickle_data(self, name='train'):
        labels = []
        text = []
        dir = os.path.join(self.imdb_dir, name)
        if name == 'train' or name == 'test':
            for label_type in ['pos', 'neg']:
                dir_name = os.path.join(dir, label_type)
                for fname in os.listdir(dir_name):
                    with open(os.path.join(dir_name, fname)) as f:
                        x = f.read()
                        x = self.punc_strip(x)
                        text.append(x)

                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)
        # Check number of examples
            print(type(text), len(text))
            print(type(labels), len(labels))
            dill.dump(text, open(name + '_text.pkd', 'wb'))
            dill.dump(labels, open(name + '_label.pkd', 'wb'))

        elif name == 'unsup':
            for fname in os.listdir(dir):
                with open(os.path.join(dir, fname)) as f:
                    x = f.read()
                    x = self.punc_strip(x)
                    text.append(x)

            print(type(text), len(text))
            dill.dump(text, open('unsup_text.pkd', 'wb'))


    #Method for stripping punctuation etc
    def punc_strip(self, x):
        x = x.lower()
        x = re.sub(r'\<br', '', x)
        x = re.sub(r'\/br\>', '', x)
        x = re.sub(r'\/\>', '', x)
        x = re.sub(r'[,.()?\']', '', x)
        x.strip()
        return x

    #Methods for looking at random data
    def random_review(self, name='train'):
        name = name + '_text.pkd'
        text, labels,_ = self.load_data(is_numpy=False)
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

    def remove_stopwords(self, sentence):
        tokens = sentence.split(" ")
        tokens_filtered = [word for word in tokens if not word in self.stopwords]
        return (" ").join(tokens_filtered)


    def pickle_data_stopwords(self, name='train'):
        save_name = name + '_text_stopwords.pkd'
        name = name + '_text.pkd'
        with open(name, 'rb') as f:
            text = dill.load(f)

        text = [self.remove_stopwords(x) for x in text]

        dill.dump(text, open(save_name, 'wb'))

    def load_data(self, name='train', is_numpy=True, maxlen=100, max_words=10000):
        #Track labels as required
        if name == 'train' or name == 'test':
            lbl = name + '_label.pkd'
            with open(lbl, 'rb') as f:
                labels = dill.load(f)
        else:
            labels = []

        #Format data based on return type
        if is_numpy:
            name = name + '_text_stopwords.pkd'
            with open(name, 'rb') as f:
                text = dill.load(f)

            sequences, word_index = self.__tokenize(text, max_words=max_words)
            data, labels = self.__data_to_numpy(sequences, labels, maxlen)
            return data, labels, word_index

        else:
            name = name + '_text.pkd'
            with open(name, 'rb') as f:
                text = dill.load(f)

            _, word_index = self.__tokenize(text, max_words=max_words)
            return text, labels, word_index




    def __data_to_numpy(self, sequences, labels, maxlen=100):
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
            return data, labels

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

    #Create files for easier loading
    data.pickle_data(name='train')
    data.pickle_data(name='test')
    data.pickle_data(name='unsup')

    data.pickle_data_stopwords(name='train')
    data.pickle_data_stopwords(name='test')
    data.pickle_data_stopwords(name='unsup')

