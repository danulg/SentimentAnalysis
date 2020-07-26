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
        self.imdb_dir = './IMDB'
        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords = self.nlp.Defaults.stop_words

    #Data augmentation methods
    def data_augmentation(self):
        #To be added
        pass

    def __load_from_source(self, name='train'):
        labels = []
        text = []
        dir = os.path.join(self.imdb_dir, name)
        if name == 'train' or name == 'test':
            for label_type in ['pos', 'neg']:
                dir_name = os.path.join(dir, label_type)
                for fname in os.listdir(dir_name):
                    with open(os.path.join(dir_name, fname)) as f:
                        x = f.read()
                        text.append(x)

                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)

            return text, labels

        elif name == 'unsup':
            for fname in os.listdir(dir):
                with open(os.path.join(dir, fname)) as f:
                    x = f.read()
                    text.append(x)

            return text, labels



    #Method for stripping punctuation etc
    def __punc_strip(self, x, strip_br=True, lower=True, rem_punc=True):
        if lower:
            x = x.lower()

        if strip_br:
            x = re.sub(r'\<br', '', x)
            x = re.sub(r'\/br\>', '', x)

        if rem_punc:
            x = re.sub(r'\/\>', '', x)
            x = re.sub(r'[,.()?\']', '', x)
            x.strip()
        return x

    #Methods for looking at random data
    def read_review(self, name='train', num=10, is_random=True):
        text, labels = self.__load_from_source(name=name)
        text = [self.__punc_strip(x, lower=False) for x in text]
        print(name, len(text))
        i = 0

        if num<0 or num>len(text):
            num = 100

        if is_random:
            while(i<num):
                j = random.randint(0, len(text) - 1)
                if name == 'unsup':
                    print(text[j], ': unlabled')
                    i+=1
                else:
                    print(text[j], ':', labels[j])
                    i+=1

        else:
            while(i<num):
                if name == 'unsup':
                    print(text[i], ': unlabled')
                    i += 1

                else:
                    print(text[i], ':', labels[i])
                    i += 1


    def __remove_stopwords(self, text):
        stripped_text = []
        for review in text:
            tokens = review.split(" ")
            tokens_filtered = [word for word in tokens if not word in self.stopwords]
            tokens_filtered = (" ").join(tokens_filtered)
            stripped_text.append(tokens_filtered)

        return stripped_text

    def load_data(self, name='train', is_numpy=True, has_stopwords=True, maxlen=200, max_words=30000):
        #Format data based on return type, strip punctuation
        text, labels = self.__load_from_source(name=name)
        text = [self.__punc_strip(x) for x in text]

        if is_numpy:
            if not has_stopwords:
                sequences, word_index = self.__tokenize(text, max_words=max_words)

            else:
                text = self.__remove_stopwords(text)
                sequences, word_index = self.__tokenize(text, max_words=max_words)

            data, labels = self.__data_to_numpy(sequences, labels, maxlen)
            return data, labels, word_index


        else:
            _, word_index = self.__tokenize(text, max_words=max_words)
            return text, labels, word_index


    def __data_to_numpy(self, sequences, labels=[], maxlen=100):
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
    data.read_review(num=100)



