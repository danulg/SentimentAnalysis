import re
import random
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy


class IMDBDataSet():
    def __init__(self,  max_len=200, max_words=20000):
        super().__init__()
        self.amazing_sub = {'amazing', 'awesome', 'stunning', 'astounding'}
        self.amazingly_sub = {'amazingly', 'stunningly', 'astoundlingly'}
        self.love_sub = {'love', 'adore'}
        self.great_sub = {'great', 'really nice'}
        self.imdb_dir = './IMDB'
        self.nlp = spacy.load('en_core_web_sm')
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.stopwords = {'a', 'and', 'for', 'of', 'that', 'it', 'are', 'i', 'am', 'on', 'this', 'the', 'try',
                          'to', 'in', 'an', 'these', 'his', 'her', 'in', 'if', 'as', 'he', 'she', 'me', 'i.e.', 'i\'ll',
                          'e.g.', 'at', 'e', 'g', 'my', 'i\'m', 'was', 'with', 'we', 'i\'ve', 'wa'}

    #Data augmentation methods
    def data_augmentation(self):
        #To be added
        pass

    def __name_extractor(self, text):
        people = []
        for _ in text:
            entry = self.nlp(_)
            temp = []
            for ent in entry.ents:
                if ent.label_ == 'ORG' or ent.label_ == 'PERSON':
                    people.append(ent.text)
                    temp.append(ent.text)
        return people


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
    def __strip(self, x, strip_tags=True, rem_punc=True):
        if strip_tags:
            x = re.sub(r'\<br', ' ', x)
            x = re.sub(r'\/br\>', ' ', x)
            x = re.sub(r'\/\>', ' ', x)

        if rem_punc:
            x = re.sub(r'[!;,.()?-]', ' ', x)


        return x

    #Methods for looking at random data
    def read_review(self, name='train', num=10, is_random=True, original=False, restricted=False, names=False):
        text, labels = self.__load_from_source(name=name)

        if original:
            text = [self.__strip(x, rem_punc=False) for x in text]

        else:
            text = [x.lower() for x in text]
            text = self.__remove_stopwords(text, self.stopwords)
            text = [self.__strip(x) for x in text]
            text = self.__remove_stopwords(text, self.stopwords)

            if restricted:
                _, word_index = self.__tokenize(text)
                word_index = list(word_index.items())
                word_index = word_index[:self.max_words]
                word_index = [x[0] for x in word_index]
                set_word_index = set(word_index)
                print(set_word_index)
                text = self.__remove_stopwords(text, set_word_index, non_invert=False)

                if names:
                    people = self.__name_extractor(text)
                    text = self.__remove_stopwords(text, people, non_invert=False)
                    text = [self.__strip(x) for x in text]

                stripped_text = []
                for review in text:
                    tokens = review.split(" ")
                    if len(tokens) > self.max_len:
                        tokens = tokens[:self.max_len]
                    tokens_truncated = (" ").join(tokens)
                    stripped_text.append(tokens_truncated)

                text = stripped_text

        print(name, len(text))
        i = 0

        if num < 0 or num > len(text):
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


    def __remove_stopwords(self, text, word_list, non_invert=True):
        stripped_text = []
        for review in text:
            tokens = review.split(" ")
            if non_invert:
                tokens_filtered = [word for word in tokens if not word in word_list]

            else:
                tokens_filtered = [word for word in tokens if word in word_list]
            tokens_filtered = (" ").join(tokens_filtered)
            stripped_text.append(tokens_filtered)

        return stripped_text

    def load_data(self, name='train', is_numpy=True, stopwords=True, names=False):

        #Format data based on return type, strip punctuation
        text, labels = self.__load_from_source(name=name)


        if is_numpy:
            if stopwords:
                text = [x.lower() for x in text]
                text = self.__remove_stopwords(text, self.stopwords)
                text = [self.__strip(x) for x in text]
                text = self.__remove_stopwords(text, self.stopwords)

                if names:
                    people = self.__name_extractor(text)
                    text = self.__remove_stopwords(text, people)
                    text = [self.__strip(x) for x in text]

                sequences, word_index = self.__tokenize(text)

            else:
                text = [self.__strip(x) for x in text]
                sequences, word_index = self.__tokenize(text)

            data, labels = self.__data_to_numpy(sequences, labels)

            # Verify sequences are as the should be!
            # data_1, labels_cp = self.__data_to_numpy(sequences, labels_cp, shuffle=False)
            # data_1 = self.tokenizer.sequences_to_texts(data_1[:100])
            #
            # for x, y in zip(data_1, labels_cp[:100]):
            #     print(x, y)

            return data, labels, word_index


        else:
            _, word_index = self.__tokenize(text)
            return text, labels, word_index


    def __data_to_numpy(self, sequences, labels=[], shuffle=True):
        #Convert to numpy
        data = pad_sequences(sequences, maxlen=self.max_len)
        print('Shape of data tensor:', data.shape)

        if labels != [] and shuffle:
            labels = np.asarray(labels)
            # print('Shape of label tensor:', labels.shape)
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            data_sh = data[indices]
            labels_sh = labels[indices]
            return data_sh, labels_sh

        else:
            return data, labels

    def __tokenize(self, text):
        # tokenize the text data
        self.tokenizer.fit_on_texts(text)
        sequences = self.tokenizer.texts_to_sequences(text)
        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        return sequences, word_index

if __name__ == "__main__":
    data = IMDBDataSet()

    # Verify encoding methods does what it is supposed
    data.read_review(num=100, is_random=False, restricted=True)
    # data.load_data()



