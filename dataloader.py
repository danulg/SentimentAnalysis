import re
import random
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
import dill
import json


class IMDBDataSet():
    def __init__(self, max_len=200, max_words=20000):
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
        self.stopwords = {'a', 'and', 'for', 'of', 'that', 'are', 'i', 'am', 'on', 'this', 'the', 'try', 'it',
                          'to', 'in', 'an', 'these', 'his', 'her', 'in', 'if', 'as', 'he', 'she', 'me', 'i.e.', 'i\'ll',
                          'e.g.', 'at', 'e', 'g', 'my', 'i\'m', 'was', 'with', 'we', 'i\'ve', 'wa', 'you'}

    # Data augmentation methods
    def data_augmentation(self):
        # To be added
        pass

    # Load data from source files.
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

    # Method for stripping punctuation etc
    def __strip(self, x, strip_tags=True, rem_punc=True):
        if strip_tags:
            x = re.sub(r'\<br', ' ', x)
            x = re.sub(r'\/br\>', ' ', x)
            x = re.sub(r'\/\>', ' ', x)

        if rem_punc:
            x = re.sub(r'[!;:,.()?-]', ' ', x)

        return x

    # Methods for looking at reviews / random reviews, before / after formatting
    def reviews(self, name='train', num=10, is_random=True, original=False, input_view=True, ret_val=False):
        text, labels = self.__load_from_source(name=name)

        if original:
            text = [self.__strip(x, rem_punc=False) for x in text]

        else:
            text = self.__text_formatting(text)

            if input_view:
                tokenizer = dill.load(open('tokenizer.pkd', 'rb'))
                word_index = tokenizer.word_index
                word_index = list(word_index.items())
                word_index = word_index[:self.max_words]
                word_index = [x[0] for x in word_index]
                set_word_index = set(word_index)
                text = self.__remove_stopwords(text, set_word_index, non_invert=False)

                stripped_text = []
                for review in text:
                    tokens = review.split(" ")
                    if len(tokens) > self.max_len:
                        tokens = tokens[:self.max_len]
                    tokens_truncated = (" ").join(tokens)
                    stripped_text.append(tokens_truncated)

                text = stripped_text

        if ret_val:
            return text, labels

        else:
            print(name, len(text))

            i = 0

            if is_random:
                while i < num:
                    j = random.randint(0, len(text) - 1)
                    if name == 'unsup':
                        print(text[j], ': unlabled')
                        i += 1
                    else:
                        print(text[j], ':', labels[j])
                        i += 1

            else:
                while i < num:
                    if name == 'unsup':
                        print(text[i], ': unlabled')
                        i += 1
                    else:
                        print(text[i], ':', labels[i])
                        i += 1

    # Method for removing unwanted lists of words
    def __remove_stopwords(self, text, word_list, non_invert=True):
        stripped_text = []
        for review in text:
            tokens = review.split(" ")
            if non_invert:
                tokens_filtered = [word for word in tokens if not (word in word_list)]

            else:
                tokens_filtered = [word for word in tokens if word in word_list]

            tokens_filtered = " ".join(tokens_filtered)
            stripped_text.append(tokens_filtered)

        return stripped_text

    # Format text
    def __text_formatting(self, text, save=False):
        text = [x.lower() for x in text]
        text = self.__remove_stopwords(text, self.stopwords)
        text = [self.__strip(x) for x in text]
        text = self.__remove_stopwords(text, self.stopwords)

        if save:
            dill.dump(text, open('saved_text.pkd', 'wb'))

        return text

    # Method for loading data
    def load_data(self, name='train', new_tokens=False, verify=False):
        text, labels = self.__load_from_source(name=name)
        text = self.__text_formatting(text)
        labels_cp = labels.copy()

        if new_tokens:
            sequences, word_index = self.__new_tokens(text)

        else:
            tokenizer = dill.load(open('tokenizer.pkd', 'rb'))
            word_index = tokenizer.word_index
            sequences = tokenizer.texts_to_sequences(text)

            # Verify sequences are as the should be!
            if verify:
                data_1, labels_cp = self.__data_to_numpy(sequences, labels_cp, shuffle=False)
                data_1 = tokenizer.sequences_to_texts(data_1[:100])

                for x, y in zip(data_1, labels_cp[:100]):
                    print(x, y)

        encoded, labels = self.__data_to_numpy(sequences, labels)

        return encoded, labels, word_index

    # Convert data to numpy: shuffle the labled data
    def __data_to_numpy(self, sequences, labels=[], shuffle=True):
        padded = pad_sequences(sequences, maxlen=self.max_len)
        print('Shape of data tensor:', padded.shape)

        if shuffle:
            indices = np.arange(padded.shape[0])
            np.random.shuffle(indices)
            padded = padded[indices]

            if labels != []:
                labels = np.asarray(labels)
                labels = labels[indices]

        return padded, labels

    # Creat tokens
    def __new_tokens(self, text):
        # tokenize the text data
        self.tokenizer.fit_on_texts(text)
        sequences = self.tokenizer.texts_to_sequences(text)
        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        dill.dump(self.tokenizer, open('tokenizer.pkd', 'wb'))
        tokenizer_json = self.tokenizer.to_json()
        with open('tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
        return sequences, word_index

if __name__ == "__main__":
    data = IMDBDataSet()

    # Create and save tokenizer on unsupervised data.
    # data.load_data(name='unsup', new_tokens=True)

    #View reviews and verify conversion porcess
    data.reviews(name='train', num=100, is_random=False)
    data.load_data(verify=True)
