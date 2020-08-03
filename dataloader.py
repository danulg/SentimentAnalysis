import random
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import dill
import json
from gensim.models import Word2Vec
from textprep import TextPrep


class IMDBDataSet():
    def __init__(self, max_len=200, max_words=20000):
        super().__init__()
        self.max_words = max_words
        self.max_len = max_len
        self.text_prep = TextPrep()
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.stopwords = {'a', 'and', 'for', 'of', 'that', 'are', 'i', 'am', 'on', 'this', 'the', 'try', 'it', 'its',
                          'it\'s',
                          'to', 'in', 'an', 'these', 'his', 'her', 'in', 'if', 'as', 'he', 'she', 'me', 'i.e.', 'i\'ll',
                          'e.g.', 'at', 'e', 'g', 'my', 'i\'m', 'was', 'with', 'we', 'i\'ve', 'wa', 'you', 'ha', 'doe'}

    # Methods for looking at reviews / random reviews, before / after formatting
    def reviews(self, name='train', num=10, is_random=True, original=False, input_view=True, ret_val=False):
        text, labels = self.text_prep.load_from_source(name=name)

        if original:
            text = [self.text_prep.strip_punc(x, rem_punc=False) for x in text]

        else:
            text = self.text_prep.text_formatting(text)

            if input_view:
                tokenizer = dill.load(open('tokenizer.pkd', 'rb'))
                word_index = tokenizer.word_index
                word_index = list(word_index.items())
                word_index = word_index[:self.max_words]
                word_index = [x[0] for x in word_index]
                set_word_index = set(word_index)
                text = self.text_prep.remove_stopwords(text, set_word_index, non_invert=False)

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

    # Method for loading data
    def load_data_default(self, name='train', new_tokens=False, verify=False):
        text, labels = self.text_prep.load_from_source(name=name)
        text = self.text_prep.text_formatting(text)
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

    # Creat tokens: 
    def __new_tokens(self, text):
        # tokenize the text data
        name = 'tokenizer.pkd'
        self.tokenizer.fit_on_texts(text)
        sequences = self.tokenizer.texts_to_sequences(text)
        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        dill.dump(self.tokenizer, open(name, 'wb'))
        return sequences, word_index


if __name__ == "__main__":
    data = IMDBDataSet()
    data.load_data_default(name='unsup', new_tokens=True)


