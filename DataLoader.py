import re
import random
import os, shutil
from collections import Counter
from tensorflow import keras as keras
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
        self.imdb_dir = '/home/danul-g/IMDB'

    #Data augmentation methods
    def data_augmentation(self):
        pass

    #Methods for looking at random data
    def random_review(self, name='train'):
        name = name + '_text.pkd'
        text, labels = self.load_data(is_numpy=False)
        print(name, len(text))
        i = 0

        if name=='unsup':
            labels=['unlabled']*10

        while(i<10):
            j = random.randint(0, len(text) - 1)
            print(text[j], ':' ,labels[j])
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
                    _ = re.sub(r'\\\>', '', _)
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
                _.strip()
                text.append(_)
        print(type(text), len(text))
        dill.dump(text, open('unsup_text.pkd', 'wb'))


    def load_data(self, name='train', is_numpy=True, maxlen=100, max_words=10000):
        data = name+'_text.pkd'
        lbl = name+'_label.pkd'
        with open(data, 'rb') as f:
            text = dill.load(f)

        if name == 'train' or name == 'test':
            with open(lbl, 'rb') as f:
                labels = dill.load(f)
            if is_numpy:
                text = self.__data_to_numpy(text, maxlen, max_words)
                return text, labels
            else:
                return text, labels

        if name == 'unsup':
            if is_numpy:
                text = self.__data_to_numpy(text, maxlen, max_words)
                return text, []
            else:
                return text, []

    def __data_to_numpy(self, data, maxlen=100, max_words=10000):
        pass

if __name__ == "__main__":
    data = IMDBDataSet()

    #Run once to create files and comment out
    #data.labled_to_pickle(val='train')
    #data.labled_to_pickle(val='test')
    #data.unlabled_to_pickle()

    data.random_review(name='train')
    data.random_review(name='unsup')

    # data.random_original_review()
    #data.word_usage(40)

''' #Rough usage statistics: used to create lexicon for data augmentation
    self.common = {'a', 'are', 'you', 'have', 'by', 'an', 'one', 'is', 'the', 'I', 'and', 'in', 'of',\
                       'to', 'that', 'it', 'this', '/><br', 'as', 'has', 'about', 'very', 'they', 'or',\
                       'with', 'was', 'for', 'The', 'but', 'his', 'her', 'on', 'film', 'movie', 'be', 'by',\
                       'an', 'at', 'who', 'from', 'all', 'had', 'up', 'story', 'will', 'would', 'my', 'if',\
                       'only', 'see', 'can', 'it\'s', 'he', 'she', 'which', 'their', 'when', 'so', 'or', 'out',\
                       'some', 'just', 'this', 'out', 'This', 'It', '-', 'even', 'were', 'more', 'what', 'than',\
                       'been', 'there', '<br', 'into', 'get', 'because', 'other', 'most', 'we', 'me', 'do',\
                       'first', 'its', 'any', 'think', 'him', 'being', 'did', 'characters', 'It\'s', 'know', 'movie',\
                       'does', 'watch', 'after', 'way', 'too', 'little', 'then', 'But', 'but', 'too', 'films', 'In',\
                       'A', 'such', 'these', 'should', 'still', 'seen', 'it.', 'them', 'And'}
    
    
    def word_usage(self, num='35'):
        text, _ = self.load_original_train()
        word_counter = Counter()
        for z in text:
            for _ in z.split():
                if _ not in self.common:
                    word_counter[_]+=1
        stats = word_counter.most_common(num)
        for _ in stats:
            print(_)
        #Add data for tokenizers?
        #Can we use bokeh and scipy to get a "nicer" look at the data?        
        '''


