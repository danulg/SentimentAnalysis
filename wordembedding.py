import os
import numpy as np
from dataloader import IMDBDataSet
from gensim.models import Word2Vec

class LoadGloVe:
    def __init__(self, words):
        super().__init__()
        self.glove_dir = './IMDB/glove.6B'
        self.embeddings_index = {}
        self.word_index = words

    def load_glove(self, max_words=20000, embedding_dim=100, name='glove.6B.'):
        file_name = name+str(embedding_dim)+'d.txt'
        with open(os.path.join(self.glove_dir, file_name)) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(self.embeddings_index))

        embedding_matrix = np.zeros((max_words, embedding_dim))
        for word, i in self.word_index.items():
            if i < max_words:
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        return embedding_matrix

class Word2VecTrain:
    def __init__(self):
        self.glove_dir = './IMDB/glove.6B'
        self.embeddings_index = {}

    def create_corpus(self):
        imdb = IMDBDataSet()
        text, _ = imdb.reviews(name='unsup', ret_val=True)
        text = [x.split() for x in text]

        # train model
        model = Word2Vec(text, min_count=1)
        # summarize the loaded model
        print(model)
        # summarize vocabulary
        words = list(model.wv.vocab)
        print(words)
        # save model
        model.save('model.bin')

    def load_word2vec(self, max_words=20000, embedding_dim=100):
        model = Word2Vec.load('model.bin')
        print('Found %s word vectors.' % len(self.embeddings_index))

        embedding_matrix = np.zeros((max_words, embedding_dim))
        length = range(0, max_words)
        for word, i in zip(list(model.wv.vocab), length):
            embedding_vector = model[word]
            embedding_matrix[i] = embedding_vector

        print(embedding_matrix)


if __name__ == '__main__':
    # Check load functionality
    # temp = IMDBDataSet()
    # x, y, words = temp.load_data()
    # glove = LoadGloVe(words)
    #
    # matrix = glove.load_glove()
    # print(matrix.shape)

    # Create word2vec
    # w2vec = Word2VecTrain()
    # w2vec.create_corpus()

    # Load and check
    # model = Word2Vec.load('model.bin')
    # print(model)
    # print(model['bad'])

    w2vec = Word2VecTrain()
    w2vec.load_word2vec()

