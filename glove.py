import os
import numpy as np
from data_loader import IMDBDataSet

class LoadGloVe():
    def __init__(self, words):
        super().__init__()
        self.glove_dir = '/home/danul-g/IMDB/glove.6B'
        # self.glove_dir = '/home/danul-g/IMDB/glove.6B'
        self.embeddings_index = {}
        self.word_index = words

    def load_glove(self,
                   max_words=10000, embedding_dim = 100):
        file_name = 'glove.6B.'+str(embedding_dim)+'d.txt'
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


if __name__ == '__main__':
    temp = IMDBDataSet()
    x, y, words = temp.load_data()
    glove = LoadGloVe(words)

    matrix = glove.load_glove()
    print(matrix.shape)
